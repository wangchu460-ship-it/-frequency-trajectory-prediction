"""Formal seed789 training for the frozen G0 ablation and external-baseline matrix."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from balance_common import SYSTEMS, atomic, load_data, make_batch, read, scene_losses, sha
from external_data import collate_external
from external_models import build_model as build_external, parameter_inventory
from spatial_metrics import metrics

ABLATIONS = ("A1_NO_BALANCE", "A2_NO_SPATIAL", "A3_F_ONLY", "A4_L_ONLY", "A5_NO_PHYSICAL", "A6_PRE_EVENT")
EXTERNALS = ("E1_GRAPHICAL_DEEPONET", "E2_GAT_H128", "E3_GCN_H128", "E4_UGCN", "E5_GAT_H384", "E6_GCN_H384")
ARMS = ABLATIONS + EXTERNALS
EXTERNAL_NAME = {
    "E1_GRAPHICAL_DEEPONET": "graphical_deeponet", "E2_GAT_H128": "gat_h128",
    "E3_GCN_H128": "gcn_h128", "E4_UGCN": "ugcn",
    "E5_GAT_H384": "gat_h384", "E6_GCN_H384": "gcn_h384",
}


def save_torch(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def rng_state(gens):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "samplers": {k: v.get_state() for k, v in gens.items()},
    }


def restore_rng(state, gens):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])
    for key, gen in gens.items(): gen.set_state(state["samplers"][key])


def transform_relation(items, relation, g, reuse, split):
    if relation == "POST_F": return items
    index = {(x["split"], str(x["sample_id"])): x for x in read(reuse / "controls" / "INPUT_INDEX.json")}
    result = []
    for item in items:
        entry = index[(split, str(item["id"]))]
        with np.load(reuse / "controls" / entry["cache"].replace("\\", "/"), allow_pickle=False) as z:
            result.append(g.apply_control(item, relation, z))
        result[-1]["_system"] = item["_system"]; result[-1]["_row"] = item["_row"]
        if "local_index" in item: result[-1]["local_index"] = item["local_index"]
    return result


def attach_external_paths(items, physical_root, split):
    for item in items:
        item["_physical_path"] = str(physical_root / split / f"{item['_system']}__{item['id']}.npz")
        if not Path(item["_physical_path"]).is_file(): raise FileNotFoundError(item["_physical_path"])


def build_g0(g, r, package, reuse, item, arm, device):
    stats = read(reuse / "formal" / "data" / "freeze" / "controller_normalization.json")
    initial = torch.load(package / "initial_seed789.pt", map_location="cpu", weights_only=False)
    internal = {"A3_F_ONLY": "NO_LAPLACIAN", "A4_L_ONLY": "NO_MESSAGE"}.get(arm, "FULL")
    return g.build(r, item["x"].shape[1], stats, 789, internal, initial, "cpu").to(device)


def model_prediction(model, data, external):
    value = model(data)
    return value if external else value["direct_trajectory"]


def main():
    p = argparse.ArgumentParser()
    for name in ("reuse", "package", "topology", "physical_root", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--target-epoch", type=int, choices=(1, 60, 70, 80), required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    device = "cuda:0"

    g, r, cfg, cache = load_data(a.reuse, a.package, a.topology, ("train", "validation"))
    train, _ = cache["train"]; validation, _ = cache["validation"]
    relation = "PRE_F" if a.arm == "A6_PRE_EVENT" else "NO_MESSAGE" if a.arm == "A5_NO_PHYSICAL" else "POST_F"
    train = transform_relation(train, relation, g, a.reuse, "train")
    validation = transform_relation(validation, relation, g, a.reuse, "validation")
    external = a.arm in EXTERNALS
    if external:
        attach_external_paths(train, a.physical_root, "train")
        attach_external_paths(validation, a.physical_root, "validation")
        # These bus-only artifacts are authoritative POST-event Y/branch inputs.
        audit = read(a.physical_root / "AUDIT_SUMMARY.json")
        assert audit["status"] == "PHYSICAL_INPUT_ARTIFACT_PASS" and audit["samples"] == 8000
        external_stats = read(a.physical_root / "NORMALIZATION.json")
    else:
        external_stats = None

    if a.smoke:
        train = [x for s in SYSTEMS for x in [v for v in train if v["_system"] == s][:4]]
        validation = [next(v for v in validation if v["_system"] == s) for s in SYSTEMS]
    by = {s: [x for x in train if x["_system"] == s] for s in SYSTEMS}
    assert len(by["IEEE39"]) == len(by["NPCC140"]) and (a.smoke or len(by["IEEE39"]) == 3500)
    steps = 1 if a.smoke else 875

    random.seed(789); np.random.seed(789); torch.manual_seed(789); torch.cuda.manual_seed_all(789)
    if external:
        model = build_external(EXTERNAL_NAME[a.arm], train[0]["x"].shape[1]).to(device)
        parameter_counts = parameter_inventory(model)
        init_source = "architecture-specific torch seed 789"
    else:
        model = build_g0(g, r, a.package, a.reuse, train[0], a.arm, device)
        parameter_counts = (sum(x.numel() for x in model.parameters() if x.requires_grad), sum(x.numel() for x in model.parameters() if not x.requires_grad), sum(x.numel() for x in model.buffers()))
        init_source = "exact G0 initial_seed789.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    base = read(a.package / "BASE_PROTOCOL.json")
    lambda_spatial = 0.0 if a.arm == "A2_NO_SPATIAL" else float(base["lambda_early"])
    sampler_mode = "random_mixed_7000" if a.arm == "A1_NO_BALANCE" else "strict_4_plus_4"
    gens = ({"MIXED": torch.Generator().manual_seed(1789000)} if a.arm == "A1_NO_BALANCE" else {
        "IEEE39": torch.Generator().manual_seed(1789039), "NPCC140": torch.Generator().manual_seed(1789140)})
    config = {
        "version": "PAPER_FORMAL_ABLATION_AND_BASELINES_V1", "arm": a.arm, "seed": 789,
        "batch_size": 8, "sampler": sampler_mode, "train_counts": {"IEEE39": len(by["IEEE39"]), "NPCC140": len(by["NPCC140"])},
        "optimizer_steps_per_epoch": steps, "lr": 1e-4, "weight_decay": 1e-4, "gradient_clip": 1.0,
        "precision": "FP32", "scheduler": None, "lambda_spatial": lambda_spatial,
        "loss_windows": [[0, 2], [2, 15]], "relation": relation,
        "initialization": init_source, "target_normalization": cfg["target_rms_hz"],
        "validation_n": len(validation), "selection": "frozen V2 U5 rule after e60; conditional +10 to max e80",
        "parameter_counts": {"trainable": parameter_counts[0], "frozen": parameter_counts[1], "buffers": parameter_counts[2]},
        "initial_seed789_sha256": sha(a.package / "initial_seed789.pt"),
        "physical_artifact_audit_sha256": sha(a.physical_root / "AUDIT_SUMMARY.json") if external else None,
        "test_reads": 0, "ood_reads": 0, "wecc_reads": 0,
    }
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "config.json").exists():
        if read(a.output / "config.json") != config: raise RuntimeError("existing run contract differs")
    else: atomic(a.output / "config.json", config)

    history, start = [], 1
    if (a.output / "last.pt").exists():
        ck = torch.load(a.output / "last.pt", map_location="cpu", weights_only=False)
        if ck["config"] != config: raise RuntimeError("resume contract differs")
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        history, start = ck["history"], ck["epoch"] + 1; restore_rng(ck["rng"], gens)

    def batch_g0(items): return make_batch(r, items, device)
    def batch_external(items): return collate_external(items, r, external_stats, device)
    batcher = batch_external if external else batch_g0

    def evaluate(epoch):
        dest = a.output / f"epoch_{epoch:03d}"; dest.mkdir(exist_ok=True)
        records = []; model.eval()
        with gzip.open(dest / "metrics.jsonl.gz", "wt", encoding="utf-8") as stream, torch.inference_mode():
            for pos in range(0, len(validation), 8):
                sub = validation[pos:pos + 8]; d, y, m, f, ids = batcher(sub)
                pred = model_prediction(model, d, external)
                pp, yy, mm, ff = [v.detach().cpu().numpy() for v in (pred, y, m, f)]
                for j, item in enumerate(sub):
                    row = item["_row"]; meta = {k: row[k] for k in ("system", "group_id", "dynamic_bin", "event", "composition")}
                    meta.update(sample_id=item["id"], run=a.arm, epoch=epoch); n = len(item["rows"])
                    rec = metrics(pp[j, :n], yy[j, :n], mm[j, :n], ff[j, :n], item["t"], meta, base["thresholds"]["spatial_" + row["system"]])
                    records.extend(rec)
                    for value in rec: stream.write(json.dumps(value, allow_nan=False) + "\n")
        summary = []
        for system in SYSTEMS:
            for window in ("0_2", "2_10", "10_30", "full"):
                specs = (("device", "differential_rmse", "HIGH"), ("device", "differential_rmse", "BELOW_HIGH"),
                         ("device", "nadir_error", "HIGH"), ("device", "max_abs_rocof_error", "HIGH"),
                         ("pair", "pair_rmse", "ALL"), ("spread", "spread_mae", "ALL"))
                for kind, endpoint, stratum in specs:
                    rows = [v for v in records if v["system"] == system and v["window"] == window and not v["dynamic_bin"].startswith("D3") and v["kind"] == kind and (v.get("stratum") == stratum if kind == "device" else v.get("pair") == "ALL" if kind == "pair" else True) and v[endpoint] is not None]
                    scenes, groups = {}, {}
                    for v in rows: scenes.setdefault((v["group_id"], v["sample_id"]), []).append(v[endpoint])
                    for (group, _), vals in scenes.items(): groups.setdefault(group, []).append(float(np.mean(vals)))
                    vals = [v[endpoint] for v in rows]
                    summary.append({"system": system, "window": window, "endpoint": endpoint, "stratum": stratum,
                                    "n": len(vals), "groups": len(groups), "median": float(np.median(vals)) if vals else None,
                                    "p90": float(np.quantile(vals, .9)) if vals else None,
                                    "group_macro_mean": float(np.mean([np.mean(v) for v in groups.values()])) if groups else None})
        atomic(dest / "summary.json", summary)
        atomic(dest / "DONE.json", {"epoch": epoch, "n_validation": len(validation), "metrics_sha256": sha(dest / "metrics.jsonl.gz"), "test_reads": 0, "ood_reads": 0, "wecc_reads": 0})

    all_train = train
    for epoch in range(start, a.target_epoch + 1):
        model.train(); losses, system_losses = [], defaultdict(list); tic = time.time()
        if sampler_mode == "random_mixed_7000":
            order = torch.randperm(len(all_train), generator=gens["MIXED"]).tolist()
            batches = [[all_train[i] for i in order[s * 8:(s + 1) * 8]] for s in range(steps)]
            order_ids = [all_train[i]["id"] for i in order]
        else:
            orders = {s: torch.randperm(len(by[s]), generator=gens[s]).tolist() for s in SYSTEMS}
            batches = [[by["IEEE39"][i] for i in orders["IEEE39"][s * 4:(s + 1) * 4]] + [by["NPCC140"][i] for i in orders["NPCC140"][s * 4:(s + 1) * 4]] for s in range(steps)]
            order_ids = {s: [by[s][i]["id"] for i in orders[s]] for s in SYSTEMS}
        for step, sub in enumerate(batches):
            torch.manual_seed(789000000 + epoch * 1000 + step); torch.cuda.manual_seed_all(789000000 + epoch * 1000 + step)
            d, y, m, f, ids = batcher(sub); optimizer.zero_grad(set_to_none=True)
            pred = model_prediction(model, d, external)
            old, spatial, total, _ = scene_losses(r, cfg, base["thresholds"], pred, y, m, f, d.phys_rollout_t, sub, lambda_spatial)
            if sampler_mode == "strict_4_plus_4":
                loss = .5 * total[:4].mean() + .5 * total[4:].mean()
            else:
                loss = total.mean()
            if not torch.isfinite(loss): raise RuntimeError(f"nonfinite loss epoch={epoch} step={step}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
            losses.append(float(loss.detach()))
            for system in SYSTEMS:
                idx = [i for i, item in enumerate(sub) if item["_system"] == system]
                if idx: system_losses[system].append(float(total[idx].mean().detach()))
            if step % 100 == 0:
                atomic(a.output / "STATUS.json", {"status": "TRAINING", "arm": a.arm, "epoch": epoch, "step": step, "steps": steps, "target_epoch": a.target_epoch})
                print(a.arm, epoch, step, steps, round(time.time() - tic, 1), flush=True)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "system_loss_mean": {s: float(np.mean(system_losses[s])) for s in SYSTEMS},
               "seconds": time.time() - tic, "batch_order_sha256": hashlib.sha256(json.dumps(order_ids, sort_keys=True).encode()).hexdigest()}
        history.append(row)
        save_torch(a.output / f"epoch_{epoch:03d}_model.pt", {"model": model.state_dict(), "completed_epoch": epoch, "config": config})
        atomic(a.output / "history.json", history)
        saved_cpu = torch.get_rng_state(); saved_cuda = torch.cuda.get_rng_state_all()
        evaluate(epoch); torch.set_rng_state(saved_cpu); torch.cuda.set_rng_state_all(saved_cuda)
        save_torch(a.output / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                   "history": history, "rng": rng_state(gens), "config": config})
        atomic(a.output / "STATUS.json", {"status": "EPOCH_COMPLETE", "arm": a.arm, "epoch": epoch, "target_epoch": a.target_epoch})
    print("TARGET_REACHED", a.arm, a.target_epoch, flush=True)


if __name__ == "__main__": main()

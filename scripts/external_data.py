"""Train/Validation-only data adapter for the external baselines.

The neural inputs are assembled from the frozen compact prospective sample and
an input-only physical artifact.  Targets and masks remain separate tensors and
are never attached to the object passed to a model.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PARAMS = ("M", "D", "R", "T1", "T2", "T3", "Dt", "m_p", "w_pf", "w_pll", "pll_zeta")
FLAGS = ("has_gov", "is_gfm", "is_gfl")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def load_formal(formal_root: Path):
    formal_root = Path(formal_root).resolve()
    if not hasattr(hashlib, "file_digest"):
        def _file_digest(stream, name):
            digest = hashlib.new(name)
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            return digest
        hashlib.file_digest = _file_digest
    sys.path.insert(0, str(formal_root))
    import runner as formal_runner

    _, train, validation = formal_runner.contract()
    return formal_runner, train, validation


class ExternalDataset(torch.utils.data.Dataset):
    def __init__(self, formal_dataset, artifact_root: Path, split: str):
        self.formal = formal_dataset
        self.rows = formal_dataset.rows
        self.stats = formal_dataset.stats
        self.artifact_root = Path(artifact_root)
        self.split = split

    def __len__(self):
        return len(self.formal)

    def __getitem__(self, index):
        item = self.formal[index]
        item["_manifest_row"] = self.rows[index]
        system = str(self.rows[index]["system"])
        item["_physical_path"] = str(self.artifact_root / self.split / f"{system}__{item['id']}.npz")
        return item


def controller_features(data: SimpleNamespace) -> torch.Tensor:
    numeric = torch.stack([getattr(data, "phys_gad_" + name).float() for name in PARAMS], dim=-1)
    known = data.controller_known.to(torch.bool)
    numeric = torch.where(known, numeric, torch.zeros_like(numeric))
    numeric = torch.sign(numeric) * torch.log1p(numeric.abs())
    flags = torch.stack([getattr(data, "phys_gad_" + name).float() for name in FLAGS], dim=-1)
    flags = torch.nan_to_num(flags, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.cat([numeric, flags], dim=-1)


def _load_physical(path: str, edge_mean: np.ndarray, edge_std: np.ndarray):
    with np.load(path, allow_pickle=False) as z:
        edge_attr = z["branch_edge_attr"].astype(np.float32)
        edge_attr[:, :2] = (edge_attr[:, :2] - edge_mean) / edge_std
        n_bus = int(z["n_bus"])
        gso = np.zeros((n_bus, n_bus), dtype=np.complex64)
        ix = z["gso_index"]
        gso[ix[0], ix[1]] = z["gso_value"]
        return {
            "edge_index": torch.from_numpy(z["branch_edge_index"].astype(np.int64)),
            "edge_attr": torch.from_numpy(edge_attr),
            "gso": torch.from_numpy(gso),
            "terminal": torch.from_numpy(z["terminal_bus_index"].astype(np.int64)),
            "bus_ids": z["bus_ids"].astype(np.int64),
            "input_sha256": str(z["input_sha256"]),
        }


def collate_external(items, formal_module, artifact_stats: dict, device: str):
    # Formal batching owns labels/masks; extras are ignored by it.
    data, y, mask, family, ids = formal_module.m.batching(items)
    edge_mean = np.asarray(artifact_stats["branch_gb_mean"], dtype=np.float32)
    edge_std = np.asarray(artifact_stats["branch_gb_std"], dtype=np.float32)
    physical = [_load_physical(item["_physical_path"], edge_mean, edge_std) for item in items]

    bus_x, bus_batch, terminal, edge_index, edge_attr = [], [], [], [], []
    offset = 0
    for graph, (item, phy) in enumerate(zip(items, physical)):
        x = torch.from_numpy(item["x"].astype(np.float32))
        bus = torch.from_numpy((item["type"] < 0))
        bx = x[bus]
        if len(bx) != len(phy["bus_ids"]):
            raise RuntimeError(f"physical bus mismatch for {ids[graph]}")
        bus_x.append(bx)
        bus_batch.append(torch.full((len(bx),), graph, dtype=torch.long))
        terminal.append(phy["terminal"] + offset)
        edge_index.append(phy["edge_index"] + offset)
        edge_attr.append(phy["edge_attr"])
        offset += len(bx)

    data.bus_x = torch.cat(bus_x)
    data.bus_batch = torch.cat(bus_batch)
    data.bus_ptr = torch.tensor([0] + list(np.cumsum([len(x) for x in bus_x])), dtype=torch.long)
    data.bus_edge_index = torch.cat(edge_index, dim=1)
    data.bus_edge_attr = torch.cat(edge_attr, dim=0)
    max_dev = y.shape[1]
    data.terminal_bus_row = torch.zeros((len(items), max_dev), dtype=torch.long)
    for i, row in enumerate(terminal):
        data.terminal_bus_row[i, : len(row)] = row
    data.controller_features = controller_features(data)
    data.gso = [p["gso"] for p in physical]
    data.input_artifact_sha256 = [p["input_sha256"] for p in physical]

    def move_value(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, list) and value and torch.is_tensor(value[0]):
            return [v.to(device) for v in value]
        return value

    data = SimpleNamespace(**{name: move_value(value) for name, value in vars(data).items()})
    return data, y.to(device), mask.to(device), family.to(device), ids


def permute_bus_input(data: SimpleNamespace, permutation: torch.Tensor) -> SimpleNamespace:
    """Consistently relabel a single-graph bus-only input."""
    if int(data.bus_ptr[-1]) != len(permutation) or len(data.gso) != 1:
        raise ValueError("bus permutation test requires one graph")
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation), device=permutation.device)
    values = dict(vars(data))
    values["bus_x"] = data.bus_x[permutation]
    values["bus_edge_index"] = inverse[data.bus_edge_index]
    values["terminal_bus_row"] = inverse[data.terminal_bus_row]
    values["gso"] = [data.gso[0][permutation][:, permutation]]
    return SimpleNamespace(**values)


def permute_devices(data: SimpleNamespace, permutation: torch.Tensor) -> SimpleNamespace:
    """Permute valid device queries without touching the encoded graph."""
    if data.phys_gad_dev_mask.shape[0] != 1:
        raise ValueError("device permutation test requires one graph")
    n = int(data.phys_gad_dev_mask[0].sum())
    if len(permutation) != n:
        raise ValueError("permutation length differs from valid device count")
    values = dict(vars(data))
    values["terminal_bus_row"] = data.terminal_bus_row.clone()
    values["terminal_bus_row"][0, :n] = data.terminal_bus_row[0, permutation]
    values["phys_dev_node_row"] = data.phys_dev_node_row.clone()
    values["phys_dev_node_row"][0, :n] = data.phys_dev_node_row[0, permutation]
    values["controller_features"] = data.controller_features.clone()
    values["controller_features"][0, :n] = data.controller_features[0, permutation]
    return SimpleNamespace(**values)

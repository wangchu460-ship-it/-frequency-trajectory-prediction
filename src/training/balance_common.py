"""Frozen data, loss, batching, and checkpoint helpers for cross-system balance V1."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

SYSTEMS = ("IEEE39", "NPCC140")
FAMILIES = ("SG", "GFM", "GFL")

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def atomic(path, obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    tmp.replace(path)

def runtime(reuse: Path, package: Path):
    sys.path.insert(0,str(package.resolve())); import run as g
    sys.path.insert(0,str((reuse/"formal").resolve())); import runner as r
    r.sha=g.sha
    return g,r

def topology_file(topology: Path, entry):
    candidate=topology/Path(entry["cache"]).name if Path(entry["cache"]).is_absolute() else topology/entry["cache"]
    return candidate if candidate.exists() else Path(entry["cache"])

def load_data(reuse: Path, package: Path, topology: Path, splits=("train","validation"), limit=None):
    g,r=runtime(reuse,package)
    cfg,tr,va=r.contract()
    g.verify(reuse/"formal",reuse/"controls",r.read(package/"REPAIRED_F_REUSE_LOCK.json"))
    topo=r.read(topology/"AUDIT.json"); assert topo["samples"]==8000
    topix={(v["split"],str(v["sample_id"])):v for v in topo["records"]}
    control={(v["split"],str(v["sample_id"])):v for v in r.read(reuse/"controls/INPUT_INDEX.json")}
    cache={}
    for split,ds in (("train",tr),("validation",va)):
        if split not in splits: continue
        for row in ds.rows: row["prepared"]=row["prepared"].replace("\\","/")
        rows=ds.rows if limit is None else ds.rows[:limit]
        items=[]
        for k,row in enumerate(rows):
            item=ds[k]; key=(split,str(item["id"])); ce=control[key]
            with np.load(reuse/"controls"/ce["cache"].replace("\\","/")) as z:
                item=g.apply_control(item,"POST_F",z)
            te=topix[key]; tf=topology_file(topology,te); assert sha(tf)==te["cache_sha256"]
            with np.load(tf) as z: item["local_index"]=z["local_index"].copy()
            item["_system"]=row["system"]; item["_row"]=row; items.append(item)
        cache[split]=(items,ds)
    return g,r,cfg,cache

def make_batch(r,items,device):
    d,y,m,f,ids=r.m.batching(items)
    d.local_index=torch.as_tensor(np.concatenate([v["local_index"]+int(d.ptr[i]) for i,v in enumerate(items)],1),dtype=torch.long)
    assert torch.equal(d.batch[d.local_index[0]],d.batch[d.local_index[1]])
    return r.m.move((d,y,m,f,ids),device)

def scene_losses(r,cfg,thresholds,p,y,m,f,t,items,lambda_spatial):
    """Exact old loss content, returned before the historical across-scene mean."""
    n=m.sum(1).clamp_min(1); yc=torch.where(m,y,0).sum(1)/n; pc=torch.where(m,p,0).sum(1)/n
    yd=y-yc[:,None]; pd=p-pc[:,None]
    dt=t[:,1:]-t[:,:-1]; w=torch.zeros_like(t); w[:,:-1]+=dt/2; w[:,1:]+=dt/2
    energy=(yd.square()*m*w[:,None]).sum(-1)/(m*w[:,None]).sum(-1).clamp_min(1e-12)
    cutoff=torch.tensor([thresholds["spatial_"+i["_system"]] for i in items],device=y.device)
    high=(energy.sqrt()>=cutoff[:,None])&m.any(-1)
    old=r.m.scenario_mse(p,y,m,f,t)/cfg["target_rms_hz"]**2
    pieces=[]
    for lo,hi in ((0,2),(2,15)):
        mk=m&high[:,:,None]&((t>=lo)&(t<=hi))[:,None]
        pieces.append(r.m.scenario_mse(pd,yd,mk,f,t.clamp(lo,hi))/cfg["target_rms_hz"]**2)
    spatial=(pieces[0]+pieces[1])/2
    return old,spatial,old+lambda_spatial*spatial,high

def build_model(g,r,reuse,package,items,arm,seed,checkpoint,device):
    from local_path import install
    stats=r.read(reuse/"formal/data/freeze/controller_normalization.json")
    initial=torch.load(package/f"initial_seed{seed}.pt",map_location="cpu",weights_only=False)
    model=g.build(r,items[0]["x"].shape[1],stats,seed,"FULL",initial,"cpu")
    model=install(model,arm,seed)
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model.load_state_dict(ck["model"])
    return model.to(device)

def module_of(name):
    if ".local_path." in name: return "local_path"
    if any(x in name for x in (".phy_","eta_lap")): return "physical_graph_path"
    if any(x in name for x in ("node_encoder","edge_encoder","global_context_encoder","direct_device_encoder","direct_time_encoder")):
        return "input_encoder"
    if "direct_decoder" in name: return "decoder_output"
    return "shared_graph_other"

def state_sha(state, names=None):
    h=hashlib.sha256()
    for name,v in sorted(state.items()):
        if names is None or name in names:
            h.update(name.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

from __future__ import annotations

import argparse, csv, gzip, hashlib, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PARAMS = ("M","D","R","T1","T2","T3","Dt","m_p","w_pf","w_pll","pll_zeta")
ARMS = ("G0_BALANCED","A1_NO_BALANCE","A2_NO_SPATIAL","A3_F_ONLY","A4_L_ONLY","A5_NO_PHYSICAL","A6_PRE_EVENT",
        "E1_GRAPHICAL_DEEPONET","E2_GAT_H128","E3_GCN_H128","E4_UGCN","E5_GAT_H384","E6_GCN_H384")

def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def atomic(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp")
    q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8");os.replace(q,p)

def remap_archived_path(text, archive_session):
    p=Path(str(text))
    if p.is_file(): return p
    marker="f-wang-gnn-codex-archive-20260730-4"
    parts=list(p.parts)
    for i,v in enumerate(parts):
        if v.lower()==marker:
            q=Path(archive_session,*parts[i+1:])
            if q.is_file(): return q
    return p

def attachment_rows(item):
    typ=item["type"]; src,dst=item["edge_index"]; buses=np.flatnonzero(typ<0); bset=set(map(int,buses)); out=[]
    for dev in item["rows"]:
        hits=[]
        hits += [int(v) for v in dst[src==dev] if int(v) in bset]
        hits += [int(v) for v in src[dst==dev] if int(v) in bset]
        hits=sorted(set(hits))
        if len(hits)!=1: raise RuntimeError(f"device terminal mismatch {item['id']} dev={dev} hits={hits}")
        out.append(hits[0])
    return np.asarray(out,np.int64), buses

def build_jacobian(V,theta,Y):
    V=np.asarray(V,float);theta=np.asarray(theta,float);Y=np.asarray(Y,complex);G=Y.real;B=Y.imag
    d=theta[:,None]-theta[None,:];c=np.cos(d);s=np.sin(d);vv=np.outer(V,V)
    H=vv*(G*s-B*c);N=V[:,None]*(G*c+B*s);J=-vv*(G*c+B*s);L=V[:,None]*(G*s-B*c)
    P=(vv*(G*c+B*s)).sum(1);Q=(vv*(G*s-B*c)).sum(1);i=np.arange(len(V))
    H[i,i]=-Q-V*V*np.diag(B);N[i,i]=P/V+V*np.diag(G);J[i,i]=P-V*V*np.diag(G);L[i,i]=Q/V-V*np.diag(B)
    return H,N,J,L

def reduce_angle(H,N,J,L,keep):
    n=len(H); keep=np.asarray(keep,int); drop=np.asarray([i for i in range(n) if i not in set(keep)],int)
    if not len(drop): return H[np.ix_(keep,keep)]
    A=H[np.ix_(keep,keep)]; B=np.hstack((H[np.ix_(keep,drop)],N[np.ix_(keep,drop)]))
    C=np.vstack((H[np.ix_(drop,keep)],J[np.ix_(drop,keep)]))
    D=np.block([[H[np.ix_(drop,drop)],N[np.ix_(drop,drop)]],[J[np.ix_(drop,drop)],L[np.ix_(drop,drop)]]])
    return np.real(A-B@np.linalg.solve(D,C))

def matrix_to_relation(F,rows):
    src=[];dst=[];raw=[]
    for i in range(len(F)):
        for j in range(len(F)):
            if i!=j and np.isfinite(F[i,j]) and abs(F[i,j])>0:
                src.append(rows[j]);dst.append(rows[i]);raw.append(F[i,j])
    src=np.asarray(src,np.int64);dst=np.asarray(dst,np.int64);raw=np.asarray(raw,np.float32);ab=np.abs(raw)
    deg={}
    for d,w in zip(dst,ab):deg[int(d)]=deg.get(int(d),0.)+float(w)
    norm=np.asarray([w/np.sqrt(max(deg.get(int(s),0.),1e-12)*max(deg.get(int(d),0.),1e-12)) for s,d,w in zip(src,dst,ab)],np.float32)
    attr=np.stack((raw,ab,np.sign(raw),norm,np.zeros_like(raw),np.zeros_like(raw)),1).astype(np.float32)
    return np.vstack((src,dst)),attr

def pre_relation(item,base):
    terminal_rows,bus_rows=attachment_rows(item); nbus=len(bus_rows)
    bus_ids=np.asarray(base["bus_ids"],int); physical_ids=bus_ids[:nbus]
    if len(physical_ids)!=nbus: raise RuntimeError("physical bus count mismatch")
    source={int(v):i for i,v in enumerate(bus_ids)}; pi=np.asarray([source[int(v)] for v in physical_ids])
    terminal_ids=physical_ids[terminal_rows]
    unique=np.asarray(sorted(set(map(int,terminal_ids)))); prow={int(v):i for i,v in enumerate(physical_ids)}
    keep=np.asarray([prow[int(v)] for v in unique]); Y=np.asarray(base["Y_pre"])[np.ix_(pi,pi)]
    H,N,J,L=build_jacobian(np.asarray(base["V0"])[pi],np.asarray(base["theta0"])[pi],Y)
    red=reduce_angle(H,N,J,L,keep); ratings=np.asarray(base["device_rating_mva"],float); active=np.asarray(base.get("device_post_active",np.ones(len(ratings))),bool)
    C=np.zeros((len(unique),len(terminal_ids))); urow={int(v):i for i,v in enumerate(unique)}
    for j,v in enumerate(terminal_ids):
        coloc=np.flatnonzero(terminal_ids==v); C[urow[int(v)],j]=ratings[j]/ratings[coloc].sum()
    F=C.T@red@C; F=np.diag(active.astype(float))@F@np.diag(active.astype(float))
    return matrix_to_relation(F,np.asarray(item["rows"],int))

def build_external_artifact(item,base):
    terminal,bus_rows=attachment_rows(item); nbus=len(bus_rows); ids=np.asarray(base["bus_ids"],int); physical=ids[:nbus]
    row={int(v):i for i,v in enumerate(physical)}; ei=np.asarray(base["edge_index"],int);ef=np.asarray(base["edge_features"],np.float32)
    active=(ef[:,13]>0.5) if ef.shape[1]>13 else np.ones(len(ei),bool)
    selected=[k for k,(u,v) in enumerate(ei) if active[k] and int(u) in row and int(v) in row]
    src=[];dst=[];attrs=[]
    for k in selected:
        u,v=map(int,ei[k]);a=np.asarray([ef[k,5],ef[k,6],1.],np.float32)
        src += [row[u],row[v]];dst += [row[v],row[u]];attrs += [a,a]
    source={int(v):i for i,v in enumerate(ids)}; keep=np.asarray([source[int(v)] for v in physical]);drop=np.asarray([i for i in range(len(ids)) if i not in set(keep)])
    Y=np.asarray(base["Y_post"],np.complex128)
    if len(drop):
        D=Y[np.ix_(drop,drop)];Yred=Y[np.ix_(keep,keep)]-Y[np.ix_(keep,drop)]@np.linalg.pinv(D)@Y[np.ix_(drop,keep)]
    else:Yred=Y[np.ix_(keep,keep)]
    ix=np.vstack(np.nonzero(np.abs(Yred)>1e-7)).astype(np.int64); val=Yred[ix[0],ix[1]].astype(np.complex64)
    return dict(bus_ids=physical.astype(np.int64),n_bus=np.asarray(nbus,np.int64),branch_edge_index=np.vstack((src,dst)).astype(np.int64),
                branch_edge_attr=np.asarray(attrs,np.float32),gso_index=ix,gso_value=val,terminal_bus_index=terminal.astype(np.int64),input_sha256=np.asarray("runtime_rebuilt_from_frozen_base"))

class FrozenDataset:
    def __init__(self,split,root,rows,paths,bases,stats): self.split=split;self.root=Path(root);self.rows=rows;self.paths=paths;self.bases=bases;self.stats=stats
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with np.load(self.paths[i],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        a["x"]=(a["x"]-np.asarray(self.stats["mean"]))/np.asarray(self.stats["std"]);a["id"]=str(self.rows[i]["sample_id"]);a["_row"]=self.rows[i];a["_system"]=self.rows[i].get("system","WECC179");a["_base_path"]=str(self.bases[i]);return a

def load_datasets(a):
    test_rows=read(a.test_root/"freeze/test.json"); wecc_rows=read(a.wecc_root/"freeze/wecc1600.json")
    old=a.archive_session; test_paths=[];test_bases=[]
    for r in test_rows:
        p=remap_archived_path(r["prepared"],old)
        if not p.is_file(): p=a.test_root/"prepared/test"/f"{r['sample_id']}.npz"
        if not p.is_file() or sha(p)!=r["prepared_sha256"]:raise RuntimeError(f"test prepared contract {r['sample_id']} {p}")
        b=remap_archived_path(r["base_file"],old)
        if not b.is_file() or (r.get("base_sha256") and sha(b)!=r["base_sha256"]):raise RuntimeError(f"test base contract {r['sample_id']} {b}")
        test_paths.append(p);test_bases.append(b)
    wecc_paths=[];wecc_bases=[]
    for r in wecc_rows:
        p=a.wecc_root/r["prepared"]
        b=a.wecc_root.parent/"WECC179_REPAIR_AND_USABLE_FREEZE_V1"/"converted"/str(r["sample_id"])/"base.npz"
        if not p.is_file() or sha(p)!=r["prepared_sha256_audit"]:raise RuntimeError(f"wecc prepared contract {r['sample_id']}")
        # This freeze records a FILE hash of the historical converted base.
        # A same-ID candidate-pool graph is not an interchangeable substitute.
        if not b.is_file() or sha(b)!=r["base_sha256"]:raise RuntimeError(f"wecc base contract {r['sample_id']}")
        r=dict(r);r["group_id"]=r["input_group_id"];r["dynamic_bin"]=r["evaluation_stratum"];r["composition"]=r["composition"].upper().replace("_","+");r["system"]="WECC179"
        wecc_paths.append(p);wecc_bases.append(b)
    stats=read(a.reuse/"formal/data/freeze/normalization.json")
    return FrozenDataset("test",a.test_root,test_rows,test_paths,test_bases,stats),FrozenDataset("wecc1600",a.wecc_root,wecc_rows,wecc_paths,wecc_bases,stats)

def materialize(ds,relation,external_cache=None):
    out=[]
    for i in range(len(ds)):
        item=ds[i]
        if relation=="NO_MESSAGE":item["phy_index"]=np.zeros((2,0),np.int64);item["phy_attr"]=np.zeros((0,6),np.float32)
        if relation=="PRE_F":
            with np.load(item["_base_path"],allow_pickle=True) as z:item["phy_index"],item["phy_attr"]=pre_relation(item,z)
        if external_cache is not None:
            p=external_cache/f"{item['_system']}__{item['id']}.npz";p.parent.mkdir(parents=True,exist_ok=True)
            if not p.is_file():
                with np.load(item["_base_path"],allow_pickle=True) as z:np.savez_compressed(p,**build_external_artifact(item,z))
            item["_physical_path"]=str(p)
        out.append(item)
        if (i+1)%100==0:print("LOAD",ds.split,relation,i+1,flush=True)
    return out

def model_spec(a):
    sel=read(a.selection_file or a.formal_results/"analysis/STABLE_SELECTION_V2.json")["models"]; specs={}
    for arm,v in sel.items():specs[arm]=(int(v["stable_best_epoch"]),a.formal_results/arm/f"epoch_{int(v['stable_best_epoch']):03d}_model.pt",v["boundary_status"])
    gsel=read(a.g0_results/"analysis/BALANCED_STABLE_SELECTION_V2.json")["models"]["G0"]
    specs["G0_BALANCED"]=(int(gsel["stable_best_epoch"]),a.g0_results/"G0_BALANCED"/f"epoch_{int(gsel['stable_best_epoch']):03d}_model.pt",gsel["boundary_status"])
    for arm,(ep,p,status) in specs.items():
        if not p.is_file():raise FileNotFoundError((arm,p))
    return specs

def create_model(arm,item,a,g,r,ft,em,device,ckpath):
    external=arm.startswith("E")
    if external:model=em.build_model(ft.EXTERNAL_NAME[arm],item["x"].shape[1]).to(device)
    elif arm=="G0_BALANCED":
        stats=read(a.reuse/"formal/data/freeze/controller_normalization.json");initial=torch.load(a.runtime/"initial_seed789.pt",map_location="cpu",weights_only=False)
        model=g.build(r,item["x"].shape[1],stats,789,"FULL",initial,"cpu")
        sys.path.insert(0,str(a.balance_package));from local_path import install
        model=install(model,"G0",789).to(device)
    else:model=ft.build_g0(g,r,a.runtime,a.reuse,item,arm,device)
    ck=torch.load(ckpath,map_location="cpu",weights_only=False);model.load_state_dict(ck["model"],strict=True);return model.eval(),external

def run_model(arm,epoch,ckpath,ds,a,g,r,ft,em,ed,sm,device,relation_override=None,result_arm=None):
    relation=relation_override or ("PRE_F" if arm=="A6_PRE_EVENT" else "NO_MESSAGE" if arm=="A5_NO_PHYSICAL" else "POST_F")
    ext=arm.startswith("E");needs_physical=arm in ("E1_GRAPHICAL_DEEPONET","E4_UGCN")
    items=materialize(ds,relation,a.output/"physical_cache"/ds.split if needs_physical else None)
    model,external=create_model(arm,items[0],a,g,r,ft,em,device,ckpath);result_arm=result_arm or arm;dest=a.output/result_arm/ds.split;dest.mkdir(parents=True,exist_ok=True)
    records=[];tic=time.time()
    with gzip.open(dest/"metrics.jsonl.gz","wt",encoding="utf-8") as stream,torch.inference_mode():
        for pos in range(0,len(items),a.batch):
            sub=items[pos:pos+a.batch]
            if external and needs_physical:
                d,y,m,f,ids=ed.collate_external(sub,r,read(a.physical_stats),device)
            else:
                d,y,m,f,ids=r.m.move(r.m.batching(sub),device)
                if external:d.controller_features=ed.controller_features(d)
            if arm=="G0_BALANCED":d.local_index=torch.zeros((2,0),dtype=torch.long,device=device)
            pred=ft.model_prediction(model,d,external); arrays=[v.detach().cpu().numpy() for v in (pred,y,m,f)]
            for j,item in enumerate(sub):
                row=item["_row"];meta=dict(system=item["_system"],group_id=row.get("group_id",row.get("input_group_id")),dynamic_bin=row.get("dynamic_bin",row.get("evaluation_stratum","")),event=row.get("event",row.get("event_family_actual","")),composition=row.get("composition",""),sample_id=item["id"],run=result_arm,epoch=epoch,dataset=ds.split,evaluation_stratum=row.get("evaluation_stratum"),frozen_input_group_weight=float(row.get("frozen_input_group_weight",1.)))
                threshold=a.wecc_threshold if ds.split=="wecc1600" else a.threshold_39 if item["_system"]=="IEEE39" else a.threshold_140
                pp,yy,mm,ff=(arrays[0][j,:len(item["rows"])],arrays[1][j,:len(item["rows"])],
                             arrays[2][j,:len(item["rows"])],arrays[3][j,:len(item["rows"])])
                rec=sm.metrics(pp,yy,mm,ff,item["t"],meta,threshold)
                # Preserve the ordinary, uncentered trajectory RMSE required by
                # the paper audit alongside spatial_metrics' centered error.
                windows={"full":(0.,30.),"0_2":(0.,2.),"2_10":(2.,10.),"10_30":(10.,30.)}
                for value in rec:
                    if value["kind"]!="device":continue
                    lo,hi=windows[value["window"]];sel=(item["t"]>=lo)&(item["t"]<=hi)
                    valid=mm[value["device_index"],sel].astype(bool);err=pp[value["device_index"],sel]-yy[value["device_index"],sel]
                    value["trajectory_rmse"]=sm.root(sm.weighted(err**2,valid,sm.weights(item["t"][sel].astype(float))))
                records+=rec
                for value in rec:stream.write(json.dumps(value,allow_nan=False)+"\n")
            if (pos//a.batch+1)%25==0:print("EVAL",arm,ds.split,min(pos+a.batch,len(items)),len(items),round(time.time()-tic,1),flush=True)
    atomic(dest/"DONE.json",dict(arm=result_arm,source_model_arm=arm,relation_intervention=relation,epoch=epoch,checkpoint=str(ckpath),checkpoint_sha256=sha(ckpath),dataset=ds.split,n_samples=len(items),n_records=len(records),seconds=time.time()-tic,metrics_sha256=sha(dest/"metrics.jsonl.gz")))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--test-root",type=Path,required=True);p.add_argument("--wecc-root",type=Path,required=True);p.add_argument("--wecc-base-root",type=Path,required=True);p.add_argument("--wecc-raw-root",type=Path);p.add_argument("--archive-session",type=Path,required=True)
    p.add_argument("--formal-results",type=Path,required=True);p.add_argument("--g0-results",type=Path,required=True);p.add_argument("--source-package",type=Path,required=True);p.add_argument("--balance-package",type=Path,required=True);p.add_argument("--reuse",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--selection-file",type=Path)
    p.add_argument("--device",default="cuda:0");p.add_argument("--batch",type=int,default=8);p.add_argument("--arms",nargs="*",default=list(ARMS));p.add_argument("--splits",nargs="*",default=["test","wecc1600"]);p.add_argument("--limit",type=int,default=0)
    p.add_argument("--threshold-39",type=float,default=.0184240117377);p.add_argument("--threshold-140",type=float,default=.012875054017);p.add_argument("--wecc-threshold",type=float,default=0.0)
    a=p.parse_args();a.runtime=a.source_package/"g0_runtime";a.physical_stats=a.source_package/"physical_inputs/NORMALIZATION.json";a.output.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(a.source_package),str(a.runtime),str((a.reuse/"formal").resolve())]
    import formal_train as ft, external_models as em, external_data as ed, spatial_metrics as sm, balance_common as bc
    g,r=bc.runtime(a.reuse,a.runtime); specs=model_spec(a);test,wecc=load_datasets(a)
    if a.limit:test.rows=test.rows[:a.limit];test.paths=test.paths[:a.limit];test.bases=test.bases[:a.limit];wecc.rows=wecc.rows[:a.limit];wecc.paths=wecc.paths[:a.limit];wecc.bases=wecc.bases[:a.limit]
    atomic(a.output/"EVALUATION_CONTRACT.json",dict(test_manifest=str(a.test_root/"freeze/test.json"),test_n=len(test),wecc_manifest=str(a.wecc_root/"freeze/wecc1600.json"),wecc_n=len(wecc),checkpoint_selection="Validation-only frozen V2",wecc_high_policy="No target-derived WECC threshold; threshold=0 labels all valid devices for common metric machinery",test_thresholds={"IEEE39":a.threshold_39,"NPCC140":a.threshold_140},arms={k:{"epoch":v[0],"checkpoint":str(v[1]),"boundary_status":v[2]} for k,v in specs.items()}))
    for arm in a.arms:
        ep,ck,_=specs[arm]
        for ds in (test,wecc):
            if ds.split not in a.splits:continue
            done=a.output/arm/ds.split/"DONE.json"
            if done.is_file():
                prior=read(done)
                if prior["epoch"]!=ep or prior["checkpoint_sha256"]!=sha(ck) or prior["n_samples"]!=len(ds) or prior["metrics_sha256"]!=sha(done.parent/"metrics.jsonl.gz"):
                    raise RuntimeError(f"Stale/incomplete result must be archived before evaluation: {done}")
                print("SKIP",arm,ds.split,flush=True);continue
            run_model(arm,ep,ck,ds,a,g,r,ft,em,ed,sm,a.device)

if __name__=="__main__":main()

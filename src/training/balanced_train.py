"""System-balanced G0/LOCAL training; architecture and loss content remain frozen."""
import argparse,gzip,json,os,time
from pathlib import Path
import numpy as np
import torch
from balance_common import *
from spatial_metrics import metrics

def main():
    p=argparse.ArgumentParser()
    for x in ("reuse","package","topology","output"): p.add_argument("--"+x,required=True,type=Path)
    p.add_argument("--arm",choices=("G0","LOCAL"),required=True); p.add_argument("--seed",type=int,default=789)
    p.add_argument("--target-epoch",type=int,required=True); p.add_argument("--smoke",action="store_true")
    a=p.parse_args(); assert a.seed==789; assert 1<=a.target_epoch<=80
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"; torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    g,r,cfg,cache=load_data(a.reuse,a.package,a.topology,("train","validation"))
    train,tr=cache["train"]; validation,va=cache["validation"]; base=read(a.package/"BASE_PROTOCOL.json")
    if a.smoke:
        train=[x for s in SYSTEMS for x in [v for v in train if v["_system"]==s][:4]]
        validation=[next(v for v in validation if v["_system"]==s) for s in SYSTEMS]
    by={s:[x for x in train if x["_system"]==s] for s in SYSTEMS}; assert all(by.values())
    old_steps=875 if not a.smoke else 1
    contract=read(Path(__file__).parent/"BALANCED_TRAINING_CONTRACT.json")
    assert contract["optimizer_steps_per_epoch"]==875 and contract["batch"]["IEEE39"]==4 and contract["batch"]["NPCC140"]==4
    a.output.mkdir(parents=True,exist_ok=True)
    immutable=dict(version="CROSS_SYSTEM_BALANCE_AND_EVIDENCE_V1",arm=a.arm,seed=a.seed,batch_size=8,per_system_batch=4,
        optimizer_steps_per_epoch=old_steps,lr=1e-4,weight_decay=base["weight_decay"],gradient_clip=1.0,precision="FP32",
        lambda_spatial=base["lambda_early"],loss_windows=[[0,2],[2,15]],post_f=True,train_n=len(train),validation_n=len(validation),
        model_init_stream=789,sampler_streams={"IEEE39":1789039,"NPCC140":1789140},dropout_stream="789000000+epoch*1000+step",
        initial_sha256=sha(a.package/"initial_seed789.pt"),topology_sha256=sha(a.topology/"AUDIT.json"),test_reads=0,ood_reads=0,wecc_reads=0)
    if (a.output/"config.json").exists(): assert read(a.output/"config.json")==immutable
    else: atomic(a.output/"config.json",immutable)
    model=build_model(g,r,a.reuse,a.package,train,a.arm,a.seed,a.package/"INIT_BALANCE_PLACEHOLDER.pt",device) if False else None
    # Build from the single frozen base initialization; LOCAL construction uses fork_rng and cannot alter shared tensors.
    from local_path import install
    stats=r.read(a.reuse/"formal/data/freeze/controller_normalization.json")
    initial=torch.load(a.package/"initial_seed789.pt",map_location="cpu",weights_only=False)
    model=g.build(r,train[0]["x"].shape[1],stats,a.seed,"FULL",initial,"cpu"); model=install(model,a.arm,a.seed).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=base["weight_decay"])
    gens={s:torch.Generator().manual_seed(1789039 if s=="IEEE39" else 1789140) for s in SYSTEMS}
    history=[]; start=1
    if (a.output/"last.pt").exists():
        ck=torch.load(a.output/"last.pt",map_location="cpu",weights_only=False); assert ck["config"]==immutable
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); history=ck["history"]; start=ck["epoch"]+1
        for s in SYSTEMS: gens[s].set_state(ck["sampler_rng"][s])
        torch.set_rng_state(ck["torch_rng"])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(ck["cuda_rng"])
    def queue(system):
        q=[]; repeats=0; cycles=0
        while len(q)<old_steps*4:
            order=torch.randperm(len(by[system]),generator=gens[system]).tolist(); cycles+=1
            take=min(len(order),old_steps*4-len(q)); q.extend(order[:take]); repeats+=max(0,take-(len(by[system]) if cycles==1 else 0))
        ids=[by[system][i]["id"] for i in q]
        return q,dict(unique_scenes=len(set(ids)),draws=len(ids),repeat_draws=len(ids)-len(set(ids)),repeat_rate=(len(ids)-len(set(ids)))/len(ids),cycles=cycles,order_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    def evaluate(epoch):
        dest=a.output/f"epoch_{epoch:03d}"; dest.mkdir(exist_ok=True); records=[]; model.eval()
        with gzip.open(dest/"metrics.jsonl.gz","wt",encoding="utf-8") as stream,torch.inference_mode():
            for pos in range(0,len(validation),8):
                sub=validation[pos:pos+8]; d,y,m,f,ids=make_batch(r,sub,device); pred=model(d)["direct_trajectory"]
                pp,yy,mm,ff=[v.cpu().numpy() for v in (pred,y,m,f)]
                for j,item in enumerate(sub):
                    row=item["_row"]; meta={k:row[k] for k in ("system","group_id","dynamic_bin","event","composition")}
                    meta.update(sample_id=item["id"],run=a.arm+"_BALANCED",epoch=epoch); n=len(item["rows"])
                    rec=metrics(pp[j,:n],yy[j,:n],mm[j,:n],ff[j,:n],item["t"],meta,base["thresholds"]["spatial_"+row["system"]]); records.extend(rec)
                    for x in rec: stream.write(json.dumps(x,allow_nan=False)+"\n")
        summary=[]
        for system in SYSTEMS:
            for window in ("0_2","2_10","10_30","full"):
                for kind,endpoint,stratum in (("device","differential_rmse","HIGH"),("device","differential_rmse","BELOW_HIGH"),("device","nadir_error","HIGH"),("device","max_abs_rocof_error","HIGH"),("pair","pair_rmse","ALL"),("spread","spread_mae","ALL")):
                    rs=[v for v in records if v["system"]==system and v["window"]==window and not v["dynamic_bin"].startswith("D3") and v["kind"]==kind and (v.get("stratum")==stratum if kind=="device" else v.get("pair")=="ALL" if kind=="pair" else True) and v[endpoint] is not None]
                    scenes={}; groups={}
                    for v in rs: scenes.setdefault((v["group_id"],v["sample_id"]),[]).append(v[endpoint])
                    for (group,sid),vals in scenes.items(): groups.setdefault(group,[]).append(float(np.mean(vals)))
                    vals=[v[endpoint] for v in rs]
                    summary.append(dict(system=system,window=window,endpoint=endpoint,stratum=stratum,n=len(vals),groups=len(groups),median=float(np.median(vals)) if vals else None,p90=float(np.quantile(vals,.9)) if vals else None,group_macro_mean=float(np.mean([np.mean(v) for v in groups.values()])) if groups else None))
        atomic(dest/"summary.json",summary); atomic(dest/"DONE.json",dict(epoch=epoch,n_validation=len(validation),metrics_sha256=sha(dest/"metrics.jsonl.gz"),checkpoint_sha256=sha(a.output/f"epoch_{epoch:03d}_model.pt"),test_reads=0,ood_reads=0,wecc_reads=0))
    for epoch in range(start,a.target_epoch+1):
        model.train(); q={}; qa={}
        for s in SYSTEMS: q[s],qa[s]=queue(s)
        losses=[]; sysloss=defaultdict(list); tic=time.time()
        for step in range(old_steps):
            sub=[by["IEEE39"][i] for i in q["IEEE39"][step*4:(step+1)*4]]+[by["NPCC140"][i] for i in q["NPCC140"][step*4:(step+1)*4]]
            torch.manual_seed(789000000+epoch*1000+step)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(789000000+epoch*1000+step)
            d,y,m,f,ids=make_batch(r,sub,device); opt.zero_grad(set_to_none=True)
            old,sp,total,_=scene_losses(r,cfg,base["thresholds"],model(d)["direct_trajectory"],y,m,f,d.phys_rollout_t,sub,base["lambda_early"])
            l39=total[:4].mean(); l140=total[4:].mean(); loss=.5*l39+.5*l140
            assert torch.isfinite(loss); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True); opt.step()
            losses.append(float(loss.detach())); sysloss["IEEE39"].append(float(l39.detach())); sysloss["NPCC140"].append(float(l140.detach()))
            if step%100==0: print(a.arm,epoch,step,old_steps,round(time.time()-tic,1),flush=True)
        hist=dict(epoch=epoch,loss=float(np.mean(losses)),system_loss_mean={s:float(np.mean(sysloss[s])) for s in SYSTEMS},seconds=time.time()-tic,queues=qa)
        history.append(hist); r.save(a.output/f"epoch_{epoch:03d}_model.pt",dict(model=model.state_dict(),completed_epoch=epoch,config=immutable))
        atomic(a.output/"history.json",history)
        r.save(a.output/"last.pt",dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler_rng={s:gens[s].get_state() for s in SYSTEMS},torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],epoch=epoch,history=history,config=immutable))
        saved_cpu=torch.get_rng_state(); saved_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        evaluate(epoch); torch.set_rng_state(saved_cpu)
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(saved_cuda)
        atomic(a.output/"STATUS.json",dict(status="EPOCH_COMPLETE",epoch=epoch,arm=a.arm,target_epoch=a.target_epoch))
    print("TARGET_REACHED",a.arm,a.target_epoch,flush=True)

if __name__=="__main__":
    import hashlib
    from collections import defaultdict
    main()

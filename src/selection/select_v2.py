"""Apply the frozen V2 five-epoch worst-case checkpoint rule."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from balance_common import atomic,read,sha

REF={"IEEE39_0_2":0.11887778015473238,"IEEE39_2_10":0.04393542913471328,"NPCC140_0_2":0.05726787790138924,"NPCC140_2_10":0.04330700979827828}
KEYS=list(REF)

def endpoint(run,epoch,key):
    system,window=key.split("_",1); window=window.replace("_","_",1)
    rows=read(run/f"epoch_{epoch:03d}"/"summary.json")
    hit=[r for r in rows if r["system"]==system and r["window"]==window and r["endpoint"]=="differential_rmse" and r["stratum"]=="HIGH"]
    assert len(hit)==1,(epoch,key,len(hit)); return float(hit[0]["group_macro_mean"])

def main():
    p=argparse.ArgumentParser(); p.add_argument("--runs",required=True,type=Path); p.add_argument("--output",required=True,type=Path); p.add_argument("--end",required=True,type=int); a=p.parse_args()
    assert a.end in (50,60,70,80); a.output.mkdir(parents=True,exist_ok=True)
    allrows=[]; models={}; need=False
    for arm in ("G0","LOCAL"):
        run=a.runs/(arm+"_BALANCED")
        vals=np.asarray([[endpoint(run,e,k) for k in KEYS] for e in range(1,a.end+1)],float)
        scores=np.mean(vals/np.asarray([REF[k] for k in KEYS]),axis=1)
        candidates=[]
        for e in range(3,a.end-1):
            w=scores[e-3:e+2]
            candidates.append(dict(epoch=e,U5=float(max(w)),M5=float(np.mean(w)),STD5=float(np.std(w,ddof=0)),center_S_HIGH=float(scores[e-1]),window_epochs=list(range(e-2,e+3)),window_S_HIGH=w.tolist()))
        best=min(candidates,key=lambda x:(x["U5"],x["M5"],x["center_S_HIGH"],x["epoch"])); raw=int(np.argmin(scores))+1
        trends={}
        for name,v in [(k,vals[:,i]) for i,k in enumerate(KEYS)]+[("S_HIGH",scores)]:
            z=v[-10:]; trends[name]=dict(values=z.tolist(),epochs=list(range(a.end-9,a.end+1)),linear_slope=float(np.polyfit(np.arange(a.end-9,a.end+1),z,1)[0]),mean_previous5=float(z[:5].mean()),mean_last5=float(z[5:].mean()),decreasing_steps=int((np.diff(z)<0).sum()),increasing_steps=int((np.diff(z)>0).sum()))
        boundary=best["epoch"] in range(a.end-4,a.end-1)
        t=trends["S_HIGH"]; declining=boundary and t["linear_slope"]<0 and t["mean_last5"]<t["mean_previous5"]
        status="TRAINING_BOUNDARY_NOT_RESOLVED" if declining else "PLATEAU/OSCILLATION" if boundary else "NOT_AT_TRAINING_BOUNDARY"
        need=need or (declining and a.end<80)
        models[arm]=dict(raw_best_epoch=raw,raw_best_S_HIGH=float(scores[raw-1]),stable_best_epoch=best["epoch"],**{k:v for k,v in best.items() if k!="epoch"},core_endpoints=dict(zip(KEYS,vals[best["epoch"]-1].tolist())),near_training_boundary=boundary,boundary_status=status,late_trends=trends)
        for e in range(1,a.end+1):
            c=next((x for x in candidates if x["epoch"]==e),{})
            allrows.append(dict(model=arm+"_BALANCED",epoch=e,**dict(zip(KEYS,vals[e-1].tolist())),S_HIGH=float(scores[e-1]),U5=c.get("U5",""),M5=c.get("M5",""),STD5=c.get("STD5",""),is_raw_best=e==raw,is_stable_best=e==best["epoch"]))
    horizon_resolved=not any(m["boundary_status"]=="TRAINING_BOUNDARY_NOT_RESOLVED" for m in models.values())
    result=dict(rule="frozen V2 lexicographic U5,M5,center S_HIGH,earlier epoch",reference=REF,end_epoch=a.end,models=models,synchronous_extension_required=need,next_end=min(a.end+10,80) if need else a.end,horizon_resolved=horizon_resolved,at_maximum_with_unresolved_boundary=(a.end==80 and not horizon_resolved),test_reads=0,ood_reads=0,wecc_reads=0)
    atomic(a.output/"BALANCED_STABLE_SELECTION_V2.json",result)
    with (a.output/"BALANCED_S_HIGH_E1_END.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0])); w.writeheader(); w.writerows(allrows)
    lines=["# Balanced training convergence and frozen V2 selection","",f"Both models have reached e{a.end}. S_HIGH is used only for checkpoint selection; it is not a performance superiority claim.","","| model | raw best | stable best | U5 | M5 | STD5 | boundary status |","|---|---:|---:|---:|---:|---:|---|"]
    for arm,m in models.items(): lines.append(f"| {arm}_BALANCED | e{m['raw_best_epoch']} | e{m['stable_best_epoch']} | {m['U5']:.9f} | {m['M5']:.9f} | {m['STD5']:.9f} | {m['boundary_status']} |")
    lines += ["","The five original S_HIGH values, four center endpoints, and complete late-trend diagnostics are stored in the JSON. The continuation decision uses the previously frozen boundary interpretation: center in the final e(end-4)..e(end-2), negative last-10 slope, and lower last-5 mean than the previous five. Both arms extend synchronously when either arm meets it. No Test/OOD/WECC179 data were read."]
    (a.output/"BALANCED_TRAINING_CONVERGENCE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(end=a.end,extend=need,next=result["next_end"],horizon_resolved=horizon_resolved,selected={k:v["stable_best_epoch"] for k,v in models.items()})),flush=True)

if __name__=="__main__": main()

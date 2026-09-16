"""Validation-only selection: no held-out arrays read here."""
import csv, json, datetime
from pathlib import Path
import numpy as np
from heldout_eval import read, sha, atomic, ARMS
from relation_transfer_boundary_audit import RUNS, G0RUN
OUT=Path(__file__).resolve().parents[1]/'FINAL_FROZEN_HELDOUT_EVALUATION_V2'
def main():
    old=read(Path(__file__).parent/'results/VALIDATION_REFERENCE/STABLE_SELECTION_V2.json')
    refs=old['reference']; keys=list(refs); models={}; table=[]
    for arm in ARMS:
        root=(G0RUN if arm=='G0_BALANCED' else RUNS/'PAPER_FORMAL_E60_RECOVERY_V1' if arm in ('A6_PRE_EVENT','E3_GCN_H128') else RUNS)/arm
        status=read(root/'STATUS.json')
        assert status['epoch']>=60,(arm,status)
        values=[]
        for e in range(1,61):
            rows=read(root/f'epoch_{e:03d}/summary.json'); v=[]
            for key in keys:
                system,window=key.split('_',1)
                hits=[r for r in rows if r['system']==system and r['window']==window and r['endpoint']=='differential_rmse' and r['stratum']=='HIGH']
                assert len(hits)==1,(arm,e,key)
                v.append(hits[0]['group_macro_mean'])
            values.append(v)
        values=np.asarray(values); scores=(values/np.array(list(refs.values()))).mean(1)
        assert np.isfinite(scores).all()
        e=min(range(3,59),key=lambda e:(scores[e-3:e+2].max(),scores[e-3:e+2].mean(),scores[e-1],e))
        w=scores[e-3:e+2]; ck=root/f'epoch_{e:03d}_model.pt'
        assert ck.is_file(),(arm,e,str(ck))
        declining=np.polyfit(np.arange(10),scores[-10:],1)[0]<0 and scores[-5:].mean()<scores[-10:-5].mean()
        models[arm]=dict(epoch=e,checkpoint=str(ck),checkpoint_sha256=sha(ck),raw_best_epoch=int(scores.argmin()+1),window_epochs=list(range(e-2,e+3)),window_S_HIGH=w.tolist(),U5=float(w.max()),M5=float(w.mean()),STD5=float(w.std()),core_endpoints=dict(zip(keys,values[e-1].tolist())),boundary_status='TRAINING_BOUNDARY_NOT_RESOLVED' if e>=56 and declining else 'PLATEAU/OSCILLATION' if e>=56 else 'NOT_AT_BOUNDARY',summary_sha256=sha(root/f'epoch_{e:03d}/summary.json'))
        table.extend(dict(model=arm,epoch=i+1,S_HIGH=float(s),**dict(zip(keys,values[i]))) for i,s in enumerate(scores))
        print(arm,e,models[arm]['boundary_status'],flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'FROZEN_CHECKPOINTS.json'
    obj=dict(frozen_at=datetime.datetime.now().astimezone().isoformat(),reference=refs,horizon=60,rule='Lexicographic min (U5, M5, center S_HIGH, epoch); centers 3..58',disclosure='V1 held-out results were already inspected; this freeze precedes V2 inference. No claim of globally unseen held-out data.',models=models)
    if target.exists(): assert read(target)['models']==models,'Frozen selection changed'
    else: atomic(target,obj)
    with (OUT/'S_HIGH_E1_E60.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
if __name__=='__main__':main()

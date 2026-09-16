"""Common inference and saved predictions for the pre-frozen V2 checkpoints."""
import sys, time, gzip, json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
import heldout_eval as he
from relation_transfer_boundary_audit import SRC, REUSE, TEST, ARCH, WECC, WINDOWS
from freeze_v2 import OUT

def main():
    frozen=he.read(OUT/'FROZEN_CHECKPOINTS.json'); freeze_sha=he.sha(OUT/'FROZEN_CHECKPOINTS.json')
    a=SimpleNamespace(source_package=SRC,runtime=SRC/'g0_runtime',reuse=REUSE,balance_package=SRC.parent/'CROSS_SYSTEM_BALANCE_AND_EVIDENCE_V1_PACKAGE',physical_stats=SRC/'physical_inputs/NORMALIZATION.json',test_root=TEST,wecc_root=WECC,archive_session=ARCH,output=OUT)
    sys.path[:0]=[str(SRC),str(a.runtime),str(REUSE/'formal')]
    import formal_train as ft, external_models as em, external_data as ed, spatial_metrics as sm, balance_common as bc
    torch.set_num_threads(4)
    g,r,cfg,cache=bc.load_data(REUSE,a.runtime,SRC/'topology',('validation',))
    validation,_=cache['validation']; assert len(validation)==1000
    test,wecc=he.load_datasets(a); assert len(test)==1000 and len(wecc)==1600
    he.atomic(OUT/'EVALUATION_CONTRACT.json',dict(freeze_sha256=freeze_sha,test_manifest_sha256=he.sha(TEST/'freeze/test.json'),wecc_manifest_sha256=he.sha(WECC/'freeze/wecc1600.json'),normalization_sha256=he.sha(REUSE/'formal/data/freeze/normalization.json'),test_primary='ALL 1000; D3 retained',wecc_primary='1402 ordinary trajectory usable; 198 risk separate',wecc_high=None,source_high={'IEEE39':.0184240117377,'NPCC140':.012875054017},top_k=[1,3,5],ranking='centered trajectory RMS within each window; deterministic device-index ties',bootstrap='10000 prospective-group paired resamples; scene means before group means; nominal endpoint-wise 95% CI',windows=WINDOWS))
    for arm,spec in frozen['models'].items():
        ck=Path(spec['checkpoint']); assert he.sha(ck)==spec['checkpoint_sha256']
        relation='PRE_F' if arm=='A6_PRE_EVENT' else 'NO_MESSAGE' if arm=='A5_NO_PHYSICAL' else 'POST_F'
        physical=arm in ('E1_GRAPHICAL_DEEPONET','E4_UGCN')
        for split in ('validation','test','wecc1600'):
            dest=OUT/arm/split; dest.mkdir(parents=True,exist_ok=True)
            if (dest/'DONE.json').exists():
                done=he.read(dest/'DONE.json'); assert done['freeze_sha256']==freeze_sha
                assert he.sha(dest/'metrics.jsonl.gz')==done['metrics_sha256']
                assert he.sha(dest/'predictions.npz')==done['predictions_sha256']
                continue
            if split=='validation':
                items=ft.transform_relation([dict(v) for v in validation],relation,g,REUSE,split)
                if physical:ft.attach_external_paths(items,SRC/'physical_inputs',split)
            else:items=he.materialize(test if split=='test' else wecc,relation,OUT/'physical_cache'/split if physical else None)
            model,external=he.create_model(arm,items[0],a,g,r,ft,em,'cuda:0',ck)
            saved={}; meta_rows=[]; tic=time.time(); count=0
            with gzip.open(dest/'metrics.jsonl.gz','wt',encoding='utf-8') as stream,torch.inference_mode():
                for pos in range(0,len(items),8):
                    sub=items[pos:pos+8]
                    if external and physical:d,y,m,f,ids=ed.collate_external(sub,r,he.read(a.physical_stats),'cuda:0')
                    else:
                        d,y,m,f,ids=r.m.move(r.m.batching(sub),'cuda:0')
                        if external:d.controller_features=ed.controller_features(d)
                    if arm=='G0_BALANCED':d.local_index=torch.zeros((2,0),dtype=torch.long,device='cuda:0')
                    pred=ft.model_prediction(model,d,external)
                    arrays=[v.detach().cpu().numpy() for v in (pred,y,m,f)]
                    for j,item in enumerate(sub):
                        n=len(item['rows']); pp,yy,mm,ff=[v[j,:n] for v in arrays]; t=item['t']; row=item['_row']
                        assert np.isfinite(pp[mm.astype(bool)]).all()
                        meta=dict(system=item['_system'],sample_id=item['id'],group_id=row.get('group_id',row.get('input_group_id')),dynamic_bin=row.get('dynamic_bin',''),event=row.get('event',row.get('event_family_actual','')),evaluation_stratum=row.get('evaluation_stratum',''),dataset=split,run=arm,epoch=spec['epoch'])
                        assert meta['group_id'] is not None
                        meta_rows.append(meta)
                        prefix=f's{pos+j:04d}_'
                        for name,value in dict(prediction=pp,target=yy,mask=mm,family=ff,t=t,device_rows=item['rows']).items():saved[prefix+name]=value
                        threshold=.0184240117377 if item['_system']=='IEEE39' else .012875054017 if item['_system']=='NPCC140' else float('inf')
                        rec=sm.metrics(pp,yy,mm,ff,t,meta,threshold)
                        for value in rec:
                            if value['kind']=='device':
                                lo,hi=WINDOWS[value['window']]; sel=(t>=lo)&(t<=hi); idx=value['device_index']
                                value['trajectory_rmse']=sm.root(sm.weighted((pp[idx,sel].astype(float)-yy[idx,sel])**2,mm[idx,sel].astype(bool),sm.weights(t[sel].astype(float))))
                                if split=='wecc1600':value['stratum']='ALL_NO_HIGH_THRESHOLD'
                            stream.write(json.dumps(value,allow_nan=False)+'\n'); count+=1
                    if pos%200==0:print('V2',arm,split,pos,len(items),round(time.time()-tic,1),flush=True)
            np.savez_compressed(dest/'predictions.npz',**saved)
            he.atomic(dest/'SCENES.json',meta_rows)
            he.atomic(dest/'DONE.json',dict(freeze_sha256=freeze_sha,checkpoint_sha256=spec['checkpoint_sha256'],epoch=spec['epoch'],n_samples=len(items),n_records=count,metrics_sha256=he.sha(dest/'metrics.jsonl.gz'),predictions_sha256=he.sha(dest/'predictions.npz'),seconds=time.time()-tic))
            del model,items,saved
            torch.cuda.empty_cache()
            print('COMPLETE',arm,split,flush=True)
if __name__=='__main__':main()

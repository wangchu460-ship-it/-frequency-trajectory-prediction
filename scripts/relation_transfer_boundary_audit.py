from __future__ import annotations
import csv,json,sys,math,hashlib
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE=Path(__file__).resolve().parent; OUT=HERE/'RELATION_TRANSFER_BOUNDARY_AUDIT_V1'
SRC=HERE.parent/'PAPER_FORMAL_ABLATION_AND_BASELINES_V1_PACKAGE_STORAGE_EFFICIENT'
REUSE=HERE.parents[1]/'work/reuse/UNIFIED_F_FCAR_FULL_V1_PACKAGE'
TEST=Path('F:/wang/GNN/codex_archive_20260730/lunwen/02_PAPER_OUTPUTS/PAPER_MAINLINE_EXECUTION_V1/FINAL9000_MASK_REPAIR_AND_TRAINING_READY_FREEZE_V1/FORMAL9000_MASK_CORRECTED_V1')
ARCH=Path('F:/wang/GNN/codex_archive_20260730/2026-09-05/f-wang-gnn-codex-archive-20260730-4')
WECC=ARCH/'outputs/PAPER_MAINLINE_EXECUTION_V1/WECC179_GENERALIZATION1600_FREEZE_V1'
RUNS=Path('F:/wang/GNN/codex_archive_20260730/lunwen/结果'); G0RUN=Path('F:/wang/GNN/codex_archive_20260730/lunwen/xinF/CROSS_SYSTEM_BALANCE_AND_EVIDENCE_V1_RUNS')
WINDOWS={'full':(0,30),'0_2':(0,2),'2_10':(2,10),'10_30':(10,30)}
def write_csv(p,rows):
    rows=list(rows);fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with p.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def domain(item,dataset):
    if dataset in ('train','validation','test'):return f'{dataset}:{item["_system"]}'
    return 'wecc1600:RISK' if item['_row']['evaluation_stratum']=='PHYSICAL_RISK_DIAGNOSTIC_ONLY' else 'wecc1600:USABLE'
def graph_relation(item,dataset):
    a=np.asarray(item['phy_attr'],float);n=len(item['rows']);src,dst=np.asarray(item['phy_index'],int);mass=np.zeros(len(item['x']))
    if len(a):np.add.at(mass,dst,a[:,3])
    def q(v,p):return float(np.quantile(v,p)) if len(v) else 0.
    return dict(dataset=dataset,domain=domain(item,dataset),sample_id=item['id'],group_id=item['_row'].get('group_id',item['_row'].get('input_group_id')),n_device=n,n_relation_edges=len(a),relation_density=len(a)/max(n*(n-1),1),raw_F_abs_median=q(np.abs(a[:,0]),.5),raw_F_abs_p90=q(np.abs(a[:,0]),.9),raw_F_abs_max=q(np.abs(a[:,0]),1),normalized_F_median=q(np.abs(a[:,3]),.5),normalized_F_p90=q(np.abs(a[:,3]),.9),normalized_F_max=q(np.abs(a[:,3]),1),incoming_normalized_mass_mean=float(mass[np.asarray(item['rows'],int)].mean()))
def amplitudes(item,dataset):
    y=np.asarray(item['y'],float);m=np.asarray(item['mask'],bool);t=np.asarray(item['t'],float);out=[]
    mean=np.divide((y*m).sum(0),m.sum(0),out=np.zeros(y.shape[1]),where=m.sum(0)>0);center=y-mean
    for w,(lo,hi) in WINDOWS.items():
        z=(t>=lo)&(t<=hi);dt=np.diff(t[z]);weights=np.zeros(z.sum());weights[:-1]+=dt/2;weights[1:]+=dt/2
        valid=m[:,z];den=(valid*weights).sum();
        spatial=math.sqrt(float(((center[:,z]**2)*valid*weights).sum()/den)) if den else float('nan')
        common=math.sqrt(float(((mean[z]**2)*(m[:,z].any(0))*weights).sum()/((m[:,z].any(0))*weights).sum()))
        out.append(dict(dataset=dataset,domain=domain(item,dataset),sample_id=item['id'],group_id=item['_row'].get('group_id',item['_row'].get('input_group_id')),window=w,true_spatial_rms_hz=spatial,true_common_mode_rms_hz=common,spatial_to_common_ratio=spatial/max(common,1e-12)))
    return out
def summaries(rows,metrics):
    out=[]
    domains=sorted({r['domain'] for r in rows})
    for d in domains:
        q=[r for r in rows if r['domain']==d]
        windows=sorted({r.get('window','') for r in q})
        for w in windows:
            z=[r for r in q if r.get('window','')==w]
            for metric in metrics:
                v=np.asarray([r[metric] for r in z if np.isfinite(r[metric])]);out.append(dict(domain=d,window=w or 'NA',metric=metric,n=len(v),mean=float(v.mean()),median=float(np.median(v)),p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9))))
    return out
def update_audit(model,items,dataset,r,device,batch=8):
    acc=defaultdict(lambda:defaultdict(lambda:[0.,0])); model.eval()
    with torch.inference_mode():
        for pos in range(0,len(items),batch):
            sub=items[pos:pos+batch]; d,y,m,f,ids=r.m.move(r.m.batching(sub),device);d.local_index=torch.zeros((2,0),dtype=torch.long,device=device)
            model(d)
            for li,b in enumerate(model.base.blocks,1):
                h=b._local_h; old=b.phy_operator
                try:
                    b.phy_operator='message';msg=type(b)._physical_branch(b,h,d.edge_index_phy,d.edge_attr_phy,node_type_id=d.node_type_id)
                    b.phy_operator='laplacian';lap=type(b)._physical_branch(b,h,d.edge_index_phy,d.edge_attr_phy,node_type_id=d.node_type_id)
                finally:b.phy_operator=old
                gamma=torch.sigmoid(b.gamma_phy_logit);diff=b.eta_lap*lap;both=msg+diff
                for name,v in [('base_hidden',h),('F_message_update',gamma*msg),('F_differential_update',gamma*diff),('F_combined_update',gamma*both)]:
                    a=acc[(domain(sub[0],dataset),li)][name];a[0]+=float(v.square().sum());a[1]+=v.numel()
            if (pos//batch+1)%50==0:print('UPDATE',dataset,pos+len(sub),len(items),flush=True)
    out=[]
    for (d,li),vals in acc.items():
        rms={k:math.sqrt(v[0]/v[1]) for k,v in vals.items()};out.append(dict(domain=d,layer=li,**{k+'_rms':v for k,v in rms.items()},combined_to_base=rms['F_combined_update']/max(rms['base_hidden'],1e-12),message_to_base=rms['F_message_update']/max(rms['base_hidden'],1e-12),differential_to_base=rms['F_differential_update']/max(rms['base_hidden'],1e-12)))
    return out
def main():
    OUT.mkdir(parents=True,exist_ok=True);sys.path[:0]=[str(HERE),str(SRC),str(SRC/'g0_runtime')]
    import heldout_eval as he, balance_common as bc, formal_train as ft, external_models as em
    a=SimpleNamespace(test_root=TEST,wecc_root=WECC,wecc_base_root=Path('.'),wecc_raw_root=None,archive_session=ARCH,formal_results=RUNS,g0_results=G0RUN,source_package=SRC,balance_package=HERE.parent/'CROSS_SYSTEM_BALANCE_AND_EVIDENCE_V1_PACKAGE',reuse=REUSE,output=OUT,runtime=SRC/'g0_runtime',physical_stats=SRC/'physical_inputs/NORMALIZATION.json',selection_file=None)
    test,wecc=he.load_datasets(a);test_items=he.materialize(test,'POST_F');wecc_items=he.materialize(wecc,'POST_F')
    g,r,cfg,cache=bc.load_data(REUSE,SRC/'g0_runtime',SRC/'topology',('train','validation'))
    train_items,val_items=cache['train'][0],cache['validation'][0]
    relation=[];amp=[]
    for name,items in [('train',train_items),('validation',val_items),('test',test_items),('wecc1600',wecc_items)]:
        for item in items:relation.append(graph_relation(item,name));amp+=amplitudes(item,name)
    write_csv(OUT/'RELATION_DESCRIPTOR_PER_SCENE.csv',relation);write_csv(OUT/'TRUE_SPATIAL_AMPLITUDE_PER_SCENE.csv',amp)
    relmetrics=['n_device','n_relation_edges','relation_density','raw_F_abs_median','raw_F_abs_p90','raw_F_abs_max','normalized_F_median','normalized_F_p90','normalized_F_max','incoming_normalized_mass_mean']
    write_csv(OUT/'RELATION_DESCRIPTOR_SUMMARY.csv',summaries(relation,relmetrics));write_csv(OUT/'TRUE_SPATIAL_AMPLITUDE_SUMMARY.csv',summaries(amp,['true_spatial_rms_hz','true_common_mode_rms_hz','spatial_to_common_ratio']))
    ep,ck,_=he.model_spec(a)['G0_BALANCED'];device='cuda:0';model,_=he.create_model('G0_BALANCED',val_items[0],a,g,r,ft,em,device,ck)
    updates=[]
    for name,items in [('validation',val_items),('test',test_items),('wecc1600',wecc_items)]:
        buckets=defaultdict(list)
        for x in items:buckets[domain(x,name)].append(x)
        for subset in buckets.values():updates+=update_audit(model,subset,name,r,device)
    write_csv(OUT/'G0_LAYER_UPDATE_RMS.csv',updates)
    report=['# RELATION_TRANSFER_BOUNDARY_AUDIT_V1','',f'G0_BALANCED e{ep} was inspected without retraining or parameter changes. Relation descriptors and target amplitudes cover Train7000, Validation1000, Test1000 and frozen WECC1600. Layer update RMS covers Validation1000, Test1000 and WECC1600. WECC usable and physical-risk strata are separated.','',
    'This audit measures distribution and activation differences. It does not assign causality to normalization, topology size, event mix or any single descriptor. A normalization explanation requires a targeted intervention trained or calibrated under a frozen protocol.','',
    'Files: `RELATION_DESCRIPTOR_SUMMARY.csv`, `TRUE_SPATIAL_AMPLITUDE_SUMMARY.csv`, `G0_LAYER_UPDATE_RMS.csv`; per-scene source tables are also retained.']
    (OUT/'RELATION_TRANSFER_BOUNDARY_AUDIT_V1.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    files=list(OUT.glob('*'));(OUT/'AUDIT.json').write_text(json.dumps({'status':'PASS','checkpoint_epoch':ep,'checkpoint_sha256':he.sha(ck),'counts':{'train':len(train_items),'validation':len(val_items),'test':len(test_items),'wecc1600':len(wecc_items)},'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}},indent=2),encoding='utf-8')
    print('RELATION_TRANSFER_BOUNDARY_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':main()

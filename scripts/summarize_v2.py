"""Post-inference statistics, using identical paired prospective-group units."""
import gzip,json,csv,hashlib
from collections import defaultdict,Counter
import numpy as np
from scipy.stats import spearmanr
from freeze_v2 import OUT
from heldout_eval import read,sha,atomic
from relation_transfer_boundary_audit import WINDOWS

def extra_metrics(dest):
    scenes=read(dest/'SCENES.json'); result=[]
    with np.load(dest/'predictions.npz',allow_pickle=False) as z:
        for i,meta in enumerate(scenes):
            p,y,m,t,f=[z[f's{i:04d}_{k}'] for k in ('prediction','target','mask','t','family')]
            m=m.astype(bool);p=p.astype(float);y=y.astype(float)
            count=m.sum(0); pc=np.divide(np.where(m,p,0).sum(0),count,out=np.zeros(len(t)),where=count>0);yc=np.divide(np.where(m,y,0).sum(0),count,out=np.zeros(len(t)),where=count>0)
            for window,(lo,hi) in WINDOWS.items():
                sel=(t>=lo)&(t<=hi);tt=t[sel].astype(float);w=np.zeros(len(tt));dt=np.diff(tt);w[:-1]+=dt/2;w[1:]+=dt/2
                mm=m[:,sel];den=(mm*w).sum(1);ok=den>0
                ta=np.sqrt(np.divide((np.where(mm,(y-yc)[:,sel]**2,0)*w).sum(1),den,out=np.zeros(len(p)),where=ok))
                pa=np.sqrt(np.divide((np.where(mm,(p-pc)[:,sel]**2,0)*w).sum(1),den,out=np.zeros(len(p)),where=ok))
                ids=np.flatnonzero(ok)
                if not len(ids):continue
                trueorder=ids[np.lexsort((ids,-ta[ids]))]; predorder=ids[np.lexsort((ids,-pa[ids]))]
                rank=float(spearmanr(ta[ids],pa[ids]).statistic) if len(ids)>1 and np.ptp(ta[ids])>0 and np.ptp(pa[ids])>0 else None
                v=dict(meta,kind='identification',window=window,true_spatial_rms=float(np.sqrt((ta[ok]**2*den[ok]).sum()/den[ok].sum())),predicted_spatial_rms=float(np.sqrt((pa[ok]**2*den[ok]).sum()/den[ok].sum())),amplitude_mae=float(np.abs(pa[ok]-ta[ok]).mean()),rank_spearman=rank)
                for k in (1,3,5):
                    if len(ids)>=k:v[f'top{k}_recall']=len(set(trueorder[:k])&set(predorder[:k]))/k
                result.append(v)
    return result

def groups(rows):
    scenes=defaultdict(list)
    for identity,(group,scene,v) in rows.items():scenes[(group,scene)].append(v)
    gg=defaultdict(list)
    for (g,s),v in scenes.items():gg[g].append(float(np.mean(v)))
    return {g:float(np.mean(v)) for g,v in gg.items()},len(scenes)

def buckets(records):
    b=defaultdict(dict)
    for r in records:
        population='RISK_198' if r['dataset']=='wecc1600' and r['evaluation_stratum']=='PHYSICAL_RISK_DIAGNOSTIC_ONLY' else 'USABLE_1402' if r['dataset']=='wecc1600' else 'ALL_1000'
        label=str(r.get('dynamic_bin') or '')
        label='EXPLICIT_D3' if label.startswith('D3') else 'LABELED_NON_D3' if label else 'UNLABELED'
        scopes=[('ALL','ALL'),('event',r['event'])]
        if r['dataset']!='wecc1600':scopes += [('label',label)]
        if r['kind']=='device':
            scopes += [('family',str(r['family']))]
            if r.get('stratum')=='HIGH':scopes += [('HIGH','HIGH')]
            names=('trajectory_rmse','differential_rmse','nadir_error','max_abs_rocof_error')
        elif r['kind']=='pair':scopes=[('pair',r['pair'])];names=('pair_rmse',)
        elif r['kind']=='spread':names=('spread_mae',)
        else:names=('true_spatial_rms','predicted_spatial_rms','amplitude_mae','rank_spearman','top1_recall','top3_recall','top5_recall')
        identity=(r['system'],r['sample_id'],r['kind'],r.get('device_index',-1),r.get('pair',''))
        for scope,value in scopes:
            for metric in names:
                v=r.get(metric)
                if v is None or not np.isfinite(v):continue
                key=(r['dataset'],population,r['system'],scope,value,r['window'],metric)
                b[key][identity]=(str(r['group_id']),r['sample_id'],float(v))
    return b

def csvwrite(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    frozen=read(OUT/'FROZEN_CHECKPOINTS.json'); checksum=sha(OUT/'FROZEN_CHECKPOINTS.json'); absolute=[];paired=[];audit=[];ref=None
    fields=('dataset','population','system','scope','scope_value','window','metric')
    for arm,spec in frozen['models'].items():
        allrecords=[]
        for split,n in [('validation',1000),('test',1000),('wecc1600',1600)]:
            dest=OUT/arm/split;done=read(dest/'DONE.json')
            assert done['n_samples']==n and done['freeze_sha256']==checksum and done['checkpoint_sha256']==spec['checkpoint_sha256']
            assert done['metrics_sha256']==sha(dest/'metrics.jsonl.gz') and done['predictions_sha256']==sha(dest/'predictions.npz')
            scenes=read(dest/'SCENES.json');assert len(scenes)==n and len({(s['system'],s['sample_id']) for s in scenes})==n
            audit.append(dict(model=arm,dataset=split,n_samples=n,epoch=spec['epoch'],prediction_sha256=done['predictions_sha256']))
            with gzip.open(dest/'metrics.jsonl.gz','rt',encoding='utf-8') as f:allrecords.extend(json.loads(line) for line in f)
            extras=extra_metrics(dest);allrecords.extend(extras)
            with gzip.open(dest/'identification_metrics.jsonl.gz','wt',encoding='utf-8') as f:
                for r in extras:f.write(json.dumps(r,allow_nan=False)+'\n')
            if split=='wecc1600':assert Counter(s['evaluation_stratum']=='PHYSICAL_RISK_DIAGNOSTIC_ONLY' for s in scenes)=={False:1402,True:198}
        b=buckets(allrecords)
        if arm=='G0_BALANCED':ref=b
        for key,values in b.items():
            header=dict(zip(fields,key));gg,ns=groups(values);v=np.array([x[2] for x in values.values()]);gv=np.array(list(gg.values()))
            absolute.append(dict(model=arm,epoch=spec['epoch'],**header,n_records=len(v),n_scenes=ns,n_groups=len(gg),group_macro=float(gv.mean()),record_median=float(np.median(v)),record_p90=float(np.quantile(v,.9))))
            if arm=='G0_BALANCED' or key[-1] in ('true_spatial_rms','predicted_spatial_rms'):continue
            baseline=ref.get(key,{})
            # Rank correlation can be undefined for constant predictions. Report
            # the coverage explicitly rather than silently intersecting errors.
            common=set(values)&set(baseline)
            if key[-1]!='rank_spearman':assert set(values)==set(baseline),(arm,key,'paired coverage differs')
            if not common:continue
            diffs={i:(values[i][0],values[i][1],values[i][2]-baseline[i][2]) for i in sorted(common)}
            gd,ns=groups(diffs);d=np.array([gd[g] for g in sorted(gd)])
            rng=np.random.default_rng(20260912);boots=np.empty(10000)
            for start in range(0,10000,250):boots[start:start+250]=d[rng.integers(0,len(d),(250,len(d)))].mean(1)
            higher=key[-1].startswith('top') or key[-1]=='rank_spearman'
            paired.append(dict(model=arm,reference='G0_BALANCED',**header,n_paired_records=len(common),model_records=len(values),reference_records=len(baseline),n_paired_scenes=ns,n_groups=len(d),group_mean_difference=float(d.mean()),group_median_difference=float(np.median(d)),group_win_rate=float(np.mean(d>0 if higher else d<0)),ci95_low=float(np.quantile(boots,.025)),ci95_high=float(np.quantile(boots,.975))))
        print('SUMMARY',arm,len(absolute),len(paired),flush=True)
        csvwrite('ABSOLUTE_METRICS.csv',absolute)
        if paired:csvwrite('PAIRED_VS_G0.csv',paired)
    atomic(OUT/'FINAL_AUDIT.json',dict(status='PASS',passes=audit,bootstrap_draws=10000,bootstrap_seed=20260912,intervals='nominal endpoint-wise; no multiplicity correction; one training seed',test_primary=1000,wecc_primary=1402,wecc_risk=198,freeze_sha256=checksum))
    (OUT/'README.md').write_text('# FINAL_FROZEN_HELDOUT_EVALUATION_V2\n\nAll 13 frozen e<=60 models evaluated on Validation1000, Test1000 and all WECC1600. Test D3 retained. WECC usable1402 and risk198 reported separately. No WECC HIGH threshold.\n\nFROZEN_CHECKPOINTS.json contains exact weights, hashes and Validation selection. A6 e56 and E3 e58 remain training-boundary unresolved. V1 held-out data were previously inspected; this freeze precedes V2 inference, not all historical held-out inspection.\n\nABSOLUTE_METRICS.csv and PAIRED_VS_G0.csv use one common metric definition and scene-to-group aggregation, separated by source system. Confidence intervals are nominal endpoint-wise 10000 group bootstraps, conditional on seed789. They do not measure training-seed variability or establish equivalence. No single-RMSE ranking.\n\nEach model/split contains predictions.npz (prediction, target, mask, family, time, device rows) and SCENES.json for subsequent figures without inference. top-k uses centered RMS within each window with fixed k=1,3,5, and is omitted when fewer than k valid devices exist. Constant-vector Spearman values are undefined and paired coverage is explicitly reported.\n',encoding='utf-8')
if __name__=='__main__':main()

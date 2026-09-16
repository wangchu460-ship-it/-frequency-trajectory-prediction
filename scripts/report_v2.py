"""Evidence tables and previous-evaluation consistency, after all V2 checks."""
import csv,gzip,json,shutil
from pathlib import Path
from freeze_v2 import OUT
from heldout_eval import read,atomic,sha
HERE=Path(__file__).resolve().parent
def records(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        return {(r['system'],r['sample_id'],r['kind'],r['window'],r.get('device_index',-1),r.get('pair','')):r for r in map(json.loads,f)}
def main():
    assert read(OUT/'FINAL_AUDIT.json')['status']=='PASS'
    freeze=read(OUT/'FROZEN_CHECKPOINTS.json'); reports=[]
    for arm,spec in freeze['models'].items():
        for split in ('test','wecc1600'):
            candidates=[HERE/'FINAL_E60_EVIDENCE'/arm/split,HERE/'WECC_SOURCE_REPAIR_V1/full_evaluation'/arm/split,HERE/'results'/arm/split]
            old=None
            for p in candidates:
                if (p/'DONE.json').exists() and read(p/'DONE.json').get('checkpoint_sha256')==spec['checkpoint_sha256']:
                    old=p;break
            if old is None:reports.append(dict(model=arm,dataset=split,status='NO_SAME_CHECKPOINT_V1_REFERENCE'));continue
            a=records(old/'metrics.jsonl.gz');b=records(OUT/arm/split/'metrics.jsonl.gz')
            assert set(a)==set(b),(arm,split,'V1/V2 identity mismatch')
            maxima={}
            for key,x in a.items():
                y=b[key]
                for metric in ('trajectory_rmse','differential_rmse','pair_rmse','nadir_error','max_abs_rocof_error','spread_mae'):
                    if x.get(metric) is not None and y.get(metric) is not None:maxima[metric]=max(maxima.get(metric,0.),abs(x[metric]-y[metric]))
            reports.append(dict(model=arm,dataset=split,status='COMPARED_SAME_CHECKPOINT',n_records=len(a),maximum_absolute_metric_differences=maxima,source=str(old)))
    atomic(OUT/'V1_V2_CONSISTENCY.json',reports)
    with (OUT/'ABSOLUTE_METRICS.csv').open(encoding='utf-8-sig') as f:absolute=list(csv.DictReader(f))
    with (OUT/'PAIRED_VS_G0.csv').open(encoding='utf-8-sig') as f:paired=list(csv.DictReader(f))
    lines=['# V2 frozen evaluation evidence','', 'All 39 inference passes and Validation/array pairing checks completed. Results are reported by scientific endpoint, without a single-score ranking.','', '## Frozen checkpoints','', '| Model | Epoch | Boundary status |','|---|---:|---|']
    for arm,spec in freeze['models'].items():lines.append(f"| {arm} | {spec['epoch']} | {spec['boundary_status']} |")
    lines += ['','## Core Test and transfer comparison','', 'Each difference is model minus G0. Negative indicates lower error. Confidence intervals are nominal group-bootstrap intervals; “unresolved” is not equivalence. Source HIGH results and WECC ordinary usable results are kept distinct.','', '| Dataset | System | Model | Metric | Window | Group macro | Δ vs G0 [95% CI] | Group win |','|---|---|---|---|---|---:|---|---:|']
    lookup={(r['model'],r['dataset'],r['population'],r['system'],r['scope'],r['scope_value'],r['window'],r['metric']):r for r in paired}
    for r in absolute:
        take=(r['dataset']=='test' and r['scope']=='HIGH' and r['metric']=='differential_rmse' and r['window'] in ('0_2','2_10')) or (r['dataset']=='wecc1600' and r['population']=='USABLE_1402' and r['scope']=='ALL' and r['metric'] in ('trajectory_rmse','differential_rmse') and r['window'] in ('0_2','2_10'))
        if not take:continue
        key=tuple(r[k] for k in ('model','dataset','population','system','scope','scope_value','window','metric'));p=lookup.get(key)
        delta=f"{float(p['group_mean_difference']):.6g} [{float(p['ci95_low']):.6g}, {float(p['ci95_high']):.6g}]" if p else 'reference'
        win=f"{float(p['group_win_rate']):.1%}" if p else '—'
        lines.append(f"| {r['dataset']} | {r['system']} | {r['model']} | {r['metric']} | {r['window']} | {float(r['group_macro']):.6g} | {delta} | {win} |")
    lines += ['','## A5/G0 transfer boundary','', 'Under the common V2 population and evaluator, A5 has higher Test NPCC140 HIGH differential error in both 0–2 and 2–10 s, with nominal group-bootstrap intervals above zero. IEEE39 corresponding intervals span zero. On WECC ordinary usable1402, A5 has lower trajectory and differential error in all four windows, with intervals below zero. This is an observed transfer limitation of the frozen G0 relative to its physical-path ablation. It does not establish that normalization is the cause, or that A5 is universally superior across source systems and endpoints.', '', '## Claim boundaries','', '- A3/A4/A5 isolate message, differential and combined physical-path contributions; A5 is the unchanged model with both physical operators disabled, not a new device-only baseline.', '- A2 isolates the spatial objective. Use its HIGH and all-device endpoints together.', '- A1 isolates per-step system balancing. Preserve separate IEEE39/NPCC140 results.', '- A6 compares PRE-event relations; its e56 training boundary remains unresolved.', '- E1–E6 are the task-adapted baseline implementations. These results do not establish equivalence to every published implementation.', '- WECC provides transfer evidence for the frozen models. Its risk-only cases and ordinary usable cases cannot be merged to claim ordinary prediction performance.', '- Single-seed group bootstrap does not establish robustness over training initialization. No model modification or reselection is based on these results.', '', 'Full trajectory, all time windows, event strata, device families, cross-family pairs, source HIGH, label sensitivity, spatial amplitude, ranking/top-k and nadir/RoCoF are available in the matched CSV tables.', '', 'V1_V2_CONSISTENCY.json compares unchanged-checkpoint numerical outputs. Changed checkpoint results have no same-weight comparison and are explicitly marked. Changes caused by D3/risk population definitions are not numerical inference errors.']
    (OUT/'EVIDENCE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    code=OUT/'code';code.mkdir(exist_ok=True)
    for name in ('freeze_v2.py','evaluate_v2.py','summarize_v2.py','check_v2.py','report_v2.py','heldout_eval.py','relation_transfer_boundary_audit.py'):
        shutil.copy2(HERE/name,code/name)
    atomic(OUT/'DELIVERY_SHA256.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='DELIVERY_SHA256.json' and p.suffix not in ('.log','.pyc')})
    print('EVIDENCE_REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()

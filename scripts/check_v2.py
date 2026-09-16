import gzip,json
from collections import defaultdict
import numpy as np
from heldout_eval import read,atomic
from freeze_v2 import OUT
def main():
    freeze=read(OUT/'FROZEN_CHECKPOINTS.json'); reports=[]
    for arm,spec in freeze['models'].items():
        dest=OUT/arm/'validation'
        if not (dest/'DONE.json').exists():continue
        with gzip.open(dest/'metrics.jsonl.gz','rt',encoding='utf-8') as f:records=[json.loads(x) for x in f]
        for key,expected in spec['core_endpoints'].items():
            system,window=key.split('_',1);scenes=defaultdict(list);groups=defaultdict(list)
            for r in records:
                if r['system']==system and r['window']==window and r['kind']=='device' and r['stratum']=='HIGH' and not r['dynamic_bin'].startswith('D3') and r['differential_rmse'] is not None:
                    scenes[(r['group_id'],r['sample_id'])].append(r['differential_rmse'])
            for (g,s),v in scenes.items():groups[g].append(np.mean(v))
            actual=float(np.mean([np.mean(v) for v in groups.values()]))
            reports.append(dict(model=arm,endpoint=key,frozen=expected,recomputed=actual,absolute_difference=abs(actual-expected)))
            assert np.isclose(actual,expected,rtol=2e-5,atol=1e-8),(arm,key,actual,expected)
        for split in ('validation','test','wecc1600'):
            folder=OUT/arm/split;reference=OUT/'G0_BALANCED'/split
            if not (folder/'DONE.json').exists():continue
            scenes=read(folder/'SCENES.json');ref=read(reference/'SCENES.json')
            assert [{k:v for k,v in x.items() if k not in ('run','epoch')} for x in scenes]==[{k:v for k,v in x.items() if k not in ('run','epoch')} for x in ref]
            if arm=='G0_BALANCED':continue
            with np.load(folder/'predictions.npz') as a,np.load(reference/'predictions.npz') as b:
                assert a.files==b.files
                for key in a.files:
                    if not key.endswith('_prediction'):assert np.array_equal(a[key],b[key]),(arm,split,key)
        print('CHECK_PASS',arm,flush=True)
    atomic(OUT/'VALIDATION_REPRODUCTION_AUDIT.json',dict(checked_models=len(reports)//4,endpoints=reports))
if __name__=='__main__':main()

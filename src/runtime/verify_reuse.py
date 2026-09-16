"""Read-only verification of previously uploaded repaired-F inputs.

No GPU, model, target-array loading or training. Hashing prepared containers
checks byte identity without deserializing target fields.
"""
import argparse,hashlib,json
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def verify(formal,controls,lock):
    formal=Path(formal);controls=Path(controls)
    for rel,h in lock['formal_files'].items():assert sha(formal/rel)==h,('formal identity',rel)
    for rel,h in lock['control_files'].items():assert sha(controls/rel)==h,('repaired F identity',rel)
    contract=read(controls/'CONTRACT.json')
    assert contract['version']=='UNIFIED_F_CONTROLS_V1'
    assert contract['post_F']=='uniform terminal formula recomputed for every sample; NEVER reuse legacy physical edges'
    audit=read(controls/'FINAL_AUDIT.json')
    assert audit['total']==8000 and audit['errors']==0 and audit['all_cache_hash_and_schema_pass']
    index=read(controls/'INPUT_INDEX.json')
    lookup={(x['split'],str(x['sample_id'])):x for x in index}
    assert len(index)==len(lookup)==8000
    count=0
    for split,expected in [('train',7000),('validation',1000)]:
        rows=read(formal/f'data/freeze/{split}.json');assert len(rows)==expected
        for row in rows:
            sid=str(row['sample_id']);x=lookup[split,sid]
            assert x['prepared_sha256']==row['prepared_sha256']
            assert sha(formal/'data'/row['prepared'].replace('\\','/'))==row['prepared_sha256'],('prepared',split,sid)
            assert sha(controls/x['cache'].replace('\\','/'))==x['cache_sha256'],('unified F',split,sid)
            count+=1
    return dict(status='REPAIRED_F_REUSE_PASS',verified_samples=count,train=7000,validation=1000,
        physical_relation='controls/POST_F override; not prepared legacy phy_index',
        formal_root=str(formal.resolve()),controls_root=str(controls.resolve()),
        target_arrays_deserialized=0,test_ood_wecc_reads=0,training_started=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--formal-root',type=Path,required=True)
    p.add_argument('--controls-root',type=Path,required=True);p.add_argument('--lock',type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.formal_root,a.controls_root,read(a.lock)),indent=2))

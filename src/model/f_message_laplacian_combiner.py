"""Six-epoch PhysicalOnly operator screen; frozen loop and branches reused."""
import os, sys, json, hashlib, argparse, types, traceback
from pathlib import Path
import numpy as np
import torch

ARMS = ['F_MESSAGE', 'F_LAPLACIAN', 'F_MESSAGE_PLUS_LAPLACIAN', 'KRON_LAPLACIAN']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def tensor_hash(v):
    v=v.detach().cpu().contiguous()
    return hashlib.sha256(str(v.dtype).encode()+str(tuple(v.shape)).encode()+v.numpy().tobytes()).hexdigest()

def combined(self, *args, **kwargs):
    # Reuse both existing implementations and leave the outer gate untouched.
    old=self.phy_operator
    try:
        self.phy_operator='message';msg=type(self)._physical_branch(self,*args,**kwargs)
        self.phy_operator='laplacian';lap=type(self)._physical_branch(self,*args,**kwargs)
        return msg+self.eta_lap*lap
    finally:self.phy_operator=old

def make(r, arm, dim, stats, device, seed=789):
    model=r.make('PhysicalOnly',seed,dim,stats,'cpu')
    assert len(model.base.blocks)==4
    if arm in ('F_LAPLACIAN','KRON_LAPLACIAN'):
        for block in model.base.blocks:block.phy_operator='laplacian'
    elif arm=='F_MESSAGE_PLUS_LAPLACIAN':
        for block in model.base.blocks:
            block.eta_lap=torch.nn.Parameter(torch.tensor(0.05))
            block._physical_branch=types.MethodType(combined,block)
    else:assert arm=='F_MESSAGE'
    return model.to(device)

def setup(root, controls):
    sys.path.insert(0,str(root));import runner as r
    r.sha=sha
    sys.path.insert(0,str(Path(__file__).parent))
    from unified_control_adapter import apply_control
    rows=r.read(controls/'INPUT_INDEX.json');assert len(rows)==8000
    audit=r.read(controls/'FINAL_AUDIT.json');assert audit['fair_mapping_status']=='PASS'
    lookup={(x['split'],str(x['sample_id'])):x for x in rows}
    for x in rows:assert sha(controls/str(x['cache']).replace('\\','/'))==x['cache_sha256']
    return r,lookup,apply_control

def wrap_dataset(ds, relation, controls, lookup, apply_control, intervention='TRUE_F_WEIGHT'):
    class Wrapped:
        def __init__(self):self.rows=ds.rows;self.stats=ds.stats
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):
            # smoke_rows changes this proxy's rows, so synchronize explicitly.
            ds.rows=self.rows
            item=ds[i];key=('train' if ds is train_marker[0] else 'validation',str(item['id']))
            x=lookup[key]
            with np.load(controls/str(x['cache']).replace('\\','/'),allow_pickle=False) as z:
                out=apply_control(item,relation,z)
            support=out['phy_index'].copy()
            if intervention=='EMPTY_RELATION':
                out['phy_index']=np.zeros((2,0),np.int64);out['phy_attr']=np.zeros((0,6),np.float32)
            elif intervention in ('UNIFORM_WEIGHT_SAME_SUPPORT','SHUFFLED_WEIGHT_SAME_SUPPORT'):
                w=out['phy_attr'][:,3].copy()
                if intervention=='UNIFORM_WEIGHT_SAME_SUPPORT':
                    if len(w):out['phy_attr'][:,3]=w.mean()
                else:
                    seed=int(hashlib.sha256(('789:'+str(item['id'])).encode()).hexdigest()[:16],16)
                    out['phy_attr'][:,3]=np.random.default_rng(seed).permutation(w)
                    assert np.array_equal(np.sort(w),np.sort(out['phy_attr'][:,3]))
                assert np.array_equal(out['phy_index'],support)
            return out
    return Wrapped()

# marker only used to distinguish the two frozen split objects.
train_marker=[None]

def main():
    p=argparse.ArgumentParser();p.add_argument('--formal-root',type=Path,required=True);p.add_argument('--controls-root',type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int,default=0)
    a=p.parse_args();torch.set_num_threads(2)
    root=a.formal_root.resolve();controls=a.controls_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=out/'RUN.lock'
    with lock.open('x') as f:json.dump(dict(pid=os.getpid(),arm=a.arm,gpu=os.environ.get('CUDA_VISIBLE_DEVICES')),f)
    try:
        print('VERIFYING_DATA_AND_CODE',flush=True)
        r,lookup,adapt=setup(root,controls);cfg,tr,va=r.contract();train_marker[0]=tr
        stats=r.read(root/'data/freeze/controller_normalization.json');dim=tr[0]['x'].shape[1]
        reference=r.make('PhysicalOnly',789,dim,stats,'cpu');model=make(r,a.arm,dim,stats,a.device)
        base=reference.state_dict();shared={k:tensor_hash(v) for k,v in base.items()}
        assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in base.items())
        extras={k:dict(shape=list(v.shape),sha256=tensor_hash(v)) for k,v in model.state_dict().items() if k not in base}
        r.atomic(out/'INITIALIZATION_AUDIT.json',dict(status='MATCHED_SHARED_INITIALIZATION_PASS',shared=shared,arm_specific=extras,
            operator=a.arm,physical_outer_gate_unchanged=True,laplacian_projection_preexists_in_all_arms=True))
        relation='POST_ELECTRICAL_KRON' if a.arm=='KRON_LAPLACIAN' else 'POST_F'
        trw=wrap_dataset(tr,relation,controls,lookup,adapt);vaw=wrap_dataset(va,relation,controls,lookup,adapt)
        cfg=dict(cfg,protocol='PHYSICAL_LAPLACIAN_OPERATOR_SHORT_SCREEN_V1',epochs=6,seeds=[789],batch_size=2,
            operator=a.arm,controls_index_sha256=sha(controls/'INPUT_INDEX.json'),screen_code_sha256=sha(Path(__file__)),development_only=True)
        r.contract=lambda:(cfg,trw,vaw)
        r.make=lambda *args:model
        a.seed=789;a.output=str(out)
        r.run(a)
        if not (out/'DONE.json').exists():return
        # Only the ordinary-selected best checkpoint; no alternative selection.
        ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
        if a.arm=='F_LAPLACIAN':
            manifest={}
            for variant in ['TRUE_F_WEIGHT','UNIFORM_WEIGHT_SAME_SUPPORT','SHUFFLED_WEIGHT_SAME_SUPPORT','EMPTY_RELATION']:
                dest=out/'sensitivity'/variant
                if variant=='TRUE_F_WEIGHT':
                    manifest[variant]=dict(predictions='best_validation_predictions',checkpoint_sha256=sha(out/'best.pt'));continue
                ds=wrap_dataset(va,relation,controls,lookup,adapt,variant);ds.rows=vaw.rows
                print('SENSITIVITY_START',variant,flush=True)
                scores,records=r.evaluate(model,ds,a.device,dest/'predictions')
                r.atomic(dest/'metrics.json',scores)
                manifest[variant]=dict(predictions=str((dest/'predictions').relative_to(out)),checkpoint_sha256=sha(out/'best.pt'),
                    files={q.name:sha(q) for q in (dest/'predictions').glob('*.npz')})
            r.atomic(out/'SENSITIVITY_INDEX.json',manifest)
        r.atomic(out/'SCREEN_DONE.json',dict(status='SMOKE_SCREEN_COMPLETE' if a.smoke else 'SCREEN_COMPLETE',epochs=2 if a.smoke else 6,arm=a.arm,test_ood_wecc_reads=0))
    except Exception:
        (out/'ERROR_SCREEN.json').write_text(json.dumps(dict(traceback=traceback.format_exc())));raise
    finally:lock.unlink()

if __name__=='__main__':main()

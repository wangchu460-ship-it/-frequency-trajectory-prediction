"""Portable G0 full/ablation training. Targets: Train only, no inference export."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import sys,json,argparse,copy,time,platform,hashlib,types
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
from verify_reuse import verify,sha
from operator_screen import make as original_make
from unified_control_adapter import apply_control

def losses(r,cfg,threshold,p,y,m,f,t,items):
 n=m.sum(1).clamp_min(1);yc=torch.where(m,y,0).sum(1)/n;pc=torch.where(m,p,0).sum(1)/n
 yd=y-yc[:,None];pd=p-pc[:,None];dt=t[:,1:]-t[:,:-1];w=torch.zeros_like(t);w[:,:-1]+=dt/2;w[:,1:]+=dt/2
 energy=(yd.square()*m*w[:,None]).sum(-1)/(m*w[:,None]).sum(-1).clamp_min(1e-12)
 cutoff=torch.tensor([threshold['spatial_'+i['_system']] for i in items],device=y.device)
 high=(energy.sqrt()>=cutoff[:,None])&m.any(-1)
 old=r.m.scenario_mse(p,y,m,f,t).mean()/cfg['target_rms_hz']**2;pieces=[]
 for lo,hi in [(0,2),(2,15)]:
  mk=m&high[:,:,None]&((t>=lo)&(t<=hi))[:,None]
  pieces.append(r.m.scenario_mse(pd,yd,mk,f,t.clamp(lo,hi)).mean()/cfg['target_rms_hz']**2)
 return old,(pieces[0]+pieces[1])/2

def only_message(self,*args,**kw):
 old=self.phy_operator
 try:self.phy_operator='message';return type(self)._physical_branch(self,*args,**kw)
 finally:self.phy_operator=old
def only_laplacian(self,*args,**kw):
 old=self.phy_operator
 try:self.phy_operator='laplacian';return self.eta_lap*type(self)._physical_branch(self,*args,**kw)
 finally:self.phy_operator=old

def build(r,dim,stats,seed,arm,initial,device):
 model=original_make(r,'F_MESSAGE_PLUS_LAPLACIAN',dim,stats,'cpu',seed)
 model.load_state_dict(initial,strict=True)
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items())
 if arm in ['NO_LAPLACIAN','NO_MESSAGE']:
  for b in model.base.blocks:
   b._physical_branch=types.MethodType(only_message if arm=='NO_LAPLACIAN' else only_laplacian,b)
 return model.to(device)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reuse-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--batch',type=int,choices=[8,16],required=True);ap.add_argument('--seed',type=int,choices=[123,456,789],required=True);ap.add_argument('--arm',choices=['FULL','NO_LAPLACIAN','NO_MESSAGE','OLD_ONLY'],required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 formal=a.reuse_root/'formal';controls=a.reuse_root/'controls';sys.path.insert(0,str(formal));import runner as r;r.sha=sha
 manifest=r.read(HERE/'PACKAGE_SHA256.json')
 for rel,h in manifest.items():assert sha(HERE/rel)==h,rel
 data_audit=verify(formal,controls,r.read(HERE/'REPAIRED_F_REUSE_LOCK.json'));cfg,tr,va=r.contract();assert torch.cuda.is_available()
 for ds in [tr,va]:
  for row in ds.rows:row['prepared']=row['prepared'].replace('\\','/')
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if torch.cuda.get_device_capability(0)==(7,0):assert 'sm_70' in torch.cuda.get_arch_list()
 base=r.read(HERE/'BASE_PROTOCOL.json');assert cfg['target_rms_hz']==base['target_rms_hz'];stats=r.read(formal/'data/freeze/controller_normalization.json')
 initial=torch.load(HERE/f'initial_seed{a.seed}.pt',map_location='cpu',weights_only=False)
 epochs=2 if a.smoke else 30;out=a.output;out.mkdir(parents=True,exist_ok=True)
 runtime=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
 protocol=dict(arm=a.arm,seed=a.seed,batch=a.batch,max_epochs=epochs,lr=.0001,weight_decay=base['weight_decay'],clip=base['clip'],lambda_early=0. if a.arm=='OLD_ONLY' else base['lambda_early'],initial_sha256=sha(HERE/f'initial_seed{a.seed}.pt'),data_lock=base['data_lock'],control_index=base['control_index'],normalization_sha256=sha(formal/'data/freeze/normalization.json') if (formal/'data/freeze/normalization.json').exists() else r.jsha(tr.stats),package_sha256=sha(HERE/'PACKAGE_SHA256.json'),runtime=runtime,smoke=a.smoke,selection='fixed epoch30; no server Validation selection',test_reads=0,ood_reads=0,wecc_reads=0)
 if (out/'DONE.json').exists():assert r.read(out/'config.json')==protocol;print('ALREADY_COMPLETE',out,flush=True);return
 with (out/'RUN.lock').open('x') as f:json.dump(dict(pid=os.getpid(),protocol=protocol),f)
 try:
  if (out/'config.json').exists():assert r.read(out/'config.json')==protocol
  r.atomic(out/'config.json',protocol);r.atomic(out/'DATA_AUDIT.json',data_audit)
  index={(x['split'],str(x['sample_id'])):x for x in r.read(controls/'INPUT_INDEX.json')};cache=[]
  for i in range(32 if a.smoke else len(tr)):
   item=tr[i];x=index['train',str(item['id'])]
   with np.load(controls/x['cache'].replace('\\','/')) as z:item=apply_control(item,'POST_F',z)
   item['_system']=tr.rows[i]['system'];cache.append(item)
  model=build(r,tr[0]['x'].shape[1],stats,a.seed,a.arm,initial,'cuda:0').train();opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=base['weight_decay']);r.m.seed_all(a.seed+500000);gen=torch.Generator().manual_seed(a.seed)
  epoch=1;cursor=0;order=None;history=[];trace=[];global_step=0
  if (out/'last.pt').exists():
   assert a.resume,'Use --resume with audited stopped run';ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert ck['protocol']==protocol
   model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);r.restore(ck['rng'],gen)
   epoch,cursor,order,history,trace,global_step=[ck[k] for k in ['epoch','cursor','order','history','trace','global_step']]
  def checkpoint():return dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=r.rng(gen),epoch=epoch,cursor=cursor,order=order,history=history,trace=trace,global_step=global_step,protocol=protocol,scheduler=None,scaler=None)
  while epoch<=epochs:
   if order is None:order=torch.randperm(len(cache),generator=gen).tolist();cursor=0;trace=[]
   start=time.monotonic()
   for pos in range(cursor,len(order),a.batch):
    items=[cache[i] for i in order[pos:pos+a.batch]];d,y,m,f,ids=r.m.move(r.m.batching(items),'cuda:0');opt.zero_grad(set_to_none=True)
    old,early=losses(r,cfg,base['thresholds'],model(d)['direct_trajectory'],y,m,f,d.phys_rollout_t,items);loss=old+protocol['lambda_early']*early
    assert torch.isfinite(loss);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),base['clip'],error_if_nonfinite=True));opt.step();cursor=min(pos+a.batch,len(order));global_step+=1;trace.append(dict(old=float(old.detach()),early=float(early.detach()),norm=norm))
    if global_step%100==0 or cursor==len(order):
     r.save(out/'last.pt',checkpoint());status=dict(arm=a.arm,batch=a.batch,seed=a.seed,epoch=epoch,step=(cursor+a.batch-1)//a.batch,steps=(len(cache)+a.batch-1)//a.batch,pid=os.getpid());r.atomic(out/'STATUS.json',status);print(json.dumps(status),flush=True)
   history.append(dict(epoch=epoch,order_sha256=r.jsha(order),seconds=time.monotonic()-start,old_loss=float(np.mean([v['old'] for v in trace])),early_loss=float(np.mean([v['early'] for v in trace])),clip_fraction=float(np.mean([v['norm']>base['clip'] for v in trace]))))
   r.atomic(out/'history.json',history);r.atomic(out/f'epoch_{epoch:03d}_trace.json',trace);r.save(out/f'epoch_{epoch:03d}_model.pt',dict(model=model.state_dict(),completed_epoch=epoch,protocol=protocol));epoch+=1;cursor=0;order=None;trace=[];r.save(out/'last.pt',checkpoint())
  r.atomic(out/'DONE.json',dict(status='SMOKE_COMPLETE' if a.smoke else 'COMPLETE',epochs=epochs,global_step=global_step,validation_inference=False))
 finally:(out/'RUN.lock').unlink(missing_ok=True)
if __name__=='__main__':main()

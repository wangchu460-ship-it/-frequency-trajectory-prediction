"""Fresh paired-arm pilot. Validation selects checkpoints; no test data is loaded here."""
import argparse,json,random,time,os,hashlib
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from data import Samples,collate,move,digest
from backbone.models import GraphLevelGNN

ROOT=Path(__file__).resolve().parent
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(4)
 torch.backends.cudnn.benchmark=False
def model_for(arm,node_dim,hidden=128,layers=4,dropout=.05):
 return GraphLevelGNN(node_dim=node_dim,edge_dim=1,hidden_dim=hidden,num_layers=layers,heads=4,dropout=dropout,
  graph_mode={'full':'haag','physical':'phy_graph','adaptive':'dyn_only'}[arm],edge_dim_phy=6,
  dyn_relation_mode='bus_dense_entmax15',haag_unified_dynamic=True,haag_compact_dynamic=True,
  unified_dyn_gate_init=.5,unified_dyn_gate_max=1.,use_system_adapter=False,use_global_context=True,
  graph_context_dim=8,architecture_screen_mode='e0_direct',prediction_heads=False,q_aware_heads=False,enable_r=False)
def time_weights(t):
 dt=t[:,1:]-t[:,:-1];w=torch.zeros_like(t);w[:,:-1]+=dt/2;w[:,1:]+=dt/2;return w[:,None,:]
def scenario_mse(pred,y,mask,family,t):
 w=time_weights(t)*mask;den=w.sum(-1);dev=((pred-y).square()*w).sum(-1)/den.clamp_min(1e-12)
 sums=[];present=[]
 for k in range(3):
  valid=(family==k)&(den>0);sums.append((dev*valid).sum(-1)/valid.sum(-1).clamp_min(1));present.append(valid.any(-1))
 present=torch.stack(present,-1);return (torch.stack(sums,-1)*present).sum(-1)/present.sum(-1).clamp_min(1)
def check_freeze(root,check_cache=True):
 root=Path(root);lock=json.loads((root/'freeze/LOCK.json').read_text())
 for name,h in lock.items():assert digest(root/'freeze'/name)==h,('freeze modified',name)
 if check_cache:
  for role in ['train','validation']:
   for r in json.loads((root/'freeze'/f'{role}.json').read_text()):assert digest(root/r['prepared'])==r['sha256'],r['prepared']
@torch.no_grad()
def evaluate(model,loader,device):
 model.eval();records=[];branch_values={}
 for batch in loader:
  d,y,m,f,ids=move(batch,device);output=model(d);pred=output['direct_trajectory'];s=scenario_mse(pred,y,m,f,d.phys_rollout_t);zero=scenario_mse(torch.zeros_like(y),y,m,f,d.phys_rollout_t)
  for layer,dbg in enumerate(output.get('haag_debug',[])):
   for key in ['R_phy_norm','R_dyn_norm','gamma_phy','gamma_dyn','dyn_to_phy_effective_ratio']:
    if key in dbg:branch_values.setdefault(f'layer{layer}_{key}',[]).append(float(dbg[key]))
  assert torch.isfinite(pred).all()
  for i,identity in enumerate(ids):
   row=dict(sample_id=identity,mse_hz2=float(s[i]),zero_mse_hz2=float(zero[i]))
   for k,name in enumerate(['sg','gfm','gfl']):
    w=time_weights(d.phys_rollout_t[i:i+1])[0]*m[i]*(f[i,:,None]==k);den=w.sum(-1);v=den>0
    row[name+'_mse_hz2']=float((((pred[i]-y[i]).square()*w).sum(-1)/den.clamp_min(1e-12))[v].mean()) if v.any() else None
   for lo,hi in [(0,2),(2,10),(10,30)]:
    tm=(d.phys_rollout_t>=lo)&(d.phys_rollout_t<=hi)
    row[f'window_{lo}_{hi}_mse_hz2']=float(scenario_mse(pred[i:i+1],y[i:i+1],m[i:i+1]&tm[i:i+1,None,:],f[i:i+1],d.phys_rollout_t[i:i+1])[0])
   records.append(row)
 mean=float(np.mean([r['mse_hz2'] for r in records]));return dict(mse_hz2=mean,rmse_hz=np.sqrt(mean),zero_rmse_hz=np.sqrt(np.mean([r['zero_mse_hz2'] for r in records])),branch_diagnostics_batch_mean={k:float(np.mean(v)) for k,v in branch_values.items()}),records
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));p.add_argument('--arm',choices=['full','physical','adaptive'],default='full');p.add_argument('--seed',type=int,default=123);p.add_argument('--epochs',type=int,default=6);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--hidden',type=int,default=128);p.add_argument('--layers',type=int,default=4);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--run-name',default=None);p.add_argument('--smoke',action='store_true',help='4 train and 4 validation records, one epoch; plumbing check only');a=p.parse_args()
 root=Path(a.root);check_freeze(root);seed_all(a.seed);tr=Samples(root,'train',a.arm);va=Samples(root,'validation',a.arm)
 if a.smoke:tr.rows=tr.rows[:4];va.rows=va.rows[:4];a.epochs=1
 g=torch.Generator().manual_seed(a.seed);train=DataLoader(tr,batch_size=a.batch_size,shuffle=True,generator=g,collate_fn=collate,num_workers=0);val=DataLoader(va,batch_size=a.batch_size,shuffle=False,collate_fn=collate,num_workers=0)
 model=model_for(a.arm,tr[0]['x'].shape[1],a.hidden,a.layers).to(a.device);optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
 run=root/'runs'/(a.run_name or f'{a.arm}_seed{a.seed}');run.mkdir(parents=True,exist_ok=False);cfg=vars(a)|dict(torch=torch.__version__,numpy=np.__version__,freeze_sha256=digest(root/'freeze/LOCK.json'),normalization_sha256=digest(root/'freeze/normalization.json'),parameters=sum(p.numel() for p in model.parameters()),output_unit='Hz deviation',code_sha256={p.name:digest(p) for p in [root/'train.py',root/'data.py',root/'backbone/models.py',root/'backbone/haag_layers.py']});(run/'config.json').write_text(json.dumps(cfg,indent=2))
 init_hash=hashlib.sha256()
 for name,v in model.state_dict().items():init_hash.update(name.encode());init_hash.update(v.detach().cpu().numpy().tobytes())
 (run/'initial_state_sha256.txt').write_text(init_hash.hexdigest());torch.save(model.state_dict(),run/'initial.pt');best=float('inf');scale=tr.stats['target_rms_hz'];history=[]
 for epoch in range(1,a.epochs+1):
  model.train();total=0.;count=0;start=time.time();order=[]
  for step,batch in enumerate(train,1):
   d,y,m,f,ids=move(batch,a.device);order.extend(ids);optimizer.zero_grad(set_to_none=True);pred=model(d)['direct_trajectory'];loss=scenario_mse(pred,y,m,f,d.phys_rollout_t).mean()/scale**2
   if not torch.isfinite(loss):raise RuntimeError('nonfinite loss '+str(ids))
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();total+=float(loss.detach())*len(ids);count+=len(ids)
   if step%100==0:print(json.dumps(dict(epoch=epoch,step=step,loss=total/count,grad_norm=float(norm))),flush=True)
  (run/f'epoch{epoch}_batch_order.json').write_text(json.dumps(order));metrics,records=evaluate(model,val,a.device);row=dict(epoch=epoch,train_scaled_mse=total/count,seconds=time.time()-start,batch_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),**metrics);history.append(row);(run/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
  state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,config=cfg,validation=metrics)
  torch.save(state,run/'last.pt')
  if metrics['mse_hz2']<best:
   best=metrics['mse_hz2'];torch.save(state,run/'best.pt');(run/'validation_best_per_sample.json').write_text(json.dumps(records,indent=2))
 print('DONE',run,flush=True)
if __name__=='__main__':main()

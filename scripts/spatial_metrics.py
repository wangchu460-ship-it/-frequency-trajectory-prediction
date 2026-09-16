def weights(t):
    import numpy as np
    dt=np.diff(t);w=np.zeros_like(t);w[:-1]+=dt/2;w[1:]+=dt/2;return w
import numpy as np
def weighted(v,m,w):
    z=m*w;den=z.sum()
    return float((np.where(m,v,0)*w).sum()/den) if den else None
def root(v):return np.sqrt(v).item() if v is not None else None
def metrics(p,y,m,f,t,meta,threshold):
    p=p.astype(float);y=y.astype(float);m=m.astype(bool)
    count=m.sum(0);yc=np.divide(np.where(m,y,0).sum(0),count,out=np.zeros(len(t)),where=count>0)
    pc=np.divide(np.where(m,p,0).sum(0),count,out=np.zeros(len(t)),where=count>0)
    yd=y-yc;ed=(p-pc)-yd;e=p-y
    w=weights(t.astype(float))
    high=np.array([root(weighted(v*v,mm,w)) or 0 for v,mm in zip(yd,m)])>=threshold
    rows=[]
    for window,lo,hi in [('full',0,30),('0_2',0,2),('2_10',2,10),('10_30',10,30)]:
        sel=(t>=lo)&(t<=hi);tt=t[sel];ww=weights(tt.astype(float));mm=m[:,sel];ee=e[:,sel];dd=ed[:,sel]
        for j in range(len(p)):
            valid=mm[j];yy=y[j,sel];pp=p[j,sel]
            if not valid.any():continue
            adjacent=valid[1:]&valid[:-1];dt=np.diff(tt)
            roc=None
            if adjacent.any():roc=float(abs(np.max(np.abs(np.diff(pp)[adjacent]/dt[adjacent]))-np.max(np.abs(np.diff(yy)[adjacent]/dt[adjacent]))))
            rows.append(dict(meta,kind='device',device_index=j,family=int(f[j]),stratum='HIGH' if high[j] else 'BELOW_HIGH',window=window,differential_rmse=root(weighted(dd[j]**2,valid,ww)),nadir_error=float(abs(pp[valid].min()-yy[valid].min())),max_abs_rocof_error=roc,terminal_differential_error=float(abs(dd[j,np.flatnonzero(valid)[-1]]))))
        for a,b,label in [(0,1,'SG_GFM'),(0,2,'SG_GFL'),(1,2,'GFM_GFL'),(-1,-1,'ALL')]:
            ia=np.arange(len(p)) if a<0 else np.flatnonzero(f==a);ib=np.arange(len(p)) if b<0 else np.flatnonzero(f==b)
            ii,jj=np.meshgrid(ia,ib,indexing='ij');ii=ii.ravel();jj=jj.ravel()
            if a<0:keep=ii<jj;ii=ii[keep];jj=jj[keep]
            if not len(ii):continue
            pm=mm[ii]&mm[jj];pe=ee[ii]-ee[jj]
            rows.append(dict(meta,kind='pair',pair=label,window=window,pair_rmse=root(weighted(pe**2,pm,ww)),pair_mae=weighted(abs(pe),pm,ww),pair_error_sup=float(np.max(np.abs(pe[pm]))) if pm.any() else None))
        ok=mm.sum(0)>=2
        if ok.any():
            yp=np.where(mm,y[:,sel],-np.inf).max(0)-np.where(mm,y[:,sel],np.inf).min(0)
            pp=np.where(mm,p[:,sel],-np.inf).max(0)-np.where(mm,p[:,sel],np.inf).min(0)
            rows.append(dict(meta,kind='spread',window=window,spread_mae=weighted(abs(pp-yp),ok,ww),spread_sup_error=float(abs(pp[ok]-yp[ok]).max())))
    return rows


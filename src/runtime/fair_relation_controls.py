"""Input-only, same-apparatus-space relation controls. No fitting/targets."""
import numpy as np

ARMS=('NO_MESSAGE','POST_ELECTRICAL_KRON','PRE_F','POST_F')

def components(y):
    adj=(np.abs(y)>0)|(np.abs(y.T)>0);np.fill_diagonal(adj,False)
    unseen=set(range(len(y)));out=[]
    while unseen:
        stack=[min(unseen)];unseen.remove(stack[0]);group=[]
        while stack:
            i=stack.pop();group.append(i)
            nxt=sorted(unseen.intersection(np.flatnonzero(adj[i]).tolist()))
            unseen.difference_update(nxt);stack.extend(nxt)
        out.append(np.array(sorted(group),int))
    return out

def ac_jacobian(v,theta,y):
    g=y.real;b=y.imag;angle=theta[:,None]-theta[None,:]
    c=np.cos(angle);s=np.sin(angle);vv=np.outer(v,v)
    H=vv*(g*s-b*c);N=v[:,None]*(g*c+b*s)
    J=-vv*(g*c+b*s);L=v[:,None]*(g*s-b*c)
    p=(vv*(g*c+b*s)).sum(1);q=(vv*(g*s-b*c)).sum(1);i=np.arange(len(v))
    H[i,i]=-q-v*v*np.diag(b);N[i,i]=p/v+v*np.diag(g)
    J[i,i]=p-v*v*np.diag(g);L[i,i]=q/v-v*np.diag(b)
    return H,N,J,L

def reduce_ac(y,v,theta,keep):
    H,N,J,L=ac_jacobian(v,theta,y)
    passive=np.array([i for i in range(len(y)) if i not in set(keep)],int)
    A=H[np.ix_(keep,keep)]
    if not len(passive):return A
    B=np.hstack((H[np.ix_(keep,passive)],N[np.ix_(keep,passive)]))
    C=np.vstack((H[np.ix_(passive,keep)],J[np.ix_(passive,keep)]))
    D=np.block([[H[np.ix_(passive,passive)],N[np.ix_(passive,passive)]],[J[np.ix_(passive,passive)],L[np.ix_(passive,passive)]]])
    return A-B@np.linalg.solve(D,C)

def reduce_electrical(y,keep):
    # Magnitude admittance Laplacian, NOT AC sensitivity and NOT a dynamic model.
    w=(np.abs(y)+np.abs(y.T))/2;np.fill_diagonal(w,0)
    lap=np.diag(w.sum(1))-w
    passive=np.array([i for i in range(len(y)) if i not in set(keep)],int)
    reduced=lap[np.ix_(keep,keep)]
    if len(passive):
        reduced=reduced-lap[np.ix_(keep,passive)]@np.linalg.solve(lap[np.ix_(passive,passive)],lap[np.ix_(passive,keep)])
    return reduced

def construct(y,v,theta,bus_ids,terminals,ratings,active,kind):
    """All candidate terminals retained, then C lift, then availability A."""
    y=np.asarray(y,np.complex128);v=np.asarray(v,float);theta=np.asarray(theta,float)
    ids=np.asarray(bus_ids);terminals=np.asarray(terminals);ratings=np.maximum(np.asarray(ratings,float),1e-6)
    active=np.asarray(active,float);out=np.zeros((len(terminals),len(terminals)))
    for group in components(y):
        lookup={int(ids[row]):i for i,row in enumerate(group)}
        dr=np.array([i for i,t in enumerate(terminals) if int(t) in lookup],int)
        if not len(dr):continue
        unique=list(dict.fromkeys(terminals[dr].tolist()));keep=np.array([lookup[int(t)] for t in unique])
        local_y=y[np.ix_(group,group)]
        f=reduce_ac(local_y,v[group],theta[group],keep) if kind=='AC' else reduce_electrical(local_y,keep)
        C=np.zeros((len(unique),len(dr)))
        for i,t in enumerate(unique):
            hits=np.flatnonzero(terminals[dr]==t);r=ratings[dr[hits]];C[i,hits]=r/r.sum()
        out[np.ix_(dr,dr)]=C.T@f@C
    out=active[:,None]*out*active[None,:]
    if not np.isfinite(out).all():raise ValueError('nonfinite relation; no fallback allowed')
    return out

def apply_control(item,arm,cache=None):
    """Adapter before frozen batching; all other inputs unchanged."""
    if arm not in ARMS:raise ValueError(arm)
    out=dict(item)
    if arm=='NO_MESSAGE':
        out['phy_index']=np.zeros((2,0),np.int64);out['phy_attr']=np.zeros((0,6),np.float32)
    elif arm!='POST_F':
        if cache is None:raise ValueError('control cache required')
        out['phy_index']=cache[arm+'_index'];out['phy_attr']=cache[arm+'_attr']
    return out

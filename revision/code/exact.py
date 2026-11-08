"""Analytic parameter sensitivities for the initialization comparison."""
import numpy as np
from model import Model, PAULI, MEAS

def exact_metrics(model,v):
    n=model.N*model.substeps;dt=model.dt;p=model.p
    v=np.repeat(np.asarray(v).reshape(model.N,3),model.substeps,axis=0)
    z=np.repeat((v+model.target[0])[:,None,:],2,axis=1);z[:,:,2]+=np.array([model.J,-model.J])
    r=np.linalg.norm(z,axis=-1);a=dt*np.sinc(r*dt/np.pi)
    mat=np.einsum('...k,kij->...ij',z,PAULI)
    u=np.cos(r*dt)[...,None,None]*np.eye(2)-1j*a[...,None,None]*mat
    da=np.empty_like(r);mask=r*dt>1e-5
    da[mask]=(dt*r[mask]*np.cos(r[mask]*dt)-np.sin(r[mask]*dt))/r[mask]**3
    da[~mask]=-dt**3/3+dt**5*r[~mask]**2/30
    du=(-dt*a[...,None,None,None]*z[..., :,None,None]*np.eye(2)
        -1j*da[...,None,None,None]*z[..., :,None,None]*mat[...,None,:,:]
        -1j*a[...,None,None,None]*PAULI)
    ts=(np.arange(n)+getattr(model,"time_shift",0))*dt;dx=np.zeros((n,p,3))
    if model.kind=='static':
        b,th,ph=model.x
        dx[:,0]=[np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)]
        dx[:,1]=[b*np.cos(th)*np.cos(ph),b*np.cos(th)*np.sin(ph),-b*np.sin(th)]
        dx[:,2]=[-b*np.sin(th)*np.sin(ph),b*np.sin(th)*np.cos(ph),0]
    elif model.kind in ['xy','xz']:
        b,w=model.x;k=1 if model.kind=='xy' else 2
        dx[:,0,0]=-np.cos(w*ts);dx[:,0,k]=-np.sin(w*ts)
        dx[:,1,0]=b*ts*np.sin(w*ts);dx[:,1,k]=-b*ts*np.cos(w*ts)
    else:
        b,w=model.x;dx[:,0,0]=np.cos(w*ts);dx[:,1,0]=-b*ts*np.sin(w*ts)
    dup=np.einsum('nbkij,npk->npbij',du,dx)
    psi=np.eye(2,dtype=complex)/np.sqrt(2);ds=np.zeros((p,2,2),complex)
    for k in range(n):
        ds=np.einsum('bij,pbj->pbi',u[k],ds)+np.einsum('pbij,bj->pbi',dup[k],psi)
        psi=np.einsum('bij,bj->bi',u[k],psi)
    psi=psi.T.ravel();ds=ds.transpose(0,2,1).reshape(p,4)
    amps=MEAS@psi;damps=ds@MEAS.T;probs=abs(amps)**2
    dp=2*np.real(amps.conj()*damps)
    f=(dp/np.maximum(probs,1e-30))@dp.T
    eig=np.linalg.eigvalsh(f);cost=float(np.trace(np.linalg.inv(f))) if eig[0]>0 else float('inf')
    proxy=float(np.sum(1/np.diag(f)))
    ov=ds.conj()@psi;tangent=ds.conj()@ds.T-np.outer(ov,ov.conj());q=4*tangent.real
    return dict(cost=cost,proxy_cost=proxy,ratio=cost/proxy,F=f.tolist(),QFIM=q.tolist(),
        min_probability=float(probs.min()),min_eigenvalue_Q_minus_F=float(np.linalg.eigvalsh(q-f)[0]),
        eigenvalues=eig.tolist(),condition=float(eig[-1]/eig[0]))

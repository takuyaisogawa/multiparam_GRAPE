"""Memory-bounded midpoint evaluation for the sensitive xy endpoint."""
from propagation import *
from numba import njit

@njit(cache=True)
def carry(u,dup,psi,ds):
 p=dup.shape[1];newpsi=np.empty_like(psi);newds=np.empty_like(ds)
 for n in range(len(u)):
  for b in range(2):
   for i in range(2):
    newpsi[b,i]=u[n,b,i,0]*psi[b,0]+u[n,b,i,1]*psi[b,1]
    for a in range(p):newds[a,b,i]=(u[n,b,i,0]*ds[a,b,0]+u[n,b,i,1]*ds[a,b,1]+dup[n,a,b,i,0]*psi[b,0]+dup[n,a,b,i,1]*psi[b,1])
  psi,newpsi=newpsi,psi;ds,newds=newds,ds
 return psi,ds

def chunked_state(kind,T,J,v,x,sub,block=16000):
 N=len(v);n=N*sub;dt=T/n;psi=np.eye(2,dtype=complex)/np.sqrt(2);ds=np.zeros((len(x),2,2),complex)
 for start in range(0,n,block):
  ix=np.arange(start,min(start+block,n));z,dx=ms.field(kind,x,(ix+.5)*dt)
  z=np.repeat((v[ix//sub]+z)[:,None,:],2,axis=1);z[:,:,2]+=np.array([J,-J]);r=np.linalg.norm(z,axis=-1)
  a=dt*np.sinc(r*dt/np.pi);mat=np.einsum('...k,kij->...ij',z,PAULI);u=np.cos(r*dt)[...,None,None]*np.eye(2)-1j*a[...,None,None]*mat
  da=np.empty_like(r);mask=r*dt>1e-5;da[mask]=(dt*r[mask]*np.cos(r[mask]*dt)-np.sin(r[mask]*dt))/r[mask]**3;da[~mask]=-dt**3/3+dt**5*r[~mask]**2/30
  du=(-dt*a[...,None,None,None]*z[..., :,None,None]*np.eye(2)-1j*da[...,None,None,None]*z[..., :,None,None]*mat[...,None,:,:]-1j*a[...,None,None,None]*PAULI)
  psi,ds=carry(u,np.einsum('nbkij,npk->npbij',du,dx),psi,ds)
 return psi.T.ravel(),ds.transpose(0,2,1).reshape(len(x),4)

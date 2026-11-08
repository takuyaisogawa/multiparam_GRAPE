"""The three recorded replacement optimizations."""
from propagation import *
from numba import njit
from scipy.optimize import minimize
import time

@njit(cache=True)
def forward(u):
 c,n,b,_,_=u.shape;s=np.zeros((n+1,c,2,2),np.complex128)
 for a in range(c):s[0,a,0,0]=1/np.sqrt(2);s[0,a,1,1]=1/np.sqrt(2)
 for i in range(n):
  for a in range(c):
   for k in range(2):
    for q in range(2):s[i+1,a,k,q]=u[a,i,k,q,0]*s[i,a,k,0]+u[a,i,k,q,1]*s[i,a,k,1]
 return s

@njit(cache=True)
def reverse(u,du,s,lam):
 c,n,_,_,_=u.shape;g=np.zeros((n,3));new=np.empty_like(lam)
 for i in range(n-1,-1,-1):
  for a in range(c):
   for b in range(2):
    for k in range(3):
     for q in range(2):g[i,k]+=2*(lam[a,b,q].conjugate()*(du[a,i,b,k,q,0]*s[i,a,b,0]+du[a,i,b,k,q,1]*s[i,a,b,1])).real
    for q in range(2):new[a,b,q]=u[a,i,b,0,q].conjugate()*lam[a,b,0]+u[a,i,b,1,q].conjugate()*lam[a,b,1]
  lam,new=new,lam
 return g

class FastModel(Model):
 def __init__(self,kind,T,N,J,h=1e-4,substeps=1,shift=.5,floor=1e-14,ridge=1e-12):
  super().__init__(kind,T,N,J,h,substeps);self.floor=floor;self.ridge=ridge
  pars=[self.x.copy()]
  for i in range(self.p):
   for sign in (1,-1):
    x=self.x.copy();x[i]+=sign*h;pars.append(x)
  self.target=np.array([ms.field(kind,x,(np.arange(N*substeps)+shift)*self.dt)[0] for x in pars])
 def objective(self,v):
  v=np.asarray(v).reshape(self.N,3);z=np.repeat(v,self.substeps,axis=0)[None]+self.target
  z=np.repeat(z[:,:,None,:],2,axis=2);z[:,:,:,2]+=np.array([self.J,-self.J]);r=np.linalg.norm(z,axis=-1);dt=self.dt
  a=dt*np.sinc(r*dt/np.pi);mat=np.einsum('...k,kij->...ij',z,PAULI);u=np.cos(r*dt)[...,None,None]*np.eye(2)-1j*a[...,None,None]*mat
  s=forward(u);psi=s[-1].transpose(0,2,1).reshape(-1,4);amps=psi@MEAS.T;probs=abs(amps)**2
  d=(probs[1::2]-probs[2::2])/(2*self.h);den=np.maximum(probs[0],self.floor);F=(d/den)@d.T
  diag=np.diag(F)+self.ridge;cost=np.sum(1/diag);gf=np.diag(-1/diag**2)/cost
  gd=2*gf@d/den;gp=np.zeros_like(probs);gp[0]=-np.sum(d*(gf@d),axis=0)/den**2;gp[0,probs[0]<self.floor]=0
  gp[1::2]=gd/(2*self.h);gp[2::2]=-gd/(2*self.h)
  da=np.empty_like(r);mask=r*dt>1e-5;da[mask]=(dt*r[mask]*np.cos(r[mask]*dt)-np.sin(r[mask]*dt))/r[mask]**3;da[~mask]=-dt**3/3+dt**5*r[~mask]**2/30
  du=(-dt*a[...,None,None,None]*z[..., :,None,None]*np.eye(2)-1j*da[...,None,None,None]*z[..., :,None,None]*mat[...,None,:,:]-1j*a[...,None,None,None]*PAULI)
  lam=((gp*amps)@MEAS.conj()).reshape(-1,2,2).transpose(0,2,1).copy()
  grad=reverse(u,du,s,lam).reshape(self.N,self.substeps,3).sum(axis=1).ravel()
  return float(np.log(cost)),grad

def optimize_case(job):
 tag,kind,T,J,N,source=job;dest=OUT/'reoptimized';dest.mkdir(exist_ok=True)
 v=input_control(source);m=FastModel(kind,T,N,J,substeps=8)
 hist=[];best=[np.inf,v.ravel().copy()];cap=1200;start=time.perf_counter()
 class Stop(Exception):pass
 def fun(z):
  if len(hist)>=cap:raise Stop()
  val,g=m.objective(z)
  if val<best[0]:best[:]=[val,z.copy()]
  hist.append(dict(evaluation=len(hist)+1,objective=val,best_objective=best[0],gradient_inf=float(np.max(abs(g)))))
  if len(hist)%100==0:save(dest/(tag+'_progress.json'),dict(calls=len(hist),elapsed_s=time.perf_counter()-start,best=best[0]))
  return val,g
 attempts=[]
 for attempt in range(3):
  before=len(hist)
  try:
   rr=minimize(fun,best[1],jac=True,method='L-BFGS-B',options=dict(ftol=0.,gtol=0.,maxls=100,maxcor=10,maxiter=cap,maxfun=cap));attempts.append(dict(message=str(rr.message),calls=len(hist)-before))
   if len(hist)==before:break
  except Stop:attempts.append(dict(message='budget exhausted',calls=len(hist)-before));break
 np.savez_compressed(OUT/'controls'/(tag+'.npz'),v=best[1].reshape(N,3))
 row=dict(id=tag,kind=kind,T=T,J=J,N=N,arm='new_reoptimization',source_pulse=source,optimization_substeps=8,budget_cap=cap,evaluations=len(hist),objective='-log f0 with probability floor 1e-14 and diagonal ridge 1e-12',method='L-BFGS-B',parameter_h=1e-4,history=hist,attempts=attempts,elapsed_s=time.perf_counter()-start,provenance='New recorded calculation, not recovered original run')
 save(dest/(tag+'.json'),row);result=evaluate(row)
 return dict(id=tag,C=result['refined']['C'],evaluations=len(hist))

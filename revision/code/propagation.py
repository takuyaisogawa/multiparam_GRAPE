"""Fixed waveform propagation, finite differences and probability checks."""
import numpy as np,json
import mismatch as ms
from model import PAULI, MEAS, Model
from optimizer import save
from paths import output, control
OUT=output('propagation')
(OUT/'records').mkdir(exist_ok=True)
(OUT/'controls').mkdir(exist_ok=True)
def input_control(tag):
    local=OUT/'controls'/(tag+'.npz')
    return np.load(local)['v'] if local.exists() else control(tag)

def state(kind,T,J,v,x,sub=1,shift=.5):
 N=len(v);n=N*sub;dt=T/n
 z,dx=ms.field(kind,x,(np.arange(n)+shift)*dt)
 z=np.repeat((np.repeat(v,sub,axis=0)+z)[:,None,:],2,axis=1);z[:,:,2]+=np.array([J,-J])
 r=np.linalg.norm(z,axis=-1);a=dt*np.sinc(r*dt/np.pi);mat=np.einsum('...k,kij->...ij',z,PAULI)
 u=np.cos(r*dt)[...,None,None]*np.eye(2)-1j*a[...,None,None]*mat
 da=np.empty_like(r);mask=r*dt>1e-5
 da[mask]=(dt*r[mask]*np.cos(r[mask]*dt)-np.sin(r[mask]*dt))/r[mask]**3
 da[~mask]=-dt**3/3+dt**5*r[~mask]**2/30
 du=(-dt*a[...,None,None,None]*z[..., :,None,None]*np.eye(2)-1j*da[...,None,None,None]*z[..., :,None,None]*mat[...,None,:,:]-1j*a[...,None,None,None]*PAULI)
 psi,ds=ms.chain(u,np.einsum('nbkij,npk->npbij',du,dx))
 return psi.T.ravel(),ds.transpose(0,2,1).reshape(len(x),4)

def metric(psi,ds,floor=0.):
 amps=MEAS@psi;dp=2*np.real(amps.conj()*(ds@MEAS.T));prob=abs(amps)**2
 return fisher(prob,dp,floor)

def fisher(prob,dp,floor=0.):
 den=np.maximum(prob,floor)
 if np.any(den<=0):return dict(C=None,exact_zero_probability=True,probabilities=prob.tolist())
 W=dp/np.sqrt(den);F=W@W.T;sv=np.linalg.svd(W,compute_uv=False);eig=np.linalg.eigvalsh(F)
 C=float(np.sum(1/sv**2));direct=float(np.trace(np.linalg.inv(F))) if eig[0]>0 else None
 return dict(C=C,F=F.tolist(),probabilities=prob.tolist(),dp=dp.tolist(),min_probability=float(prob.min()),
  singular_values=sv.tolist(),eigenvalues=eig.tolist(),condition=float((sv[0]/sv[-1])**2),
  C_direct_inverse=direct,inverse_relative_difference=abs(direct/C-1) if direct else None,
  f0=float(1/np.sum(1/np.diag(F))))

def evaluate(row):
 dest=OUT/'records'/(row['id']+'.json')
 if dest.exists():return json.loads(dest.read_text())
 kind=row['kind'];T=row['T'];J=row['J'];N=row['N'];x=np.array(row.get('x',ms.nominal(kind)))
 v=input_control(row.get('pulse_id',row['id']))
 if row.get('arm')=='analytical':v=Model(kind,T,N,J).initial()
 native=metric(*state(kind,T,J,v,x,1,0.));levels=[];previous=None;stable=0
 for sub in (1,2,4,8,16,32,64,128,256,512):
  met=metric(*state(kind,T,J,v,x,sub,.5));C=met['C']
  delta=abs(C/previous['C']-1) if previous else None
  fdelta=float(np.linalg.norm(np.array(met['F'])-previous['F'])/np.linalg.norm(previous['F'])) if previous else None
  stable=stable+1 if delta is not None and max(delta,fdelta)<1e-4 else 0
  levels.append(dict(substeps=sub,C=C,C_change=delta,F_change=fdelta))
  if stable>=2:break
  previous=met
 # Independently vary parameter differences at the same refined propagation.
 hs=[];floor_tests=[]
 p=np.array(met['probabilities']);dp=np.array(met['dp']);F=np.array(met['F'])
 for h in (1e-3,1e-4,1e-5,1e-6):
  ds=[]
  for k in range(len(x)):
   xp=x.copy();xm=x.copy();xp[k]+=h;xm[k]-=h
   pp=abs(MEAS@state(kind,T,J,v,xp,sub)[0])**2;pm=abs(MEAS@state(kind,T,J,v,xm,sub)[0])**2
   ds.append((pp-pm)/(2*h))
  fd=fisher(p,np.array(ds));hs.append(dict(h=h,C=fd['C'],C_relative_error=fd['C']/met['C']-1,F_relative_error=float(np.linalg.norm(np.array(fd['F'])-F)/np.linalg.norm(F)),dp_relative_error=float(np.linalg.norm(np.array(ds)-dp)/np.linalg.norm(dp))))
 for floor in (0.,1e-30,1e-16,1e-14,1e-12,1e-10,1e-8):
  fm=fisher(p,dp,floor);floor_tests.append(dict(floor=floor,C=fm['C'],relative_change=fm['C']/met['C']-1))
 r=dict(**row,native=native,refined=met,levels=levels,substeps=sub,grid_converged=stable>=2,
  native_to_refined_relative_change=met['C']/native['C']-1,parameter_difference_checks=hs,floor_checks=floor_tests)
 if row.get('arm','saved')=='saved' and 'C_plot' in row:r['plot_to_refined_relative_change']=met['C']/row['C_plot']-1
 save(dest,r);return r

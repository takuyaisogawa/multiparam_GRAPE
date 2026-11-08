"""Independent propagation and directional-gradient checks for the comparison."""
import numpy as np
from scipy.integrate import solve_ivp
from model import Model,CASES,random_initial,PAULI,MEAS
from optimizer import objective,save
from paths import load,initialization_control,output
P=np.array([np.kron(q,np.eye(2)) for q in PAULI])
ZZ=np.kron(PAULI[2],PAULI[2])

def ode_metric(kind,T,N,J,v):
 m=Model(kind,T,N,J);x=m.x;p=m.p;y=np.zeros((p+1,4),complex);y[0]=np.array([1,0,0,1])/np.sqrt(2);dt=T/N;nfev=0
 def rhs(t,flat,control):
  z=np.zeros(3);dz=np.zeros((p,3))
  if kind=='static':
   b,th,ph=x;z[:]=[b*np.sin(th)*np.cos(ph),b*np.sin(th)*np.sin(ph),b*np.cos(th)];dz[0]=z/b;dz[1]=[b*np.cos(th)*np.cos(ph),b*np.cos(th)*np.sin(ph),-b*np.sin(th)];dz[2]=[-b*np.sin(th)*np.sin(ph),b*np.sin(th)*np.cos(ph),0]
  elif kind in ['xz','xy']:
   b,w=x;k=2 if kind=='xz' else 1;z[0]=-b*np.cos(w*t);z[k]=-b*np.sin(w*t);dz[0]=z/b;dz[1,0]=b*t*np.sin(w*t);dz[1,k]=-b*t*np.cos(w*t)
  else:
   b,w=x;z[0]=b*np.cos(w*t);dz[0,0]=np.cos(w*t);dz[1,0]=-b*t*np.sin(w*t)
  H=np.einsum('k,kij->ij',z+control,P)+J*ZZ;dH=np.einsum('pk,kij->pij',dz,P);yy=flat.reshape(p+1,4);dy=-1j*(yy@H.T);dy[1:]+=-1j*np.einsum('pij,j->pi',dH,yy[0]);return dy.ravel()
 for i in range(N):
  sol=solve_ivp(rhs,[i*dt,(i+1)*dt],y.ravel(),args=(v[i],),method='DOP853',rtol=1e-10,atol=1e-12);assert sol.success;y=sol.y[:,-1].reshape(p+1,4);nfev+=sol.nfev
 amps=MEAS@y[0];damps=y[1:]@MEAS.T;prob=abs(amps)**2;dp=2*np.real(amps.conj()*damps);F=(dp/np.maximum(prob,1e-30))@dp.T
 return dict(C=float(np.trace(np.linalg.inv(F))),F=F.tolist(),norm=float(np.linalg.norm(y[0])),nfev=nfev)

def gradient_checks():
    checks=[]
    for kind,cfg in CASES.items():
     for J in cfg['path']:
      m=Model(kind,cfg['T'],cfg['N'],J);rng=np.random.default_rng(773);v=random_initial(m,rng);d=rng.normal(size=v.shape);d/=np.linalg.norm(d);val,g,_=objective(m,v,'log_proxy');ad=g@d.ravel();tests=[]
      for h in [1e-3,3e-4,1e-4]:
       fd=(objective(m,v+h*d,'log_proxy')[0]-objective(m,v-h*d,'log_proxy')[0])/(2*h);tests.append(float(abs(fd-ad)/max(abs(fd),abs(ad),1e-12)))
      checks.append(dict(model=kind,J=J,best_relative_error=min(tests)))
    assert max(r["best_relative_error"] for r in checks)<.005
    return checks


def run(record_id=None):
    gradients=[]
    if record_id is None:
        gradients=gradient_checks()
    byid={r['tag']:r for r in load('initialization.json')}
    records=load('initialization_checks.json')['independent_ODE']
    if record_id:records=[r for r in records if r['tag']==record_id]
    if not records:raise ValueError('No independent-check record with that ID.')
    results=[]
    for r in records:
        saved=byid[r['tag']];kind=saved['job']['kind']
        ode=ode_metric(kind,saved['T'],saved['N'],saved['J'],initialization_control(r['tag']))
        difference=saved['reference']['cost']/ode['C']-1
        assert abs(difference)<1e-4
        results.append(dict(id=r['tag'],ODE=ode,reference_C=saved['reference']['cost'],relative_difference=difference))
    save(output('initialization_checks')/'checks.json',dict(gradient=gradients,independent_ODE=results))
    print('Passed',len(gradients),'gradient checks and',len(results),'independent ODE checks.')

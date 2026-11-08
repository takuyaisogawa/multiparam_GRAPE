"""Independent full-state integration of saved controls."""
from propagation import *
from scipy.integrate import solve_ivp
(OUT/"ode_checks").mkdir(exist_ok=True)

def run_ode(row):
 kind=row['kind'];T=row['T'];J=row['J'];N=row['N'];x=np.array(row.get('x',ms.nominal(kind)));p=len(x)
 v=input_control(row.get('pulse_id',row['id']))
 if row.get('arm')=='analytical':v=Model(kind,T,N,J).initial()
 P=np.array([np.kron(s,np.eye(2)) for s in PAULI]);ZZ=np.kron(PAULI[2],PAULI[2]);dt=T/N
 y=np.zeros((p+1,4),complex);y[0]=np.array([1,0,0,1])/np.sqrt(2)
 def rhs(t,flat,c):
  z,dz=ms.field(kind,x,np.array([t]));H=np.einsum('k,kij->ij',z[0]+c,P)+J*ZZ
  dH=np.einsum('pk,kij->pij',dz[0],P);yy=flat.reshape(p+1,4);dy=-1j*(yy@H.T);dy[1:]+=-1j*np.einsum('pij,j->pi',dH,yy[0]);return dy.ravel()
 for i in range(N):
  sol=solve_ivp(rhs,[i*dt,(i+1)*dt],y.ravel(),args=(v[i],),method='DOP853',rtol=2e-11,atol=2e-13)
  assert sol.success;y=sol.y[:,-1].reshape(p+1,4)
 met=metric(y[0],y[1:]);r=dict(id=row['id'],ODE=met,C_relative_difference=met['C']/row['refined']['C']-1,F_relative_difference=float(np.linalg.norm(np.array(met['F'])-row['refined']['F'])/np.linalg.norm(met['F'])))
 save(OUT/'ode_checks'/(row['id']+'.json'),r);return r

"""Fixed-control parameter mismatch; reruns write to build/mismatch/."""
import numpy as np,json
from numba import njit
from scipy.integrate import solve_ivp
from model import PAULI, MEAS, Model
from exact import exact_metrics
from optimizer import save
from paths import output, control
CALC=output('mismatch')
(CALC/'records').mkdir(exist_ok=True)
(CALC/'ode_checks').mkdir(exist_ok=True)

CASES={'static':(8.,.2,50),'xz':(8.,10.,500),'xy':(6.,10.,500),'linear':(1.4,20.,1000)}

def nominal(kind):
    return np.array([1.,np.pi/4,np.pi/4]) if kind=='static' else np.array([1.,100. if kind=='linear' else 1.])

def field(kind,x,ts):
    p=len(x);z=np.zeros((len(ts),3));dx=np.zeros((len(ts),p,3))
    if kind=='static':
        b,th,ph=x
        z[:]=[b*np.sin(th)*np.cos(ph),b*np.sin(th)*np.sin(ph),b*np.cos(th)]
        dx[:,0]=z[0]/b
        dx[:,1]=[b*np.cos(th)*np.cos(ph),b*np.cos(th)*np.sin(ph),-b*np.sin(th)]
        dx[:,2]=[-b*np.sin(th)*np.sin(ph),b*np.sin(th)*np.cos(ph),0]
    elif kind in ('xz','xy'):
        b,w=x;k=2 if kind=='xz' else 1
        z[:,0]=-b*np.cos(w*ts);z[:,k]=-b*np.sin(w*ts)
        dx[:,0]=z/b;dx[:,1,0]=b*ts*np.sin(w*ts);dx[:,1,k]=-b*ts*np.cos(w*ts)
    else:
        b,w=x;z[:,0]=b*np.cos(w*ts)
        dx[:,0,0]=np.cos(w*ts);dx[:,1,0]=-b*ts*np.sin(w*ts)
    return z,dx

@njit(cache=True)
def chain(u,dup):
    p=dup.shape[1];psi=np.eye(2,dtype=np.complex128)/np.sqrt(2.)
    ds=np.zeros((p,2,2),np.complex128)
    newpsi=np.empty_like(psi);newds=np.empty_like(ds)
    for n in range(len(u)):
        for b in range(2):
            for i in range(2):
                newpsi[b,i]=u[n,b,i,0]*psi[b,0]+u[n,b,i,1]*psi[b,1]
                for a in range(p):
                    newds[a,b,i]=(u[n,b,i,0]*ds[a,b,0]+u[n,b,i,1]*ds[a,b,1]
                        +dup[n,a,b,i,0]*psi[b,0]+dup[n,a,b,i,1]*psi[b,1])
        psi,newpsi=newpsi,psi
        ds,newds=newds,ds
    return psi,ds

def state(kind,x,v,sub):
    T,J,N=CASES[kind];n=N*sub;dt=T/n
    target,dx=field(kind,x,(np.arange(n)+.5)*dt)
    z=np.repeat((np.repeat(v,sub,axis=0)+target)[:,None,:],2,axis=1)
    z[:,:,2]+=np.array([J,-J]);r=np.linalg.norm(z,axis=-1)
    a=dt*np.sinc(r*dt/np.pi);mat=np.einsum('...k,kij->...ij',z,PAULI)
    u=np.cos(r*dt)[...,None,None]*np.eye(2)-1j*a[...,None,None]*mat
    da=np.empty_like(r);mask=r*dt>1e-5
    da[mask]=(dt*r[mask]*np.cos(r[mask]*dt)-np.sin(r[mask]*dt))/r[mask]**3
    da[~mask]=-dt**3/3+dt**5*r[~mask]**2/30
    du=(-dt*a[...,None,None,None]*z[..., :,None,None]*np.eye(2)
        -1j*da[...,None,None,None]*z[..., :,None,None]*mat[...,None,:,:]
        -1j*a[...,None,None,None]*PAULI)
    dup=np.einsum('nbkij,npk->npbij',du,dx)
    psi,ds=chain(u,dup)
    return psi.T.ravel(),ds.transpose(0,2,1).reshape(len(x),4)

def metrics(psi,ds):
    amps=MEAS@psi;damps=ds@MEAS.T;prob=abs(amps)**2;dp=2*np.real(amps.conj()*damps)
    if np.min(prob)<=1e-25:raise FloatingPointError('near-zero probability requires an explicit limiting treatment')
    F=(dp/prob)@dp.T;eig=np.linalg.eigvalsh(F)
    C=float(np.trace(np.linalg.inv(F))) if eig[0]>0 else None
    ov=ds.conj()@psi;tangent=ds.conj()@ds.T-np.outer(ov,ov.conj());Q=4*tangent.real
    return dict(C=C,F=F.tolist(),probabilities=prob.tolist(),probability_derivatives=dp.tolist(),
                eigenvalues=eig.tolist(),condition=float(eig[-1]/eig[0]) if eig[0]>0 else None,
                Q=Q.tolist(),min_eigenvalue_Q_minus_F=float(np.linalg.eigvalsh(Q-F)[0]),
                norm=float(np.vdot(psi,psi).real))

def pulse(kind,arm):
    if arm=='saved':return control(f'mismatch_{kind}_saved')
    return control(f'mismatch_{kind}_analytical')

def evaluate(job):
    dest=CALC/'records'/(job['id']+'.json')
    if dest.exists():return json.loads(dest.read_text())
    kind=job['kind'];v=pulse(kind,job['arm']);x=np.array(job['true_parameters'])
    levels=[];previous=None;stable=0
    for sub in (1,2,4,8,16,32,64,128,256):
        met=metrics(*state(kind,x,v,sub));C=met['C']
        change=abs(C/previous['C']-1) if previous and C is not None and previous['C'] else None
        fchange=float(np.linalg.norm(np.array(met['F'])-np.array(previous['F']))/np.linalg.norm(previous['F'])) if previous else None
        stable=stable+1 if change is not None and max(change,fchange)<1e-4 else 0
        levels.append(dict(substeps=sub,C=C,C_relative_change=change,F_relative_change=fchange))
        if stable>=2:break
        previous=met
    result=dict(**job,metrics=met,grid_converged=stable>=2,levels=levels,substeps=sub)
    save(dest,result);return result

def independent_ode(row):
    kind=row['kind'];T,J,N=CASES[kind];x=np.array(row['true_parameters']);p=len(x);v=pulse(kind,row['arm'])
    P=np.array([np.kron(s,np.eye(2)) for s in PAULI]);ZZ=np.kron(PAULI[2],PAULI[2])
    y=np.zeros((p+1,4),complex);y[0]=np.array([1,0,0,1])/np.sqrt(2);dt=T/N
    def rhs(t,flat,control):
        z,dz=field(kind,x,np.array([t]));H=np.einsum('k,kij->ij',z[0]+control,P)+J*ZZ
        dH=np.einsum('pk,kij->pij',dz[0],P);yy=flat.reshape(p+1,4);dy=-1j*(yy@H.T)
        dy[1:]+=-1j*np.einsum('pij,j->pi',dH,yy[0]);return dy.ravel()
    for i in range(N):
        sol=solve_ivp(rhs,[i*dt,(i+1)*dt],y.ravel(),args=(v[i],),method='DOP853',rtol=2e-11,atol=2e-13)
        assert sol.success;y=sol.y[:,-1].reshape(p+1,4)
    met=metrics(y[0],y[1:]);ref=row['metrics']
    delta=abs(met['C']/ref['C']-1);fdelta=float(np.linalg.norm(np.array(met['F'])-np.array(ref['F']))/np.linalg.norm(met['F']))
    result=dict(id=row['id'],ODE=met,grid_C=ref['C'],C_relative_difference=delta,F_relative_difference=fdelta)
    save(CALC/'ode_checks'/(row['id']+'.json'),result)
    return result

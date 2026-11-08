"""SLD and Holevo bounds for a fixed pure output state, with W=I."""
import numpy as np,itertools
from numba import njit
from scipy.linalg import null_space
from scipy.integrate import solve_ivp
from scipy.optimize import minimize,minimize_scalar,brentq
from optimizer import save
from paths import output
OUT=output('holevo')

PAULI = np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)

BELL = np.array([[1,0,0,-1],[0,1,1,0],[0,1,-1,0],[1,0,0,1]],complex)/np.sqrt(2)

ROT2 = np.cos(np.pi/3)*np.eye(2)-1j*np.sin(np.pi/3)*PAULI.sum(axis=0)/np.sqrt(3)

MEAS = BELL.conj()@np.kron(ROT2,np.eye(2))

def nominal(kind):
    return np.array([1,np.pi/4,np.pi/4]) if kind=='static' else np.array([1,100 if kind=='linear' else 1],float)

def field(kind,x,t):
    n=len(t);z=np.zeros((n,3));d=np.zeros((n,len(x),3))
    if kind=='static':
        b,th,ph=x
        z[:]=[b*np.sin(th)*np.cos(ph),b*np.sin(th)*np.sin(ph),b*np.cos(th)]
        d[:,0]=z[0]/b
        d[:,1]=[b*np.cos(th)*np.cos(ph),b*np.cos(th)*np.sin(ph),-b*np.sin(th)]
        d[:,2]=[-b*np.sin(th)*np.sin(ph),b*np.sin(th)*np.cos(ph),0]
    elif kind in ('xz','xy'):
        b,w=x;k=2 if kind=='xz' else 1
        z[:,0]=-b*np.cos(w*t);z[:,k]=-b*np.sin(w*t)
        d[:,0]=z/b;d[:,1,0]=b*t*np.sin(w*t);d[:,1,k]=-b*t*np.cos(w*t)
    else:
        b,w=x;z[:,0]=b*np.cos(w*t)
        d[:,0,0]=np.cos(w*t);d[:,1,0]=-b*t*np.sin(w*t)
    return z,d

@njit(cache=True)
def chain(u,du):
    p=du.shape[1];psi=np.eye(2,dtype=np.complex128)/np.sqrt(2)
    ds=np.zeros((p,2,2),np.complex128)
    newpsi=np.empty_like(psi);newds=np.empty_like(ds)
    for n in range(len(u)):
        for b in range(2):
            for i in range(2):
                newpsi[b,i]=u[n,b,i,0]*psi[b,0]+u[n,b,i,1]*psi[b,1]
                for a in range(p):
                    newds[a,b,i]=(u[n,b,i,0]*ds[a,b,0]+u[n,b,i,1]*ds[a,b,1]
                        +du[n,a,b,i,0]*psi[b,0]+du[n,a,b,i,1]*psi[b,1])
        psi,newpsi=newpsi,psi;ds,newds=newds,ds
    return psi.T.copy().ravel(),ds.transpose(0,2,1).copy().reshape(p,4)

def midpoint(kind,T,J,v,x,sub=1,shift=.5):
    n=len(v)*sub;dt=T/n;z,dx=field(kind,x,(np.arange(n)+shift)*dt)
    z=np.repeat((np.repeat(v,sub,axis=0)+z)[:,None,:],2,axis=1)
    z[:,:,2]+=np.array([J,-J]);r=np.linalg.norm(z,axis=-1)
    a=dt*np.sinc(r*dt/np.pi);mat=np.einsum('...k,kij->...ij',z,PAULI)
    u=np.cos(r*dt)[...,None,None]*np.eye(2)-1j*a[...,None,None]*mat
    da=np.empty_like(r);mask=r*dt>1e-5
    da[mask]=(dt*r[mask]*np.cos(r[mask]*dt)-np.sin(r[mask]*dt))/r[mask]**3
    da[~mask]=-dt**3/3+dt**5*r[~mask]**2/30
    du=(-dt*a[...,None,None,None]*z[..., :,None,None]*np.eye(2)
        -1j*da[...,None,None,None]*z[..., :,None,None]*mat[...,None,:,:]
        -1j*a[...,None,None,None]*PAULI)
    return chain(u,np.einsum('nbkij,npk->npbij',du,dx))

def independent_ode(kind,T,J,v,x):
    p=len(x);n=len(v);dt=T/n
    ops=np.array([np.kron(s,np.eye(2)) for s in PAULI]);ZZ=np.kron(PAULI[2],PAULI[2])
    y=np.zeros((p+1,4),complex);y[0]=np.array([1,0,0,1])/np.sqrt(2)
    def rhs(t,flat,c):
        z,d=field(kind,x,np.array([t]));H=np.einsum('k,kij->ij',z[0]+c,ops)+J*ZZ
        dH=np.einsum('pk,kij->pij',d[0],ops);yy=flat.reshape(p+1,4)
        dy=-1j*(yy@H.T);dy[1:]+=-1j*np.einsum('pij,j->pi',dH,yy[0])
        return dy.ravel()
    for i in range(n):
        sol=solve_ivp(rhs,[i*dt,(i+1)*dt],y.ravel(),args=(v[i],),method='DOP853',rtol=2e-12,atol=2e-14)
        assert sol.success;y=sol.y[:,-1].reshape(p+1,4)
    return y[0],y[1:]

def holevo(psi,ds):
    norm=float(np.vdot(psi,psi).real);psi=psi/np.sqrt(norm);ds=ds/np.sqrt(norm)
    p=len(ds);assert p in (2,3)
    basis=null_space(psi.conj()[None,:]);d=(basis.conj().T@ds.T).T
    m=d.shape[1];B=2*np.hstack([d.real,d.imag]);A=np.kron(np.eye(p),B);b=np.eye(p).ravel()
    feasible=A.T@np.linalg.solve(A@A.T,b);free=null_space(A)
    J=np.block([[np.zeros((m,m)),np.eye(m)],[-np.eye(m),np.zeros((m,m))]])
    pairs=list(itertools.combinations(range(p),2));nk=len(pairs)
    gram=d.conj()@d.T;Q=4*gram.real;CS=float(np.trace(np.linalg.inv(Q)))
    def evaluate(t):
        skew=np.zeros((p,p))
        for value,(i,j) in zip(t,pairs):skew[i,j]=value;skew[j,i]=-value
        H=np.eye(p*2*m)+np.kron(skew,J)
        u=feasible-free@np.linalg.solve(free.T@H@free,free.T@H@feasible)
        ur=u.reshape(p,2*m);uc=ur[:,:m]+1j*ur[:,m:]
        G=uc.conj()@uc.T
        imag=np.array([G[i,j].imag for i,j in pairs])
        lower=float(u@H@u)
        upper=float(np.trace(G).real+np.linalg.svd(G.imag,compute_uv=False).sum())
        return lower,upper,u,2*imag,G,H
    def fun(t):
        e=evaluate(t);return -e[0]/CS,-e[3]/CS
    scalar=None
    if nk==1:
        sc=minimize_scalar(lambda z:-evaluate([z])[0]/CS,bounds=(-1+1e-9,1-1e-9),method='bounded',options={'xatol':1e-13})
        scalar=-sc.fun*CS
        left,right=-1+1e-7,1-1e-7
        if evaluate([left])[3][0]*evaluate([right])[3][0]<0:
            z=brentq(lambda z:evaluate([z])[3][0],left,right,xtol=1e-14)
            t=np.array([z])
        else:t=np.array([sc.x])
        success=sc.success;message=sc.message
    else:
        def mapped(z):
            scale=np.sqrt(1+np.dot(z,z));t=z/scale
            val,grad=fun(t)
            jac=np.eye(nk)/scale-np.outer(z,z)/scale**3
            return val,jac@grad
        opt=minimize(mapped,np.zeros(nk),jac=True,method='BFGS',
            options={'gtol':1e-12,'maxiter':1000})
        t=opt.x/np.sqrt(1+np.dot(opt.x,opt.x))
        success=opt.success;message=opt.message
    lo,hi,u,g,G,H=evaluate(t)
    # Recover Hermitian estimator observables and verify original constraints.
    ur=u.reshape(p,2*m);uv=(basis@(ur[:,:m]+1j*ur[:,m:]).T).T
    X=np.array([np.outer(a,psi.conj())+np.outer(psi,a.conj()) for a in uv])
    drho=np.array([np.outer(a,psi.conj())+np.outer(psi,a.conj()) for a in ds])
    loc=np.einsum('iab,jba->ij',drho,X).real
    expect=np.einsum('a,iab,b->i',psi.conj(),X,psi)
    r=4*gram.imag/np.sqrt(np.outer(np.diag(Q),np.diag(Q)))
    gap=(hi-lo)/CS
    assert np.linalg.eigvalsh(H)[0]>0
    assert np.max(abs(A@u-b))<1e-9 and np.max(abs(loc-np.eye(p)))<1e-9
    assert gap>-1e-9 and gap<2e-7,(gap,message,t)
    assert lo>=CS*(1-1e-8)
    return dict(C_SLD=CS,C_H=hi,dual_lower=lo,primal_upper=hi,relative_primal_dual_gap=gap,
        dual_variables=t.tolist(),dual_success=bool(success),dual_message=message,
        scalar_dual=scalar,scalar_difference=abs(scalar-lo) if scalar is not None else None,
        locally_unbiased_residual=float(np.max(abs(loc-np.eye(p)))),centering_residual=float(abs(expect).max()),
        Q=Q.tolist(),weak_commutator_scaled=r.tolist(),weak_commutator_max=float(abs(r).max()),
        gram_imag=gram.imag.tolist(),primal_gram_imag=G.imag.tolist(),state_norm=norm)

def metrics(psi,ds):
    h=holevo(psi,ds)
    a=MEAS@psi;prob=abs(a)**2;dp=2*np.real(a.conj()*(ds@MEAS.T))
    F=(dp/prob)@dp.T;sv=np.linalg.svd(dp/np.sqrt(prob),compute_uv=False)
    C=float(np.sum(1/sv**2));Q=np.array(h['Q'])
    assert C>=h['C_H']*(1-1e-7),(C,h['C_H'])
    return dict(**h,C=C,F=F.tolist(),min_probability=float(prob.min()),
        min_eigenvalue_Q_minus_F=float(np.linalg.eigvalsh(Q-F).min()),
        incompatibility_percent=100*(h['C_H']/h['C_SLD']-1),measurement_gap_percent=100*(C/h['C_H']-1))

def selfcheck():
    psi=np.array([1,0,0,0],complex)
    compatible=holevo(psi,np.eye(4,dtype=complex)[1:]/2)
    assert abs(compatible['C_H']-3)<1e-10
    coherent=holevo(psi,np.array([[0,.5,0,0],[0,.5j,0,0]]))
    assert abs(coherent['C_H']-4)<1e-7
    rng=np.random.default_rng(42);checks=[]
    for p in (2,3):
        ds=np.zeros((p,4),complex);ds[:,1:]=rng.normal(size=(p,3))+1j*rng.normal(size=(p,3))
        h=holevo(psi,ds);checks.append(dict(p=p,result=h))
    save(OUT/'solver_selfcheck.json',dict(compatible=compatible,coherent=coherent,generic=checks))

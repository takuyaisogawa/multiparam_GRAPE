"""Sensor/ancilla model and the original adjoint forward model."""
import numpy as np

PAULI=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)

BELL=np.array([[1,0,0,-1],[0,1,1,0],[0,1,-1,0],[1,0,0,1]],complex)/np.sqrt(2)

ROT2=np.cos(np.pi/3)*np.eye(2)-1j*np.sin(np.pi/3)*PAULI.sum(axis=0)/np.sqrt(3)

MEAS=BELL.conj()@np.kron(ROT2,np.eye(2))

class Model:
    def __init__(self,kind,T,N,J,h=1e-4,substeps=1):
        self.kind,self.T,self.N,self.J,self.h,self.substeps=kind,T,N,J,h,substeps
        self.x=np.array([1.,np.pi/4,np.pi/4]) if kind=='static' else np.array([1.,100. if kind=='linear' else 1.])
        self.p=len(self.x); self.dt=T/(N*substeps)
        pars=[self.x.copy()]
        for i in range(self.p):
            for s in [1,-1]:
                x=self.x.copy(); x[i]+=s*h; pars.append(x)
        pars=np.array(pars); ts=np.arange(N*substeps)*self.dt
        target=np.zeros((len(pars),len(ts),3))
        if kind=='static':
            b,th,ph=pars.T
            target[:]=np.stack((b*np.sin(th)*np.cos(ph),b*np.sin(th)*np.sin(ph),b*np.cos(th)),axis=-1)[:,None,:]
        elif kind in ['xy','xz']:
            b,w=pars.T; target[:,:,0]=-b[:,None]*np.cos(w[:,None]*ts)
            target[:,:,1 if kind=='xy' else 2]=-b[:,None]*np.sin(w[:,None]*ts)
        else:
            b,w=pars.T; target[:,:,0]=b[:,None]*np.cos(w[:,None]*ts)
        self.target=target

    def initial(self):
        v=-self.target[0,::self.substeps].copy()
        if self.kind=='xy':v[:,2]+=self.x[1]/2
        if self.kind=='xz':v[:,1]-=self.x[1]/2
        if self.kind=='linear':v[:,1]+=self.x[1]/2
        return v

    def propagate(self,v,derivative=False):
        v=np.asarray(v).reshape(self.N,3)
        z=np.repeat(v,self.substeps,axis=0)[None,:,:]+self.target
        z=np.repeat(z[:,:,None,:],2,axis=2)
        z[:,:,:,2]+=np.array([self.J,-self.J])
        r=np.linalg.norm(z,axis=-1); dt=self.dt
        a=dt*np.sinc(r*dt/np.pi)
        mat=np.einsum('...k,kij->...ij',z,PAULI)
        u=np.cos(r*dt)[...,None,None]*np.eye(2)-1j*a[...,None,None]*mat
        m=u.shape[1]; c=u.shape[0]
        states=np.empty((m+1,c,2,2),complex)
        states[0]=np.array([[1,0],[0,1]],complex)/np.sqrt(2)
        for n in range(m):states[n+1]=np.einsum('cbij,cbj->cbi',u[:,n],states[n])
        psi=states[-1].transpose(0,2,1).reshape(c,4)
        amps=psi@MEAS.T; probs=np.abs(amps)**2
        if derivative:
            da=np.empty_like(r); mask=r*dt>1e-5
            da[mask]=(dt*r[mask]*np.cos(r[mask]*dt)-np.sin(r[mask]*dt))/r[mask]**3
            da[~mask]=-dt**3/3+dt**5*r[~mask]**2/30
            du=(-dt*a[...,None,None,None]*z[..., :,None,None]*np.eye(2)
                -1j*da[...,None,None,None]*z[..., :,None,None]*mat[...,None,:,:]
                -1j*a[...,None,None,None]*PAULI)
            return probs,psi,(u,du,states,amps)
        return probs,psi

    def fisher(self,probs):
        d=(probs[1::2]-probs[2::2])/(2*self.h)
        den=np.maximum(probs[0],1e-14)
        return (d/den)@d.T,d,den

    def objective(self,v,objective='proxy'):
        probs,psi,cache=self.propagate(v,True)
        f,d,den=self.fisher(probs)
        # Tiny fixed ridge is used only inside the optimizer, not reported metrics.
        f=f+1e-12*np.eye(self.p)
        if objective=='proxy':
            cost=np.sum(1/np.diag(f)); gf=np.diag(-1/np.diag(f)**2)/cost
        else:
            inv=np.linalg.inv(f); cost=np.trace(inv); gf=-(inv@inv)/cost
        if not np.isfinite(cost) or cost<=0:raise FloatingPointError('invalid Fisher objective')
        gd=2*gf@d/den
        gp=np.zeros_like(probs)
        gp[0]=-np.sum(d*(gf@d),axis=0)/den**2
        gp[0,probs[0]<1e-14]=0
        gp[1::2]=gd/(2*self.h); gp[2::2]=-gd/(2*self.h)
        u,du,states,amps=cache
        lam=((gp*amps)@MEAS.conj()).reshape(-1,2,2).transpose(0,2,1)
        grad=np.empty((u.shape[1],3))
        for n in range(u.shape[1]-1,-1,-1):
            step=np.einsum('cbkij,cbj->cbki',du[:,n],states[n])
            grad[n]=2*np.real(np.einsum('cbi,cbki->k',lam.conj(),step))
            lam=np.einsum('cbji,cbj->cbi',u[:,n].conj(),lam)
        grad=grad.reshape(self.N,self.substeps,3).sum(axis=1)
        return float(np.log(cost)),grad.ravel()

    def metrics(self,v):
        probs,psi=self.propagate(v)
        f,d,den=self.fisher(probs); eig=np.linalg.eigvalsh(f)
        inv=np.linalg.inv(f) if eig[0]>0 else np.full_like(f,np.nan)
        cost=float(np.trace(inv)); proxy=float(np.sum(1/np.diag(f)))
        corr=f/np.sqrt(np.outer(np.diag(f),np.diag(f)))
        ds=(psi[1::2]-psi[2::2])/(2*self.h)
        overlap=ds.conj()@psi[0]
        tangent=ds.conj()@ds.T-np.outer(overlap,overlap.conj())
        q=4*np.real(tangent)
        return dict(kind=self.kind,J=self.J,T=self.T,N=self.N,h=self.h,substeps=self.substeps,
                    F=f.tolist(),eigenvalues=eig.tolist(),condition=float(eig[-1]/eig[0]),
                    cost=cost,proxy_cost=proxy,ratio=cost/proxy,
                    max_abs_correlation=float(np.max(np.abs(corr-np.eye(self.p)))),
                    probabilities=probs[0].tolist(),QFIM=q.tolist(),
                    normalized_incompatibility=float(np.max(np.abs(np.imag(tangent))/np.sqrt(np.outer(np.diag(tangent).real,np.diag(tangent).real)))),
                    max_amplitude=float(np.max(np.linalg.norm(np.asarray(v).reshape(-1,3),axis=1))))

CASES={
 'static':dict(T=8.,N=50,path=[.01,.10,.15,.20]),
 'xz':dict(T=8.,N=500,path=[.01,.1,1.,5.,10.]),
 'xy':dict(T=6.,N=500,path=[.01,.1,1.,5.,10.]),
 'linear':dict(T=1.6,N=1000,path=[.01,.1,1.,5.,10.,20.]),
}

def random_initial(model,rng):
    # Smooth, reproducible perturbations to the same analytical starting pulse.
    ts=np.linspace(0,1,model.N,endpoint=False)
    coefficients=rng.normal(size=(3,5,2))
    noise=np.zeros((model.N,3))
    for k in range(5):
        noise+=(np.cos(2*np.pi*(k+1)*ts)[:,None]*coefficients[:,k,0]+np.sin(2*np.pi*(k+1)*ts)[:,None]*coefficients[:,k,1])/(k+1)
    noise/=np.sqrt(np.mean(noise**2))
    scale=max(1.,model.J)
    return model.initial()+scale*noise

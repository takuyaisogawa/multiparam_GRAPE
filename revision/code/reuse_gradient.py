"""Original finite-difference objective/gradient, with shared propagation cached.
No analytic parameter or control derivatives, new regularization, optimizer,
line search, adaptive learning rate, or pulse constraints are introduced.
Only matrix-product association differs from full repropagation; validate.py
quantifies those floating-point differences against the notebook functions.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def unitary(vx,vy,vz,J,dt,model):
    hz=np.array([vz+J,vz-J])
    U=np.zeros((4,4),np.complex128)
    for blk in range(2):
        mag=np.sqrt(vx*vx+vy*vy+hz[blk]*hz[blk])
        if model!=3 and mag<1e-14:
            cs=1.;ss=0.
        else:
            cs=np.cos(mag*dt);ss=np.sin(mag*dt)/mag
        a=cs-1j*ss*hz[blk];d=cs+1j*ss*hz[blk]
        b=-1j*ss*(vx-1j*vy);c=-1j*ss*(vx+1j*vy)
        U[blk,blk]=a;U[blk,blk+2]=b;U[blk+2,blk]=c;U[blk+2,blk+2]=d
    return U

@njit(cache=True)
def setup(model,T,N):
    """Use the notebook's left endpoint time samples and 1e-4 parameter stencil."""
    D=3 if model==0 else 2
    par=np.array([1.,np.pi/4,np.pi/4]) if model==0 else np.array([1.,100. if model==3 else 1.])
    stencil=np.zeros((1+2*D,D))
    stencil[0]=par
    for d in range(D):
        stencil[1+2*d]=par;stencil[2+2*d]=par
        stencil[1+2*d,d]+=1e-4;stencil[2+2*d,d]-=1e-4
    fields=np.zeros((1+2*D,N,3))
    V=np.zeros((3,N));dt=T/N
    for q in range(1+2*D):
        B=stencil[q,0]
        for t in range(N):
            now=t*dt
            if model==0:
                theta=stencil[q,1];phi=stencil[q,2]
                fields[q,t,0]=B*np.sin(theta)*np.cos(phi)
                fields[q,t,1]=B*np.sin(theta)*np.sin(phi)
                fields[q,t,2]=B*np.cos(theta)
            elif model==1:
                fields[q,t,0]=-B*np.cos(stencil[q,1]*now)
                fields[q,t,2]=-B*np.sin(stencil[q,1]*now)
            elif model==2:
                fields[q,t,0]=-B*np.cos(stencil[q,1]*now)
                fields[q,t,1]=-B*np.sin(stencil[q,1]*now)
            else:
                fields[q,t,0]=B*np.cos(stencil[q,1]*now)
    for t in range(N):
        for a in range(3):V[a,t]=-fields[0,t,a]
        if model==1:V[1,t]=-par[1]/2
        elif model==2:V[2,t]=par[1]/2
        elif model==3:V[1,t]=par[1]/2
    return fields,V

@njit(cache=True)
def measurement():
    bell=np.array([[1,0,0,-1],[0,1,1,0],[0,1,-1,0],[1,0,0,1]],np.complex128)/np.sqrt(2.)
    projectors=np.empty((4,4,4),np.complex128)
    for k in range(4):projectors[k]=np.outer(bell[k],bell[k].conj())
    psi=np.array([1,0,0,1],np.complex128)/np.sqrt(2.)
    rho=np.outer(psi,psi.conj())
    c=1/np.sqrt(3.)
    rot=unitary(c,c,c,0.,np.pi/3,0)
    return rho,rot,projectors

@njit(cache=True)
def measured_prob(rho,rot,projectors):
    r=rot@rho@rot.conj().T
    p=np.empty(4)
    for k in range(4):p[k]=np.real(np.trace(r@projectors[k]))
    return p

@njit(cache=True)
def fisher(P,model):
    D=(len(P)-1)//2
    grad=np.empty((D,4))
    for d in range(D):grad[d]=(P[1+2*d]-P[2+2*d])/(2e-4)
    F=np.zeros((D,D))
    for a in range(D):
        for b in range(D):
            acc=0.
            for k in range(4):
                denom=P[0,k] if model==0 else max(P[0,k],1e-12)
                acc+=grad[a,k]*grad[b,k]/denom
            F[a,b]=acc
    return F

@njit(cache=True)
def objective(P,model):
    F=fisher(P,model)
    denom=1./F[0,0]+1./F[1,1]
    if model==0:denom+=1./F[2,2]
    return 1./denom

@njit(cache=True)
def evaluate(V,fields,J,dt,model):
    rho0,rot,projectors=measurement()
    P=np.empty((len(fields),4))
    for q in range(len(fields)):
        rho=rho0.copy()
        for i in range(V.shape[1]):
            U=unitary(fields[q,i,0]+V[0,i],fields[q,i,1]+V[1,i],fields[q,i,2]+V[2,i],J,dt,model)
            rho=U@rho@U.conj().T
        P[q]=measured_prob(rho,rot,projectors)
    F=fisher(P,model)
    cov=np.linalg.inv(F)
    tr=cov[0,0]+cov[1,1]
    if model==0:tr+=cov[2,2]
    return objective(P,model),tr,F,P

@njit(cache=True)
def gradient(V,fields,J,dt,model):
    Q,N,_=fields.shape
    rho0,rot,projectors=measurement()
    forwards=np.empty((Q,N+1,4,4),np.complex128)
    suffix=np.empty((Q,N+1,4,4),np.complex128)
    unitaries=np.empty((Q,N,4,4),np.complex128)
    baseP=np.empty((Q,4))
    for q in range(Q):
        forwards[q,0]=rho0
        for i in range(N):
            U=unitary(fields[q,i,0]+V[0,i],fields[q,i,1]+V[1,i],fields[q,i,2]+V[2,i],J,dt,model)
            unitaries[q,i]=U
            forwards[q,i+1]=U@forwards[q,i]@U.conj().T
        baseP[q]=measured_prob(forwards[q,N],rot,projectors)
        suffix[q,N]=np.eye(4,dtype=np.complex128)
        for i in range(N-1,-1,-1):suffix[q,i]=suffix[q,i+1]@unitaries[q,i]
    base=objective(baseP,model)
    g=np.empty_like(V);p=np.empty_like(baseP);eps=1e-5
    for a in range(3):
        for i in range(N):
            V[a,i]+=eps
            for q in range(Q):
                U=unitary(fields[q,i,0]+V[0,i],fields[q,i,1]+V[1,i],fields[q,i,2]+V[2,i],J,dt,model)
                r=U@forwards[q,i]@U.conj().T
                S=suffix[q,i+1]
                r=S@r@S.conj().T
                p[q]=measured_prob(r,rot,projectors)
            g[a,i]=(objective(p,model)-base)/eps
            V[a,i]-=eps
    return g

def learning_rate(model,epoch):
    if model==0:return 1e-8 if epoch<=300 else 1e-6
    if model==1:return 1e-6
    if model==2:return 1e-8 if epoch<=500 else 1e-6
    return 50. if epoch<=10000000 else 1e-6

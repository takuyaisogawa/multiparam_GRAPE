"""Original notebook evaluation for the saved-control reuse comparison."""
import numpy as np,importlib.util,sys
from pathlib import Path
from functools import lru_cache
@lru_cache(None)
def evaluator(model):
    path=Path(__file__).resolve().parent/'legacy'/(model+'.py')
    spec=importlib.util.spec_from_file_location('original_'+model,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

MODELS = {'static': 'SMF', 'xz': 'CPL_xz', 'xy': 'CPL_xy', 'linear': 'LPL'}

def original_evaluate(m,model,V,T,J,h=1e-4):
    N=V.shape[1]
    args=(m.theta_rot,T/N,N,m.B0,m.theta0,m.phi0,h,J,m.projectors) if model=='static' else (m.theta_rot,T/N,N,m.B0,m.omega0,h,J,m.projs)
    return m.f0_and_trace(*V,*args)

def details(m,model,V,T,J,h=1e-4):
    N=V.shape[1]
    par=np.array([m.B0,m.theta0,m.phi0]) if model=='static' else np.array([m.B0,m.omega0])
    projs=m.projectors if model=='static' else m.projs
    def prob(q):return m.probabilities(m.theta_rot,T/N,N,*q,*V,J,projs)
    P=prob(par); G=[]
    for k in range(len(par)):
        pp=par.copy();pm=par.copy();pp[k]+=h;pm[k]-=h
        G.append((prob(pp)-prob(pm))/(2*h))
    G=np.array(G); denom=P if model=='static' else np.maximum(P,1e-12)
    F=(G/denom)@G.T
    f0,C=original_evaluate(m,model,V,T,J,h)
    return dict(f0=float(f0),C=float(C),diag_cost=float(1/f0),F=F.tolist(),eigenvalues=np.linalg.eigvalsh(F).tolist(),condition=float(np.linalg.cond(F)),P=P.tolist(),prob_sum=float(P.sum()),correlation_penalty=float(C*f0))

def analytic(model,T,N):
    t=np.arange(N)*T/N
    if model=='static':return np.tile(np.array([-.5,-.5,-1/np.sqrt(2)])[:,None],(1,N))
    if model=='xz':return np.array([np.cos(t),-.5*np.ones(N),np.sin(t)])
    if model=='xy':return np.array([np.cos(t),np.sin(t),.5*np.ones(N)])
    return np.array([-np.cos(100*t),50*np.ones(N),np.zeros(N)])

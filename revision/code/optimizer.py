"""The recorded L-BFGS-B objective and stopping policy."""
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from model import Model, CASES, MEAS, random_initial

def clean(x):
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer): return int(x)
    return x

def save(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)

def objective(model,v,form):
    """Same forward model, FD sensitivities, ridge, probability floor and adjoint as audit #1."""
    probs,psi,cache=model.propagate(v,True)
    F,d,den=model.fisher(probs)
    f=F+1e-12*np.eye(model.p)
    proxy=float(np.sum(1/np.diag(f)))
    if form=='log_full':
        inv=np.linalg.inv(f);cost=float(np.trace(inv));gf=-(inv@inv)/cost
    else:
        cost=proxy;gf=np.diag(-1/np.diag(f)**2)/cost
    if not np.isfinite(cost) or cost<=0: raise FloatingPointError('invalid objective')
    val=float(np.log(cost))
    if form=='raw_proxy':
        val=-1/cost;gf/=cost
    gd=2*gf@d/den
    gp=np.zeros_like(probs)
    gp[0]=-np.sum(d*(gf@d),axis=0)/den**2
    gp[0,probs[0]<1e-14]=0
    gp[1::2]=gd/(2*model.h);gp[2::2]=-gd/(2*model.h)
    u,du,states,amps=cache
    lam=((gp*amps)@MEAS.conj()).reshape(-1,2,2).transpose(0,2,1)
    grad=np.empty((u.shape[1],3))
    for n in range(u.shape[1]-1,-1,-1):
        grad[n]=2*np.real(np.einsum('cbi,cbki->k',lam.conj(),np.einsum('cbkij,cbj->cbki',du[:,n],states[n])))
        lam=np.einsum('cbji,cbj->cbi',u[:,n].conj(),lam)
    grad=grad.reshape(model.N,model.substeps,3).sum(axis=1).ravel()
    if not np.all(np.isfinite(grad)): raise FloatingPointError('nonfinite gradient')
    eig=np.linalg.eigvalsh(F)
    C=float(np.trace(np.linalg.inv(F))) if eig[0]>0 else float('inf')
    info=dict(C_fd=C,f0=1/float(np.sum(1/np.diag(F))),min_probability=float(probs[0].min()),gradient_inf=float(np.max(abs(grad))))
    return val,grad,info

class BudgetStop(Exception): pass

def optimize(model,v,budget,form):
    t=time.perf_counter();cpu=time.process_time();best=None;calls=0;history=[];attempts=[]
    def fun(x):
        nonlocal calls,best
        if calls>=budget: raise BudgetStop()
        val,g,info=objective(model,x,form);calls+=1
        if best is None or val<best['value']:
            best=dict(value=val,v=np.array(x).copy(),info=info.copy())
        history.append(dict(evaluation=calls,objective=val,candidate_C_fd=info['C_fd'],incumbent_objective=best['value'],incumbent_C_fd=best['info']['C_fd'],gradient_inf=info['gradient_inf']))
        return val,g
    x=np.array(v).ravel().copy();termination='not_started'
    # Identical restart policy for all three forms; no objective rescaling.
    for attempt in range(3):
        before=calls
        try:
            r=minimize(fun,x,method='L-BFGS-B',jac=True,options=dict(maxiter=budget,maxfun=budget,ftol=0.,gtol=0.,maxls=100,maxcor=10))
            msg=str(r.message);attempts.append(dict(attempt=attempt+1,message=msg,calls=calls-before,success=bool(r.success),iterations=int(r.nit),final_gradient_inf=float(np.max(abs(r.jac)))))
            termination='numerical_stop' if r.success else 'abnormal_stop'
            if calls>=budget: termination='budget_exhausted';break
            if np.max(abs(r.jac))==0:termination='zero_gradient';break
            if calls==before:break
            x=best['v'].copy()
        except BudgetStop:
            attempts.append(dict(attempt=attempt+1,message='evaluation budget exhausted',calls=calls-before));termination='budget_exhausted';break
        except (FloatingPointError,np.linalg.LinAlgError) as exc:
            attempts.append(dict(attempt=attempt+1,message=str(exc),calls=calls-before));termination='invalid_numeric';break
    if best is None: raise RuntimeError('No valid objective evaluation')
    return best['v'].reshape(model.N,3),dict(evaluations=calls,budget_cap=budget,termination=termination,attempts=attempts,history=history,objective_best=best['value'],optimization_wall_seconds=time.perf_counter()-t,optimization_cpu_seconds=time.process_time()-cpu)

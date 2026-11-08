"""Matched-budget comparison; reruns write to build/initialization/."""
import time,json
import numpy as np
from scipy.integrate import quad
from model import Model, CASES, random_initial
from optimizer import optimize, objective, save, clean
from exact import exact_metrics
from paths import output
OUT=output('initialization')
(OUT/'runs').mkdir(exist_ok=True)

def bound(kind,T):
 if kind=='static':return 1/T**2
 if kind in ('xz','xy'):return 1/(4*T*T)+1/T**4
 w=100.;pts=np.arange(0,w*T/np.pi+1)*np.pi/w;pts=pts[(pts>0)&(pts<T)]
 Ac=quad(lambda t:abs(np.cos(w*t)),0,T,points=np.arange(np.pi/(2*w),T,np.pi/w),epsabs=1e-12,limit=300)[0]
 As=quad(lambda t:t*abs(np.sin(w*t)),0,T,points=pts,epsabs=1e-12,limit=300)[0]
 return 1/(4*Ac**2)+1/(4*As**2)

def refined_path(path,factor):
 return np.concatenate([np.linspace(a,b,factor+1)[:-1] for a,b in zip(path[:-1],path[1:])]+[np.array([path[-1]])]).tolist() if len(path)>1 else list(path)

def refmodel(kind,T,N,J,sub):
 m=Model(kind,T,N,J,substeps=sub);m.time_shift=.5
 ts=(np.arange(N*sub)+.5)*m.dt;x=m.x
 if kind in ('xz','xy'):
  z=np.zeros((len(ts),3));z[:,0]=-x[0]*np.cos(x[1]*ts);z[:,1 if kind=='xy' else 2]=-x[0]*np.sin(x[1]*ts);m.target[0]=z
 elif kind=='linear':m.target[0,:,0]=x[0]*np.cos(x[1]*ts)
 return m

def reference(kind,T,N,J,v):
 levels=[];stable=0;previous=None
 for sub in (2,4,8,16,32,64,128,256):
  m=refmodel(kind,T,N,J,sub);e=exact_metrics(m,v);cost=e['cost'];change=abs(cost/previous-1) if previous is not None and np.isfinite(previous) else None
  stable=stable+1 if change is not None and change<1e-4 else 0
  levels.append(dict(substeps=sub,C=cost,relative_change=change))
  if stable>=2:break
  previous=cost
 e['levels']=levels;e['grid_converged']=stable>=2;e['propagation_substeps']=sub
 e['SLD_cost']=float(np.trace(np.linalg.inv(np.array(e['QFIM']))))
 e['lower_bound']=bound(kind,T);e['gap_to_lower_bound']=cost-e['lower_bound'];e['ratio_to_lower_bound']=cost/e['lower_bound'];e['attains_2x_bound']=bool(np.isfinite(cost) and cost<=2*e['lower_bound']);e['attains_1p1x_bound']=bool(np.isfinite(cost) and cost<=1.1*e['lower_bound'])
 return e

def jobs():
 alljobs=[]
 for kind,cfg in CASES.items():
  for idx,J in enumerate(cfg['path']):
   for B in ([600,1200,2400] if idx==len(cfg['path'])-1 else [600]):
    for arm,rep in [('direct',None),('continuation',None)]+[('random',r) for r in range(20)]:
     alljobs.append(dict(kind=kind,target_index=idx,budget=B,arm=arm,rep=rep,factor=1,mode='target'))
  for B in [600,1200,2400]:
   for fac in [2,4]:alljobs.append(dict(kind=kind,target_index=len(cfg['path'])-1,budget=B,arm='continuation',rep=None,factor=fac,mode='target'))
  for fac in [1,2,4]:alljobs.append(dict(kind=kind,target_index=len(cfg['path'])-1,budget=600*len(cfg['path']),arm='continuation',rep=None,factor=fac,mode='scan'))
 return alljobs

def tag(j):return f"{j['kind']}_{j['mode']}_j{j['target_index']}_b{j['budget']}_{j['arm']}_p{j['factor']}"+(f"_r{j['rep']:02d}" if j['rep'] is not None else '')

def run(j):
 name=tag(j);dest=OUT/'runs'/f'{name}.json'
 if dest.exists():return dict(tag=name,status='cached')
 cfg=CASES[j['kind']];kind=j['kind'];J=cfg['path'][j['target_index']];m=Model(kind,cfg['T'],cfg['N'],J);v=m.initial();logs=[];snapshots=[];candidates=[];pulses={};start=time.perf_counter()
 B=j['budget'];path=refined_path(cfg['path'][:j['target_index']+1],j['factor'])
 if j['arm']=='direct':
  v,log=optimize(m,v,B,'log_proxy');log['J']=J;logs.append(log)
 elif j['arm']=='random':
  rng=np.random.default_rng(41000+j['rep'])
  for idx in range(4):
   initial=random_initial(m,rng);share=B//4 if idx<3 else B-sum(l['evaluations'] for l in logs)
   vv,log=optimize(m,initial,share,'log_proxy');log['J']=J;log['restart']=idx;logs.append(log);candidates.append((log['objective_best'],vv.copy()))
  v=min(candidates,key=lambda a:a[0])[1]
 else:
  for idx,Jstage in enumerate(path):
   share=B//len(path) if idx<len(path)-1 else B-sum(l['evaluations'] for l in logs)
   mm=Model(kind,cfg['T'],cfg['N'],Jstage);v,log=optimize(mm,v,share,'log_proxy');log['J']=Jstage;logs.append(log)
   key=f'stage_{idx}';pulses[key]=v.copy()
   if j['mode']=='scan' and any(np.isclose(Jstage,base,rtol=0,atol=1e-12) for base in cfg['path']):
    snapshots.append(dict(J=Jstage,evaluations=sum(l['evaluations'] for l in logs),pulse_key=key,native=exact_metrics(mm,v),reference=reference(kind,cfg['T'],cfg['N'],Jstage,v)))
   elif j['mode']=='target':
    snapshots.append(dict(J=Jstage,evaluations=sum(l['evaluations'] for l in logs),C_at_final_J_fd=m.metrics(v)['cost']))
 native=exact_metrics(m,v);ref=reference(kind,cfg['T'],cfg['N'],J,v)
 pulses['v']=v;np.savez_compressed(OUT/'runs'/f'{name}.npz',**pulses)
 row=dict(tag=name,job=j,T=cfg['T'],N=cfg['N'],J=J,J_path=path,logs=logs,snapshots=snapshots,native=native,reference=ref,evaluations=sum(l['evaluations'] for l in logs),optimization_CPU_s=sum(l['optimization_cpu_seconds'] for l in logs),optimization_wall_s=sum(l['optimization_wall_seconds'] for l in logs),total_wall_s=time.perf_counter()-start)
 save(dest,row);return dict(tag=name,status='done',evaluations=row['evaluations'],C=ref['cost'],native_C=native['cost'],grid_converged=ref['grid_converged'])

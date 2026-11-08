#!/usr/bin/env python3
"""Reproduce the saved results, or explicitly rerun a numerical calculation."""
import os,sys,argparse,runpy,json
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
CODE=Path(__file__).resolve().parent/'revision/code'
sys.path.insert(0,str(CODE))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',nargs='?',default='all',choices=['all','tables','figures','verify','baselines','coupling','holevo','mismatch','propagation','initialization','initialization-checks','replacements','reuse'])
    parser.add_argument('--id',help='Restrict a numerical rerun to a saved record ID.')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--model',choices=['static','xz','xy','linear'])
    parser.add_argument('--arm',choices=['analytic','saved_previous'])
    parser.add_argument('--literal',action='store_true',help='Use full original repropagation for the static reuse test.')
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    from paths import load,control,output
    if args.task in ['all','tables']:
        from summarize import tables
        tables()
    if args.task in ['all','figures']:
        from figures import make_figures
        make_figures()
    if args.task in ['all','verify']:
        from verify import verify
        verify()
    if args.task=='coupling':runpy.run_path(str(CODE/'coupling.py'),run_name='__main__')
    if args.task=='baselines':
        from baselines import recalculate
        recalculate()
    if args.task=='initialization-checks':
        from initialization_checks import run
        run(args.id)
    if args.task=='holevo':
        import holevo
        from optimizer import save
        holevo.selfcheck()
        rows=[r for r in load('holevo.json') if not args.id or r['id']==args.id]
        if not rows:parser.error('No matching record ID.')
        for row in rows:
            psi,ds=holevo.independent_ode(row['kind'],row['T'],row['J'],control(row['id']),holevo.nominal(row['kind']))
            result=holevo.metrics(psi,ds);save(output('holevo')/(row['id']+'.json'),result)
            print(row['id'],result['C_H'],flush=True)
    if args.task in ['mismatch','propagation','initialization']:
        import concurrent.futures
        if args.task=='mismatch':
            from mismatch import evaluate as run
            jobs=load('mismatch_settings.json')['jobs'];key=lambda x:x['id']
        elif args.task=='propagation':
            from propagation import evaluate as run
            jobs=[{k:v for k,v in r.items() if k in ['id','kind','T','J','N','C_plot','array_sha256']} for r in load('saved_controls.json')];key=lambda x:x['id']
        else:
            import initialization
            run=initialization.run;jobs=[j for j in initialization.jobs() if j['mode']=='target'];key=initialization.tag
        if args.id:jobs=[j for j in jobs if key(j)==args.id]
        if not jobs:parser.error('No matching record ID.')
        if args.workers==1:
            for j in jobs:r=run(j);print(key(j),'done',flush=True)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                for j,r in zip(jobs,pool.map(run,jobs)):print(key(j),'done',flush=True)
    if args.task=='replacements':
        from reoptimize import optimize_case
        from refinement import chunked_state
        from propagation import metric,input_control
        from mismatch import nominal
        from optimizer import save
        rows=[r for r in load('replacement_points.json')['points'] if not args.id or r['adopted']['new_record']==args.id]
        if not rows:parser.error('No matching record ID.')
        for x in rows:
            a=x['adopted'];r=x['original_refinement'];tag=a['new_record']
            # The optimizer writes a new pulse, so invalidate only its previous derived evaluation.
            (output('propagation')/'records'/(tag+'.json')).unlink(missing_ok=True)
            optimize_case((tag,r['kind'],r['T'],r['J'],r['N'],a['new_source_pulse']))
            # Refine the newly generated pulse until two successive C/F changes pass.
            previous=None;stable=0;levels=[]
            import numpy as np
            for sub in [1,2,4,8,16,32,64,128,256,512,1024,2048,4096]:
                m=metric(*chunked_state(r['kind'],r['T'],r['J'],input_control(tag),nominal(r['kind']),sub))
                dc=abs(m['C']/previous['C']-1) if previous else None
                df=float(np.linalg.norm(np.array(m['F'])-previous['F'])/np.linalg.norm(previous['F'])) if previous else None
                stable=stable+1 if dc is not None and max(dc,df)<1e-4 else 0
                levels.append(dict(substeps=sub,C=m['C'],C_change=dc,F_change=df));previous=m
                if stable>=2:break
            save(output('propagation')/(tag+'_final.json'),dict(levels=levels,refined=m,converged=stable>=2))
            print(tag,m['C'],flush=True)
    if args.task=='reuse':
        if not args.model or not args.arm:parser.error('reuse requires --model and --arm')
        from reuse import run
        run(args.model,args.arm,args.literal)

if __name__=='__main__':main()

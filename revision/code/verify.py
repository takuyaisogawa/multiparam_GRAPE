"""Check released arrays, reported statistics and representative numerical evaluations."""
import hashlib,json,platform,time
import numpy as np
from paths import ROOT,DATA,load,control,initialization_control,output
from optimizer import save

def close(a,b,tol=1e-8):
    error=abs(a/b-1) if b else abs(a)
    assert error<tol,(a,b,error,tol)
    return float(error)

def verify():
    start=time.perf_counter();report={'python':platform.python_version(),'platform':platform.platform()}
    manifest=ROOT/'MANIFEST.json'
    if manifest.exists():
        files=json.loads(manifest.read_text())
        for name,h in files.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
        report['file_hashes_checked']=len(files)
    for r in load('control_index.json'):
        a=control(r['key']);assert list(a.shape)==r['shape']
        assert hashlib.sha256(a.tobytes()).hexdigest()==r['array_sha256'],r['key']
    report['control_arrays_checked']=len(load('control_index.json'))
    rows=load('initialization.json');assert len(rows)==640
    for r in rows:
        a=initialization_control(r['tag']);assert list(a.shape)==[r['N'],3]
        assert hashlib.sha256(a.tobytes()).hexdigest()==r['array_sha256']
        assert r['evaluations']==sum(x['evaluations'] for x in r['logs'])<=r['job']['budget']
        assert r['reference']['grid_converged']
        if r['job']['arm']=='random':assert r['seed']==41000+r['job']['rep']
    from summarize import initialization_summary
    summary=initialization_summary();assert len(summary['target'])==28 and len(summary['paths'])==36
    cases=load('initialization_settings.json')['cases']
    for r in summary['target']:
        if r['index']==len(cases[r['model']]['path'])-1:
            assert (r['continuation']<r['direct'])==(r['model']=='static' and r['budget'] in [1200,2400])
    report['initialization_trials_checked']=len(rows)
    waves=load('saved_controls.json');assert len(waves)==99
    for r in waves:
        assert all(max(x['C_change'],x['F_change'])<1e-4 for x in r['levels'][-2:])
        F=np.array(r['refined']['F']);C=float(np.trace(np.linalg.inv(F)))
        close(C,r['refined']['C'],1e-7)
    report['saved_control_convergence_records_checked']=len(waves)
    from holevo import metrics,selfcheck
    selfcheck();hh=[]
    for r in load('holevo.json'):
        m=metrics(control(r['id']+'__psi'),control(r['id']+'__ds'));old=r[r['adopted']]
        errors={k:close(m[k],old[k],1e-8) for k in ['C','C_H','C_SLD']}
        hh.append(dict(id=r['id'],relative_errors=errors))
    report['Holevo_from_saved_states']=hh
    for r in load('representatives.json'):
        F=np.array(r['CFIM']);C=float(np.trace(np.linalg.inv(F)));D=float(np.sum(1/np.diag(F)))
        close(C,r['C'],1e-12);close(D,r['D'],1e-12)
        close(100*(C/D-1),r['delta_percent'],1e-6)
        assert r['C_S']<=r['C_H']<=C
    from legacy_io import evaluator,original_evaluate
    reuse=[]
    for r in load('reuse.json')['comparisons']:
        model,arm=r['model'],r['arm'];v=control('reuse_'+model+'_'+arm)
        f,C=original_evaluate(evaluator(model),model,v,r['T'],r['J'])
        reuse.append(dict(model=model,arm=arm,relative_error=close(C,r['final']['C'],2e-7)))
        assert r['history'][-1]['epoch']==r['updates']
    report['reuse_endpoints']=reuse
    print('Checked stored records, Holevo bounds and eight reuse endpoints.',flush=True)
    from initialization import refmodel
    from exact import exact_metrics
    checks=[]
    for kind,cfg in cases.items():
        r=next(r for r in rows if r['job']['kind']==kind and r['job']['arm']=='direct' and r['job']['budget']==600 and r['job']['target_index']==len(cfg['path'])-1)
        m=refmodel(kind,r['T'],r['N'],r['J'],r['reference']['propagation_substeps'])
        e=exact_metrics(m,initialization_control(r['tag']))
        checks.append(dict(id=r['tag'],relative_error=close(e['cost'],r['reference']['cost'],2e-7)))
    report['initialization_endpoint_checks']=checks
    import mismatch
    mm=[]
    for r in load('mismatch.json'):
        if r['category']!='nominal' or r['arm']!='saved':continue
        m=mismatch.metrics(*mismatch.state(r['kind'],np.array(r['true_parameters']),control('mismatch_'+r['kind']+'_saved'),r['substeps']))
        mm.append(dict(id=r['id'],relative_error=close(m['C'],r['metrics']['C'],2e-7)))
    report['mismatch_nominal_checks']=mm
    from refinement import chunked_state
    from propagation import metric
    from mismatch import nominal
    checks=[]
    for x in load('replacement_points.json')['points']:
        a=x['adopted'];tag=a['new_record'];r=x['original_refinement']
        m=metric(*chunked_state(r['kind'],r['T'],r['J'],control(tag),nominal(r['kind']),a['new_refined_substeps']))
        checks.append(dict(id=tag,C=m['C'],relative_error=close(m['C'],a['new_refined_C'],2e-7)))
        print('Checked',tag,m['C'],flush=True)
    report['replacement_endpoint_checks']=checks
    report['elapsed_seconds']=time.perf_counter()-start
    report['new_optimizations']=0
    save(output('verification')/'checks.json',report)
    print('Numerical checks passed; no optimization was run.',flush=True)
    return report

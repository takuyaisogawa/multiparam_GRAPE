"""Rebuild the article/reply tables from the released full-precision records."""
import csv,json
import numpy as np
from paths import load,output

def writecsv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def initialization_summary():
    rows=load('initialization.json');cases=load('initialization_settings.json')['cases']
    target=[];paths=[]
    for kind,cfg in cases.items():
        for index,J in enumerate(cfg['path']):
            for budget in ([600,1200,2400] if index==len(cfg['path'])-1 else [600]):
                subset=[r for r in rows if r['job']['kind']==kind and r['job']['target_index']==index and r['job']['budget']==budget and r['job']['factor']==1]
                direct=next(r for r in subset if r['job']['arm']=='direct')
                continuation=next(r for r in subset if r['job']['arm']=='continuation')
                random=[r for r in subset if r['job']['arm']=='random'];assert len(random)==20
                values=[r['reference']['cost'] for r in random]
                target.append(dict(model=kind,T=cfg['T'],J=J,index=index,budget=budget,
                    direct=direct['reference']['cost'],continuation=continuation['reference']['cost'],
                    random_median=float(np.median(values)),random_q25=float(np.quantile(values,.25)),
                    random_q75=float(np.quantile(values,.75)),random_min=min(values),random_max=max(values),
                    direct_evaluations=direct['evaluations'],continuation_evaluations=continuation['evaluations'],
                    random_evaluations_min=min(r['evaluations'] for r in random),random_evaluations_max=max(r['evaluations'] for r in random),
                    direct_cpu=direct['optimization_CPU_s'],continuation_cpu=continuation['optimization_CPU_s'],
                    random_cpu_median=float(np.median([r['optimization_CPU_s'] for r in random]))))
        for budget in [600,1200,2400]:
            for factor in [1,2,4]:
                r=next(r for r in rows if r['job']==dict(kind=kind,target_index=len(cfg['path'])-1,budget=budget,arm='continuation',rep=None,factor=factor,mode='target'))
                paths.append(dict(model=kind,budget=budget,factor=factor,stage_count=len(r['J_path']),
                    C=r['reference']['cost'],evaluations=r['evaluations'],cpu=r['optimization_CPU_s']))
    return dict(target=target,paths=paths)

def tables():
    out=output('tables');summary=initialization_summary()
    writecsv(out/'initialization.csv',summary['target']);writecsv(out/'coupling_steps.csv',summary['paths'])
    writecsv(out/'runtimes.csv',[r for r in summary['target'] if r['budget']==2400])
    (out/'initialization_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    flat=[]
    for r in load('initialization.json'):
        j=r['job'];flat.append(dict(id=r['tag'],model=j['kind'],T=r['T'],J=r['J'],method=j['arm'],budget=j['budget'],
            factor=j['factor'],repeat=j['rep'],seed=r.get('seed'),C=r['reference']['cost'],evaluations=r['evaluations'],cpu_seconds=r['optimization_CPU_s']))
    writecsv(out/'initialization_trials.csv',flat)
    reps=[]
    for r in load('representatives.json'):
        F=np.array(r['CFIM']);eig=np.linalg.eigvalsh(F)
        C=float(np.trace(np.linalg.inv(F)));D=float(np.sum(1/np.diag(F)))
        reps.append(dict(model=r['model'],J=r['J'],T=r['T'],C=C,D=D,
            discrepancy_percent=100*(C/D-1),condition=float(eig[-1]/eig[0]),C_S=r['C_S'],C_H=r['C_H'],
            gap_percent=100*(C/r['C_H']-1),incompatibility_percent=100*(r['C_H']/r['C_S']-1)))
    writecsv(out/'representatives.csv',reps)
    reuse=load('reuse.json')['comparisons'];rr=[]
    for kind in ['static','xz','xy','linear']:
        a=next(r for r in reuse if r['model']==kind and r['arm']=='analytic')
        b=next(r for r in reuse if r['model']==kind and r['arm']=='saved_previous')
        rr.append(dict(model=kind,T=a['T'],J=a['J'],source_J=a['source_J'],updates=a['updates'],
            analytical_C=a['final']['C'],saved_C=b['final']['C']))
    writecsv(out/'saved_control_reuse.csv',rr)
    waveforms=load('saved_controls.json');nr=[];ns=[]
    for r in waveforms:
        last=r['levels'][-1];fd=next(x for x in r['parameter_difference_checks'] if x['h']==1e-4)
        nr.append(dict(id=r['id'],model=r['kind'],T=r['T'],J=r['J'],N=r['N'],C_plot=r['C_plot'],
            C_native=r['native']['C'],C_refined=r['refined']['C'],substeps=r['substeps'],
            grid_change_percent=100*r['native_to_refined_relative_change'],
            finite_difference_error_percent=100*fd['C_relative_error'],last_C_change=last['C_change'],last_F_change=last['F_change']))
    writecsv(out/'propagation_checks.csv',nr)
    for kind in ['static','xz','xy','linear']:
        rows=[r for r in nr if r['model']==kind]
        ns.append(dict(model=kind,controls=len(rows),max_substeps=max(r['substeps'] for r in rows),
            max_grid_change_percent=max(abs(r['grid_change_percent']) for r in rows),
            max_FD_error_percent=max(abs(r['finite_difference_error_percent']) for r in rows)))
    writecsv(out/'propagation_summary.csv',ns)
    replacements=[x['adopted'] for x in load('replacement_points.json')['points']]
    writecsv(out/'replacement_points.csv',replacements)
    mismatch=load('mismatch.json');mr=[];ms=[]
    for r in mismatch:
        m=r['metrics'];mr.append(dict(id=r['id'],model=r['kind'],arm=r['arm'],category=r['category'],axis=r['axis'],
            offsets=json.dumps(r['offset']),parameters=json.dumps(r['true_parameters']),C=m['C'],condition=m['condition'],substeps=r['substeps']))
    writecsv(out/'parameter_mismatch.csv',mr)
    for kind in ['static','xz','xy','linear']:
        rows=[r for r in mismatch if r['kind']==kind];c0=next(r['metrics']['C'] for r in rows if r['category']=='nominal' and r['arm']=='saved')
        byid={r['id']:r for r in rows};saved=[r for r in rows if r['arm']=='saved']
        ms.append(dict(model=kind,C0=c0,
            maximum_individual_C_over_C0=max(r['metrics']['C']/c0 for r in saved if r['category']=='individual'),
            maximum_joint_C_over_C0=max(r['metrics']['C']/c0 for r in saved if r['category']=='joint'),
            maximum_C_over_analytical=max(r['metrics']['C']/byid[r['id'].replace('_saved','_analytical')]['metrics']['C'] for r in saved)))
    writecsv(out/'mismatch_summary.csv',ms)
    gaps=[];plots=load('figure_data.json')['original_plot_values']
    for kind,T,J,ref in [('static',8.,.2,1/64),('xz',8.,10.,1/(4*64)+1/8**4),('xy',6.,10.,1/(4*36)+1/6**4)]:
        C=next(r['C_plot'] for r in plots[kind] if r['T']==T and r['J']==J)
        gaps.append(dict(model=kind,T=T,J=J,C_plot=C,C_reference=ref,absolute_gap=C-ref))
    writecsv(out/'reference_gaps.csv',gaps)
    print('Rebuilt tables in',out)
    return summary

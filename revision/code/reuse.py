"""Repeat the target-J saved-control reuse test with the original f0 updates."""
import time
import numpy as np
from legacy_io import evaluator,original_evaluate,details,analytic,MODELS
from reuse_gradient import setup,gradient,learning_rate
from optimizer import save
from paths import load,control,output

def run(model,arm,literal=False):
    case=next(r for r in load('reuse.json')['comparisons'] if r['model']==model and r['arm']==arm)
    T,J,E=case['T'],case['J'],case['updates'];seed=control('reuse_'+model+'_seed')
    V=analytic(model,T,seed.shape[1]) if arm=='analytic' else seed.copy()
    idx=list(MODELS).index(model);m=evaluator(model);N=V.shape[1];dt=T/N
    if literal and model!='static':raise ValueError('The literal check is for the static case.')
    fields,_=setup(idx,T,N);history=[];start=time.perf_counter()
    for epoch in range(E+1):
        if epoch:
            if literal:m.literal_step(*V,J,dt,epoch)
            else:V+=learning_rate(idx,epoch)*gradient(V,fields,J,dt,idx)
        if epoch%100==0 or epoch==E:
            f,C=original_evaluate(m,model,V,T,J)
            history.append(dict(epoch=epoch,f0=float(f),C=float(C),lr=0. if epoch==0 else learning_rate(idx,epoch)))
            if epoch%1000==0:print(model,arm,epoch,float(C),flush=True)
    out=output('reuse');tag=f'{model}_{arm}'+('_literal' if literal else '')
    np.savez_compressed(out/(tag+'.npz'),v=V)
    save(out/(tag+'.json'),dict(model=model,arm=arm,T=T,J=J,N=N,updates=E,literal=literal,
        final=details(m,model,V,T,J),history=history,seconds=time.perf_counter()-start))

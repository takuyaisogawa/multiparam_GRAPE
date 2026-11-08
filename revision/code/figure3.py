"""Draw Fig. 3 from the corrected baseline samples and adopted points."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from paths import load, output
O=output('figures')
record=load('figure_data.json')['circular']
D=record['original'];curves=record['curves'];points=record['points']
times=np.array(curves['xz']['T'])

colors={.1:(.122,.467,.706),1:(1.,.498,.055),10:(.173,.627,.173)}
plt.rcParams.update({'font.size':11,'axes.labelsize':13,'font.family':'DejaVu Sans','svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(12.8,4.6),layout='constrained')
for ax,k,label in zip(axes,['xz','xy'],['(a)','(b)']):
    for c in D[k]['curves']:
        if c['color']==0:
            a=np.array(c['points']);ax.plot(a[:,0],a[:,1],color='black',lw=1.1)
    for J in (.1,1.,10.):ax.plot(times,curves[k][f'J{J:g}'],color=colors[J],lw=1.1)
    for p in points[k]:ax.plot(p['T'],p['C'],'o',color=p['color'],ms=5,zorder=5)
    ax.set(xlim=(1,11),ylim=(1e-3,1e2),yscale='log',xlabel=r'$T$',ylabel=r'$\mathrm{Tr}[F_{\mathrm{cl}}^{-1}]$',xticks=[2,4,6,8,10])
    ax.text(-.13,1.02,label,transform=ax.transAxes,fontsize=12)
    ax.grid(alpha=.22)
    handles=[Line2D([0],[0],color='k',label=r'$J=0$')]+[Line2D([0],[0],color=colors[j],marker='o',ms=4,lw=1,label=f'$J={j:g}$') for j in [.1,1,10]]
    ax.legend(handles=handles,loc='upper left',fontsize=8,framealpha=.96,borderpad=.4)
    ins=ax.inset_axes([.52,.50,.455,.455]);ins.set_facecolor('white')
    valid=times<=11
    for J in (.1,1.,10.):ins.plot(times[valid],np.array(curves[k][f'J{J:g}'])[valid],color=colors[J],lw=.7)
    lo=1e-3 if k=='xz' else 1e-2;hi=1e8 if k=='xz' else 1e16
    assert max(max(np.array(curves[k][f'J{j:g}'])[valid]) for j in [.1,1,10])<hi
    assert min(min(np.array(curves[k][f'J{j:g}'])[valid]) for j in [.1,1,10])>lo
    ins.set(xlim=(1,11),ylim=(lo,hi),yscale='log',xticks=[2,6,10],yticks=([1e-2,1e2,1e6] if k=='xz' else [1e-1,1e4,1e9,1e14]))
    ins.tick_params(labelsize=7,length=2,pad=1)
    ins.set_title('Unoptimized controls',fontsize=8,pad=3,bbox=dict(facecolor='white',edgecolor='none',pad=1))
    ins.grid(alpha=.17)
fig.savefig(O/'fig3_circular.pdf')
fig.savefig(O/'figure3_preview.png',dpi=190)

plt.close(fig)

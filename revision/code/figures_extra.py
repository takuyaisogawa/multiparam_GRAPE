from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import to_hex
from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
from paths import DATA as D, output, load
from summarize import initialization_summary
F=output('figures');OUT=F;CALC=F
CASES=load('initialization_settings.json')['cases']
s=initialization_summary()
plt.rcdefaults();plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','pdf.fonttype':42,'svg.fonttype':'none'})
original_save=Figure.savefig
checks={}
def data(fig):
 return [{'lines':[{'x':np.asarray(l.get_xdata()).tolist(),'y':np.asarray(l.get_ydata()).tolist()} for l in ax.lines],
          'bands':[[p.vertices.tolist() for p in c.get_paths()] for c in ax.collections],
          'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),'xscale':ax.get_xscale(),'yscale':ax.get_yscale()} for ax in fig.axes]
colors={'#2563a4':'#1f77b4','#c36722':'#ff7f0e','#16816c':'#2ca02c','#1263a0':'#1f77b4'}
def save(fig,path,*args,**kwargs):
 name=Path(path).stem
 if not getattr(fig,'_njp_styled',False):
  before=data(fig)
  mismatch=name=='parameter_mismatch'
  fig.set_size_inches((7.3,7.1) if mismatch else (7.8,4.1 if name=='q3_budgets' else 4.8))
  for i,ax in enumerate(fig.axes):
   for sp in ax.spines.values():sp.set_visible(True);sp.set_color('black');sp.set_linewidth(.8)
   ax.tick_params(axis='both',which='both',direction='in',labelsize=9 if mismatch else 10,width=.8)
   ax.grid(True,which='major',color='#b0b0b0',alpha=.3,lw=.8,linestyle='-')
   ax.set_axisbelow(True)
   ax.set_ylabel(r'$\mathrm{Tr}\mathcal{F}_{\mathrm{cl}}^{-1}$',fontsize=10)
   ax.xaxis.label.set_fontsize(10)
   for line in ax.lines:
    old=to_hex(line.get_color());line.set_color(colors.get(old,old))
    if line.get_label().startswith('_'):continue
    line.set_linewidth(1.2)
    if line.get_marker() not in ['None',None,'',' ']:line.set_markersize(3 if mismatch else 5)
   for collection in ax.collections:
    if len(collection.get_facecolors()):
     c=to_hex(collection.get_facecolors()[0]);collection.set_facecolor(colors.get(c,c));collection.set_edgecolor(colors.get(c,c))
   if mismatch:
    ax.title.set_fontsize(10);ax._left_title.set_fontsize(10)
   else:
    oldtitle=ax.get_title();kind=oldtitle.split(':')[0]
    names={'static':'Static','xz':r'Circular $xz$','xy':r'Circular $xy$','linear':'Linear'}
    tail=oldtitle.split(':',1)[1].strip() if ':' in oldtitle else ''
    if 'final J=' in tail:tail='$J='+tail.split('=')[1]+'$'
    elif tail.startswith('T='):tail='$'+tail+'$'
    ax.set_title('');ax.set_title(f'({chr(97+i)}) '+names[kind]+(', '+tail if tail else ''),loc='left',fontsize=10,pad=7)
  handles,labels=fig.axes[0].get_legend_handles_labels()
  for legend in list(fig.legends):legend.remove()
  labels=[{'direct':'Direct','continuation':'Continuation','random median':'Random median'}.get(x,x) for x in labels]
  fig.legend(handles,labels,ncol=2 if mismatch else (3 if len(labels)<=3 else 3),loc='upper center',bbox_to_anchor=(.5,.998),fontsize=9,frameon=True,facecolor='white',edgecolor='#cccccc',framealpha=.9)
  fig.tight_layout(rect=(0,0,1,.95 if mismatch else .92),h_pad=1.4 if mismatch else 1.8,w_pad=1.3 if mismatch else 1.7)
  after=data(fig);assert before==after,name
  checks[name]={'all_line_coordinates_IQR_bands_limits_and_scales_unchanged':True,'axes':len(fig.axes),'before_data_sha256':hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),'after_data_sha256':hashlib.sha256(json.dumps(after,sort_keys=True).encode()).hexdigest()}
  fig._njp_styled=True
 newpath=F/(name+Path(path).suffix)
 original_save(fig,newpath,*args,**kwargs)
Figure.savefig=save

def reduce_crowded_ticks(axs):
 for ax in axs.ravel():
  lo,hi=ax.get_ylim()
  if hi/lo<3:
   ticks=np.unique(np.r_[ax.get_yticks(),ax.get_yticks(minor=True)])
   ticks=ticks[(ticks>=lo)&(ticks<=hi)]
   if len(ticks)>5:
    keep=ticks[np.linspace(0,len(ticks)-1,4,dtype=int)]
    ax.yaxis.set_major_locator(FixedLocator(keep))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(lo,hi)
def plots(target,pathrows,scan):
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 cols={'direct':'#2563a4','continuation':'#c36722','random_median':'#16816c'}
 fig,axs=plt.subplots(2,2,figsize=(7,3.7))
 for ax,(kind,cfg) in zip(axs.ravel(),CASES.items()):
  rr=[r for r in target if r['model']==kind and r['index']==len(cfg['path'])-1];xx=[r['budget'] for r in rr]
  for k,c in cols.items():ax.plot(xx,[r[k] for r in rr],'-o',color=c,label=k.replace('_',' '))
  ax.fill_between(xx,[r['random_q25'] for r in rr],[r['random_q75'] for r in rr],alpha=.15,color=cols['random_median']);ax.set_xticks(xx);ax.set_yscale('log');ax.grid(alpha=.15);ax.set_xlabel('Total evaluation cap');ax.set_ylabel('Refined C');ax.set_title(f"{kind}: final J={cfg['path'][-1]:g}")
 fig.legend(*axs[0,0].get_legend_handles_labels(),ncol=3,loc='upper center');reduce_crowded_ticks(axs);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'q3_budgets.png',dpi=200);fig.savefig(OUT/'q3_budgets.pdf');plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(7,3.7))
 for ax,kind in zip(axs.ravel(),CASES):
  for budget,col in zip([600,1200,2400],['#2563a4','#c36722','#16816c']):
   rr=[r for r in pathrows if r['model']==kind and r['budget']==budget];ax.plot([r['factor'] for r in rr],[r['C'] for r in rr],'-o',color=col,label=f'{budget} evaluations')
  ax.set_xticks([1,2,4]);ax.set_yscale('log');ax.grid(alpha=.15);ax.set_xlabel('Subdivisions per original J interval');ax.set_ylabel('Refined C');ax.set_title(kind)
 fig.legend(*axs[0,0].get_legend_handles_labels(),ncol=3,loc='upper center');reduce_crowded_ticks(axs);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'q3_paths.png',dpi=200);fig.savefig(OUT/'q3_paths.pdf');plt.close(fig)
plots(s["target"],s["paths"],[])
plt.rcdefaults();plt.rcParams.update({'pdf.fonttype':42})
rows=load('mismatch.json')
names={'static':'Static','xz':'Circular $xz$','xy':'Circular $xy$','linear':'Linear'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                     'legend.fontsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42})
panels=[('static',0),('static',1),('static',2),('xz',0),('xz',1),('xy',0),('xy',1),('linear',0),('linear',1)]
fig,axs=plt.subplots(3,3,figsize=(7.3,7.1))
for idx,((kind,axis),ax) in enumerate(zip(panels,axs.ravel())):
    parameter=['B',r'\theta',r'\phi'][axis] if kind=='static' else ['B',r'\omega'][axis]
    for arm,label,color,style in [('saved','Saved GRAPE','#1263A0','-'),('analytical','Analytical initial control','#666666','--')]:
        rr=[r for r in rows if r['kind']==kind and r['arm']==arm and (r['category']=='nominal' or (r['category']=='individual' and r['axis']==axis))]
        rr.sort(key=lambda r:r['offset'][axis])
        x=np.array([r['offset'][axis] for r in rr]);y=[r['metrics']['C'] for r in rr]
        if kind=='static' and axis>0:xlabel=rf'$\Delta {parameter}$ (rad)'
        else:
            x=x/rr[0]['nominal_parameters'][axis]*100
            xlabel=rf'$\Delta {parameter}/{parameter}_0$ (%)'
        ax.plot(x,y,color=color,ls=style,lw=1.2,marker='o' if arm=='saved' else None,ms=2.2,label=label)
    ax.set_yscale('log');ax.set_xlabel(xlabel);ax.set_ylabel(r'$C=\mathrm{Tr}(F_{\rm cl}^{-1})$')
    ax.set_title(f'({chr(97+idx)}) '+names[kind]+rf': ${parameter}$',loc='left')
    ax.grid(True,which='major',alpha=.2,lw=.5);ax.axvline(0,color='k',alpha=.15,lw=.5)
    ax.tick_params(axis='both',which='both',direction='in')
handles,labels=axs[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.01))
fig.tight_layout(rect=[0,0,1,.97],h_pad=1.4,w_pad=1.3)
fig.savefig(CALC/'parameter_mismatch.pdf',bbox_inches='tight')
fig.savefig(CALC/'parameter_mismatch.png',dpi=180,bbox_inches='tight')

plt.close('all')
(F/'plot_data_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print('Regenerated figures from preserved results; data and axis ranges verified unchanged.')

Figure.savefig=original_save

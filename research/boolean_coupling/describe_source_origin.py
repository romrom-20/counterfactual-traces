"""All-case282 prediction and signed generation/transmission figures; zero model calls."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'results/source2-origin-282'
FIG=ROOT/'docs/figures/source2-origin'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(n):
    with (DEST/(n+'.csv')).open() as stream:
        return list(csv.DictReader(stream))


def main():
    assert not FIG.exists()
    FIG.mkdir(parents=True)
    audit=json.loads((DEST/'audit.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9.5))
    arms=['B','K','W','S','G0','G1','G2','X','NC','NW','U']
    colors=['#87919d','#247d99','#bc8a33','#568a70','#b6bec5','#b6bec5','#b6bec5','#a86b8b','#7372ae','#82705c','#245c44']
    x=np.arange(len(arms))
    axes[0,0].bar(x-.18,[audit['native_counts'][a] for a in arms],.36,label='Original native response',color='#247d99')
    axes[0,0].bar(x+.18,[audit['joint_counts'][a] for a in arms],.36,label='Joint + paired write account',color='#bc8a33')
    axes[0,0].axhline(24,color='#666',linestyle='--',linewidth=1)
    axes[0,0].set(xticks=x,xticklabels=arms,ylim=(0,34),ylabel='Cases passing / 32')
    axes[0,0].set_title('Retain the smaller causal target',loc='left',weight='bold')
    axes[0,0].legend(fontsize=8,loc='lower left',framealpha=.95)
    summary={}
    for ax,dataset,field,title in ((axes[0,1],'local','relative','Missing source2 write · 32 cases'),
                                   (axes[1,0],'effect','all_relative','Actual receiving effect · 32 cases'),
                                   (axes[1,1],'paired','local_relative','Paired source-write contrast · 16 pairs')):
        data=rows(dataset)
        actual=[[float(r[field])*100 for r in data if r['arm']==a] for a in arms]
        values=[[max(v,.1) for v in group] for group in actual]
        boxes=ax.boxplot(values,patch_artist=True,tick_labels=arms,showfliers=False)
        for patch,color in zip(boxes['boxes'],colors,strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(.7)
        for i,(group,color) in enumerate(zip(values,colors,strict=True)):
            ax.scatter(np.linspace(i+.86,i+1.14,len(group)),group,s=9,alpha=.65,color=color,zorder=3)
        ax.axhline(25,color='#666',linestyle='--',linewidth=1)
        ax.set_yscale('log')
        ax.set_ylabel('Relative error (%) · log scale')
        ax.set_title(title,loc='left',weight='bold')
        summary[dataset]={a:float(np.median(v))/100 for a,v in zip(arms,actual,strict=True)}
    fig.suptitle('Where does the source2 curvature correction come from?',fontsize=17,weight='bold',ha='left',x=.07)
    fig.text(.07,.929,'32 old contexts / 16 dependent pairs · source1 cubic · direct source2 product interventions · all later work finite',fontsize=10,color='#444')
    fig.text(.07,.025,'K: carry · W: attention · S: additive · X: local interaction · NC/NW: remove carry/attention · U: full positive control.\nAll cases retained. Joint success includes the paired local target. Zero errors plotted at 0.1%; exact values remain in CSV.',fontsize=8.5,color='#444')
    fig.subplots_adjust(top=.865,bottom=.13,hspace=.43,wspace=.28)
    for ext in ('png','svg'):
        fig.savefig(FIG/('source2-origin.'+ext),dpi=170,bbox_inches='tight',metadata={'Creator':'Counterfactual Traces'})
    p=FIG/'source2-origin.svg'
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig)
    def geom(name):
        with (DEST/'geometry-posthoc'/(name+'.csv')).open() as stream:
            return list(csv.DictReader(stream))
    fig,axes=plt.subplots(1,2,figsize=(12,5.8))
    norms=geom('input-output-norms')
    fields=['attention_to_carry','attention_write_relative','mixed_write_relative']
    groups=[[float(r[n])*100 for r in norms] for n in fields]
    ax=axes[0]
    ax.bar(range(3),[np.median(v) for v in groups],color=['#87919d','#bc8a33','#a86b8b'],alpha=.7)
    for i,vals in enumerate(groups):
        ax.scatter(np.linspace(i-.12,i+.12,len(vals)),vals,s=14,color='#444',alpha=.55)
    ax.set(xticks=range(3),xticklabels=['Attention input\n/ carry input','Attention error\n/ full error','Mixed error\n/ full error'],yscale='log',ylabel='Norm ratio (%) · log scale')
    ax.set_title('Small standalone error, larger mixed error',loc='left',weight='bold')
    ax=axes[1]
    led=geom('signed-ledger')
    components=['carry','attention','mixed_gate_remainder','mixed_up_reweighting','rounding']
    groups=[[float(r['signed_projection']) for r in led if r['kind']=='paired_missing_write' and r['component']==c] for c in components]
    ax.bar(range(5),[np.median(v) for v in groups],color=['#247d99','#bc8a33','#a86b8b','#568a70','#87919d'],alpha=.75)
    for i,vals in enumerate(groups):
        ax.scatter(np.linspace(i-.12,i+.12,len(vals)),vals,s=14,color='#444',alpha=.55)
    ax.axhline(0,color='#555',linewidth=.8)
    ax.axhline(1,color='#777',linestyle='--',linewidth=.8)
    ax.set(xticks=range(5),xticklabels=['Carry','Attention','Mixed\ngate\nremainder','Mixed up\nreweighting','Rounding'],ylabel='Signed projection onto paired missing write')
    ax.tick_params(axis='x',labelsize=9)
    ax.set_title('What the paired contrast contains',loc='left',weight='bold')
    fig.suptitle('Attention changes the response around the carried state',fontsize=16,weight='bold',ha='left',x=.07)
    fig.text(.07,.88,'Posthoc saved-array geometry · 32 old cases / 16 dependent pairs · no new model forwards',fontsize=10,color='#444')
    fig.text(.07,.025,'Left: denominators differ as labeled. Right: exact signed product/write ledger with rounding; separate medians need not add.\nThe gate remainder includes normalization and SiLU. This identity is not a primitive causal intervention or learned feature.',fontsize=8.5,color='#444')
    fig.subplots_adjust(top=.79,bottom=.22,wspace=.37)
    for ext in ('png','svg'):
        fig.savefig(FIG/('source2-origin-geometry.'+ext),dpi=170,bbox_inches='tight',metadata={'Creator':'Counterfactual Traces'})
    p=FIG/'source2-origin-geometry.svg'
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig)
    (FIG/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (FIG/'provenance.json').write_text(json.dumps(dict(posthoc=True,model_calls=0,predictions_modified=False,script_sha256=sha(__file__),
        source_csv_sha256={n+'.csv':sha(DEST/(n+'.csv')) for n in ('native','local','effect','joint','ledger','paired')},
        geometry_provenance_sha256=sha(DEST/'geometry-posthoc/provenance.json'),
        audit_sha256=sha(DEST/'audit.json'),figure_sha256={p.name:sha(p) for p in FIG.iterdir() if p.suffix in ('.png','.svg')}),indent=2)+'\n')


if __name__=='__main__':
    main()

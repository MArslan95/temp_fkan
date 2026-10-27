"""Re-layout archived maps without changing their class colors or predictions."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from .protocol import sha256

ROOT=Path(__file__).resolve().parents[1]
LABELS=['GT','HViT','SSF','SSFTTKAN','HybridKAN','DF','SSMM','MHM','WM','FKANMamba']


def spans(mask):
    delta=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(delta==1),np.flatnonzero(delta==-1)))


def main():
    out=ROOT/'paper/revision_assets/figures'
    out.mkdir(parents=True,exist_ok=True)
    records={}
    for dataset in ('HC','HH','TH','PI','QUH'):
        source=ROOT/'paper/Fig'/f'{dataset}.png'
        im=Image.open(source).convert('RGBA')
        array=np.asarray(im)
        rows=[(a,b) for a,b in spans((array[:,:,3]>0).mean(1)>.3) if b-a>100]
        panels=[]
        boxes=[]
        for y0,y1 in rows:
            columns=[(a,b) for a,b in spans(array[(y0+y1)//2,:,3]>0) if b-a>100]
            for x0,x1 in columns:
                crop=im.crop((x0,y0,x1,y1))
                white=Image.new('RGB',crop.size,'white')
                white.paste(crop,mask=crop.getchannel('A'))
                panels.append(white)
                boxes.append([int(x0),int(y0),int(x1),int(y1)])
        if len(panels)!=10:
            raise ValueError(f'{dataset}: expected 10 unambiguous map panels, found {len(panels)}')
        nrows,ncols=(1,10) if dataset=='HC' else ((4,3) if dataset=='QUH' else (2,5))
        heights={'HC':6.2,'HH':11.7,'TH':11.7,'PI':10.8,'QUH':12.0}
        fig,axes=plt.subplots(nrows,ncols,figsize=(14.4,heights[dataset]),squeeze=False)
        for ax in axes.ravel():
            ax.set_axis_off()
        positions=list(range(10)) if dataset!='QUH' else list(range(9))+[10]
        for index,panel,label in zip(positions,panels,LABELS):
            ax=axes.ravel()[index]
            ax.imshow(panel,interpolation='nearest')
            ax.text(.5,-.025,label,ha='center',va='top',transform=ax.transAxes,
                    fontsize=14 if dataset=='HC' else 17,color='black',fontweight='bold')
        fig.subplots_adjust(left=.005,right=.995,top=.995,bottom=.06,wspace=.06,hspace=.10)
        fig.savefig(out/f'{dataset}.png',dpi=220,facecolor='white')
        plt.close(fig)
        records[dataset]=dict(source=str(source.relative_to(ROOT)).replace('\\','/'),
                             source_sha256=sha256(source),map_boxes=boxes,
                             labels=LABELS,note='Order visually checked against original captions; original pixels retained inside each map crop. Only layout, white transparency background and external labels changed.')
    (out/'provenance.json').write_text(json.dumps(records,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(16,5.2))
    ax.set(xlim=(0,1),ylim=(0,1));ax.set_axis_off()
    def box(x,y,w,h,label):
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.008',
                                   facecolor='#f4f7fa',edgecolor='#25364a',linewidth=1.2))
        ax.text(x,y,label,ha='center',va='center',fontsize=13)
    def arrow(x0,y0,x1,y1):
        ax.annotate('',(x1,y1),(x0,y0),arrowprops={'arrowstyle':'->','lw':1.2,'color':'#25364a'})
    box(.065,.5,.11,.22,'Input patch\n12 x 12 x 15')
    for y,label in [(.74,'Transpose [0,2,3,1]\nReshape [B,144,15]\nIndependent Dense 32'),
                    (.26,'Reshape [B,144,15]\nIndependent Dense 32')]:
        box(.255,y,.22,.24,label)
        box(.465,y,.15,.24,'FEB\nIndependent\nFastKAN gate')
        box(.655,y,.16,.24,'FSSA\n8 heads, width 32\nShared across views')
        arrow(.123,.5,.138,y);arrow(.371,y,.383,y);arrow(.548,y,.568,y)
        arrow(.742,y,.788,.5)
    ax.text(.465,.5,'Gates use spatial-view token 6\nNot the target-pixel spectrum',
            ha='center',va='center',fontsize=11,color='#843000')
    arrow(.255,.612,.255,.5);arrow(.255,.5,.383,.74);arrow(.255,.5,.383,.26)
    box(.823,.5,.075,.25,'Concat\n144 x 64')
    arrow(.867,.5,.887,.5)
    box(.945,.5,.10,.34,'Affine scan\nstate 64\nFinal state\nDense softmax')
    ax.text(.5,.965,'Implemented FKANMamba: corrected operation diagram',ha='center',fontsize=17,fontweight='bold')
    ax.text(.5,.035,'Fixed learned recurrent matrices; no selective parameters or post-recurrence normalization',ha='center',fontsize=12)
    fig.subplots_adjust(left=.01,right=.99,top=.98,bottom=.02)
    fig.savefig(out/'architecture.png',dpi=220,facecolor='white');plt.close(fig)
    print('Rebuilt five map figures with explicit labels and saved crop provenance.')


if __name__=='__main__':
    main()

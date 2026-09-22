"""Generate CPU-only backbone summaries, schematic, and torchviz DOT graph."""
import json
from dataclasses import asdict, replace
from pathlib import Path
import importlib.metadata
import torch
from torch import nn
from torchinfo import summary
from torchviz import make_dot
from semi_mlip.model import ModelConfig, Potential
from semi_mlip.graph import neighbor_list, collate

ROOT = Path(__file__).resolve().parents[1]

class EnergyBackbone(nn.Module):
    """Expose the energy path without force/stress differentiation in torchinfo."""
    def __init__(self, config):
        super().__init__()
        self.potential = Potential(config)
    def forward(self, graph):
        return self.potential.energy(graph)

def draw(config, counts, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#f6f8fc')
    ax.set(xlim=(0,14), ylim=(0,10)); ax.axis('off')
    def box(x,y,w,h,title,body,color='#e4edfa'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10,rounding_size=0.12',facecolor=color,edgecolor='#a2b1c5',linewidth=1.1))
        ax.text(x+w/2,y+h-.25,title,ha='center',va='top',fontsize=12,weight='bold',color='#152944')
        ax.text(x+w/2,y+h/2-.15,body,ha='center',va='center',fontsize=10,color='#233b55',linespacing=1.5)
    def arrow(a,b,label=''):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color='#486581',linewidth=1.5))
        if label: ax.text((a[0]+b[0])/2+.08,(a[1]+b[1])/2,label,fontsize=9,color='#486581')
    ax.text(.4,9.65,'Semi-E3-MLIP | Shared energy backbone',fontsize=22,weight='bold',color='#152944')
    ax.text(.4,9.2,'354,545 trainable parameters   /   4 interaction blocks   /   5.0 A periodic cutoff',fontsize=12,color='#52677f')
    box(.5,7.45,3.3,1.25,'Periodic atomic structure','Atomic numbers Z [N]\nPositions [N,3] + cell [B,3,3]')
    box(5,7.45,3.6,1.25,'Geometry encoding','Directed periodic edges + image shifts\nGaussian radial [E,32] + directions [E,3]')
    box(9.7,7.45,3.7,1.25,'Species embedding','119 x 64 lookup | 7,616 parameters\nScalars [N,64]; zero vectors / tensors')
    arrow((3.9,8.05),(4.9,8.05))
    ax.plot([2.15,2.15,11.55],[8.82,8.98,8.98],color='#486581',linewidth=1.5)
    arrow((11.55,8.98),(11.55,8.8))
    ax.text(9.2,9.01,'atomic numbers Z',fontsize=8,color='#486581')
    ax.add_patch(FancyBboxPatch((.5,3.1),12.9,3.6,boxstyle='round,pad=0.12',facecolor='#ffffff',edgecolor='#6f8cae',linewidth=1.6))
    ax.text(.8,6.3,'Interaction block x 4',fontsize=16,weight='bold',color='#152944')
    ax.text(13.1,6.3,'85,684 parameters each | independent weights',ha='right',fontsize=11,color='#52677f')
    box(.85,4.2,3.6,1.5,'Scalar conditioning','LayerNorm + SwiGLU source MLP\nTarget gate x radial projection\nSmooth attention: 4 heads')
    box(5.05,4.2,3.65,1.5,'Equivariant messages','Scalar / polar-vector / rank-2 tensor\nDirection-based geometric mixing\nSmooth cutoff + neighbor sum', '#e0f1ec')
    box(9.3,4.2,3.65,1.5,'Residual feature update','Invariant contractions + scalar MLP\nVector / tensor channel mixing\nResidual scalar, vector, tensor states', '#e0f1ec')
    arrow((4.55,4.95),(4.95,4.95)); arrow((8.8,4.95),(9.2,4.95))
    ax.text(7,3.5,'State after each block: scalars [N,64]   |   vectors [N,3,32]   |   tensors [N,3,3,16]',ha='center',fontsize=12,color='#233b55')
    arrow((11.5,7.35),(11.5,6.8)); arrow((6.7,7.35),(6.7,6.8))
    box(.6,.95,3.65,1.35,'Atomic energy readout','SwiGLU MLP: 64 -> 1\n4,193 parameters; fixed species offsets')
    box(5.05,.95,3.65,1.35,'Total energy [B]','Sum atomic energies per structure\nRotation / reflection invariant', '#fff0db')
    box(9.5,.95,3.65,1.35,'Conservative derivatives','Forces [N,3] = -dE/dR\nStress [B,3,3] = (1/V) dE/dstrain', '#fff0db')
    arrow((2.4,3),(2.4,2.4)); arrow((4.35,1.6),(4.95,1.6)); arrow((8.8,1.6),(9.4,1.6))
    ax.text(.5,.35,'Module-level schematic of the published 1x configuration. N: atoms; E: directed edges; B: structures.',fontsize=10,color='#52677f')
    fig.savefig(destination.with_suffix('.png'),dpi=170,bbox_inches='tight')
    fig.savefig(destination.with_suffix('.svg'),bbox_inches='tight')
    plt.close(fig)

def main():
    torch.set_num_threads(1); torch.manual_seed(0)
    out=ROOT/'reports/architecture'; out.mkdir(parents=True,exist_ok=True)
    assets=ROOT/'docs/assets'; assets.mkdir(parents=True,exist_ok=True)
    config=ModelConfig(attention=True,tensor_channels=16,average_neighbors=43.825026652452024)
    row=dict(z=[22,22,22], positions=[[0.,0.,0.],[2.6,0.,0.],[1.3,2.25,0.]],cell=[[6.,0.,0.],[0.,6.,0.],[0.,0.,6.]],pbc=[True]*3)
    graph=collate([row],[neighbor_list(row['positions'],row['cell'],row['pbc'],config.cutoff)])
    report={'description':'Published shared configuration; fresh random weights for topology inspection only', 'config':asdict(config),'input':{'atoms':3,'edges':len(graph['i']),'structures':1},'source_model_sha256':__import__('hashlib').sha256((ROOT/'semi_mlip/model.py').read_bytes()).hexdigest(),'versions':{p:importlib.metadata.version(p) for p in ('torch','torchinfo','torchviz','graphviz')},'models':{}}
    for factor,expected in ((1,354545),(2,1356513),(4,5288873)):
        c=replace(config,scalar_channels=64*factor,vector_channels=32*factor,tensor_channels=16*factor)
        wrapped=EnergyBackbone(c).eval()
        stats=summary(wrapped,input_data=(graph,),depth=5,col_names=('input_size','output_size','num_params'),row_settings=('var_names',),device='cpu',verbose=0)
        count=sum(p.numel() for p in wrapped.parameters())
        assert stats.total_params==count==expected
        (out/f'torchinfo_{factor}x.txt').write_text(str(stats)+'\n',encoding='utf-8')
        counts={name:sum(p.numel() for p in module.parameters()) for name,module in wrapped.potential.named_children()}
        report['models'][f'{factor}x']={'parameters':count,'modules':counts,'config':asdict(c)}
        if factor==1:
            graph['positions'].requires_grad_(True)
            energy=wrapped(graph)
            dot=make_dot(energy,params={**dict(wrapped.named_parameters()),'positions':graph['positions']})
            dot.save(str(out/'backbone_autograd.dot'))
            # Rendering is optional: DOT remains usable without native Graphviz.
            import shutil
            if shutil.which('dot'):
                dot.render(filename='backbone_autograd',directory=str(assets),format='svg',cleanup=True)
            draw(c,counts,assets/'backbone_summary')
            result=wrapped.potential(graph)
            assert result['forces'].shape==(3,3) and result['stress'].shape==(1,3,3)
            assert all(torch.isfinite(result[k]).all() for k in ('energy','forces','stress'))
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report['models'],indent=2))

if __name__=='__main__': main()

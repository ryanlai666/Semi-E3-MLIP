"""Flowchart of the implemented computation, exported as vector artwork."""
from pathlib import Path
from .visualize import plotting

def draw_backbone(config, output):
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    plt=plotting(); c=config
    fig,ax=plt.subplots(figsize=(16,10),dpi=160)
    fig.patch.set_facecolor('white');ax.set_facecolor('white')
    ax.set(xlim=(0,16),ylim=(0,10));ax.axis('off')
    colors={'scalar':'#e4effb','geometry':'#e7f3ee','attention':'#eee7f8','energy':'#f9edda'}
    def node(x,y,w,label,kind='scalar',h=.62):
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.035,rounding_size=.09',
                                   facecolor=colors[kind],edgecolor='#45566b',lw=1.1))
        ax.text(x,y,label,ha='center',va='center',fontsize=10.5,color='#172c44',linespacing=1.35)
    def edge(points,dashed=False):
        for p,q in zip(points[:-2],points[1:-1]):
            ax.plot([p[0],q[0]],[p[1],q[1]],color='#52657a',lw=1.2,ls='--' if dashed else '-')
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=12,
                                    lw=1.2,color='#52657a',linestyle='--' if dashed else '-'))
    def operator(x,y,label):
        ax.add_patch(Circle((x,y),.20,facecolor='white',edgecolor='#45566b',lw=1.2))
        ax.text(x,y,label,ha='center',va='center',fontsize=14)
    ax.text(.45,9.65,'E(3)-equivariant graph neural network potential',fontsize=21,weight='bold',color='#172c44')
    ax.text(.45,9.13,'a   Full model',fontsize=12,weight='bold')
    node(1.4,8.38,2.1,'Atoms + periodic cell','geometry')
    node(4.1,8.38,2.25,'Node / edge encoding','geometry')
    # Stacked silhouettes indicate repeated computation, not parallel models.
    for dx,dy in [(.11,.11),(.055,.055)]:
        ax.add_patch(FancyBboxPatch((6.0+dx,8.04+dy),2.3,.62,boxstyle='round,pad=.035',facecolor='white',edgecolor='#8192a5',lw=.8))
    node(7.15,8.35,2.3,f'Interaction × {c.blocks}')
    node(10.0,8.35,2.1,'Atomwise MLP')
    node(12.55,8.35,2.0,r'$E=\sum_i(\epsilon_i+b_{Z_i})$','energy')
    for p,q in [((2.45,8.38),(2.96,8.38)),((5.23,8.38),(5.98,8.38)),((8.32,8.35),(8.94,8.35)),((11.07,8.35),(11.53,8.35))]:edge([p,q])
    node(14.7,8.8,1.65,r'$\mathbf{F}=-\nabla_R E$','energy')
    node(14.7,7.9,1.65,r'$\sigma=V^{-1}\partial_\eta E$','energy')
    edge([(13.57,8.35),(13.75,8.35),(13.75,8.8),(13.85,8.8)])
    edge([(13.57,8.35),(13.75,8.35),(13.75,7.9),(13.85,7.9)])
    ax.plot([.4,15.6],[7.3,7.3],color='#d8dfe7',lw=.8)
    ax.text(.45,6.96,'b   Inside one interaction block',fontsize=12,weight='bold')
    node(4.25,6.25,2.8,r'Node features  $h_i,\,v_i,\,T_i$')
    node(1.45,5.08,2.25,r'$r_{ij},\,\hat r_{ij},\,Q(\hat r_{ij})$','geometry')
    node(1.45,3.75,2.25,'Radial basis\n+ smooth cutoff','geometry',h=.8)
    edge([(1.45,4.76),(1.45,4.17)])
    edge([(.30,5.08),(.16,5.08),(.16,1.48),(2.86,1.48)])
    node(4.25,5.08,2.6,'Scalar LayerNorm')
    edge([(4.25,5.93),(4.25,5.41)])
    node(4.25,3.75,2.6,'Source MLP\n× central gate',h=.8)
    edge([(4.25,4.76),(4.25,4.17)])
    node(7.75,5.08,2.45,'Scalar Q, K + radial bias','attention')
    edge([(5.57,5.08),(6.49,5.08)])
    edge([(2.61,3.75),(2.83,3.75),(2.83,5.76),(7.75,5.76),(7.75,5.41)],dashed=True)
    heads=c.attention_heads if c.attention else 1
    if c.attention:
        for k in range(heads):
            x=6.55+k*(2.4/max(heads-1,1))
            node(x,3.75,.62,f'H{k+1}','attention',h=.6)
            edge([(7.75,4.76),(7.75,4.42),(x,4.42),(x,4.07)])
            edge([(x,3.43),(x,3.05),(7.75,3.05),(7.75,2.86)])
        node(7.75,2.52,2.7,'Smooth head weights','attention')
    else:
        node(7.75,3.75,2.4,'Gated baseline','attention')
        edge([(7.75,4.76),(7.75,4.08)])
        node(7.75,2.52,2.7,'Neighbor normalization','attention')
        edge([(7.75,3.43),(7.75,2.86)])
    operator(4.25,2.52,'×')
    edge([(4.25,3.33),(4.25,2.74)])
    edge([(1.45,3.33),(1.45,2.52),(4.03,2.52)])
    edge([(6.38,2.52),(4.47,2.52)])
    node(4.25,1.48,2.7,'Equivariant messages\n+ neighbor sum',h=.78)
    edge([(4.25,2.3),(4.25,1.89)])
    node(7.75,1.48,2.7,'Invariant contractions\n+ gated residual update',h=.78)
    edge([(5.62,1.48),(6.37,1.48)])
    node(11.45,1.48,2.65,r'$h_i^{\ell+1},\,v_i^{\ell+1},\,T_i^{\ell+1}$')
    edge([(9.12,1.48),(10.09,1.48)])
    edge([(4.25,6.58),(10.05,6.58),(10.05,2.08),(8.65,2.08),(8.65,1.9)],dashed=True)
    ax.text(8.35,6.71,'residual path',fontsize=9,color='#52657a')
    # A compact legend, separate from the computational flow.
    ax.text(11.2,6.5,'Feature types',fontsize=12,weight='bold',color='#172c44')
    ax.text(11.2,6.04,r'$h$: invariant scalar',fontsize=11)
    ax.text(11.2,5.60,r'$v$: equivariant vector',fontsize=11)
    ax.text(11.2,5.16,r'$T$: symmetric traceless tensor',fontsize=11)
    ax.text(11.2,4.45,f'Channels: {c.scalar_channels} / {c.vector_channels} / {c.tensor_channels}\nRadial basis: {c.radial_basis}   |   Cutoff: {c.cutoff:g} Å\nActivation: {c.activation.upper()}',fontsize=10.5,linespacing=1.8,color='#52657a',va='top')
    ax.text(11.2,2.82,'Scalar attention weights preserve\nvector / tensor transformation laws.',fontsize=10,color='#52657a',linespacing=1.5)
    ax.text(.5,.45,'Tensor branch is optional. T = 0 channels disables it. Forces and stress are derivatives of the same scalar energy.',fontsize=10,color='#52657a')
    fig.subplots_adjust(left=.015,right=.985,top=.99,bottom=.015)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg','pdf'):fig.savefig(output.with_suffix('.'+suffix),dpi=180,facecolor='white')
    plt.close(fig)

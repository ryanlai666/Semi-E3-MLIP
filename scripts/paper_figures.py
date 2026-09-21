"""Publication-style, vector-exported explanations of our implemented model."""
import argparse
import json
from pathlib import Path
import numpy as np
from semi_mlip.visualize import plotting
from semi_mlip.train import load_potential

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');a=p.parse_args()
    model,payload=load_potential(a.checkpoint);c=model.config
    out=Path('reports/figures');out.mkdir(exist_ok=True)
    plt=plotting()
    from semi_mlip.architecture import draw_backbone
    draw_backbone(c,out/"backbone")
    trials=[]
    for name in ('device_search.json','tensor_search.json','expanded_search.json'):
        path=Path('reports')/name
        if path.exists():trials+=json.loads(path.read_text())['trials']
    fig,axes=plt.subplots(1,2,figsize=(13,5.5));x=np.arange(len(trials))
    for ax,key,title,target in [(axes[0],'force_mae_eV_A','Validation force MAE (eV/Å)',.1),
                               (axes[1],'energy_mae_eV_atom','Validation energy MAE (eV/atom)',.01)]:
        values=[t['validation']['overall'][key] for t in trials]
        ax.bar(x,values,color=['#377e91' if t['model']['attention'] else '#ac956c' for t in trials])
        ax.axhline(target,color='#b74747',ls='--',label='Working target')
        ax.set_xticks(x,[t['name'] for t in trials],rotation=45,ha='right',fontsize=8)
        ax.set_title(title);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Validation-only pilots | unequal durations; expanded runs add MP-ALOE data',weight='bold')
    fig.tight_layout()
    for suffix in ('png','svg','pdf'):fig.savefig(out/f'hyperparameter_comparison.{suffix}',dpi=180)
    plt.close(fig)
    history=[json.loads(l) for l in (Path(a.checkpoint).parent/'history.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,3,figsize=(14,4.3))
    for ax,key,label in [(axes[0],'force_mae_eV_A','Force MAE (eV/Å)'),(axes[1],'energy_mae_eV_atom','Energy MAE (eV/atom)')]:
        ax.plot([r['epoch'] for r in history],[r['valid'][key] for r in history],label='Validation')
        h=[r for r in history if 'train_metrics' in r]
        ax.plot([r['epoch'] for r in h],[r['train_metrics'][key] for r in h],label='Training')
        ax.axvline(payload['epoch']+1,color='gray',ls=':',label='Selected checkpoint');ax.set(xlabel='Epoch',ylabel=label);ax.legend(fontsize=8)
    axes[2].plot([r['updates'] for r in history],[r['lr'] for r in history]);axes[2].set(xlabel='Optimizer update',ylabel='Learning rate',title='AdamW: warmup + cosine')
    fig.tight_layout()
    for suffix in ('png','svg','pdf'):fig.savefig(out/f'generalization.{suffix}',dpi=180)
    plt.close(fig)
    (out/'configuration.json').write_text(json.dumps({'model':payload['model_config'],'training':payload['train_config'],
                          'selected_epoch':payload['epoch']+1},indent=2))

if __name__=='__main__':main()

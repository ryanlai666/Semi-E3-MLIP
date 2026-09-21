"""Publish the owner's authorized final milestone only after all studies and checks pass."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
GIT = ['git', '-c', 'safe.directory='+ROOT.as_posix()]
REPO = 'ryanlai666/Semi-E3-MLIP'

def run(args):
    subprocess.run(args, check=True)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify():
    progress=read('reports/material_studies/progress.json')
    protocol=read('reports/material_studies/protocol.json')
    frozen=read('reports/material_studies/frozen.json')
    results=read('reports/material_studies/results.json')
    assert progress['complete'] and progress['completed']==progress['expected']==54
    assert len(protocol['studies'])==18 and len(results)==len(frozen['trials'])==54
    assert frozen['protocol_sha256']==sha('reports/material_studies/protocol.json')
    for name,study in protocol['studies'].items():
        trials=[t for t in frozen['trials'] if t['study']==name]
        assert sorted(t['seed'] for t in trials)==[42,43,44]
        for part in study['partitions'].values():assert sha(part['path'])==part['sha256']
        for trial in trials:
            assert sha(trial['checkpoint'])==trial['sha256']
            metrics=results[f'{name}_s{trial["seed"]}']['splits']
            assert set(metrics)==set(study['partitions'])
            for split,metric in metrics.items():
                assert metric['frames']==study['partitions'][split]['frames']
                for key in ('energy_mae_eV_atom','force_mae_eV_A','force_rmse_eV_A'):
                    assert math.isfinite(metric[key]),f'Nonfinite result: {name} {split} {key}'
    assert read('reports/benchmark/documentation.json')['complete']
    for name in ('test_forces','distributions','learning_curves'):
        with Image.open(f'docs/assets/material_studies/{name}.png') as im:im.verify()
    assert Path('docs/material_studies.md').is_file()
    return protocol

def api(method='GET',body=None):
    result=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,
        capture_output=True,check=True,env={**os.environ,'GIT_TERMINAL_PROMPT':'0','GCM_INTERACTIVE':'never'})
    credential=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    request=urllib.request.Request('https://api.github.com/repos/'+REPO,
        data=json.dumps(body).encode() if body is not None else None,method=method,
        headers={'Authorization':'Bearer '+credential['password'],'Accept':'application/vnd.github+json',
                 'Content-Type':'application/json','User-Agent':'Semi-E3-MLIP'})
    with urllib.request.urlopen(request,timeout=60) as response:return json.load(response)

def main():
    verify()
    branch=subprocess.check_output(GIT+['branch','--show-current'],text=True).strip()
    assert branch=='experiment/002-independent-material-studies',f'Unexpected branch: {branch}'
    assert not subprocess.check_output(GIT+['diff','--cached','--name-only'],text=True).strip(),'Existing staged changes require review'
    run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','--basetemp=.test-tmp-material-final'])
    run([sys.executable,'scripts/build_project_report.py','--visibility','public'])
    summary=read('reports/material_studies/summary.json')
    block=['## Independent material studies','',
        'Separate models for six additional TM23 metals, elemental Al/Si, and ten oxide chemical systems: 18 studies and 54 fits. Each uses three initialization seeds and validation-only checkpoint selection.', '',
        '[Full train/test results, distributions and learning curves](docs/material_studies.md). Al remains exploratory because its MatPES test contains only one frame. Known Cu/Ti molten-transfer failures remain documented above.', '',
        '![Independent-model test results](docs/assets/material_studies/test_forces.png)','']
    p=Path('README.md'); text=p.read_text(encoding='utf-8');text+='\n'+'\n'.join(block);p.write_text(text,encoding='utf-8')
    p=Path('docs/usage.md');text=p.read_text(encoding='utf-8').replace('(private).','(public).');p.write_text(text,encoding='utf-8')
    p=Path('docs/README.md');text=p.read_text(encoding='utf-8')+'\nFollow-up: [independent material results](material_studies.md) and [fixed protocol](next_material_studies.md).\n';p.write_text(text,encoding='utf-8')
    for p in Path('docs/assets').rglob('*.svg'):
        p.write_text('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    for p in [Path('README.md'),Path('docs/results.md'),Path('docs/gallery.md'),Path('docs/material_studies.md')]:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if '://' not in target and not target.startswith('#'):
                assert (p.parent/target.split('#')[0]).exists(),f'Broken link in {p}: {target}'
    assert api()['private'], 'Expected the repository to remain private until this completed milestone'
    paths=['README.md','docs/results.md','docs/gallery.md','docs/usage.md','docs/README.md','docs/material_studies.md',
        'docs/assets','reports/material_studies','reports/benchmark/documentation.json','reports/benchmark/inference.json',
        'reports/benchmark/pilot_completion.json','reports/device_search.json','reports/expanded_search.json','reports/tensor_search.json']
    run(GIT+['add','--',*paths]);run(GIT+['diff','--cached','--check'])
    staged=subprocess.check_output(GIT+['diff','--cached','--name-only'],text=True).splitlines()
    for name in staged:
        assert not name.startswith(('data/','runs/')) and not name.endswith('.pt')
        p=Path(name);assert p.stat().st_size<100_000_000
        if p.suffix in ('.json','.md','.py','.yml','.ps1'):
            assert not re.search(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (RSA |OPENSSH )?PRIVATE KEY',p.read_text(encoding='utf-8-sig'))
    run(GIT+['commit','-m','Complete independent metal and oxide studies with measured results'])
    run(GIT+['push','origin','HEAD:refs/heads/main','HEAD:refs/heads/'+branch])
    # Public visibility is explicitly authorized by the owner after all work completes.
    repository=api('PATCH',{'private':False})
    assert repository['full_name']==REPO and repository['private'] is False
    request=urllib.request.Request('https://api.github.com/repos/'+REPO,headers={'User-Agent':'Semi-E3-MLIP'})
    with urllib.request.urlopen(request,timeout=60) as response:assert json.load(response)['private'] is False
    receipt={'complete':True,'repository':'https://github.com/'+REPO,'private':False,
             'commit':subprocess.check_output(GIT+['rev-parse','HEAD'],text=True).strip(),'studies':18,'fits':54}
    Path('runs/material_publication.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()

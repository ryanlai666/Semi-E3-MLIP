"""Download only data from the 2025 Al/Si paper at a pinned Git tree revision."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    meta = json.loads(Path('reports/alsi_interface_source.json').read_text(encoding='utf-8'))
    revision = meta['sha']
    root = Path('data/raw/alsi_interface_2025')
    def download(entry):
        path = root / entry['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/krutarth24/Al-Si-DeePMD-NNP/{revision}/{entry['path']}"
        if not path.exists():
            temporary = path.with_suffix(path.suffix+'.partial')
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        content = path.read_bytes()
        actual = hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
        assert actual == entry['sha'] and len(content) == entry['size']
        print(entry['path'], len(content), flush=True)
        return {'path': str(path), 'url': url, 'git_blob_sha1': actual,
                'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    entries = [x for x in meta['tree'] if x['type']=='blob' and x['path'].endswith(('.npy','.raw'))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        downloaded = list(pool.map(download, entries))
    Path('reports/alsi_interface_download.json').write_text(json.dumps({
        'revision': revision, 'files': downloaded,
        'citation': 'https://doi.org/10.1063/5.0243641',
        'license': 'No explicit license file in pinned repository tree; retain source attribution.'},indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()

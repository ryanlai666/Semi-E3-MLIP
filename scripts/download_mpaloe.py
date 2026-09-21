"""Download only the pinned MP-ALOE v2 dataset, never pretrained model weights."""
import hashlib
import json
from pathlib import Path
import urllib.request
from semi_mlip.data import write_json

ARTICLE=29452190
FILE_ID=57488704
SIZE=675635633
MD5='0cac41b76cdc936848d360a14bdc9bf2'


def digest(path):
    h=hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    with urllib.request.urlopen(f'https://api.figshare.com/v2/articles/{ARTICLE}',timeout=60) as response:
        metadata=json.load(response)
    if metadata['version']!=2:
        raise ValueError('Latest source version changed; review and explicitly pin it before importing')
    source=next(f for f in metadata['files'] if f['id']==FILE_ID)
    if source['size']!=SIZE or source['computed_md5']!=MD5:
        raise ValueError('Pinned source file metadata changed')
    path=Path('data/raw/MP_ALOE_data.jsonl.gz');path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        partial=path.with_suffix('.gz.part');start=partial.stat().st_size if partial.exists() else 0
        if start<SIZE:
            request=urllib.request.Request(source['download_url'],headers={'Range':f'bytes={start}-'} if start else {})
            with urllib.request.urlopen(request,timeout=120) as response:
                append=start>0 and response.status==206
                if append and not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                    raise ValueError('Unexpected download range')
                with partial.open('ab' if append else 'wb') as out:
                    for block in iter(lambda:response.read(8*1024**2),b''):out.write(block)
        if partial.stat().st_size!=SIZE or digest(partial)!=MD5:
            raise ValueError('Dataset integrity check failed; partial retained')
        partial.replace(path)
    if path.stat().st_size!=SIZE or digest(path)!=MD5:
        raise ValueError('Existing archive differs from pinned MP-ALOE v2; not overwritten')
    write_json('reports/mpaloe_metadata.json',metadata)
    write_json('reports/mpaloe_download.json',{'version':2,'file_id':FILE_ID,'bytes':SIZE,'md5':MD5,
        'verified':True,'source':metadata['doi'],'path':str(path),'pretrained_weights_downloaded':False})
    print(f'Verified MP-ALOE v2: {path} ({SIZE:,} bytes)',flush=True)


if __name__=='__main__':main()

"""Pinned, checksum-verified download; network access only on explicit invocation."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

REVISION = "47d2cc020cf913b5a48a3480136a128dddc0a92c"
FILENAME = "MatPES-R2SCAN-2025.2.jsonl"
SHA256 = "cce36109689d75446b720d21a11faf3ed980df1a32ce6387237629774a770606"
SIZE = 1946162981
BASE = f"https://huggingface.co/datasets/materialyze/matpes/resolve/{REVISION}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def download(directory="data/raw"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / FILENAME
    if destination.exists():
        if sha256(destination) != SHA256:
            raise ValueError("Existing dataset checksum mismatch; refusing to overwrite")
    else:
        partial = destination.with_suffix(".jsonl.part")
        start = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(f"{BASE}/{FILENAME}", headers={"Range": f"bytes={start}-"} if start else {})
        with urllib.request.urlopen(request, timeout=120) as response:
            append = start > 0 and response.status == 206
            if append and not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                raise ValueError("Unexpected download range")
            done = start if append else 0
            last = time.monotonic()
            with partial.open("ab" if append else "wb") as out:
                while chunk := response.read(8 * 1024**2):
                    out.write(chunk)
                    done += len(chunk)
                    if time.monotonic() - last > 10:
                        print(f"Downloaded {done / 1e6:.0f}/{SIZE / 1e6:.0f} MB", flush=True)
                        last = time.monotonic()
        if partial.stat().st_size != SIZE or sha256(partial) != SHA256:
            raise ValueError("Dataset integrity check failed; partial file retained")
        partial.replace(destination)
    with urllib.request.urlopen(f"{BASE}/README.md", timeout=60) as response:
        (directory / "SOURCE_README.md").write_bytes(response.read())
    manifest = {"dataset": "MatPES", "release": "2025.2", "functional": "r2SCAN", "revision": REVISION,
                "url": f"{BASE}/{FILENAME}", "sha256": SHA256, "bytes": SIZE,
                "license_from_dataset_card": "BSD-3-Clause", "verified": True}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(destination, flush=True)
    return destination


if __name__ == "__main__":
    download()

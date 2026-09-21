"""Small, strict extended-XYZ reader for public DFT data (no ASE dependency)."""
import shlex
import numpy as np


def read_extxyz(stream):
    """Yield (metadata, per-atom columns); preserve source fields without inference."""
    while line := stream.readline():
        if not line.strip():
            continue
        n = int(line)
        if n <= 0:
            raise ValueError('Invalid atom count')
        metadata = dict(item.split('=', 1) for item in shlex.split(stream.readline()) if '=' in item)
        schema = metadata.get('Properties', 'species:S:1:pos:R:3').split(':')
        if len(schema) % 3:
            raise ValueError('Invalid Properties schema')
        specs = [(schema[i], schema[i+1], int(schema[i+2])) for i in range(0, len(schema), 3)]
        width = sum(s[2] for s in specs)
        lines = [stream.readline().split() for _ in range(n)]
        if any(len(row) != width for row in lines):
            raise ValueError('Truncated or malformed atom data')
        columns, offset = {}, 0
        for name, kind, count in specs:
            dtype = {'S': str, 'R': float, 'I': int}.get(kind)
            if dtype is None:
                raise ValueError(f'Unsupported field type {kind}')
            values = np.asarray([row[offset:offset+count] for row in lines], dtype=dtype)
            if kind == 'R' and not np.isfinite(values).all():
                raise ValueError('Nonfinite atom data')
            columns[name] = values[:, 0] if count == 1 else values
            offset += count
        yield metadata, columns

"""Periodic radius graphs without external graph or atomistic packages."""
import itertools
import numpy as np
import torch


def neighbor_list(positions, cell, pbc, cutoff):
    """Directed i <- j edges and integer image shifts, with no neighbor cap.

    Reciprocal-plane bounds enumerate every image within the sphere, including
    cells shorter than twice the cutoff and skewed cells. Geometry remains
    differentiable in the model; only discrete connectivity is detached.
    """
    p, c = np.asarray(positions, float), np.asarray(cell, float)
    periodic = np.asarray(pbc, bool)
    if cutoff <= 0 or p.ndim != 2 or p.shape[1] != 3 or not len(p):
        raise ValueError("Invalid graph geometry or cutoff")
    if not np.isfinite(p).all() or not np.isfinite(c).all():
        raise ValueError("Nonfinite graph geometry")
    if abs(np.linalg.det(c)) < 1e-10:
        raise ValueError("A nonsingular cell is required")
    inverse = np.linalg.inv(c)
    frac = p @ inverse
    wrap = np.zeros_like(frac, dtype=np.int64)
    wrap[:, periodic] = np.floor(frac[:, periodic]).astype(np.int64)
    wrapped = p - wrap @ c
    # Wrapped fractional differences are strictly between -1 and 1.
    # Any included image satisfies |shift_k| < 1 + cutoff*|b_k|.
    # floor(R*|b_k|)+1 is a conservative integer bound (including exact ties).
    bounds = np.floor(cutoff * np.linalg.norm(inverse, axis=0)).astype(int) + 1
    ranges = [range(-b, b + 1) if flag else [0] for b, flag in zip(bounds, periodic)]
    ii, jj, shifts = [], [], []
    # Iterate images, keeping only N x N temporaries instead of images x N x N.
    for shift in itertools.product(*ranges):
        d = wrapped[None, :, :] - wrapped[:, None, :] + np.array(shift) @ c
        r2 = np.einsum("ijk,ijk->ij", d, d)
        mask = r2 < cutoff**2
        if shift == (0, 0, 0):
            np.fill_diagonal(mask, False)
        if np.any(mask & (r2 < 1e-16)):
            raise ValueError("Overlapping distinct atoms/images")
        a, b = np.nonzero(mask)
        if len(a):
            ii.append(a)
            jj.append(b)
            shifts.append(np.array(shift)[None, :] + wrap[a] - wrap[b])
    if not ii:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty((0, 3), np.int64)
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(shifts)


def collate(records, graphs, device="cpu", dtype=torch.float32):
    z, p, ids, src, dst, shifts, edge_batch = [], [], [], [], [], [], []
    offset = 0
    for batch, (r, (i, j, s)) in enumerate(zip(records, graphs)):
        n = len(r["z"])
        z.extend(r["z"])
        p.extend(r["positions"])
        ids.extend([batch] * n)
        src.extend(i + offset)
        dst.extend(j + offset)
        shifts.extend(s.tolist())
        edge_batch.extend([batch] * len(i))
        offset += n
    def tensor(x, dt=dtype):
        return torch.as_tensor(x, dtype=dt, device=device)
    auxiliary, auxiliary_mask = [], []
    for r in records:
        for a in range(len(r["z"])):
            targets = [r.get("auxiliary", {}).get(k) for k in ("bader_population", "bader_abs_magmom")]
            auxiliary.append([values[a] if values is not None else 0. for values in targets])
            auxiliary_mask.append([values is not None for values in targets])
    return {"z": tensor(z, torch.long), "positions": tensor(p), "batch": tensor(ids, torch.long),
            "auxiliary": tensor(auxiliary), "auxiliary_mask": tensor(auxiliary_mask, torch.bool),
            "cell": tensor([r["cell"] for r in records]), "n_atoms": tensor([len(r["z"]) for r in records]),
            "i": tensor(src, torch.long), "j": tensor(dst, torch.long),
            "shifts": tensor(shifts).reshape(-1, 3), "edge_batch": tensor(edge_batch, torch.long),
            "energy": tensor([r.get("energy", 0) for r in records]),
            "forces": tensor([v for r in records for v in r.get("forces", [[0., 0., 0.]] * len(r["z"]))]),
            "stress": tensor([r.get("stress") if r.get("stress") is not None else np.zeros((3, 3)).tolist() for r in records]),
            "stress_mask": tensor([r.get("stress") is not None for r in records], torch.bool)}

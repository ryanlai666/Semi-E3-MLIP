"""MatPES ingestion, leakage-resistant splits, and training-only statistics."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np

ELEMENTS = {"O": 8, "Al": 13, "Si": 14, "Ti": 22, "Co": 27, "Cu": 29,
            "Zr": 40, "Ru": 44, "Hf": 72, "Ta": 73, "W": 74}
SYMBOLS = {v: k for k, v in ELEMENTS.items()}
GPA_PER_EV_A3 = 160.21766208


def stress_from_matpes(stress):
    """VASP compressive-positive kbar; Voigt xx,yy,zz,yz,xz,xy."""
    if stress is None:
        return None
    s = np.asarray(stress, dtype=float)
    if s.shape == (6,):
        xx, yy, zz, yz, xz, xy = s
        s = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    if s.shape != (3, 3) or not np.isfinite(s).all():
        raise ValueError("invalid_stress")
    return (-0.1 / GPA_PER_EV_A3 * s).tolist()


def convert(raw):
    elements = set(raw["elements"])
    if not elements or not elements <= ELEMENTS.keys():
        raise ValueError("outside_elements")
    if len(elements) > 1 and "O" not in elements:
        raise ValueError("oxygen_free_alloy")
    if raw.get("functional", "").lower() != "r2scan":
        raise ValueError("wrong_functional")
    structure = raw["structure"]
    if structure.get("charge", 0) not in (None, 0, 0.0):
        raise ValueError("charged_structure")
    z, positions = [], []
    cell = np.asarray(structure["lattice"]["matrix"], dtype=float)
    for site in structure["sites"]:
        species = site["species"]
        if len(species) != 1 or species[0].get("occu", 1) != 1:
            raise ValueError("partial_occupancy")
        z.append(ELEMENTS[species[0]["element"]])
        positions.append(site["xyz"] if "xyz" in site else np.asarray(site["abc"]) @ cell)
    p, f = np.asarray(positions), np.asarray(raw["forces"], dtype=float)
    e = float(raw["energy"])
    if not z or p.shape != (len(z), 3) or f.shape != p.shape or cell.shape != (3, 3):
        raise ValueError("invalid_shape")
    if not all(np.isfinite(x).all() for x in (p, f, cell, e)):
        raise ValueError("nonfinite")
    if abs(np.linalg.det(cell)) < 1e-6:
        raise ValueError("singular_cell")
    parent = raw.get("provenance", {}).get("original_mp_id")
    if not parent:
        raise ValueError("missing_parent")
    auxiliary = {}
    for source_key, target_key in (("bader_charges", "bader_population"), ("bader_magmoms", "bader_abs_magmom")):
        values = raw.get(source_key)
        if values is not None:
            values = np.asarray(values, float)
            if values.shape == (len(z),) and np.isfinite(values).all():
                auxiliary[target_key] = (np.abs(values) if target_key.endswith("magmom") else values).tolist()
    return {"id": raw["matpes_id"], "parent": parent, "source": "MatPES-r2SCAN-2025.2",
            "z": z, "positions": p.tolist(), "cell": cell.tolist(),
            "pbc": structure["lattice"].get("pbc", [True] * 3),
            "energy": e, "forces": f.tolist(), "stress": stress_from_matpes(raw.get("stress")),
            "chemsys": "-".join(sorted(elements)), "formula": raw.get("formula_pretty"),
            "provenance": raw.get("provenance", {}), "auxiliary": auxiliary}


def fingerprint(record):
    """Same-basis rotation/translation/permutation invariant geometry hash.

    Does not identify arbitrary equivalent lattice bases or supercells. Parent
    grouping remains the main protection against related-frame leakage.
    """
    cell = np.array(record["cell"])
    frac = np.array(record["positions"]) @ np.linalg.inv(cell)
    pbc = np.array(record["pbc"], bool)
    candidates = []
    for origin in frac:
        relative = frac - origin
        relative[:, pbc] %= 1
        relative = np.round(relative, 6)
        relative[:, pbc] %= 1
        candidates.append(tuple(sorted((z, *r) for z, r in zip(record["z"], relative))))
    value = (tuple(np.round(cell @ cell.T, 6).flat), tuple(pbc), min(candidates))
    return hashlib.sha256(repr(value).encode()).hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def assign_splits(records, seed=42):
    # Union parent groups when a duplicate spans different original MP IDs.
    roots = {r["parent"]: r["parent"] for r in records}
    def root(x):
        while roots[x] != x:
            roots[x] = roots[roots[x]]
            x = roots[x]
        return x
    seen = {}
    for r in records:
        key = r["fingerprint"]
        if key in seen:
            a, b = root(r["parent"]), root(seen[key])
            roots[max(a, b)] = min(a, b)
        seen[key] = r["parent"]
    for r in records:
        r["group"] = root(r["parent"])
    systems = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        systems[r.get("chemsys", "unknown")][r["group"]].append(r)
    assignments = {}
    for groups in systems.values():
        ordered = sorted(groups, key=lambda group: (-len(groups[group]),
                         hashlib.sha256(f"{seed}:{group}".encode()).digest()))
        total = sum(len(v) for v in groups.values())
        targets = {"train": .8*total, "valid": .1*total, "test": .1*total}
        counts = {key: 0 for key in targets}
        required = ["train", "valid", "test"][:min(3, len(ordered))]
        for index, group in enumerate(ordered):
            missing = [s for s in required if counts[s] == 0]
            if index == 0:
                split = "train"  # Largest trajectory must inform the fitted model.
            elif len(ordered) - index == len(missing):
                split = missing[0]
            else:
                size = len(groups[group])
                split = min(required, key=lambda s: (counts[s]+size-targets[s])**2 - (counts[s]-targets[s])**2)
            if group in assignments and assignments[group] != split:
                # Conservative for unexpected cross-system duplicate groups.
                split = assignments[group]
            assignments[group] = split
            counts[split] += len(groups[group])
    for r in records:
        r["split"] = assignments[r["group"]]
    return records


def prepare(source, output="data/processed", seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, rejected = [], collections.Counter()
    scanned = 0
    with (output / "rejected.jsonl").open("w", encoding="utf-8") as log:
        for raw in read_jsonl(source):
            scanned += 1
            try:
                r = convert(raw)
                r["fingerprint"] = fingerprint(r)
                records.append(r)
            except (ValueError, KeyError, TypeError) as exc:
                reason = str(exc)
                rejected[reason] += 1
                if reason not in ("outside_elements", "oxygen_free_alloy"):
                    log.write(json.dumps({"id": raw.get("matpes_id"), "reason": reason}) + "\n")
            if scanned % 50000 == 0:
                print(f"Scanned {scanned:,}; eligible {len(records):,}", flush=True)
    assign_splits(records, seed)
    unique, duplicates = {}, 0
    for r in sorted(records, key=lambda r: r["id"]):
        if r["fingerprint"] in unique:
            duplicates += 1
        else:
            unique[r["fingerprint"]] = r
    records = list(unique.values())
    coverage = {}
    for r in records:
        row = coverage.setdefault(r["chemsys"], {"frames": 0, "parents": set(), "split": collections.Counter(),
                                               "min_atoms": len(r["z"]), "max_atoms": 0})
        row["frames"] += 1
        row["parents"].add(r["parent"])
        row["split"][r["split"]] += 1
        row["min_atoms"] = min(row["min_atoms"], len(r["z"]))
        row["max_atoms"] = max(row["max_atoms"], len(r["z"]))
    for row in coverage.values():
        row["parents"] = len(row["parents"])
        row["status"] = "sparse" if row["parents"] < 10 else "available_unvalidated"
    for split in ("train", "valid", "test"):
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                if r["split"] == split:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
    report = {"source": str(source), "seed": seed, "scanned": scanned, "retained": len(records),
              "duplicates_removed": duplicates, "rejected": dict(rejected), "coverage": coverage,
              "split_counts": dict(collections.Counter(r["split"] for r in records)),
              "split_method": "Chemical-system stratified, indivisible parent/duplicate groups; target frame fractions 80/10/10. One group: train only; two: train/valid; three or more: all splits.",
              "units": {"energy": "eV total", "forces": "eV/A", "stress": "eV/A^3 tensile-positive"},
              "stress_conversion": "MatPES Voigt xx yy zz yz xz xy; multiply by -0.1/160.21766208",
              "dedup_limit": "Same lattice basis only; equivalent-basis/supercell duplicates not guaranteed"}
    write_json(output / "audit.json", report)
    print(json.dumps({k: report[k] for k in ("scanned", "retained", "split_counts")}), flush=True)
    return report


def training_statistics(records):
    if not records:
        raise ValueError("No training structures")
    zs = sorted({z for r in records for z in r["z"]})
    counts = np.array([[r["z"].count(z) for z in zs] for r in records], dtype=float)
    n = counts.sum(axis=1)
    energies = np.array([r["energy"] for r in records])
    # Per-atom fit prevents large cells from dominating the offsets.
    offsets, _, rank, _ = np.linalg.lstsq(counts / n[:, None], energies / n, rcond=1e-10)
    residual = (energies - counts @ offsets) / n
    force = np.concatenate([np.array(r["forces"]).reshape(-1) for r in records])
    stress = [np.array(r["stress"]).reshape(-1) for r in records if r["stress"] is not None]
    aux_stats = {}
    for key in ("bader_population", "bader_abs_magmom"):
        by_z = collections.defaultdict(list)
        for r in records:
            values = r.get("auxiliary", {}).get(key)
            if values is not None:
                for z, value in zip(r["z"], values):
                    by_z[z].append(value)
        means = {str(z): float(np.mean(values)) for z, values in by_z.items()}
        residuals = [v - means[str(z)] for z, values in by_z.items() for v in values]
        aux_stats[key] = {"element_means": means,
                          "scale": max(float(np.std(residuals)), .1) if residuals else 1.,
                          "labelled_atoms": len(residuals)}
    return {"offsets": {str(z): float(e) for z, e in zip(zs, offsets)}, "composition_rank": int(rank),
            "auxiliary": aux_stats,
            "scales": {"energy": max(float(np.std(residual)), 0.05),
                       "forces": max(float(np.sqrt(np.mean(force**2))), 0.1),
                       "stress": max(float(np.sqrt(np.mean(np.concatenate(stress)**2))), 0.001) if stress else 1.0}}

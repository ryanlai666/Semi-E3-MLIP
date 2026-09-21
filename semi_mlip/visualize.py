"""Scientific plots and GIFs. Matplotlib/Pillow are optional visualization deps."""
import collections
import html
import itertools
import json
from pathlib import Path
import numpy as np
from .data import read_jsonl, SYMBOLS, write_json

COLORS = {8: "#df4a50", 13: "#7893ae", 14: "#d7a349", 22: "#8478b8", 27: "#408caf",
          29: "#c47b42", 40: "#54a49b", 44: "#748083", 72: "#8769a5", 73: "#526faf", 74: "#65727c"}
CASES = [("silicon", "Si", "Si"), ("copper", "Cu", "Cu"), ("silica", "O-Si", "SiO2"),
         ("alumina", "Al-O", "Al2O3"), ("hafnia", "Hf-O", "HfO2"), ("titania", "O-Ti", "TiO2")]


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
                         "figure.facecolor": "#f7f9fc", "axes.facecolor": "#f7f9fc"})
    return plt


def draw_structure(ax, record, forces=None, force_scale=1., title="", azimuth=35., bounds=None):
    cell = np.array(record["cell"])
    p = np.array(record["positions"])
    frac = p @ np.linalg.inv(cell)
    for axis, periodic in enumerate(record.get("pbc", [True]*3)):
        if periodic:
            frac[:, axis] %= 1
    p = frac @ cell
    corners = np.array(list(itertools.product((0,1), repeat=3)))
    vertices = corners @ cell
    for a, b in itertools.combinations(range(8), 2):
        if np.abs(corners[a]-corners[b]).sum() == 1:
            ax.plot(*vertices[[a,b]].T, color="#aab3bf", lw=.8, alpha=.65)
    for z in sorted(set(record["z"])):
        mask = np.array(record["z"]) == z
        ax.scatter(*p[mask].T, s=65 if z == 8 else 110, color=COLORS[z], edgecolor="white", linewidth=.7,
                   label=SYMBOLS[z], depthshade=True)
    if forces is not None:
        arrows = np.array(forces) * force_scale
        ax.quiver(*p.T, *arrows.T, color="#25496e", arrow_length_ratio=.22, linewidth=1.2, normalize=False)
    if bounds is None:
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        center = (lower + upper)/2
        span = max(float((upper-lower).max()), 2.) * .65
    else:
        center, span = bounds
    ax.set(xlim=(center[0]-span,center[0]+span), ylim=(center[1]-span,center[1]+span), zlim=(center[2]-span,center[2]+span))
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=23, azim=azimuth)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, pad=3)


def choose_cases(records):
    selected = {}
    for name, chemistry, formula in CASES:
        rows = [r for r in records if r["chemsys"] == chemistry and r.get("formula") == formula]
        if not rows:
            continue
        groups = collections.defaultdict(list)
        for r in rows:
            groups[(r["parent"], tuple(r["z"]))].append(r)
        group = max(groups.values(), key=lambda v: (len(v), v[0]["parent"]))
        # If only static examples exist, show separate structures, never interpolate.
        selected[name] = sorted(group if len(group)>1 else rows[:6],
                               key=lambda r: (r["parent"], r.get("provenance", {}).get("md_step") or 0, r["id"]))[:12]
    return selected


def compare_cases(checkpoint, data_file="data/processed/test.jsonl", output="reports/visualizations", device="cpu"):
    from .simulate import Calculator
    from matplotlib.animation import FuncAnimation, PillowWriter
    plt = plotting()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    calculator = Calculator(checkpoint, device)
    cases = choose_cases(list(read_jsonl(data_file)))
    summaries, cards = {}, []
    for name, rows in cases.items():
        predicted = [calculator(row) for row in rows]
        errors = [float(np.mean(np.abs(pred["forces"]-np.array(row["forces"])))) for row,pred in zip(rows,predicted)]
        max_force = max(float(np.linalg.norm(f,axis=1).max()) for r,p in zip(rows,predicted) for f in [np.array(r["forces"]),p["forces"]])
        scale = 1.5 / max(max_force, .1)
        fig = plt.figure(figsize=(11,5.6), dpi=100)
        axes = [fig.add_subplot(1,2,i+1,projection="3d") for i in range(2)]
        heading = fig.suptitle("", fontsize=16, fontweight="bold", y=.98)
        footer = fig.text(.5,.055,"", ha="center", fontsize=9, color="#34445a")
        fig.subplots_adjust(left=.015,right=.985,bottom=.15,top=.82,wspace=.01)
        def update(frame):
            index = frame // 4 % len(rows)
            row, pred = rows[index], predicted[index]
            for ax in axes:
                ax.clear()
            azimuth = 32 + (frame % 4)*4
            draw_structure(axes[0], row, row["forces"], scale, "DFT reference forces", azimuth)
            draw_structure(axes[1], row, pred["forces"], scale, "GNN predicted forces", azimuth)
            axes[0].legend(loc="lower left", frameon=False, ncol=3, fontsize=9)
            energy_error = (pred["energy"]-row["energy"])/len(row["z"])
            heading.set_text(f"{row.get('formula', name)}  |  held-out configuration {index+1}/{len(rows)}")
            footer.set_text(f"{row['id']}  |  force MAE {errors[index]:.3f} eV/A  |  energy error {energy_error:+.3f} eV/atom\n"
                            f"Same geometry, camera and arrow scale on both sides. 1 eV/A = {scale:.2f} A arrow.\n"
                            "Sampled structures + camera orbit; NOT a continuous AIMD trajectory.")
        preview = max(range(len(rows)), key=lambda k: float(np.linalg.norm(rows[k]["forces"],axis=1).max()))
        update(preview * 4)
        fig.savefig(output / f"{name}.png", dpi=130)
        animation = FuncAnimation(fig, update, frames=max(4,len(rows)*4), interval=250)
        animation.save(output / f"{name}.gif", writer=PillowWriter(fps=4))
        plt.close(fig)
        summaries[name] = {"ids": [r["id"] for r in rows], "mean_force_mae_eV_A": float(np.mean(errors)),
                           "force_scale_A_per_eV_A": scale,
                           "source": "MatPES splits: " + ",".join(sorted({r.get("split", "unspecified") for r in rows})),
                           "continuous_AIMD": False}
        cards.append(f'<section><h2>{html.escape(name.title())}</h2><img src="{name}.gif" alt="DFT versus predicted forces for {name}"><p>Mean case force MAE: {np.mean(errors):.3f} eV/Angstrom.</p></section>')
    write_json(output / "cases.json", summaries)
    (output / "index.html").write_text('<!doctype html><html><head><meta charset="utf-8"><title>MLIP case comparisons</title>'
        '<style>body{font:16px system-ui;background:#f7f9fc;color:#182b45;max-width:1140px;margin:32px auto;padding:0 20px}'
        'section{background:white;border:1px solid #dce2eb;border-radius:12px;padding:16px;margin:24px 0}img{width:100%;height:auto}p{line-height:1.6}</style></head><body>'
        '<h1>DFT vs GNN: case-by-case force comparisons</h1><p>Identical held-out geometries and shared arrow scales. '
        'Animation cycles through sampled configurations with a camera orbit, not continuous DFT dynamics. '
        'Cases are selected by available reference frames, not prediction accuracy.</p>' + ''.join(cards) + '</body></html>', encoding="utf-8")
    return summaries


def learning_curves(run_dir, output):
    plt = plotting()
    rows = list(read_jsonl(Path(run_dir)/"history.jsonl"))
    fig, axes = plt.subplots(1,3,figsize=(12,3.7),layout="constrained")
    epochs = [r["epoch"] for r in rows]
    axes[0].plot(epochs,[r["train_loss"] for r in rows],color="#25496e")
    axes[0].set(title="Training objective",ylabel="Normalized weighted loss",yscale="log")
    for ax,key,label,baseline in [(axes[1],"energy_mae_eV_atom","Energy MAE (eV/atom)","offset_baseline_mae_eV_atom"),
                                   (axes[2],"force_mae_eV_A","Force MAE (eV/A)","zero_force_baseline_mae_eV_A")]:
        ax.plot(epochs,[r["valid"][key] for r in rows],color="#176c8f",label="Validation")
        ax.axhline(rows[0]["valid"][baseline],ls="--",color="#bd7451",label="Simple baseline")
        ax.set(title=label,ylabel=label)
        ax.legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160)
    plt.close(fig)


def md_comparison(coarse_file, fine_file, output):
    from matplotlib.animation import FuncAnimation, PillowWriter
    plt = plotting()
    coarse, fine = list(read_jsonl(coarse_file)), list(read_jsonl(fine_file))
    fine_by_time = {round(r["time_fs"],7):r for r in fine}
    rows = [(r,fine_by_time[round(r["time_fs"],7)]) for r in coarse if round(r["time_fs"],7) in fine_by_time]
    sample = np.linspace(0,len(rows)-1,min(48,len(rows))).astype(int)
    fig = plt.figure(figsize=(11,5.4),dpi=100)
    axes = [fig.add_subplot(1,2,k+1,projection="3d") for k in range(2)]
    heading = fig.suptitle("Model-only NVE timestep comparison",fontsize=15,fontweight="bold")
    footer = fig.text(.5,.06,"",ha="center",fontsize=10)
    fig.subplots_adjust(left=.02,right=.98,bottom=.13,top=.84)
    dt_coarse,dt_fine = coarse[1]["time_fs"],fine[1]["time_fs"]
    def update(frame):
        pair = rows[sample[frame]]
        for ax,row,dt in zip(axes,pair,(dt_coarse,dt_fine)):
            ax.clear()
            draw_structure(ax,row,title=f"GNN MD: timestep {dt:g} fs")
        footer.set_text(f"Time {pair[0]['time_fs']:.1f} fs | Identical initial positions and velocities\nBoth panels are model predictions, not AIMD reference trajectories.")
    update(0)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix(".png"),dpi=130)
    FuncAnimation(fig,update,frames=len(sample),interval=100).save(output,writer=PillowWriter(fps=10))
    plt.close(fig)

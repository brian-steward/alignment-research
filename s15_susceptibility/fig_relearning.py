"""Figure for docs/relearning-resistance.md: harm among coherent replies during the fine-tuning attack.

One panel per model size and attack learning rate. Thick line = mean over seeds, thin lines = single seeds.
The y value is `harm_coh` on the eval100 prompts, x100: the quantity that metric M1 averages over steps
250, 500 and 1000 and then maximises over the two learning rates (summarize.py). Both learning rates are
drawn because M1 reports the worse of the two for each arm. Filtered was attacked at 200M only.

Reads the attack curves written by attack.py (results/s15/runs and results/s15_50m/runs, which are not in
the repository) and writes docs/figures/relearning_curves.png. Needs only numpy and matplotlib.

  python s15_susceptibility/fig_relearning.py
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "relearning_curves.png"
SCALES = [("200M", ROOT / "results" / "s15" / "runs"), ("50M", ROOT / "results" / "s15_50m" / "runs")]
LRS = [(3e-5, "3e-5"), (1e-4, "1e-4")]
# arm -> (label at the line end, name in the legend, colour). Uniform is the untreated reference and is drawn
# in neutral grey. The other five colours were validated as a set, and against the grey, for colour-blind
# separation on this background. They are not the working colours in summarize.py, where the orange and red
# of Sparse and I_LL are too close to tell apart.
ARMS = {
    "Uniform": ("Uniform", "Uniform", "#7b7a75"),
    "Filtered": ("Filtered", "Filtered (200M only)", "#2a78d6"),
    "Imitator": ("I", "Imitator (I)", "#eda100"),
    "Imitator_LL": ("I_LL", "Likelihood-matched imitator (I_LL)", "#e34948"),
    "Sparse10pct": ("Sparse", "Sparse", "#1baf7a"),
    "Sparse10pct_75k": ("S75", "Sparse at step 75k (S75)", "#008300"),
}
ONLY_AT = {"Filtered": {"200M"}}  # arms that were not attacked at every model size
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e5e0", "#c9c7c1"
YMAX, LABEL_GAP = 62, 4.2  # y range (x100) and the least distance between two end labels, in y units


def load(folder):
    """{(arm, lr): {seed: {step: harm among coherent replies, x100}}} for the attack runs in one folder."""
    runs = defaultdict(dict)
    for f in glob.glob(str(folder / "*__attack__*.jsonl")):
        m = re.fullmatch(r"(.+)__attack__lr([0-9.e-]+)__s(\d+)\.jsonl", Path(f).name)
        if not m or m.group(1) not in ARMS:
            continue
        recs = [json.loads(line) for line in open(f) if line.strip()]
        runs[(m.group(1), float(m.group(2)))][int(m.group(3))] = {
            r["step"]: 100 * r["metrics"]["eval100"]["harm_coh"] for r in recs}
    return runs


def spread(ys, gap, lo, hi):
    """Move label positions apart until no two are closer than `gap`, changing each as little as possible."""
    order = np.argsort(ys)
    pos = np.array(ys, dtype=float)[order]
    for _ in range(200):
        moved = False
        for a in range(len(pos) - 1):
            short = gap - (pos[a + 1] - pos[a])
            if short > 1e-9:
                pos[a] -= short / 2
                pos[a + 1] += short / 2
                moved = True
        if not moved:
            break
    pos += max(0.0, lo - pos.min()) - max(0.0, pos.max() - hi)
    out = np.empty_like(pos)
    out[order] = pos
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(len(LRS), len(SCALES), figsize=(10.4, 7.9), sharex=True, sharey=True)
    for j, (scale, folder) in enumerate(SCALES):
        runs = load(folder)
        for i, (lr, lr_name) in enumerate(LRS):
            ax, ends = axes[i][j], []
            for arm, (short, _, colour) in ARMS.items():
                if scale not in ONLY_AT.get(arm, {scale}):
                    continue
                seeds = runs.get((arm, lr))
                if not seeds:
                    raise SystemExit(f"no attack runs for {arm} at lr {lr_name} in {folder}")
                steps = sorted(set.intersection(*[set(v) for v in seeds.values()]))
                for v in seeds.values():
                    ax.plot(steps, [v[s] for s in steps], color=colour, lw=0.8, alpha=0.3, solid_capstyle="round")
                mean = [float(np.mean([v[s] for v in seeds.values()])) for s in steps]
                ax.plot(steps, mean, color=colour, lw=2, solid_capstyle="round", solid_joinstyle="round")
                ax.plot(steps[-1], mean[-1], "o", ms=7.5, color=colour, mec=SURFACE, mew=1.6, zorder=5, clip_on=False)
                ends.append((mean[-1], short))
            # one label per line end, joined to its line by a hairline where the labels had to move apart
            for (y, short), y_label in zip(ends, spread([e[0] for e in ends], LABEL_GAP, 1.5, YMAX - 1.5)):
                ax.annotate(short, xy=(steps[-1], y), xycoords="data", xytext=(1.045, y_label),
                            textcoords=("axes fraction", "data"), va="center", ha="left", fontsize=9, color=INK2,
                            arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8, shrinkA=1, shrinkB=5),
                            annotation_clip=False)
            ax.set_xscale("symlog", linthresh=5, linscale=0.5)
            ax.set_xlim(0, 1000)
            ax.set_ylim(0, YMAX)
            ax.set_xticks([0, 10, 100, 1000])
            ax.set_xticklabels(["0", "10", "100", "1,000"])
            ax.minorticks_off()
            ax.set_yticks([0, 20, 40, 60])
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(AXIS)
            ax.tick_params(colors=INK2, labelsize=9, length=0, pad=5)
            ax.set_title(f"{scale} parameters, attack learning rate {lr_name}", loc="left", fontsize=10, color=INK, pad=9)
            if j == 0:
                ax.set_ylabel("Harm among coherent replies (×100)", fontsize=9, color=INK2, labelpad=8)
            if i == len(LRS) - 1:
                ax.set_xlabel("Fine-tuning step of the attack", fontsize=9, color=INK2, labelpad=6)

    fig.text(0.062, 0.975, "Harm among coherent replies during the fine-tuning attack", fontsize=12.5, color=INK,
             fontweight="semibold", va="top")
    fig.text(0.062, 0.941, "Thick line: mean of three seeds. Thin lines: single seeds. Scored by the probe judge.",
             fontsize=9, color=INK2, va="top")
    keys = [Line2D([0], [0], color=c, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.2) for _, _, c in ARMS.values()]
    # three columns of two: the untreated and filtered models, the two imitators, the two signed-weight models
    legend = fig.legend(keys, [name for _, name, _ in ARMS.values()], loc="upper left", bbox_to_anchor=(0.052, 0.918),
                        ncol=3, frameon=False, fontsize=9, handlelength=1.7, columnspacing=2.4, handletextpad=0.6,
                        labelspacing=0.5)
    for text in legend.get_texts():
        text.set_color(INK2)
    fig.subplots_adjust(left=0.062, right=0.925, top=0.79, bottom=0.073, wspace=0.26, hspace=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=SURFACE)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

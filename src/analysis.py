"""Post-training analysis and comparison plots for the signed-weight experiment."""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size": 12})


def load_training_log(run_dir: Path) -> list[dict]:
    """Load training log entries from a run directory."""
    log_file = run_dir / "training_log.jsonl"
    if not log_file.exists():
        return []
    entries = []
    with open(log_file) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def load_final_metrics(run_dir: Path) -> dict | None:
    """Load final metrics from a run directory."""
    path = run_dir / "final_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect_runs(models_dir: Path) -> dict[str, list[dict]]:
    """Collect results from all runs, grouped by scheme.

    Returns:
        {"baseline": [metrics_seed42, metrics_seed137, ...], "filtered": [...], ...}
    """
    runs = defaultdict(list)
    for run_dir in sorted(models_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics = load_final_metrics(run_dir)
        if metrics is None:
            continue

        name = run_dir.name
        # Parse scheme from directory name (e.g., "baseline_seed42" -> "baseline")
        scheme = name.rsplit("_seed", 1)[0]
        runs[scheme].append(metrics)

    return dict(runs)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(models_dir: Path, output_path: Path):
    """Overlay training loss curves for all runs."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = {"baseline": "C0", "filtered": "C1", "negative_a0.5": "C2",
              "negative_a0.1": "C3", "negative_a0.3": "C4",
              "negative_a0.7": "C5", "negative_a1.0": "C6"}

    plotted_labels = set()
    for run_dir in sorted(models_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        log = load_training_log(run_dir)
        train_entries = [e for e in log if "type" not in e]
        if not train_entries:
            continue

        name = run_dir.name
        scheme = name.rsplit("_seed", 1)[0]
        color = colors.get(scheme, "gray")
        label = scheme if scheme not in plotted_labels else None
        plotted_labels.add(scheme)

        steps = [e["step"] for e in train_entries]
        losses = [e["loss_total"] for e in train_entries]
        ax.plot(steps, losses, color=color, alpha=0.4, linewidth=0.8, label=label)

    ax.set_xlabel("Step")
    ax.set_ylabel("Weighted Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to {output_path}")


# ---------------------------------------------------------------------------
# Perplexity comparison
# ---------------------------------------------------------------------------

def plot_perplexity_comparison(runs: dict[str, list[dict]], output_path: Path):
    """Bar chart of safe/unsafe perplexity and gap for each scheme."""
    schemes = list(runs.keys())
    n = len(schemes)

    safe_ppls = []
    unsafe_ppls = []
    gaps = []
    safe_errs = []
    unsafe_errs = []

    for scheme in schemes:
        metrics_list = runs[scheme]
        safe = [m["eval"]["perplexity"]["ppl_safe"] for m in metrics_list
                if "eval" in m and "perplexity" in m["eval"]]
        unsafe = [m["eval"]["perplexity"]["ppl_unsafe"] for m in metrics_list
                  if "eval" in m and "perplexity" in m["eval"]]
        gap = [m["eval"]["perplexity"]["ppl_gap"] for m in metrics_list
               if "eval" in m and "perplexity" in m["eval"]]

        safe_ppls.append(np.mean(safe) if safe else 0)
        unsafe_ppls.append(np.mean(unsafe) if unsafe else 0)
        gaps.append(np.mean(gap) if gap else 0)
        safe_errs.append(np.std(safe) / max(np.sqrt(len(safe)), 1) if len(safe) > 1 else 0)
        unsafe_errs.append(np.std(unsafe) / max(np.sqrt(len(unsafe)), 1) if len(unsafe) > 1 else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(n)
    w = 0.35

    ax1.bar(x - w/2, safe_ppls, w, yerr=safe_errs, label="Safe PPL", color="C0", capsize=4)
    ax1.bar(x + w/2, unsafe_ppls, w, yerr=unsafe_errs, label="Unsafe PPL", color="C3", capsize=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(schemes, rotation=30, ha="right")
    ax1.set_ylabel("Perplexity")
    ax1.set_title("Perplexity by Scheme")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, gaps, color="C2")
    ax2.set_xticks(x)
    ax2.set_xticklabels(schemes, rotation=30, ha="right")
    ax2.set_ylabel("PPL Gap (unsafe / safe)")
    ax2.set_title("Perplexity Gap (higher = more aligned)")
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved perplexity comparison to {output_path}")


# ---------------------------------------------------------------------------
# Toxicity comparison
# ---------------------------------------------------------------------------

def plot_toxicity_comparison(runs: dict[str, list[dict]], output_path: Path):
    """Bar chart of toxic completion rate for each scheme."""
    schemes = list(runs.keys())
    n = len(schemes)

    rates = []
    errs = []
    for scheme in schemes:
        metrics_list = runs[scheme]
        tox = [m["eval"]["toxicity"]["toxic_rate"] for m in metrics_list
               if "eval" in m and "toxicity" in m["eval"]]
        rates.append(np.mean(tox) if tox else 0)
        errs.append(np.std(tox) / max(np.sqrt(len(tox)), 1) if len(tox) > 1 else 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(n)
    ax.bar(x, rates, yerr=errs, color="C3", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(schemes, rotation=30, ha="right")
    ax.set_ylabel("Toxic Completion Rate")
    ax.set_title("Toxic Completion Rate by Scheme (lower = more aligned)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved toxicity comparison to {output_path}")


# ---------------------------------------------------------------------------
# Alpha ablation
# ---------------------------------------------------------------------------

def plot_alpha_ablation(runs: dict[str, list[dict]], output_path: Path):
    """Plot alignment metrics vs. negative weight magnitude."""
    # Find all negative_a* schemes
    alpha_schemes = {k: v for k, v in runs.items() if k.startswith("negative_a")}
    if not alpha_schemes:
        print("No alpha ablation runs found, skipping plot.")
        return

    alphas = []
    ppl_gaps = []
    tox_rates = []
    safe_ppls = []

    for scheme in sorted(alpha_schemes.keys()):
        alpha = float(scheme.split("_a")[1])
        metrics_list = alpha_schemes[scheme]

        gap = [m["eval"]["perplexity"]["ppl_gap"] for m in metrics_list
               if "eval" in m and "perplexity" in m["eval"]]
        tox = [m["eval"]["toxicity"]["toxic_rate"] for m in metrics_list
               if "eval" in m and "toxicity" in m["eval"]]
        sppl = [m["eval"]["perplexity"]["ppl_safe"] for m in metrics_list
                if "eval" in m and "perplexity" in m["eval"]]

        alphas.append(alpha)
        ppl_gaps.append(np.mean(gap) if gap else 0)
        tox_rates.append(np.mean(tox) if tox else 0)
        safe_ppls.append(np.mean(sppl) if sppl else 0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    ax1.plot(alphas, ppl_gaps, "o-", color="C2")
    ax1.set_xlabel("Negative Weight (alpha)")
    ax1.set_ylabel("PPL Gap (unsafe / safe)")
    ax1.set_title("Alignment vs. Alpha")
    ax1.grid(True, alpha=0.3)

    ax2.plot(alphas, tox_rates, "o-", color="C3")
    ax2.set_xlabel("Negative Weight (alpha)")
    ax2.set_ylabel("Toxic Completion Rate")
    ax2.set_title("Toxicity vs. Alpha")
    ax2.grid(True, alpha=0.3)

    ax3.plot(alphas, safe_ppls, "o-", color="C0")
    ax3.set_xlabel("Negative Weight (alpha)")
    ax3.set_ylabel("Safe PPL")
    ax3.set_title("Capability (safe PPL) vs. Alpha")
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Alpha Ablation: Alignment vs. Capability Tradeoff", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved alpha ablation to {output_path}")


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def run_analysis(models_dir: Path, results_dir: Path):
    """Generate all comparison plots from completed experiment runs."""
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting experiment results...")
    runs = collect_runs(models_dir)
    print(f"Found schemes: {list(runs.keys())}")
    for scheme, metrics_list in runs.items():
        print(f"  {scheme}: {len(metrics_list)} runs")

    if not runs:
        print("No completed runs found.")
        return

    plot_training_curves(models_dir, results_dir / "training_curves.png")
    plot_perplexity_comparison(runs, results_dir / "perplexity_comparison.png")
    plot_toxicity_comparison(runs, results_dir / "toxicity_comparison.png")
    plot_alpha_ablation(runs, results_dir / "alpha_ablation.png")

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Scheme':<20} {'Safe PPL':>10} {'Unsafe PPL':>12} {'PPL Gap':>10} {'Tox Rate':>10}")
    print("-" * 70)
    for scheme, metrics_list in runs.items():
        safe = np.mean([m["eval"]["perplexity"]["ppl_safe"] for m in metrics_list
                        if "eval" in m and "perplexity" in m["eval"]] or [0])
        unsafe = np.mean([m["eval"]["perplexity"]["ppl_unsafe"] for m in metrics_list
                          if "eval" in m and "perplexity" in m["eval"]] or [0])
        gap = np.mean([m["eval"]["perplexity"]["ppl_gap"] for m in metrics_list
                       if "eval" in m and "perplexity" in m["eval"]] or [0])
        tox = np.mean([m["eval"]["toxicity"]["toxic_rate"] for m in metrics_list
                       if "eval" in m and "toxicity" in m["eval"]] or [0])
        print(f"{scheme:<20} {safe:>10.2f} {unsafe:>12.2f} {gap:>10.3f} {tox:>10.1%}")
    print("=" * 70)


if __name__ == "__main__":
    from .config import PROJECT_ROOT
    run_analysis(
        PROJECT_ROOT / "models" / "signed_weight_exp",
        PROJECT_ROOT / "results" / "signed_weight_exp",
    )

from __future__ import annotations

from typing import Dict, Tuple, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_discovery(curves: Dict[str, Tuple[Sequence, Sequence]], out_path: str) -> None:
    """curves: label -> (n, V). Plots cumulative distinct vs attempts (log-log)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, (n, V) in curves.items():
        ax.plot(n, V, marker=".", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("valid solutions drawn (n)")
    ax.set_ylabel("distinct solutions (V)")
    ax.set_title("Discovery curve (Heaps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_scaling(series: Dict[str, Tuple[Sequence, Sequence]], out_path: str,
                 xlabel: str = "difficulty", ylabel: str = "beta") -> None:
    """series: label -> (x, y). Plots a metric vs difficulty per model."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, (x, y) in series.items():
        ax.plot(x, y, marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {xlabel}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_transition(tiers: Sequence[str], distinct_counts: Sequence[int], out_path: str) -> None:
    """Saturating -> non-saturating: distinct solution count vs operator tier."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(tiers), list(distinct_counts))
    ax.set_xlabel("operator tier (richer ->)")
    ax.set_ylabel("distinct ground-truth solutions")
    ax.set_title("Saturating -> non-saturating transition")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

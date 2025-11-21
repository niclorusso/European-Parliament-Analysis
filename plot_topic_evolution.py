"""Generate analysis plots for topic networks across EP6-EP10."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS_CSV = Path("results/resin/topics/topic_evolution/topic_network_metrics.csv")
OUTPUT_DIR = Path("results/resin/topics/topic_evolution")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EP_RANGE = list(range(6, 11))


def normalize_topic_filter(topic_filters: Iterable[str]) -> set[str]:
    cleaned = set()
    for item in topic_filters:
        slug = item.strip().lower().replace(" ", "_")
        if slug:
            cleaned.add(slug)
    return cleaned


def plot_topic(df: pd.DataFrame, topic_slug: str) -> Path:
    topic_df = df[df["topic_slug"] == topic_slug].sort_values("ep")
    if topic_df.empty:
        raise ValueError(f"Topic '{topic_slug}' not found in metrics CSV.")

    label = topic_df["topic_label"].iloc[0]
    x = topic_df["ep"].to_numpy()
    topic_df = topic_df.copy()
    topic_df["avg_degree_norm"] = topic_df["avg_degree"] / topic_df["node_count"].replace(0, pd.NA)

    # Count how many modularity columns we have
    mod_cols = [c for c in topic_df.columns if c.startswith("q") and c != "qmax_std"]
    has_comprehensive_mod = "qmax" in topic_df.columns and not topic_df["qmax"].isna().all()
    
    # Determine number of subplots
    n_basic = 4  # avg_degree_norm, density, louvain_modularity, n_communities
    n_modularity = 0
    if has_comprehensive_mod:
        # Count available modularity columns
        if "qmax" in topic_df.columns:
            n_modularity += 1
        if "qparty" in topic_df.columns:
            n_modularity += 1
        if "qcountry" in topic_df.columns:
            n_modularity += 1
        if "q_left_right" in topic_df.columns:
            n_modularity += 1
        if "q_extreme_centrist" in topic_df.columns:
            n_modularity += 1
    n_node_count = 1  # Always show node count
    n_total = n_basic + n_modularity + n_node_count
    
    fig, axes = plt.subplots(
        n_total,
        1,
        figsize=(10, 2.5 * n_total),
        sharex=True,
        gridspec_kw={"hspace": 0.3},
    )
    
    # Ensure axes is always a list/array
    if n_total == 1:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = np.array(axes).flatten()
    
    idx = 0

    # Basic metrics
    axes[idx].plot(x, topic_df["avg_degree_norm"], marker="o", color="#1f77b4", linewidth=1.5, markersize=6)
    axes[idx].set_ylabel("Avg degree / nodes", fontsize=10)
    axes[idx].set_title(f"EP6-EP10 Evolution — {label}", fontsize=12, fontweight="bold")
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    axes[idx].plot(x, topic_df["density"], marker="o", color="#ff7f0e", linewidth=1.5, markersize=6)
    axes[idx].set_ylabel("Density", fontsize=10)
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    axes[idx].plot(x, topic_df["louvain_modularity"], marker="o", color="#2ca02c", linewidth=1.5, markersize=6)
    axes[idx].set_ylabel("Louvain Q", fontsize=10)
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    axes[idx].plot(x, topic_df["n_communities"], marker="o", color="#d62728", linewidth=1.5, markersize=6)
    axes[idx].set_ylabel("# communities", fontsize=10)
    axes[idx].grid(True, alpha=0.3)
    idx += 1

    # Comprehensive modularity measures (if available)
    if has_comprehensive_mod:
        # Qmax
        if "qmax" in topic_df.columns:
            qmax_vals = topic_df["qmax"]
            qmax_std = topic_df.get("qmax_std", pd.Series([0] * len(qmax_vals)))
            axes[idx].plot(x, qmax_vals, marker="o", color="#9467bd", linewidth=1.5, markersize=6, label="Qmax")
            if not qmax_std.isna().all():
                axes[idx].fill_between(x, qmax_vals - qmax_std, qmax_vals + qmax_std, alpha=0.2, color="#9467bd")
            axes[idx].set_ylabel("Qmax (mean±std)", fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend(fontsize=8)
            idx += 1
        
        # Qparty
        if "qparty" in topic_df.columns:
            axes[idx].plot(x, topic_df["qparty"], marker="o", color="#8c564b", linewidth=1.5, markersize=6)
            axes[idx].set_ylabel("Qparty", fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            idx += 1
        
        # Qcountry
        if "qcountry" in topic_df.columns:
            axes[idx].plot(x, topic_df["qcountry"], marker="o", color="#e377c2", linewidth=1.5, markersize=6)
            axes[idx].set_ylabel("Qcountry", fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            idx += 1
        
        # Q_left_right
        if "q_left_right" in topic_df.columns:
            axes[idx].plot(x, topic_df["q_left_right"], marker="o", color="#7f7f7f", linewidth=1.5, markersize=6)
            axes[idx].set_ylabel("Q Left-Right", fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            idx += 1
        
        # Q_extreme_centrist
        if "q_extreme_centrist" in topic_df.columns:
            axes[idx].plot(x, topic_df["q_extreme_centrist"], marker="o", color="#bcbd22", linewidth=1.5, markersize=6)
            axes[idx].set_ylabel("Q Extreme-Centrist", fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            idx += 1

    # Node count (always last)
    axes[idx].bar(x, topic_df["node_count"], color="#4e79a7", alpha=0.7)
    axes[idx].set_ylabel("Node count", fontsize=10)
    axes[idx].set_xlabel("EP legislature (6-10)", fontsize=11)
    axes[idx].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Topic Network Evolution Analysis", fontsize=14, fontweight="bold", y=0.995)

    # Save to topic-specific folder
    topic_dir = OUTPUT_DIR / topic_slug
    topic_dir.mkdir(parents=True, exist_ok=True)
    out_path = topic_dir / "metrics_plot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(topics: Iterable[str] | None):
    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {METRICS_CSV}")

    df = pd.read_csv(METRICS_CSV)

    if topics:
        allowed = normalize_topic_filter(topics)
    else:
        allowed = set(sorted(df["topic_slug"].unique()))

    created = []
    for topic_slug in allowed:
        if topic_slug not in df["topic_slug"].values:
            print(f"⚠️  Topic '{topic_slug}' missing from metrics, skipping.")
            continue
        out = plot_topic(df, topic_slug)
        created.append(out)
        print(f"Saved {out}")

    if not created:
        print("No plots generated — check topic names or metrics file.")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot topic metrics across EP legislatures.")
    parser.add_argument(
        "--topic",
        "-t",
        action="append",
        dest="topics",
        help="Topic slug or label to plot (repeatable). Defaults to all topics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.topics)

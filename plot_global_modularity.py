#!/usr/bin/env python3
"""Plot global network modularity evolution across EPs."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

INPUT_CSV = Path("results/modularity/global_modularity_results.csv")
OUTPUT_DIR = Path("results/modularity")
EP_RANGE = range(6, 11)

def plot_global_modularity():
    """Plot global network modularity evolution."""
    if not INPUT_CSV.exists():
        print(f"⚠️  Global modularity CSV not found: {INPUT_CSV}")
        print("   Run: python analyze_global_modularity.py first")
        return
    
    df = pd.read_csv(INPUT_CSV)
    
    # Check available metrics
    mod_cols = [
        "qmax",
        "qparty",
        "qcountry",
        "q_left_right",
        "q_left_center_right",
        "q_extreme_centrist",
        "q_majority_opposition",
    ]
    available_cols = [c for c in mod_cols if c in df.columns and df[c].notna().any()]
    
    if not available_cols:
        print("⚠️  No modularity data found in CSV.")
        return
    
    # Sort by EP
    df = df.sort_values("ep")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "qmax": "#fee090",
        "q_extreme_centrist": "#fc8d59",
        "q_majority_opposition": "#8c564b",
        "qparty": "#d73027",
        "q_left_right": "#91bfdb",
        "q_left_center_right": "#a55194",
        "qcountry": "#4575b4",
    }
    
    labels = {
        "qmax": "Qmax",
        "q_extreme_centrist": "Extreme–Centrist",
        "q_majority_opposition": "Majority-Opposition",
        "qparty": "Party",
        "q_left_right": "Left–Right",
        "q_left_center_right": "Left–Center–Right",
        "qcountry": "Country",
    }
    
    x = df["ep"].to_numpy()
    
    # Check for subsample std (preferred) or regular std
    if "qmax_subsample_std" in df.columns:
        qmax_std_series = df["qmax_subsample_std"]
    elif "qmax_std" in df.columns:
        qmax_std_series = df["qmax_std"]
    else:
        qmax_std_series = None
    
    # Plot each measure
    for measure in available_cols:
        y = df[measure].to_numpy()
        valid = np.isfinite(y)
        if valid.sum() < 2:
            continue
        y_valid = y[valid]
        ax.plot(x[valid], y_valid, marker="o", linewidth=2.5, markersize=10,
               color=colors[measure], label=labels[measure], alpha=0.8)
        
        # Add confidence interval for Qmax
        if measure == "qmax" and qmax_std_series is not None:
            std_vals = qmax_std_series.to_numpy()[valid]
            if len(std_vals) == len(y_valid):
                ax.fill_between(
                    x[valid],
                    y_valid - std_vals,
                    y_valid + std_vals,
                    color=colors[measure],
                    alpha=0.15,
                    linewidth=0,
                )
    
    # Create tick labels with vote counts and flags
    tick_labels = []
    for ep in x:
        label = f"{int(ep)}"
        row = df[df["ep"] == ep].iloc[0]
        
        votes_val = row.get("votes_used")
        flags = ""
        if row.get("bootstrapped", False):
            flags += "B"
        if row.get("downsampled", False):
            flags += "D"
        
        if pd.notna(votes_val) and not (isinstance(votes_val, float) and np.isnan(votes_val)):
            votes_label = f"{int(votes_val)}"
            if flags:
                votes_label += f"({flags})"
            label += f"\n{votes_label}"
        elif flags:
            label += f"\n({flags})"
        tick_labels.append(label)
    
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    
    ax.set_xlabel("EP Legislature (6-10)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Modularity Q", fontsize=12, fontweight="bold")
    ax.set_title("Global Network Modularity Evolution: EP6-EP10\n(All Votes Combined)", 
                 fontsize=14, fontweight="bold")
    ax.legend(frameon=False, loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "modularity_aggregated_evolution.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved plot: {out_path}")


if __name__ == "__main__":
    plot_global_modularity()


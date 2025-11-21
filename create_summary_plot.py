#!/usr/bin/env python3
"""Create a summary plot with key metrics for discussion."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

MODULARITY_DIR = Path("results/modularity")
METRICS_CSV = Path("results/resin/topics/topic_evolution/topic_network_metrics.csv")
EP_RANGE = range(6, 11)


def create_summary_plot():
    """Create a multi-panel summary plot with key findings."""
    
    # Load global modularity data (computed from global networks, not averaged)
    print("Computing global network modularity...")
    try:
        from plot_modularity_evolution import compute_global_modularity_per_ep
        global_df = compute_global_modularity_per_ep()
    except Exception as e:
        print(f"⚠️  Could not compute global modularity: {e}")
        global_df = None
    
    # Load topic-level data for other panels
    if not METRICS_CSV.exists():
        print(f"⚠️  Metrics file not found: {METRICS_CSV}")
        df = None
    else:
        df = pd.read_csv(METRICS_CSV)
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Colors
    colors = {
        "qmax": "#fee090",
        "q_majority_opposition": "#8c564b",
        "q_left_center_right": "#a55194",
        "q_left_right": "#91bfdb",
        "q_extreme_centrist": "#fc8d59",
        "qparty": "#d73027",
        "qcountry": "#4575b4",
    }
    
    labels = {
        "qmax": "Qmax",
        "q_majority_opposition": "Majority-Opposition",
        "q_left_center_right": "Left–Center–Right",
        "q_left_right": "Left–Right",
        "q_extreme_centrist": "Extreme–Centrist",
        "qparty": "Party",
        "qcountry": "Country",
    }
    
    # ===== PANEL 1: Modularity Evolution Over Time (Global Networks) =====
    ax1 = fig.add_subplot(gs[0, :])
    
    if global_df is not None and not global_df.empty:
        # Use global network modularity - include all key metrics
        key_metrics = ["qmax", "q_majority_opposition", "q_left_center_right", "q_left_right", 
                      "q_extreme_centrist", "qparty", "qcountry"]
        available_key = [m for m in key_metrics if m in global_df.columns and global_df[m].notna().any()]
        
        if available_key:
            global_df_sorted = global_df.sort_values("ep")
            x = global_df_sorted["ep"].to_numpy()
            
            for measure in available_key:
                y = global_df_sorted[measure].to_numpy()
                valid = np.isfinite(y)
                if valid.sum() >= 2:
                    ax1.plot(x[valid], y[valid], marker="o", linewidth=2.5, markersize=10,
                            color=colors[measure], label=labels[measure], alpha=0.8)
            
            # Add Qmax confidence interval if available
            if "qmax" in available_key and "qmax_std" in global_df_sorted.columns:
                qmax_idx = available_key.index("qmax")
                y_qmax = global_df_sorted["qmax"].to_numpy()
                std_qmax = global_df_sorted["qmax_std"].to_numpy()
                valid_qmax = np.isfinite(y_qmax) & np.isfinite(std_qmax)
                if valid_qmax.sum() >= 2:
                    ax1.fill_between(
                        x[valid_qmax],
                        y_qmax[valid_qmax] - std_qmax[valid_qmax],
                        y_qmax[valid_qmax] + std_qmax[valid_qmax],
                        color=colors["qmax"],
                        alpha=0.15,
                        linewidth=0,
                    )
            
            ax1.set_xlabel("EP Legislature", fontsize=12, fontweight="bold")
            ax1.set_ylabel("Modularity Q", fontsize=12, fontweight="bold")
            ax1.set_title("Modularity Evolution: Global Networks (All Votes Combined)", 
                         fontsize=13, fontweight="bold")
            ax1.legend(frameon=False, loc="best", fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(EP_RANGE)
            ax1.set_xticklabels([f"EP{ep}" for ep in EP_RANGE])
    elif df is not None:
        # Fallback: use topic-level data (averaged)
        mod_cols = ["qmax", "qparty", "q_majority_opposition", "q_left_center_right", 
                    "q_left_right", "q_extreme_centrist", "qcountry"]
        available_cols = [c for c in mod_cols if c in df.columns and df[c].notna().any()]
        
        if available_cols:
            key_metrics = ["qmax", "q_majority_opposition", "q_left_center_right", "q_left_right",
                          "q_extreme_centrist", "qparty", "qcountry"]
            available_key = [m for m in key_metrics if m in available_cols]
            
            if available_key:
                aggregated = df.groupby("ep")[available_key].mean()
                x = aggregated.index.to_numpy()
                
                for measure in available_key:
                    y = aggregated[measure].to_numpy()
                    valid = np.isfinite(y)
                    if valid.sum() >= 2:
                        ax1.plot(x[valid], y[valid], marker="o", linewidth=2.5, markersize=10,
                                color=colors[measure], label=labels[measure], alpha=0.8)
                
                ax1.set_xlabel("EP Legislature", fontsize=12, fontweight="bold")
                ax1.set_ylabel("Modularity Q", fontsize=12, fontweight="bold")
                ax1.set_title("Modularity Evolution: Key Metrics (Averaged Across Topics - Fallback)", 
                             fontsize=13, fontweight="bold")
                ax1.legend(frameon=False, loc="best", fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.set_xticks(EP_RANGE)
                ax1.set_xticklabels([f"EP{ep}" for ep in EP_RANGE])
    
    # ===== PANEL 2: Cleavage Importance (Ratios to Qmax) =====
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Use global network data if available, otherwise topic-level
    data_for_ratios = global_df if global_df is not None and not global_df.empty else df
    
    if data_for_ratios is not None:
        # Compute ratios
        ratio_cols = []
        for col in ["qparty", "q_majority_opposition", "q_left_center_right", 
                    "q_left_right", "q_extreme_centrist", "qcountry"]:
            if col in data_for_ratios.columns and f"{col}_qmax_ratio" in data_for_ratios.columns:
                ratio_cols.append(f"{col}_qmax_ratio")
            elif col in data_for_ratios.columns and "qmax" in data_for_ratios.columns:
                # Compute ratio on the fly
                data_for_ratios[f"{col}_qmax_ratio"] = data_for_ratios[col] / data_for_ratios["qmax"]
                ratio_cols.append(f"{col}_qmax_ratio")
        
        if ratio_cols:
            # Average ratios (across EPs if global, across topics+EPs if topic-level)
            avg_ratios = data_for_ratios[ratio_cols].mean()
        avg_ratios = avg_ratios.sort_values(ascending=False)
        
        # Map back to original names
        ratio_labels = {
            "qparty_qmax_ratio": "Party",
            "q_majority_opposition_qmax_ratio": "Majority-Opposition",
            "q_left_center_right_qmax_ratio": "Left–Center–Right",
            "q_left_right_qmax_ratio": "Left–Right",
            "q_extreme_centrist_qmax_ratio": "Extreme–Centrist",
            "qcountry_qmax_ratio": "Country",
        }
        
        bars = ax2.barh(range(len(avg_ratios)), avg_ratios.values, 
                       color=[colors.get(col.replace("_qmax_ratio", ""), "#999999") 
                             for col in avg_ratios.index], alpha=0.8)
        ax2.set_yticks(range(len(avg_ratios)))
        ax2.set_yticklabels([ratio_labels.get(col, col.replace("_qmax_ratio", "")) 
                            for col in avg_ratios.index])
        ax2.set_xlabel("Average Ratio to Qmax", fontsize=11, fontweight="bold")
        ax2.set_title("Cleavage Importance\n(How well each partition explains modularity)", 
                     fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="x")
        ax2.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Qmax = 1.0")
        ax2.legend(frameon=False, fontsize=9)
    
    # ===== PANEL 3: Top Topics by Modularity (EP10) =====
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Get EP10 data from topic-level metrics
    if df is not None:
        ep10_data = df[df["ep"] == 10].copy()
        if not ep10_data.empty and "qmax" in ep10_data.columns:
            ep10_data = ep10_data[ep10_data["qmax"].notna()].copy()
            ep10_data = ep10_data.sort_values("qmax", ascending=False).head(10)
            
            topics = ep10_data["topic_label"].tolist()
            # Truncate long topic names
            topics_short = [t[:40] + "..." if len(t) > 40 else t for t in topics]
            qmax_values = ep10_data["qmax"].tolist()
            
            bars = ax3.barh(range(len(qmax_values)), qmax_values, 
                           color=colors["qmax"], alpha=0.8)
            ax3.set_yticks(range(len(qmax_values)))
            ax3.set_yticklabels(topics_short, fontsize=9)
            ax3.set_xlabel("Qmax (Maximum Modularity)", fontsize=11, fontweight="bold")
            ax3.set_title("Top 10 Topics by Modularity (EP10)", 
                         fontsize=12, fontweight="bold")
            ax3.grid(True, alpha=0.3, axis="x")
        else:
            ax3.text(0.5, 0.5, "No EP10 topic data available", 
                    ha="center", va="center", transform=ax3.transAxes, fontsize=12)
            ax3.set_title("Top 10 Topics by Modularity (EP10)", 
                         fontsize=12, fontweight="bold")
    else:
        ax3.text(0.5, 0.5, "No topic data available", 
                ha="center", va="center", transform=ax3.transAxes, fontsize=12)
        ax3.set_title("Top 10 Topics by Modularity (EP10)", 
                     fontsize=12, fontweight="bold")
    
    # Add overall title
    fig.suptitle("European Parliament Voting Networks: Key Findings", 
                fontsize=16, fontweight="bold", y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save
    out_path = MODULARITY_DIR / "summary_key_findings.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved summary plot: {out_path}")


if __name__ == "__main__":
    create_summary_plot()


#!/usr/bin/env python3
"""Regenerate modularity plots with vote counts in brackets, using existing CSV files."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

MODULARITY_OUTPUT_DIR = Path("results/modularity")
METRICS_CSV = Path("results/resin/topics/topic_evolution/topic_network_metrics.csv")
EP_RANGE = range(6, 11)


def load_vote_counts():
    """Load vote counts from comprehensive metrics file."""
    vote_counts = {}
    if METRICS_CSV.exists():
        try:
            metrics_df = pd.read_csv(METRICS_CSV)
            if "votes_used" in metrics_df.columns and "topic_slug" in metrics_df.columns and "ep" in metrics_df.columns:
                for _, row in metrics_df.iterrows():
                    if pd.notna(row.get("votes_used")):
                        key = (str(row["topic_slug"]), int(row["ep"]))
                        vote_counts[key] = int(row["votes_used"])
                print(f"Loaded vote counts for {len(vote_counts)} topic-EP combinations")
        except Exception as e:
            print(f"⚠️  Could not load vote counts: {e}")
    else:
        print(f"⚠️  Metrics file not found: {METRICS_CSV}")
    return vote_counts


def regenerate_plots():
    """Regenerate modularity plots with vote counts."""
    vote_counts = load_vote_counts()
    
    # Colors matching modularity.py
    colors = {
        "qmax": "#fee090",
        "q_extreme_centrist": "#fc8d59",
        "qparty": "#d73027",
        "q_left_right": "#91bfdb",
        "qcountry": "#4575b4",
        "q_majority_opposition": "#e0f3f8",
        "q_left_center_right": "#74add1",
    }
    
    label_map = {
        "qmax": "Qmax",
        "q_extreme_centrist": "Extreme–Centrist",
        "qparty": "Party",
        "q_left_right": "Left–Right",
        "qcountry": "Country",
        "q_majority_opposition": "Majority vs Opposition",
        "q_left_center_right": "Left–Center–Right",
    }
    
    for ep in EP_RANGE:
        csv_path = MODULARITY_OUTPUT_DIR / f"modularity_results_EP{ep}.csv"
        if not csv_path.exists():
            print(f"⚠️  EP{ep}: CSV file not found, skipping")
            continue
        
        print(f"\nProcessing EP{ep}...")
        df = pd.read_csv(csv_path)
        
        if df.empty:
            print(f"  ⚠️  EP{ep}: Empty dataframe, skipping")
            continue
        
        # Filter out ratio columns for plotting
        mod_cols = [c for c in df.columns if c.startswith("q") and c not in (
            "qmax_std", "qparty_qmax_ratio", "qcountry_qmax_ratio", 
            "q_left_right_qmax_ratio", "q_extreme_centrist_qmax_ratio",
            "q_majority_opposition_qmax_ratio", "q_left_center_right_qmax_ratio"
        )]
        
        # Filter to rows with modularity data
        has_mod = df[mod_cols].notna().any(axis=1)
        ep_df = df[has_mod].copy()
        
        if ep_df.empty:
            print(f"  ⚠️  EP{ep}: No modularity data, skipping")
            continue
        
        # Sort by Qmax (descending)
        if "qmax" in ep_df.columns:
            ep_df = ep_df.sort_values("qmax", ascending=False, na_position='last')
        
        # Prepare topic labels with vote counts
        topics = []
        for _, row in ep_df.iterrows():
            topic_label = row["topic_label"]
            topic_slug = row.get("topic_slug", "")
            # Try to get vote count
            vote_count = vote_counts.get((str(topic_slug), ep), None)
            if vote_count is not None:
                topics.append(f"{topic_label} ({vote_count})")
            else:
                topics.append(topic_label)
        
        n_topics = len(topics)
        if n_topics == 0:
            continue
        
        # Create bar plot with more space between topics
        fig, ax = plt.subplots(figsize=(max(14, n_topics * 0.8), 6))
        # Add spacing between topic groups
        spacing = 1.2  # Space multiplier between topics
        x = np.arange(n_topics) * spacing
        width = 0.15
        
        # Collect which measures we have (in desired order)
        available_measures = []
        measure_order = ["qmax", "q_majority_opposition", "q_left_center_right", "q_left_right", 
                        "q_extreme_centrist", "qparty", "qcountry"]
        for measure in measure_order:
            if measure in ep_df.columns and ep_df[measure].notna().any():
                available_measures.append(measure)
        
        n_measures = len(available_measures)
        if n_measures == 0:
            print(f"  ⚠️  EP{ep}: No available measures, skipping")
            plt.close(fig)
            continue
        
        # Center bars around x
        total_width = n_measures * width
        start_offset = -total_width / 2 + width / 2
        
        # Plot each modularity measure
        for idx, measure in enumerate(available_measures):
            offset = start_offset + idx * width
            values = ep_df[measure].fillna(0).tolist()
            ax.bar(x + offset, values, width, color=colors.get(measure, "#999999"), 
                   label=label_map.get(measure, measure), alpha=0.9)
        
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(topics, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Modularity Q", fontsize=12, fontweight="bold")
        ax.set_title(f"Modularity values by topic: EP{ep}", fontsize=14, fontweight="bold")
        ax.legend(frameon=False, loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = MODULARITY_OUTPUT_DIR / f"modularity_by_topic_EP{ep}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ Saved: {plot_path}")


if __name__ == "__main__":
    regenerate_plots()
    print("\n✅ All plots regenerated!")


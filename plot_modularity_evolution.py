"""Plot modularity evolution per topic across EP6-EP10."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS_CSV = Path("results/resin/topics/topic_evolution/topic_network_metrics.csv")
OUTPUT_DIR = Path("results/modularity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EP_RANGE = list(range(6, 11))


def normalize_topic_filter(topic_filters: Iterable[str]) -> set[str]:
    cleaned = set()
    for item in topic_filters:
        slug = item.strip().lower().replace(" ", "_")
        if slug:
            cleaned.add(slug)
    return cleaned


def plot_modularity_evolution_all_topics(df: pd.DataFrame):
    """Create a comprehensive plot showing all modularity measures evolving over time for all topics."""
    mod_cols = [
        "qmax",
        "qparty",
        "qcountry",
        "q_left_right",
        "q_left_center_right",
        "q_extreme_centrist",
        "q_majority_opposition",
    ]
    available_cols = [c for c in mod_cols if c in df.columns]
    
    if not available_cols:
        print("⚠️  No modularity data found.")
        return
    
    # Filter to topics with data across multiple EPs
    topics_with_data = []
    for topic_slug, group in df.groupby("topic_slug"):
        if group["ep"].nunique() >= 2 and group[available_cols].notna().any(axis=1).any():
            topics_with_data.append(topic_slug)
    
    if not topics_with_data:
        print("⚠️  No topics with sufficient data.")
        return
    
    # Create subplots: one per modularity measure
    n_measures = len(available_cols)
    fig, axes = plt.subplots(
        n_measures, 1,
        figsize=(14, 4 * n_measures),
        sharex=True,
        gridspec_kw={"hspace": 0.3}
    )
    
    if n_measures == 1:
        axes = [axes]
    
    colors_map = {
        "qmax": "#fee090",
        "q_extreme_centrist": "#fc8d59",
        "q_majority_opposition": "#8c564b",
        "qparty": "#d73027",
        "q_left_right": "#91bfdb",
        "q_left_center_right": "#a55194",
        "qcountry": "#4575b4",
    }
    
    labels_map = {
        "qmax": "Qmax (Maximum Modularity)",
        "q_extreme_centrist": "Extreme–Centrist Modularity",
        "q_majority_opposition": "Majority vs Opposition",
        "qparty": "Party Modularity",
        "q_left_right": "Left–Right Modularity",
        "q_left_center_right": "Left–Center–Right Modularity",
        "qcountry": "Country Modularity",
    }
    
    # Plot each measure
    for idx, measure in enumerate(available_cols):
        ax = axes[idx]
        
        # Plot each topic
        for topic_slug in topics_with_data:
            topic_df = df[df["topic_slug"] == topic_slug].sort_values("ep")
            if topic_df[measure].notna().sum() < 2:
                continue
            
            x = topic_df["ep"].to_numpy()
            y = topic_df[measure].to_numpy()
            
            # Only plot if we have at least 2 data points
            valid = np.isfinite(y)
            if valid.sum() < 2:
                continue
            
            label = topic_df["topic_label"].iloc[0]
            ax.plot(x[valid], y[valid], marker="o", linewidth=1.5, markersize=5, 
                   label=label, alpha=0.7)
        
        ax.set_ylabel(labels_map[measure], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if idx == 0:  # Only show legend on first subplot
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=2, frameon=False)
    
    xtick_vals = sorted(df["ep"].unique())
    tick_labels = [str(int(v)) for v in xtick_vals]
    for ax in axes:
        ax.set_xticks(xtick_vals)
        ax.set_xticklabels(tick_labels)
    axes[-1].set_xlabel("EP Legislature (6-10)", fontsize=12, fontweight="bold")
    fig.suptitle("Modularity Evolution Across Topics (EP6-EP10)", 
                 fontsize=14, fontweight="bold", y=0.995)
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.995])
    
    out_path = OUTPUT_DIR / "modularity_evolution_all_topics.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out_path}")


def plot_modularity_evolution_per_topic(df: pd.DataFrame, topic_slug: str):
    """Plot all modularity measures for a single topic over time."""
    topic_df = df[df["topic_slug"] == topic_slug].sort_values("ep")
    if topic_df.empty:
        raise ValueError(f"Topic '{topic_slug}' not found.")
    
    mod_cols = [
        "qmax",
        "qparty",
        "qcountry",
        "q_left_right",
        "q_left_center_right",
        "q_extreme_centrist",
        "q_majority_opposition",
    ]
    available_cols = [c for c in mod_cols if c in topic_df.columns and topic_df[c].notna().any()]
    
    if not available_cols:
        print(f"⚠️  No modularity data for topic '{topic_slug}'.")
        return None
    
    label = topic_df["topic_label"].iloc[0]
    x = topic_df["ep"].to_numpy()
    
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
        "q_majority_opposition": "Majority vs Opposition",
        "qparty": "Party",
        "q_left_right": "Left–Right",
        "q_left_center_right": "Left–Center–Right",
        "qcountry": "Country",
    }
    
    # Plot each measure
    for measure in available_cols:
        y = topic_df[measure].to_numpy()
        valid = np.isfinite(y)
        if valid.sum() < 2:
            continue
        
        ax.plot(x[valid], y[valid], marker="o", linewidth=2, markersize=8,
               color=colors[measure], label=labels[measure], alpha=0.8)
    
    eps_unique = sorted(topic_df["ep"].unique())
    votes_series = topic_df["votes_used"] if "votes_used" in topic_df else None
    down_series = topic_df["downsampled"] if "downsampled" in topic_df else None
    boot_series = topic_df["bootstrap_iterations"] if "bootstrap_iterations" in topic_df else None
    tick_labels = []
    for ep_val in eps_unique:
        label = f"{int(ep_val)}"
        mask = topic_df["ep"] == ep_val
        votes_val = votes_series[mask].iloc[0] if votes_series is not None and mask.any() else None
        flags = ""
        if boot_series is not None and mask.any():
            boot_val = boot_series[mask].iloc[0]
            if pd.notna(boot_val) and boot_val > 0:
                flags += "B"
        if down_series is not None and mask.any():
            down_val = down_series[mask].iloc[0]
            if isinstance(down_val, str):
                down_flag = down_val.lower() == "true"
            else:
                down_flag = bool(down_val)
            if down_flag:
                flags += "D"
        if votes_val is not None and not (isinstance(votes_val, float) and np.isnan(votes_val)):
            votes_label = f"{int(votes_val)}"
            if flags:
                votes_label += f"({flags})"
            label += f"\n{votes_label}"
        elif flags:
            label += f"\n({flags})"
        tick_labels.append(label)
    ax.set_xticks(eps_unique)
    ax.set_xticklabels(tick_labels)
    
    ax.set_xlabel("EP Legislature (6-10)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Modularity Q", fontsize=12, fontweight="bold")
    ax.set_title(f"Modularity Evolution: {label}", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save directly in evolution folder
    evolution_dir = OUTPUT_DIR / "evolution"
    evolution_dir.mkdir(parents=True, exist_ok=True)
    out_path = evolution_dir / f"{topic_slug}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def compute_global_modularity_per_ep(ep_list=None):
    """Compute modularity from the entire (global) MEP network for each EP.
    
    Args:
        ep_list: Optional list of EP numbers to process. If None, processes all EPs in EP_RANGE.
    """
    from analyze_topic_evolution import (
        SCHEMA,
        HAS_MODULARITY_MODULE,
        compute_comprehensive_modularity_with_subsampling,
        similarity_matrix,
        graph_from_similarity,
        extract_metadata_from_mep_graph,
        MAX_NODES,
        RNG,
    )
    from pathlib import Path
    import pandas as pd
    import networkx as nx
    
    if not HAS_MODULARITY_MODULE:
        print("⚠️  modularity.py not available.")
        return None
    
    results = []
    
    # Use provided EP list or default to EP_RANGE
    eps_to_process = ep_list if ep_list is not None else EP_RANGE
    
    for ep in eps_to_process:
        print(f"  Computing global modularity for EP{ep}...")
        vote_matrix_path = Path(f"data/all_votes_main_EP{ep}.csv")
        votes_used = None
        if vote_matrix_path.exists():
            try:
                header = pd.read_csv(vote_matrix_path, nrows=0)
                vote_cols_header = [
                    c for c in header.columns if c.isdigit() or c.replace(".0", "").isdigit()
                ]
                votes_used = len(vote_cols_header)
            except Exception:
                votes_used = None
        
        # Load global MEP network if it exists
        matches = list(Path("results/resin_mep").glob(f"EP{ep}_global_*.gexf"))
        loaded = False
        if matches:
            gexf_path = sorted(matches)[-1]
            if gexf_path.stat().st_size == 0:
                print(f"    ⚠️  Global GEXF file is empty, building from votes...")
            else:
                try:
                    G_global = nx.read_gexf(gexf_path)
                    if G_global.number_of_nodes() > 0:
                        print(
                            f"    Loaded global network: {G_global.number_of_nodes()} nodes, {G_global.number_of_edges()} edges"
                        )
                        df_meta = extract_metadata_from_mep_graph(G_global)
                        downsampled_flag = G_global.number_of_nodes() > MAX_NODES
                        if downsampled_flag:
                            print(f"    Network has {G_global.number_of_nodes()} nodes, will subsample to {MAX_NODES} nodes")
                        
                        mod_metrics = compute_comprehensive_modularity_with_subsampling(
                            G_global, df_meta, ep, needs_subsampling=downsampled_flag
                        )
                        mod_metrics["ep"] = ep
                        mod_metrics["votes_used"] = votes_used
                        mod_metrics["downsampled"] = downsampled_flag
                        mod_metrics["bootstrapped"] = False
                        results.append(mod_metrics)
                        print(f"    Qmax = {mod_metrics.get('qmax', 'N/A')}")
                        loaded = True
                    else:
                        print(f"    ⚠️  Global network is empty, building from votes...")
                except Exception as e:
                    print(f"    ⚠️  Error loading GEXF: {e}, building from votes...")
        if loaded:
            continue
        
        if not vote_matrix_path.exists():
            print(f"    ⚠️  Vote matrix not found, skipping EP{ep}")
            continue
        
        try:
            df_votes = pd.read_csv(vote_matrix_path, low_memory=False)
            vote_cols = [c for c in df_votes.columns if c.isdigit() or c.replace(".0", "").isdigit()]
            if len(vote_cols) < 10:
                print(f"    ⚠️  Too few votes, skipping EP{ep}")
                continue
            votes_used = len(vote_cols)
            
            df_vote_subset = df_votes[vote_cols].copy()
            valid_mask = df_vote_subset.isna().sum(axis=1) < 0.7 * len(vote_cols)
            df_vote_subset = df_vote_subset.loc[valid_mask]
            df_meta = df_votes.loc[valid_mask].copy()
            if len(df_vote_subset) < 10:
                continue
            
            print(f"    Building network from {len(vote_cols)} votes, {len(df_vote_subset)} MEPs...")
            A = similarity_matrix(df_vote_subset)
            schema = SCHEMA[ep]
            mep_ids = df_meta[schema["member_id"]].astype(str).tolist()
            G_global = graph_from_similarity(A, mep_ids)
            
            isolated = list(nx.isolates(G_global))
            if isolated:
                G_global.remove_nodes_from(isolated)
                df_meta = df_meta[~df_meta[schema["member_id"]].astype(str).isin(isolated)].copy()
            
            if G_global.number_of_nodes() < 10:
                continue
            
            downsampled_flag = G_global.number_of_nodes() > MAX_NODES
            if downsampled_flag:
                print(f"    Network has {G_global.number_of_nodes()} nodes, will subsample to {MAX_NODES} nodes")
            
            mod_metrics = compute_comprehensive_modularity_with_subsampling(
                G_global, df_meta, ep, needs_subsampling=downsampled_flag
            )
            mod_metrics["ep"] = ep
            mod_metrics["votes_used"] = votes_used
            mod_metrics["downsampled"] = downsampled_flag
            mod_metrics["bootstrapped"] = False
            results.append(mod_metrics)
            print(f"    Qmax = {mod_metrics.get('qmax', 'N/A')}")
            print(f"    Qparty = {mod_metrics.get('qparty', 'N/A')}")
            print(f"    Qcountry = {mod_metrics.get('qcountry', 'N/A')}")
            print(f"    Q_left_right = {mod_metrics.get('q_left_right', 'N/A')}")
            print(f"    Q_extreme_centrist = {mod_metrics.get('q_extreme_centrist', 'N/A')}")
            print(f"    Qmajority_opposition = {mod_metrics.get('q_majority_opposition', 'N/A')}")
        except Exception as e:
            print(f"    ⚠️  Error building global network: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        return pd.DataFrame(results)
    return None


def plot_modularity_aggregated(df: pd.DataFrame):
    """Plot aggregated modularity measures from the global (entire) graph for each EP."""
    print("  Computing modularity from global networks (all topics combined)...")
    global_df = compute_global_modularity_per_ep()
    
    votes_info = None
    downsample_info = None
    boot_info = None
    
    if global_df is None or global_df.empty:
        print("  ⚠️  Could not compute global modularity, falling back to averaging topic-level values...")
        # Fallback to averaging
        mod_cols = [
            "qmax",
            "qparty",
            "qcountry",
            "q_left_right",
            "q_left_center_right",
            "q_extreme_centrist",
            "q_majority_opposition",
        ]
        available_cols = [c for c in mod_cols if c in df.columns]
        if not available_cols:
            print("⚠️  No modularity data found.")
            return
        aggregated = df.groupby("ep")[available_cols].mean()
        qmax_std_series = None
    else:
        # Use global modularity
        mod_cols = [
            "qmax",
            "qparty",
            "qcountry",
            "q_left_right",
            "q_left_center_right",
            "q_extreme_centrist",
            "q_majority_opposition",
        ]
        available_cols = [c for c in mod_cols if c in global_df.columns]
        if not available_cols:
            print("⚠️  No modularity data in global networks.")
            return
        aggregated = global_df.set_index("ep")[available_cols]
        # Prefer subsample_std if available (from multiple subsamples), otherwise use qmax_std
        if "qmax_subsample_std" in global_df.columns:
            qmax_std_series = global_df.set_index("ep")["qmax_subsample_std"]
        elif "qmax_std" in global_df.columns:
            qmax_std_series = global_df.set_index("ep")["qmax_std"]
        else:
            qmax_std_series = None
        aux_global = global_df.set_index("ep")
        votes_info = aux_global.get("votes_used")
        downsample_info = aux_global.get("downsampled")
        boot_info = aux_global.get("bootstrapped")
    
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
    
    x = aggregated.index.to_numpy()
    qmax_std_array = (
        qmax_std_series.reindex(aggregated.index).to_numpy()
        if qmax_std_series is not None
        else None
    )
    
    # Plot each measure
    for measure in available_cols:
        y = aggregated[measure].to_numpy()
        valid = np.isfinite(y)
        if valid.sum() < 2:
            continue
        y_valid = y[valid]
        ax.plot(x[valid], y_valid, marker="o", linewidth=2.5, markersize=10,
               color=colors[measure], label=labels[measure], alpha=0.8)
        
        if measure == "qmax" and qmax_std_array is not None:
            std_vals = qmax_std_array[valid]
            if len(std_vals) == len(y_valid):
                ax.fill_between(
                    x[valid],
                    y_valid - std_vals,
                    y_valid + std_vals,
                    color=colors[measure],
                    alpha=0.15,
                    linewidth=0,
                )
    
    tick_labels = []
    for ep in x:
        label = f"{int(ep)}"
        votes_val = votes_info.loc[ep] if votes_info is not None and ep in votes_info.index else None
        flags = ""
        if boot_info is not None and ep in boot_info.index:
            boot_val = boot_info.loc[ep]
            if pd.notna(boot_val) and bool(boot_val):
                flags += "B"
        if downsample_info is not None and ep in downsample_info.index:
            down_val = downsample_info.loc[ep]
            if isinstance(down_val, str):
                down_flag = down_val.lower() == "true"
            else:
                down_flag = bool(down_val)
            if down_flag:
                flags += "D"
        if votes_val is not None and not (isinstance(votes_val, float) and np.isnan(votes_val)):
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
    ax.set_ylabel("Modularity", fontsize=12, fontweight="bold")
    ax.set_title("Modularity Evolution: EP6-EP10", 
                 fontsize=14, fontweight="bold")
    ax.legend(frameon=False, loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "modularity_aggregated_evolution.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out_path}")


def plot_modularity_comparison_heatmap(df: pd.DataFrame):
    """Create a heatmap showing modularity values across topics and EPs."""
    mod_cols = [
        "qmax",
        "qparty",
        "qcountry",
        "q_left_right",
        "q_left_center_right",
        "q_extreme_centrist",
        "q_majority_opposition",
    ]
    available_cols = [c for c in mod_cols if c in df.columns]
    
    if not available_cols:
        print("⚠️  No modularity data found.")
        return
    
    # Focus on Qmax for the heatmap (most important)
    if "qmax" not in available_cols:
        measure = available_cols[0]
    else:
        measure = "qmax"
    
    # Pivot: topics as rows, EPs as columns
    pivot_df = df.pivot_table(
        index="topic_label",
        columns="ep",
        values=measure,
        aggfunc="mean"
    )
    
    # Sort by average Qmax across EPs
    pivot_df["avg"] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values("avg", ascending=False)
    pivot_df = pivot_df.drop("avg", axis=1)
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot_df) * 0.4)))
    
    im = ax.imshow(pivot_df.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    
    # Get actual EP columns that have data
    ep_columns = pivot_df.columns.tolist()
    
    # Set ticks
    ax.set_xticks(np.arange(len(ep_columns)))
    ax.set_xticklabels([f"EP{int(ep)}" for ep in ep_columns])
    ax.set_yticks(np.arange(len(pivot_df)))
    ax.set_yticklabels(pivot_df.index, fontsize=9)
    
    # Add text annotations
    for i in range(len(pivot_df)):
        for j in range(len(ep_columns)):
            val = pivot_df.iloc[i, j]
            if pd.notna(val):
                text = ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                             color="black" if val < pivot_df.values.max() * 0.6 else "white",
                             fontsize=8)
    
    ax.set_title(f"{measure.upper()} Evolution Heatmap (EP6-EP10)", 
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("EP Legislature", fontsize=12, fontweight="bold")
    ax.set_ylabel("Topic", fontsize=12, fontweight="bold")
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Modularity Q", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / f"modularity_heatmap_{measure}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out_path}")


def main(topics: Iterable[str] | None):
    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {METRICS_CSV}")
    
    df = pd.read_csv(METRICS_CSV)
    
    print("Generating modularity evolution plots...")
    
    # 1. Comprehensive plot: all topics, all measures
    print("  → Creating comprehensive evolution plot (all topics)...")
    plot_modularity_evolution_all_topics(df)
    
    # 2. Aggregated plot (all topics averaged)
    print("  → Creating aggregated evolution plot (all topics averaged)...")
    plot_modularity_aggregated(df)
    
    # 3. Heatmap
    print("  → Creating heatmap...")
    plot_modularity_comparison_heatmap(df)
    
    # 4. Per-topic plots (if specific topics requested)
    if topics:
        allowed = normalize_topic_filter(topics)
        for topic_slug in allowed:
            if topic_slug not in df["topic_slug"].values:
                print(f"  ⚠️  Topic '{topic_slug}' missing, skipping.")
                continue
            print(f"  → Creating plot for '{topic_slug}'...")
            plot_modularity_evolution_per_topic(df, topic_slug)
    else:
        # Create plots for all topics with sufficient data
        topics_with_data = []
        for topic_slug, group in df.groupby("topic_slug"):
            if group["ep"].nunique() >= 3:  # At least 3 EPs
                topics_with_data.append(topic_slug)
        
        print(f"  → Creating per-topic plots for {len(topics_with_data)} topics...")
        for topic_slug in topics_with_data:
            plot_modularity_evolution_per_topic(df, topic_slug)
    
    print("\n✅ All plots generated!")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot modularity evolution across EP legislatures.")
    parser.add_argument(
        "--topic",
        "-t",
        action="append",
        dest="topics",
        help="Topic slug to plot (repeatable). Defaults to all topics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.topics)


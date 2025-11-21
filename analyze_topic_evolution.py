"""Analyze topic network evolution across EP6-EP10 using existing GEXF files."""
from __future__ import annotations

import os
import re
import math
from collections import defaultdict
import random
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import networkx as nx
import numpy as np
import pandas as pd

try:
    import community as community_louvain  # python-louvain
    HAS_LOUVAIN = True
except ImportError:  # pragma: no cover
    HAS_LOUVAIN = False

# Import modularity functions from modularity.py
try:
    from modularity import (
        compute_qmax,
        compute_partition_modularity,
        create_left_right_partition,
        create_left_center_right_partition,
        create_extreme_centrist_partition,
        create_coalition_partition,
        similarity_matrix,
        graph_from_similarity,
    )
    HAS_MODULARITY_MODULE = True
except ImportError:
    HAS_MODULARITY_MODULE = False
    print("⚠️  modularity.py not found. Will compute basic modularity only.")

BASE_DIR = Path("results/resin/topics")
MEP_BASE_DIR = Path("results/resin_mep/topics")  # Existing MEP networks
OUTPUT_DIR = BASE_DIR / "topic_evolution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODULARITY_OUTPUT_DIR = Path("results/modularity")  # Modularity plots and CSVs
MODULARITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_NODES = 200
BOOTSTRAP_THRESHOLD = 30
BOOTSTRAP_ITERATIONS = 100
BOOTSTRAP_MIN_UNIQUE_NODES = 10
QMAX_RUNS = 20  # Number of runs for Qmax computation (like modularity.ipynb)
SUBSAMPLE_ITERATIONS = 10  # Number of subsamples for large networks
RNG = random.Random(42)

EP_RANGE = range(6, 11)
EP_ORDER = {ep: idx for idx, ep in enumerate(EP_RANGE)}

EP_PATTERN = re.compile(r"ep(\d+)_", re.IGNORECASE)
DATE_PATTERN = re.compile(r"\d{8}")

# Schema for loading MEP metadata (matching modularity.py)
# Note: EP6 CSV uses "Vote ID" not "euro_act_id" in the harmonized files
SCHEMA = {
    6: {"vote_id": "Vote ID", "policy": "main_policy_name", "member_id": "member.id", "country": "member.country.label"},
    7: {"vote_id": "Vote ID", "policy": "De", "member_id": "member.id", "country": "member.country.label"},
    8: {"vote_id": "Vote ID", "policy": "De", "member_id": "member.id", "country": "member.country.label"},
    9: {"vote_id": "id", "policy": "committees", "member_id": "member.id", "country": "member.country.code"},
    10: {"vote_id": "id", "policy": "committees", "member_id": "member.id", "country": "member.country.code"},
}


def extract_metadata(path: Path) -> Tuple[int | None, str]:
    """Return (ep_number, topic_slug) parsed from the filename."""
    name = path.stem.lower()
    ep_match = EP_PATTERN.search(name)
    ep_number = int(ep_match.group(1)) if ep_match else None
    topic_part = name
    if ep_match:
        topic_part = name[ep_match.end():]
    parts = [p for p in topic_part.split("_") if p]
    filtered = [p for p in parts if not DATE_PATTERN.fullmatch(p)]
    topic_slug = "_".join(filtered) if filtered else topic_part
    return ep_number, topic_slug


def maybe_subsample_graph(G: nx.Graph, topic_slug: str, ep: int) -> nx.Graph:
    """Return G or a subgraph limited to MAX_NODES."""
    n = G.number_of_nodes()
    if n <= MAX_NODES:
        return G
    nodes = list(G.nodes())
    sampled_nodes = RNG.sample(nodes, MAX_NODES)
    subgraph = G.subgraph(sampled_nodes).copy()
    print(f"  ↳ Subsampled EP{ep} {topic_slug}: {n}→{MAX_NODES} nodes")
    return subgraph


def bootstrap_metrics(G: nx.Graph, topic_slug: str, ep: int) -> Dict[str, float]:
    """Bootstrap metrics for small graphs to estimate stability."""
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}

    stats: Dict[str, List[float]] = defaultdict(list)
    iterations = 0

    for _ in range(BOOTSTRAP_ITERATIONS):
        sample_nodes = [RNG.choice(nodes) for _ in range(n)]
        unique_nodes = list(dict.fromkeys(sample_nodes))
        if len(unique_nodes) < BOOTSTRAP_MIN_UNIQUE_NODES:
            continue
        G_sample = G.subgraph(unique_nodes).copy()
        metrics = compute_graph_metrics(G_sample)
        for key, value in metrics.items():
            stats[key].append(value)
        iterations += 1

    result: Dict[str, float] = {}
    if iterations == 0:
        print(f"  ↳ Bootstrapping skipped (insufficient unique nodes) EP{ep} {topic_slug}")
        return result

    print(f"  ↳ Bootstrapping EP{ep} {topic_slug}: {iterations} iterations")
    for key, values in stats.items():
        arr = np.array(values, dtype=float)
        result[f"{key}_bootstrap_mean"] = float(np.nanmean(arr))
        result[f"{key}_bootstrap_std"] = float(np.nanstd(arr))
    result["bootstrap_iterations"] = iterations
    return result


def find_mep_network_path(ep: int, topic_slug: str) -> Path | None:
    """Find existing MEP network GEXF file for the topic."""
    ep_dir = MEP_BASE_DIR / f"EP{ep}"
    if not ep_dir.exists():
        return None
    
    # Try different filename patterns
    # Pattern 1: EP{n}_{topic_slug}_*.gexf
    pattern1 = f"EP{ep}_{topic_slug}_*.gexf"
    matches = list(ep_dir.glob(pattern1))
    if matches:
        return sorted(matches)[-1]  # Get most recent
    
    # Pattern 2: EP{n}_{topic_slug_short}_*.gexf (e.g., "agriculture" instead of "agriculture_and_rural_development")
    topic_short = topic_slug.split("_")[0]  # First word
    pattern2 = f"EP{ep}_{topic_short}_*.gexf"
    matches = list(ep_dir.glob(pattern2))
    if matches:
        return sorted(matches)[-1]
    
    # Pattern 3: Try matching with variations (underscores, spaces, etc.)
    topic_variations = [
        topic_slug.replace("_", "___"),  # EP6 uses triple underscores
        topic_slug.replace("_", "__"),
        topic_slug.replace("_", " ").title().replace(" ", "_"),
    ]
    for variation in topic_variations:
        pattern = f"EP{ep}_{variation}_*.gexf"
        matches = list(ep_dir.glob(pattern))
        if matches:
            return sorted(matches)[-1]
    
    return None


def extract_metadata_from_mep_graph(G: nx.Graph) -> pd.DataFrame:
    """Extract MEP metadata from GEXF node attributes."""
    records = []
    for node_id, attrs in G.nodes(data=True):
        record = {"member.id": str(node_id)}
        if "party" in attrs:
            record["member.group.short_label"] = attrs["party"]
        if "country" in attrs:
            record["member.country.label"] = attrs["country"]
        if "country_code" in attrs:
            record["member.country.code"] = attrs["country_code"]
        records.append(record)
    return pd.DataFrame(records)


def load_topic_vote_data(ep: int, topic_slug: str) -> Tuple[pd.DataFrame | None, List[str] | None]:
    """Load vote data and return (df_votes, vote_ids) for the topic."""
    schema = SCHEMA.get(ep)
    if not schema:
        return None, None
    
    # Load metadata
    meta_path = Path(f"data/votewatch_csv/EP{ep}_Voted main docs.csv")
    if not meta_path.exists():
        return None, None
    
    try:
        meta_df = pd.read_csv(meta_path, low_memory=False)
    except Exception:
        return None, None
    
    policy_col = schema["policy"]
    vote_id_col = schema["vote_id"]
    
    # For EP9/EP10, try both "committees" and "policy_area" (harmonized files use policy_area)
    if policy_col not in meta_df.columns:
        if ep in (9, 10) and "policy_area" in meta_df.columns:
            policy_col = "policy_area"
        else:
            return None, None
    
    if vote_id_col not in meta_df.columns:
        return None, None
    
    # Normalize topic name for matching
    # topic_slug is like "agriculture_and_rural_development"
    # CSV has harmonized names like "Agriculture and Rural Development"
    topic_normalized = topic_slug.lower().replace("_", " ").strip()
    topic_title_case = topic_slug.replace("_", " ").title()  # "Agriculture And Rural Development"
    
    # Find votes for this topic
    vote_ids = set()
    for _, row in meta_df.iterrows():
        topics_raw = str(row.get(policy_col, ""))
        if pd.isna(topics_raw) or topics_raw.lower() == "nan":
            continue
        
        # Handle semicolon-separated topics
        topics = [t.strip() for t in topics_raw.split(";") if t.strip()]
        topics_lower = [t.lower() for t in topics]
        
        # Check if topic matches (case-insensitive, handle variations)
        topic_matches = False
        for topic_val in topics:
            topic_val_lower = topic_val.lower()
            # Exact match (normalized)
            if topic_normalized == topic_val_lower:
                topic_matches = True
                break
            # Substring match
            if topic_normalized in topic_val_lower or topic_val_lower in topic_normalized:
                topic_matches = True
                break
            # Title case match (for harmonized names)
            if topic_title_case.lower() == topic_val_lower:
                topic_matches = True
                break
        
        if topic_matches:
            vote_id_val = row.get(vote_id_col)
            if pd.notna(vote_id_val):
                vote_id = str(vote_id_val).replace(".0", "").strip()
                if vote_id:
                    vote_ids.add(vote_id)
    
    if not vote_ids:
        print(f"    ⚠️  No votes found for topic '{topic_slug}' (normalized: '{topic_normalized}')")
        print(f"    Available policy names (sample): {list(meta_df[policy_col].dropna().unique()[:5])}")
        return None, None
    
    print(f"    Found {len(vote_ids)} votes for topic '{topic_slug}'")
    
    # Load vote matrix
    vote_matrix_path = Path(f"data/all_votes_main_EP{ep}.csv")
    if not vote_matrix_path.exists():
        print(f"    ⚠️  Vote matrix file not found: {vote_matrix_path}")
        return None, None
    
    try:
        df_votes = pd.read_csv(vote_matrix_path, low_memory=False)
    except Exception:
        return None, None
    
    # Filter to topic votes
    vote_cols = [c for c in df_votes.columns if c.isdigit() or c.replace(".0", "").isdigit()]
    topic_vote_cols = [c for c in vote_cols if c.replace(".0", "") in vote_ids]
    
    if len(topic_vote_cols) < 10:
        print(f"    ⚠️  Only {len(topic_vote_cols)} vote columns found (need ≥10)")
        return None, None
    
    print(f"    Loaded {len(topic_vote_cols)} vote columns from {len(vote_ids)} vote IDs")
    return df_votes, topic_vote_cols


def build_mep_network_from_votes(df_votes: pd.DataFrame, vote_cols: List[str], ep: int) -> Tuple[nx.Graph | None, pd.DataFrame | None]:
    """Build MEP-level network from vote data."""
    if not HAS_MODULARITY_MODULE:
        return None, None
    
    # Filter MEPs with sufficient participation
    df_vote_subset = df_votes[vote_cols].copy()
    valid_mask = df_vote_subset.isna().sum(axis=1) < 0.7 * len(vote_cols)
    df_vote_subset = df_vote_subset.loc[valid_mask]
    df_meta = df_votes.loc[valid_mask].copy()
    
    if len(df_vote_subset) < 10:
        return None, None
    
    try:
        # Compute similarity matrix
        A = similarity_matrix(df_vote_subset)
        
        # Get MEP IDs
        schema = SCHEMA[ep]
        mep_ids = df_meta[schema["member_id"]].astype(str).tolist()
        
        # Build graph (modularity.py version doesn't take df_meta)
        G = graph_from_similarity(A, mep_ids)
        
        # Remove isolated nodes
        isolated = list(nx.isolates(G))
        if isolated:
            G.remove_nodes_from(isolated)
            df_meta = df_meta[~df_meta[schema["member_id"]].astype(str).isin(isolated)].copy()
        
        if G.number_of_nodes() < 10:
            return None, None
        
        return G, df_meta
    except Exception as e:
        print(f"    ⚠️  Error building MEP network: {e}")
        return None, None


def compute_comprehensive_modularity_with_subsampling(G_mep_full: nx.Graph, df_meta_full: pd.DataFrame, ep: int, needs_subsampling: bool) -> Dict[str, float]:
    """Compute modularity with multiple subsamples if network needs subsampling, return max and std."""
    if not HAS_MODULARITY_MODULE or G_mep_full.number_of_edges() == 0:
        return {}
    
    # If network needs subsampling, compute modularity multiple times with different subsamples
    if needs_subsampling and G_mep_full.number_of_nodes() > MAX_NODES:
        print(f"    Computing modularity with {SUBSAMPLE_ITERATIONS} different subsamples...")
        print(f"    For each subsample, running Louvain {QMAX_RUNS} times and taking the maximum Qmax...")
        all_metrics = []
        nodes_full = list(G_mep_full.nodes())
        
        for i in range(SUBSAMPLE_ITERATIONS):
            # Create a different random subsample each time
            rng_sample = random.Random(42 + i)
            sampled_nodes = rng_sample.sample(nodes_full, MAX_NODES)
            G_subsample = G_mep_full.subgraph(sampled_nodes).copy()
            df_meta_subsample = df_meta_full[df_meta_full["member.id"].astype(str).isin(sampled_nodes)].copy()
            
            # Compute modularity for this subsample
            # For each subsample, run Louvain 20 times and take the maximum Qmax
            metrics = compute_comprehensive_modularity(G_subsample, df_meta_subsample, ep, random_seed=42, use_max_qmax=True)
            if metrics:
                all_metrics.append(metrics)
        
        if not all_metrics:
            return {}
        
        # Aggregate: take max for each metric, compute std
        result = {}
        metric_names = set()
        for m in all_metrics:
            metric_names.update(m.keys())
        
        for metric_name in metric_names:
            values = [m.get(metric_name, math.nan) for m in all_metrics if metric_name in m]
            values = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
            if values:
                result[metric_name] = max(values)  # Take maximum
                if len(values) > 1:
                    result[f"{metric_name}_subsample_std"] = float(np.std(values))
                else:
                    result[f"{metric_name}_subsample_std"] = 0.0
        
        return result
    else:
        # Not subsampled, compute normally
        return compute_comprehensive_modularity(G_mep_full, df_meta_full, ep)


def compute_comprehensive_modularity(G_mep: nx.Graph, df_meta: pd.DataFrame, ep: int, random_seed: int = 42, use_max_qmax: bool = False) -> Dict[str, float]:
    """Compute all modularity measures like in modularity.py.
    
    Args:
        G_mep: NetworkX graph
        df_meta: Metadata dataframe
        ep: EP number
        random_seed: Random seed
        use_max_qmax: If True, take max Qmax across runs; if False, take mean (default: False)
    """
    metrics: Dict[str, float] = {}
    
    if not HAS_MODULARITY_MODULE or G_mep.number_of_edges() == 0:
        return metrics
    
    try:
        # Qmax with multiple runs
        # Temporarily suppress print statements from compute_qmax
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            Qmax_value, Qmax_std = compute_qmax(G_mep, R=QMAX_RUNS, gamma=1.0, random_state=random_seed, return_max=use_max_qmax)
        finally:
            sys.stdout = old_stdout
        metrics["qmax"] = Qmax_value
        metrics["qmax_std"] = Qmax_std
        
        # Party modularity
        if "member.group.short_label" in df_meta.columns:
            party_partition = {
                str(row["member.id"]): str(row["member.group.short_label"])
                for _, row in df_meta.iterrows()
                if pd.notna(row.get("member.group.short_label"))
            }
            if party_partition:
                metrics["qparty"] = compute_partition_modularity(G_mep, party_partition)
                metrics["qparty_qmax_ratio"] = metrics["qparty"] / Qmax_value if Qmax_value > 0 else math.nan
        
        # Country modularity
        country_cols = ["member.country.label", "member.country.code"]
        country_col = next((c for c in country_cols if c in df_meta.columns), None)
        if country_col:
            country_partition = {
                str(row["member.id"]): str(row[country_col])
                for _, row in df_meta.iterrows()
                if pd.notna(row.get(country_col))
            }
            if country_partition:
                metrics["qcountry"] = compute_partition_modularity(G_mep, country_partition)
                metrics["qcountry_qmax_ratio"] = metrics["qcountry"] / Qmax_value if Qmax_value > 0 else math.nan
        
        # Left-Right modularity
        lr_partition = create_left_right_partition(df_meta)
        if lr_partition:
            metrics["q_left_right"] = compute_partition_modularity(G_mep, lr_partition)
            metrics["q_left_right_qmax_ratio"] = metrics["q_left_right"] / Qmax_value if Qmax_value > 0 else math.nan
        
        # Left-Center-Right modularity
        lcr_partition = create_left_center_right_partition(df_meta)
        if lcr_partition:
            metrics["q_left_center_right"] = compute_partition_modularity(G_mep, lcr_partition)
            metrics["q_left_center_right_qmax_ratio"] = metrics["q_left_center_right"] / Qmax_value if Qmax_value > 0 else math.nan
        
        # Extreme-Centrist modularity
        ec_partition = create_extreme_centrist_partition(df_meta)
        if ec_partition:
            metrics["q_extreme_centrist"] = compute_partition_modularity(G_mep, ec_partition)
            metrics["q_extreme_centrist_qmax_ratio"] = metrics["q_extreme_centrist"] / Qmax_value if Qmax_value > 0 else math.nan

        # Majority vs Opposition modularity
        coalition_partition = create_coalition_partition(df_meta, ep)
        if coalition_partition:
            metrics["q_majority_opposition"] = compute_partition_modularity(G_mep, coalition_partition)
            metrics["q_majority_opposition_qmax_ratio"] = metrics["q_majority_opposition"] / Qmax_value if Qmax_value > 0 else math.nan
        
    except Exception as e:
        print(f"    ⚠️  Error computing modularity: {e}")
        import traceback
        traceback.print_exc()
    
    return metrics


def compute_graph_metrics(G: nx.Graph) -> Dict[str, float]:
    """Compute basic graph metrics (for vote-based networks)."""
    metrics: Dict[str, float] = {}
    n = G.number_of_nodes()
    m = G.number_of_edges()
    metrics["node_count"] = n
    metrics["edge_count"] = m
    metrics["density"] = nx.density(G) if n > 1 else 0.0

    if n:
        degrees = np.array([deg for _, deg in G.degree()], dtype=float)
        metrics["avg_degree"] = float(degrees.mean())
        metrics["degree_std"] = float(degrees.std(ddof=0))
    else:
        metrics["avg_degree"] = 0.0
        metrics["degree_std"] = 0.0

    metrics["transitivity"] = nx.transitivity(G) if n > 2 else math.nan
    metrics["avg_clustering"] = nx.average_clustering(G) if n > 1 else math.nan

    components = list(nx.connected_components(G)) if n else []
    metrics["component_count"] = len(components)
    if components:
        largest = max(components, key=len)
        metrics["largest_component_ratio"] = len(largest) / n if n else math.nan
        if len(largest) > 1:
            metrics["avg_path_length_lcc"] = nx.average_shortest_path_length(G.subgraph(largest))
        else:
            metrics["avg_path_length_lcc"] = math.nan
    else:
        metrics["largest_component_ratio"] = math.nan
        metrics["avg_path_length_lcc"] = math.nan

    # Basic Louvain modularity (single run, for vote networks)
    if HAS_LOUVAIN and m > 0:
        try:
            partition = community_louvain.best_partition(G, weight="weight", random_state=42)
            metrics["louvain_modularity"] = community_louvain.modularity(partition, G, weight="weight")
            metrics["n_communities"] = len(set(partition.values()))
        except Exception:
            metrics["louvain_modularity"] = math.nan
            metrics["n_communities"] = math.nan
    else:
        metrics["louvain_modularity"] = math.nan
        metrics["n_communities"] = math.nan

    return metrics


def generate_modularity_plots(df: pd.DataFrame):
    """Generate modularity bar plots per EP (like modularity.py)."""
    import matplotlib.pyplot as plt
    
    # Check if we have modularity data
    mod_cols = [c for c in df.columns if c.startswith("q") and c not in ("qmax_std", "qparty_qmax_ratio", "qcountry_qmax_ratio", "q_left_right_qmax_ratio", "q_extreme_centrist_qmax_ratio")]
    if not mod_cols:
        print("  ⚠️  No modularity data found, skipping plots.")
        return
    
    # Try to load vote counts from comprehensive metrics file
    vote_counts = {}
    metrics_path = OUTPUT_DIR / "topic_network_metrics.csv"
    if metrics_path.exists():
        try:
            metrics_df = pd.read_csv(metrics_path)
            if "votes_used" in metrics_df.columns and "topic_slug" in metrics_df.columns and "ep" in metrics_df.columns:
                for _, row in metrics_df.iterrows():
                    if pd.notna(row.get("votes_used")):
                        key = (str(row["topic_slug"]), int(row["ep"]))
                        vote_counts[key] = int(row["votes_used"])
                print(f"  Loaded vote counts for {len(vote_counts)} topic-EP combinations")
        except Exception as e:
            print(f"  ⚠️  Could not load vote counts: {e}")
    
    for ep in EP_RANGE:
        ep_df = df[df["ep"] == ep].copy()
        if ep_df.empty:
            continue
        
        # Filter to rows with modularity data
        has_mod = ep_df[mod_cols].notna().any(axis=1)
        ep_df = ep_df[has_mod].copy()
        
        if ep_df.empty:
            print(f"  ⚠️  EP{ep}: No modularity data, skipping plot.")
            continue
        
        # Sort by Qmax (descending)
        if "qmax" in ep_df.columns:
            ep_df = ep_df.sort_values("qmax", ascending=False, na_position='last')
        
        # Prepare data for plotting with vote counts in brackets
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
        
        # Collect which measures we have (in desired order: majority, left-center-right, left-right, extreme, party, country)
        available_measures = []
        measure_order = ["qmax", "q_majority_opposition", "q_left_center_right", "q_left_right", 
                        "q_extreme_centrist", "qparty", "qcountry"]
        for measure in measure_order:
            if measure in ep_df.columns and ep_df[measure].notna().any():
                available_measures.append(measure)
        
        n_measures = len(available_measures)
        if n_measures == 0:
            continue
        
        # Center bars around x
        total_width = n_measures * width
        start_offset = -total_width / 2 + width / 2
        
        # Plot each modularity measure
        for idx, measure in enumerate(available_measures):
            offset = start_offset + idx * width
            values = ep_df[measure].fillna(0).tolist()
            label_map = {
                "qmax": "Qmax",
                "q_extreme_centrist": "Extreme–Centrist",
                "qparty": "Party",
                "q_left_right": "Left–Right",
                "qcountry": "Country",
                "q_majority_opposition": "Majority vs Opposition",
                "q_left_center_right": "Left–Center–Right",
            }
            ax.bar(x + offset, values, width, color=colors[measure], label=label_map[measure], alpha=0.9)
        
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
        print(f"  ✅ Saved modularity plot: {plot_path}")
        
        # Save CSV
        csv_path = MODULARITY_OUTPUT_DIR / f"modularity_results_EP{ep}.csv"
        # Select relevant columns
        csv_cols = ["topic_slug", "topic_label", "ep"] + mod_cols
        csv_cols = [c for c in csv_cols if c in ep_df.columns]
        ep_df[csv_cols].to_csv(csv_path, index=False)
        print(f"  ✅ Saved modularity CSV: {csv_path}")


def normalize_topic_filter(topic_filters: Iterable[str]) -> set[str]:
    cleaned = set()
    for item in topic_filters:
        slug = item.strip().lower().replace(" ", "_")
        if slug:
            cleaned.add(slug)
    return cleaned


def analyze_topics(topic_filters: Iterable[str] | None = None, ep_filter: int | None = None):
    records: List[Dict[str, float]] = []

    print(f"Scanning topic networks under {BASE_DIR.resolve()}...")

    allowed_topics = normalize_topic_filter(topic_filters or [])
    if allowed_topics:
        print(f"Topic filter enabled: {sorted(allowed_topics)}")
    
    if ep_filter is not None:
        print(f"EP filter enabled: EP{ep_filter}")
        ep_range = [ep_filter]
    else:
        ep_range = list(EP_RANGE)

    for ep in ep_range:
        ep_dir = BASE_DIR / f"EP{ep}"
        if not ep_dir.exists():
            print(f"⚠️  EP{ep} directory not found, skipping.")
            continue
        ep_files = sorted(ep_dir.glob("*.gexf"))
        if not ep_files:
            print(f"⚠️  EP{ep} has no GEXF files, skipping.")
            continue

        print(f"\n=== EP{ep}: {len(ep_files)} GEXF files ===")

        for path in ep_files:
            ep_number, topic_slug = extract_metadata(path)
            if ep_number is None or ep_number != ep:
                print(f"  → Skip {path.name}: mismatch ep metadata.")
                continue

            if allowed_topics and topic_slug not in allowed_topics:
                continue

            try:
                G = nx.read_gexf(path)
            except Exception as exc:
                print(f"⚠️ Failed to read {path}: {exc}")
                continue

            votes_used = G.number_of_nodes()
            G_proc = maybe_subsample_graph(G, topic_slug, ep)
            downsampled = G_proc.number_of_nodes() < votes_used
            metrics = compute_graph_metrics(G_proc)
            metrics["bootstrap_iterations"] = 0
            if metrics["node_count"] <= BOOTSTRAP_THRESHOLD:
                boot_stats = bootstrap_metrics(G_proc, topic_slug, ep)
                metrics.update(boot_stats)
            metrics["votes_used"] = votes_used
            metrics["downsampled"] = downsampled
            metrics["votes_used"] = votes_used
            metrics["downsampled"] = downsampled
            
            # Compute comprehensive modularity on existing MEP networks
            print(f"  Loading MEP network for comprehensive modularity analysis...")
            mep_gexf_path = find_mep_network_path(ep, topic_slug)
            if mep_gexf_path and mep_gexf_path.exists():
                try:
                    G_mep_full = nx.read_gexf(mep_gexf_path)
                    print(f"    Loaded MEP network: {mep_gexf_path.name} ({G_mep_full.number_of_nodes()} nodes)")
                    
                    # Extract metadata from node attributes
                    df_meta_full = extract_metadata_from_mep_graph(G_mep_full)
                    
                    # Check if we need to subsample
                    needs_subsampling = G_mep_full.number_of_nodes() > MAX_NODES
                    if needs_subsampling:
                        print(f"    Network has {G_mep_full.number_of_nodes()} nodes, will subsample to {MAX_NODES} nodes")
                    
                    # Compute modularity (with multiple subsamples if needed)
                    mod_metrics = compute_comprehensive_modularity_with_subsampling(
                        G_mep_full, df_meta_full, ep, needs_subsampling=needs_subsampling
                    )
                    metrics.update(mod_metrics)
                    qmax_val = mod_metrics.get('qmax', math.nan)
                    qmax_std = mod_metrics.get('qmax_subsample_std', mod_metrics.get('qmax_std', math.nan))
                    if needs_subsampling and not math.isnan(qmax_std):
                        print(f"    Modularity computed (max across {SUBSAMPLE_ITERATIONS} subsamples): Qmax={qmax_val:.4f} ± {qmax_std:.4f}")
                    else:
                        print(f"    Modularity computed: Qmax={qmax_val:.4f}")
                except Exception as exc:
                    print(f"    ⚠️  Error loading MEP network: {exc}")
            else:
                print(f"    ⚠️  MEP network not found for topic '{topic_slug}' (searched in {MEP_BASE_DIR}/EP{ep}/)")
            
            record = {
                "topic_slug": topic_slug,
                "topic_label": topic_slug.replace("_", " ").title(),
                "ep": ep,
                "gexf_path": str(path),
            }
            record.update(metrics)
            records.append(record)
            print(f"  ✓ Processed {path.name}: nodes={metrics['node_count']} edges={metrics['edge_count']}")

    if not records:
        print("No GEXF files found for analysis.")
        return

    df = pd.DataFrame(records)
    df.sort_values(["topic_slug", "ep"], inplace=True)

    metrics_path = OUTPUT_DIR / "topic_network_metrics.csv"
    df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")
    
    # Generate modularity plots per EP (like modularity.py)
    print("\n=== Generating modularity plots ===")
    generate_modularity_plots(df)

    trend_records: List[Dict[str, float]] = []
    metrics_cols = [c for c in df.columns if c not in {"topic_slug", "topic_label", "ep", "gexf_path"}]

    for topic, group in df.groupby("topic_slug"):
        eps = group["ep"].to_numpy()
        if len(eps) < 2:
            continue  # need at least two points for a trend
        x = np.array([EP_ORDER[e] for e in eps], dtype=float)
        for metric in metrics_cols:
            y = group[metric].to_numpy(dtype=float)
            if np.isnan(y).all():
                continue
            valid = np.isfinite(y)
            if valid.sum() < 2:
                continue
            coeffs = np.polyfit(x[valid], y[valid], 1)
            trend_records.append({
                "topic_slug": topic,
                "topic_label": group["topic_label"].iloc[0],
                "metric": metric,
                "slope_per_ep": coeffs[0],
                "intercept": coeffs[1],
                "min_value": float(np.nanmin(y)),
                "max_value": float(np.nanmax(y)),
            })

    if trend_records:
        trend_df = pd.DataFrame(trend_records)
        trend_path = OUTPUT_DIR / "topic_network_trends.csv"
        trend_df.to_csv(trend_path, index=False)
        print(f"Saved trend stats to {trend_path}")
    else:
        print("No trend data generated (insufficient overlapping topics).")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze topic network evolution across EP6-EP10."
    )
    parser.add_argument(
        "--topic",
        "-t",
        action="append",
        dest="topics",
        help="Topic slug or label to analyze (can repeat). Defaults to all topics.",
    )
    parser.add_argument(
        "--ep",
        type=int,
        choices=[6, 7, 8, 9, 10],
        help="Filter by specific EP legislature (6-10). Defaults to all EPs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_topics(args.topics, ep_filter=args.ep)

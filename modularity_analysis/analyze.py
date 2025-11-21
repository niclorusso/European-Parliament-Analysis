#!/usr/bin/env python3
"""
Modularity Analysis Script - Standalone

Flexible script to compute modularity metrics for:
- Global networks (all votes combined)
- Per-topic networks

All computation starts from raw vote data in the data/ folder.
No dependencies on external analysis scripts.

Usage:
    # Global analysis for all EPs
    python analyze.py --global
    
    # Global analysis for specific EPs
    python analyze.py --global --ep 9 10
    
    # Per-topic analysis for all topics and all EPs
    python analyze.py --topics
    
    # Per-topic analysis for specific topics and EPs
    python analyze.py --topics --ep 9 10 --topic "environment" "budget"
"""

import argparse
import sys
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Import local modules
from utils import (
    similarity_matrix,
    graph_from_similarity,
    load_vote_data,
    get_topic_vote_map,
    filter_votes_by_topic,
    extract_mep_metadata,
    match_topic,
    SCHEMA,
)
from modularity_functions import (
    compute_qmax,
    compute_partition_modularity,
    create_left_right_partition,
    create_left_center_right_partition,
    create_extreme_centrist_partition,
    create_coalition_partition,
    HAS_LOUVAIN,
)

# Configuration
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_NODES = 200
BOOTSTRAP_THRESHOLD = 30
BOOTSTRAP_ITERATIONS = 100
QMAX_RUNS = 20  # Number of Louvain runs per subsample
SUBSAMPLE_ITERATIONS = 10  # Number of subsamples for large networks
RNG = random.Random(42)


def compute_comprehensive_modularity(
    G_mep: nx.Graph,
    df_meta: pd.DataFrame,
    ep: int,
    random_seed: int = 42,
    use_max_qmax: bool = False,
    qmax_runs: int = 20
) -> Dict[str, float]:
    """
    Compute all modularity measures.
    
    Args:
        G_mep: NetworkX graph
        df_meta: MEP metadata dataframe
        ep: EP number
        random_seed: Random seed for Qmax
        use_max_qmax: If True, take max Qmax across runs; if False, take mean
        qmax_runs: Number of Louvain runs for Qmax computation
    
    Returns:
        Dictionary of modularity metrics
    """
    metrics: Dict[str, float] = {}
    
    if not HAS_LOUVAIN or G_mep.number_of_edges() == 0:
        return metrics
    
    try:
        # Qmax with multiple runs
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            Qmax_value, Qmax_std = compute_qmax(
                G_mep, R=qmax_runs, gamma=1.0, random_state=random_seed, return_max=use_max_qmax
            )
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
        return {}
    
    return metrics


def compute_comprehensive_modularity_with_subsampling(
    G_mep_full: nx.Graph,
    df_meta_full: pd.DataFrame,
    ep: int,
    needs_subsampling: bool,
    max_nodes: int = 200,
    subsample_iterations: int = 10,
    qmax_runs: int = 20
) -> Dict[str, float]:
    """
    Compute modularity with multiple subsamples if network needs subsampling.
    
    Args:
        G_mep_full: Full network graph
        df_meta_full: Full MEP metadata dataframe
        ep: EP number
        needs_subsampling: Whether subsampling is needed
        max_nodes: Maximum nodes before subsampling (default: 200)
        subsample_iterations: Number of subsamples for large networks (default: 10)
        qmax_runs: Number of Louvain runs per subsample (default: 20)
    
    Returns:
        Dictionary of modularity metrics (max and std across subsamples)
    """
    if not HAS_LOUVAIN or G_mep_full.number_of_edges() == 0:
        return {}
    
    # If network needs subsampling, compute modularity multiple times with different subsamples
    if needs_subsampling and G_mep_full.number_of_nodes() > max_nodes:
        print(f"    Computing modularity with {subsample_iterations} different subsamples...")
        print(f"    For each subsample, running Louvain {qmax_runs} times and taking the maximum Qmax...")
        all_metrics = []
        nodes_full = list(G_mep_full.nodes())
        
        for i in range(subsample_iterations):
            # Create a different random subsample each time
            rng_sample = random.Random(42 + i)
            sampled_nodes = rng_sample.sample(nodes_full, max_nodes)
            G_subsample = G_mep_full.subgraph(sampled_nodes).copy()
            df_meta_subsample = df_meta_full[df_meta_full["member.id"].astype(str).isin(sampled_nodes)].copy()
            
            # Compute modularity for this subsample
            # For each subsample, run Louvain qmax_runs times and take the maximum Qmax
            metrics = compute_comprehensive_modularity(
                G_subsample, df_meta_subsample, ep, random_seed=42, use_max_qmax=True, qmax_runs=qmax_runs
            )
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
        return compute_comprehensive_modularity(G_mep_full, df_meta_full, ep, qmax_runs=qmax_runs)


def process_single_ep_global(
    ep: int,
    max_nodes: int,
    subsample_iterations: int,
    qmax_runs: int,
    min_weight: float
) -> Optional[Dict]:
    """
    Worker function to process a single EP for global analysis.
    This function is designed to be called in parallel.
    
    Returns:
        Dictionary with modularity metrics or None if processing failed
    """
    try:
        print(f"[EP{ep}] Starting processing...")
        
        # Load vote data
        df_votes = load_vote_data(ep)
        if df_votes is None:
            print(f"[EP{ep}] ⚠️  Could not load vote data, skipping...")
            return None
        
        # Get vote columns
        vote_cols = [c for c in df_votes.columns if c.isdigit() or c.replace(".0", "").isdigit()]
        if len(vote_cols) < 1:
            print(f"[EP{ep}] ⚠️  Too few votes ({len(vote_cols)}), skipping...")
            return None
        
        votes_used = len(vote_cols)
        print(f"[EP{ep}] Found {votes_used} votes")
        
        # Filter to valid MEPs (at least 90% participation)
        df_vote_subset = df_votes[vote_cols].copy()
        valid_mask = df_vote_subset.isna().sum(axis=1) < 0.1 * len(vote_cols)
        df_vote_subset = df_vote_subset.loc[valid_mask]
        df_meta = extract_mep_metadata(df_votes.loc[valid_mask], ep)
        
        if len(df_vote_subset) < 10:
            print(f"[EP{ep}] ⚠️  Too few MEPs after filtering, skipping...")
            return None
        
        print(f"[EP{ep}] Building network from {len(df_vote_subset)} MEPs...")
        
        # Build similarity matrix and graph
        A = similarity_matrix(df_vote_subset)
        mep_ids = df_meta["member.id"].astype(str).tolist()
        G_global = graph_from_similarity(A, mep_ids, min_weight=min_weight)
        
        # Remove isolated nodes
        isolated = list(nx.isolates(G_global))
        if isolated:
            G_global.remove_nodes_from(isolated)
            df_meta = df_meta[~df_meta["member.id"].astype(str).isin(isolated)]
        
        if G_global.number_of_nodes() == 0 or G_global.number_of_edges() == 0:
            print(f"[EP{ep}] ⚠️  Network is empty or has no edges, skipping...")
            return None
        
        print(f"[EP{ep}] Network: {G_global.number_of_nodes()} nodes, {G_global.number_of_edges()} edges")
        
        # Determine if subsampling is needed
        needs_subsampling = G_global.number_of_nodes() > max_nodes
        if needs_subsampling:
            print(f"[EP{ep}] Network has {G_global.number_of_nodes()} nodes, will subsample to {max_nodes}")
        
        # Compute modularity
        mod_metrics = compute_comprehensive_modularity_with_subsampling(
            G_global, df_meta, ep, needs_subsampling=needs_subsampling,
            max_nodes=max_nodes, subsample_iterations=subsample_iterations, qmax_runs=qmax_runs
        )
        
        if not mod_metrics:
            print(f"[EP{ep}] ⚠️  Could not compute modularity, skipping...")
            return None
        
        # Add metadata
        mod_metrics["ep"] = ep
        mod_metrics["votes_used"] = votes_used
        mod_metrics["downsampled"] = needs_subsampling
        mod_metrics["bootstrapped"] = False
        
        print(f"[EP{ep}] ✅ Qmax = {mod_metrics.get('qmax', 'N/A')}")
        return mod_metrics
        
    except Exception as e:
        print(f"[EP{ep}] ⚠️  Error processing: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_global(ep_list: Optional[List[int]] = None, max_nodes: int = 200, 
                   subsample_iterations: int = 10, qmax_runs: int = 20, min_weight: float = 0.01):
    """Analyze global network modularity for specified EPs (parallelized).
    
    Args:
        ep_list: List of EP numbers to process
        max_nodes: Maximum nodes before subsampling (default: 200)
        subsample_iterations: Number of subsamples for large networks (default: 10)
        qmax_runs: Number of Louvain runs per subsample (default: 20)
        min_weight: Minimum edge weight threshold (default: 0.01)
    """
    print("=" * 80)
    print("GLOBAL NETWORK MODULARITY ANALYSIS (PARALLEL)")
    print("=" * 80)
    print(f"Configuration: max_nodes={max_nodes}, subsample_iterations={subsample_iterations}, qmax_runs={qmax_runs}, min_weight={min_weight}")
    
    if not HAS_LOUVAIN:
        print("⚠️  python-louvain not installed. Cannot compute modularity.")
        return None
    
    if ep_list:
        print(f"Processing EPs: {ep_list}")
    else:
        print("Processing all EPs (6-10)")
        ep_list = list(range(6, 11))
    
    # Determine number of workers (use all CPUs - 1, but at least 1)
    num_workers = max(1, min(len(ep_list), multiprocessing.cpu_count() - 1))
    print(f"Using {num_workers} parallel workers\n")
    
    # Process EPs in parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_ep = {
            executor.submit(
                process_single_ep_global,
                ep, max_nodes, subsample_iterations, qmax_runs, min_weight
            ): ep
            for ep in ep_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_ep):
            ep = future_to_ep[future]
            try:
                result = future.result()
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                print(f"[EP{ep}] ⚠️  Exception occurred: {e}")
    
    if not all_results:
        print("\n⚠️  No global modularity data computed.")
        return None
    
    # Sort results by EP number
    all_results.sort(key=lambda x: x["ep"])
    
    # Save to CSV
    results_df = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / "global_modularity.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved global modularity results to: {csv_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    summary_cols = ["ep", "qmax", "qparty", "q_majority_opposition", 
                    "q_left_center_right", "q_left_right", 
                    "q_extreme_centrist", "qcountry", "votes_used", "downsampled"]
    available_cols = [c for c in summary_cols if c in results_df.columns]
    print(results_df[available_cols].to_string(index=False))
    
    return results_df


def process_single_ep_topics(ep: int, topic_list: Optional[List[str]], max_nodes: int,
                             subsample_iterations: int, qmax_runs: int, min_weight: float) -> List[Dict]:
    """
    Process all topics for a single EP (worker function for parallel processing).
    
    Returns:
        List of modularity metric dictionaries (one per topic)
    """
    try:
        print(f"[EP{ep}] Starting topic analysis...")
        
        # Load vote data
        df_votes = load_vote_data(ep)
        if df_votes is None:
            print(f"[EP{ep}] ⚠️  Could not load vote data, skipping...")
            return []
        
        # Get topic-vote mapping
        topic_vote_map = get_topic_vote_map(ep)
        if not topic_vote_map:
            print(f"[EP{ep}] ⚠️  No topic data found, skipping...")
            return []
        
        # Filter topics if specified
        if topic_list:
            # Match topics
            available_topics = list(topic_vote_map.keys())
            topics_to_process = []
            for topic_query in topic_list:
                matches = match_topic(topic_query, available_topics)
                topics_to_process.extend(matches)
            topics_to_process = list(set(topics_to_process))  # Remove duplicates
        else:
            topics_to_process = list(topic_vote_map.keys())
        
        if not topics_to_process:
            print(f"[EP{ep}] ⚠️  No matching topics found, skipping...")
            return []
        
        print(f"[EP{ep}] Processing {len(topics_to_process)} topic(s)")
        
        ep_results = []
        
        for topic_name in topics_to_process:
            print(f"[EP{ep}] Processing topic: {topic_name}")
            
            vote_ids = topic_vote_map[topic_name]
            print(f"[EP{ep}]   Found {len(vote_ids)} votes for this topic")
            
            # Filter votes to this topic
            df_votes_filtered = filter_votes_by_topic(df_votes, vote_ids, ep)
            
            if len(df_votes_filtered) == 0:
                print(f"[EP{ep}]   ⚠️  No votes found after filtering, skipping...")
                continue
            
            # Get vote columns
            vote_cols = [c for c in df_votes_filtered.columns if c.isdigit() or c.replace(".0", "").isdigit()]
            if len(vote_cols) < 1:
                print(f"[EP{ep}]   ⚠️  Too few votes ({len(vote_cols)}), skipping...")
                continue
            
            # Filter to valid MEPs
            df_vote_subset = df_votes_filtered[vote_cols].copy()
            valid_mask = df_vote_subset.isna().sum(axis=1) < 0.1 * len(vote_cols)
            df_vote_subset = df_vote_subset.loc[valid_mask]
            df_meta = extract_mep_metadata(df_votes_filtered.loc[valid_mask], ep)
            
            if len(df_vote_subset) < 10:
                print(f"[EP{ep}]   ⚠️  Too few MEPs after filtering, skipping...")
                continue
            
            print(f"[EP{ep}]   Building network from {len(df_vote_subset)} MEPs...")
            
            # Build similarity matrix and graph
            A = similarity_matrix(df_vote_subset)
            mep_ids = df_meta["member.id"].astype(str).tolist()
            G_topic = graph_from_similarity(A, mep_ids, min_weight=min_weight)
            
            # Remove isolated nodes
            isolated = list(nx.isolates(G_topic))
            if isolated:
                G_topic.remove_nodes_from(isolated)
                df_meta = df_meta[~df_meta["member.id"].astype(str).isin(isolated)]
            
            if G_topic.number_of_nodes() == 0 or G_topic.number_of_edges() == 0:
                print(f"[EP{ep}]   ⚠️  Network is empty or has no edges, skipping...")
                continue
            
            print(f"[EP{ep}]   Network: {G_topic.number_of_nodes()} nodes, {G_topic.number_of_edges()} edges")
            
            # Determine if subsampling is needed
            needs_subsampling = G_topic.number_of_nodes() > max_nodes
            if needs_subsampling:
                print(f"[EP{ep}]   Network has {G_topic.number_of_nodes()} nodes, will subsample to {max_nodes}")
            
            # Compute modularity
            mod_metrics = compute_comprehensive_modularity_with_subsampling(
                G_topic, df_meta, ep, needs_subsampling=needs_subsampling,
                max_nodes=max_nodes, subsample_iterations=subsample_iterations, qmax_runs=qmax_runs
            )
            
            if not mod_metrics:
                print(f"[EP{ep}]   ⚠️  Could not compute modularity, skipping...")
                continue
            
            # Add metadata
            mod_metrics["ep"] = ep
            mod_metrics["topic"] = topic_name
            mod_metrics["votes_used"] = len(vote_ids)
            mod_metrics["downsampled"] = needs_subsampling
            mod_metrics["bootstrapped"] = False
            
            ep_results.append(mod_metrics)
            print(f"[EP{ep}]   ✅ Qmax = {mod_metrics.get('qmax', 'N/A')}")
        
        print(f"[EP{ep}] ✅ Completed {len(ep_results)} topic(s)")
        return ep_results
        
    except Exception as e:
        print(f"[EP{ep}] ⚠️  Error processing: {e}")
        import traceback
        traceback.print_exc()
        return []


def analyze_topics(ep_list: Optional[List[int]] = None, topic_list: Optional[List[str]] = None,
                   max_nodes: int = 200, subsample_iterations: int = 10, qmax_runs: int = 20, min_weight: float = 0.01):
    """Analyze per-topic network modularity for specified EPs and topics (parallelized).
    
    Args:
        ep_list: List of EP numbers to process
        topic_list: List of topic names to process
        max_nodes: Maximum nodes before subsampling (default: 200)
        subsample_iterations: Number of subsamples for large networks (default: 10)
        qmax_runs: Number of Louvain runs per subsample (default: 20)
        min_weight: Minimum edge weight threshold (default: 0.01)
    """
    print("=" * 80)
    print("PER-TOPIC NETWORK MODULARITY ANALYSIS (PARALLEL)")
    print("=" * 80)
    print(f"Configuration: max_nodes={max_nodes}, subsample_iterations={subsample_iterations}, qmax_runs={qmax_runs}, min_weight={min_weight}")
    
    if not HAS_LOUVAIN:
        print("⚠️  python-louvain not available. Cannot compute modularity.")
        return None
    
    if ep_list:
        print(f"Processing EPs: {ep_list}")
    else:
        print("Processing all EPs (6-10)")
        ep_list = list(range(6, 11))
    
    # Determine number of workers (use all CPUs - 1, but at least 1)
    num_workers = max(1, min(len(ep_list), multiprocessing.cpu_count() - 1))
    print(f"Using {num_workers} parallel workers\n")
    
    # Process EPs in parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_ep = {
            executor.submit(
                process_single_ep_topics,
                ep, topic_list, max_nodes, subsample_iterations, qmax_runs, min_weight
            ): ep
            for ep in ep_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_ep):
            ep = future_to_ep[future]
            try:
                ep_results = future.result()
                if ep_results:
                    all_results.extend(ep_results)
            except Exception as e:
                print(f"[EP{ep}] ⚠️  Exception occurred: {e}")
    
    if not all_results:
        print("\n⚠️  No topic modularity data computed.")
        return None
    
    # Save to CSV
    results_df = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / "topic_modularity.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved topic modularity results to: {csv_path}")
    print(f"   Total topics analyzed: {len(all_results)}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute modularity metrics for global or per-topic networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Global analysis for all EPs (default parameters)
  python analyze.py --global
  
  # Global analysis for EP9 and EP10 with custom parameters
  python analyze.py --global --ep 9 10 --max-nodes 150 --subsample-iterations 15 --qmax-runs 30 --min-weight 0.05
  
  # Per-topic analysis for all topics and EPs
  python analyze.py --topics
  
  # Per-topic analysis for specific topics
  python analyze.py --topics --topic "environment" "budget"
  
  # Per-topic analysis with custom parameters
  python analyze.py --topics --ep 9 10 --topic "environment" --max-nodes 250 --qmax-runs 25 --min-weight 0.02
        """
    )
    
    # Analysis type
    analysis_group = parser.add_mutually_exclusive_group(required=True)
    analysis_group.add_argument(
        "--global",
        dest="do_global",
        action="store_true",
        help="Analyze global networks (all votes combined)"
    )
    analysis_group.add_argument(
        "--topics",
        dest="do_topics",
        action="store_true",
        help="Analyze per-topic networks"
    )
    
    # EP selection
    parser.add_argument(
        "--ep", "-e",
        type=int,
        nargs="+",
        default=None,
        help="EP numbers to process (e.g., --ep 9 10). Default: all EPs (6-10)"
    )
    
    # Topic selection (only for per-topic analysis)
    parser.add_argument(
        "--topic", "-t",
        type=str,
        nargs="+",
        default=None,
        help="Topic names to process (case-insensitive, partial match). Only used with --topics"
    )
    
    # Modularity computation parameters
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=200,
        help="Maximum nodes before subsampling (default: 200)"
    )
    parser.add_argument(
        "--subsample-iterations",
        type=int,
        default=10,
        help="Number of subsamples for large networks (default: 10)"
    )
    parser.add_argument(
        "--qmax-runs",
        type=int,
        default=20,
        help="Number of Louvain runs per subsample (default: 20)"
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.01,
        help="Minimum edge weight threshold for graph construction (default: 0.01)"
    )
    
    args = parser.parse_args()
    
    # Validate topic argument
    if args.topic and not args.do_topics:
        parser.error("--topic can only be used with --topics")
    
    # Run analysis
    if args.do_global:
        analyze_global(
            ep_list=args.ep,
            max_nodes=args.max_nodes,
            subsample_iterations=args.subsample_iterations,
            qmax_runs=args.qmax_runs,
            min_weight=args.min_weight
        )
    elif args.do_topics:
        analyze_topics(
            ep_list=args.ep,
            topic_list=args.topic,
            max_nodes=args.max_nodes,
            subsample_iterations=args.subsample_iterations,
            qmax_runs=args.qmax_runs,
            min_weight=args.min_weight
        )


if __name__ == "__main__":
    main()

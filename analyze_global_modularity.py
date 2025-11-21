#!/usr/bin/env python3
"""Compute modularity for global networks (all votes) for each EP with subsampling."""

from plot_modularity_evolution import compute_global_modularity_per_ep
from pathlib import Path
import pandas as pd
import argparse

OUTPUT_DIR = Path("results/modularity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute modularity for global MEP networks")
    parser.add_argument(
        "--ep", "-e",
        type=int,
        nargs="+",
        default=None,
        help="EP numbers to process (e.g., --ep 9 10 or --ep 10). Default: all EPs (6-10)"
    )
    args = parser.parse_args()
    
    ep_list = args.ep if args.ep else None
    
    print("=" * 80)
    if ep_list:
        print(f"Computing Global Network Modularity for EPs: {ep_list}")
    else:
        print("Computing Global Network Modularity for All EPs (6-10)")
    print("=" * 80)
    
    global_df = compute_global_modularity_per_ep(ep_list=ep_list)
    
    if global_df is not None and not global_df.empty:
        # Save to CSV
        csv_path = OUTPUT_DIR / "global_modularity_results.csv"
        global_df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved global modularity results to: {csv_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(global_df[["ep", "qmax", "qparty", "q_majority_opposition", "q_left_center_right", 
                        "q_left_right", "q_extreme_centrist", "qcountry", "votes_used", "downsampled"]].to_string(index=False))
        
        # Check for subsample std
        if "qmax_subsample_std" in global_df.columns:
            print("\n" + "=" * 80)
            print("SUBSAMPLING STATISTICS (Standard Deviation across subsamples)")
            print("=" * 80)
            std_cols = [c for c in global_df.columns if "_subsample_std" in c]
            if std_cols:
                print(global_df[["ep"] + std_cols].to_string(index=False))
    else:
        print("\n⚠️  No global modularity data computed.")


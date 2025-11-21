#!/usr/bin/env python3
"""Check what results exist for modularity and network measures across all network types."""

import pandas as pd
from pathlib import Path
import os

print("=" * 80)
print("RESULTS STATUS CHECK")
print("=" * 80)

# 1. Check modularity results per EP per topic
print("\n1. MODULARITY RESULTS (per EP per topic)")
print("-" * 80)
mod_dir = Path("results/modularity")
if mod_dir.exists():
    csv_files = sorted(mod_dir.glob("modularity_results_EP*.csv"))
    if csv_files:
        print(f"   Found {len(csv_files)} CSV files")
        # Check columns in latest file
        latest = csv_files[-1]
        df = pd.read_csv(latest, nrows=0)
        cols = list(df.columns)
        print(f"   Columns in {latest.name}: {len(cols)}")
        print(f"   Metrics: {', '.join([c for c in cols if c.startswith('q')])}")
        
        # Check if new metrics are present
        has_majority = "q_majority_opposition" in cols
        has_lcr = "q_left_center_right" in cols
        print(f"   ✓ q_majority_opposition: {'YES' if has_majority else 'NO'}")
        print(f"   ✓ q_left_center_right: {'YES' if has_lcr else 'NO'}")
    else:
        print("   ⚠️  No CSV files found")
else:
    print("   ⚠️  Directory not found")

# 2. Check topic network metrics (comprehensive)
print("\n2. TOPIC NETWORK METRICS (comprehensive)")
print("-" * 80)
topic_metrics = Path("results/resin/topics/topic_evolution/topic_network_metrics.csv")
if topic_metrics.exists():
    df = pd.read_csv(topic_metrics, nrows=0)
    cols = list(df.columns)
    print(f"   File: {topic_metrics.name}")
    print(f"   Total columns: {len(cols)}")
    
    # Check metric categories
    mod_cols = [c for c in cols if c.startswith('q')]
    network_cols = [c for c in cols if c in ['node_count', 'edge_count', 'density', 'avg_degree', 'transitivity', 'avg_clustering', 'louvain_modularity', 'n_communities']]
    bootstrap_cols = [c for c in cols if 'bootstrap' in c]
    
    print(f"   Modularity metrics: {len(mod_cols)} ({', '.join(mod_cols[:5])}...)")
    print(f"   Network metrics: {len(network_cols)}")
    print(f"   Bootstrap metrics: {len(bootstrap_cols)}")
    
    # Check if new metrics are present
    has_majority = "q_majority_opposition" in cols
    has_lcr = "q_left_center_right" in cols
    print(f"   ✓ q_majority_opposition: {'YES' if has_majority else 'NO'}")
    print(f"   ✓ q_left_center_right: {'YES' if has_lcr else 'NO'}")
    
    # Check if data is populated
    df_full = pd.read_csv(topic_metrics)
    if has_majority:
        non_null = df_full["q_majority_opposition"].notna().sum()
        print(f"   Data rows with q_majority_opposition: {non_null}/{len(df_full)}")
    if has_lcr:
        non_null = df_full["q_left_center_right"].notna().sum()
        print(f"   Data rows with q_left_center_right: {non_null}/{len(df_full)}")
else:
    print("   ⚠️  File not found")

# 3. Check MEP network summaries
print("\n3. MEP NETWORK SUMMARIES (global)")
print("-" * 80)
mep_dir = Path("results/resin_mep")
if mep_dir.exists():
    csv_files = sorted(mep_dir.glob("mep_network_summary_EP*.csv"))
    if csv_files:
        print(f"   Found {len(csv_files)} CSV files")
        latest = csv_files[-1]
        df = pd.read_csv(latest, nrows=0)
        cols = list(df.columns)
        print(f"   Columns in {latest.name}: {len(cols)}")
        print(f"   Columns: {', '.join(cols)}")
        
        # Check if modularity metrics are present
        has_mod = any('modularity' in c.lower() for c in cols)
        print(f"   Contains modularity metrics: {'YES' if has_mod else 'NO'}")
    else:
        print("   ⚠️  No CSV files found")
else:
    print("   ⚠️  Directory not found")

# 4. Check global modularity computation
print("\n4. GLOBAL MODULARITY (computed in plot_modularity_evolution.py)")
print("-" * 80)
# Check if global modularity CSV exists
global_mod_csv = Path("results/modularity/global_modularity.csv")
if global_mod_csv.exists():
    df = pd.read_csv(global_mod_csv, nrows=0)
    cols = list(df.columns)
    print(f"   File: {global_mod_csv.name}")
    print(f"   Columns: {', '.join(cols)}")
    has_majority = "q_majority_opposition" in cols
    has_lcr = "q_left_center_right" in cols
    print(f"   ✓ q_majority_opposition: {'YES' if has_majority else 'NO'}")
    print(f"   ✓ q_left_center_right: {'YES' if has_lcr else 'NO'}")
else:
    print("   ⚠️  Global modularity CSV not found (computed on-the-fly in plot_modularity_evolution.py)")

# 5. Check resin (vote-level) networks
print("\n5. RESIN NETWORKS (vote-level, per topic)")
print("-" * 80)
resin_dir = Path("results/resin/topics")
if resin_dir.exists():
    ep_dirs = sorted([d for d in resin_dir.iterdir() if d.is_dir() and d.name.startswith("EP")])
    print(f"   Found {len(ep_dirs)} EP directories")
    for ep_dir in ep_dirs:
        gexf_files = list(ep_dir.glob("*.gexf"))
        png_files = list(ep_dir.glob("*.png"))
        csv_files = list(ep_dir.glob("*.csv"))
        print(f"   {ep_dir.name}: {len(gexf_files)} GEXF, {len(png_files)} PNG, {len(csv_files)} CSV")
else:
    print("   ⚠️  Directory not found")

# 6. Check resin_mep (MEP-level) networks
print("\n6. RESIN MEP NETWORKS (MEP-level, per topic)")
print("-" * 80)
resin_mep_dir = Path("results/resin_mep/topics")
if resin_mep_dir.exists():
    ep_dirs = sorted([d for d in resin_mep_dir.iterdir() if d.is_dir() and d.name.startswith("EP")])
    print(f"   Found {len(ep_dirs)} EP directories")
    for ep_dir in ep_dirs:
        gexf_files = list(ep_dir.glob("*.gexf"))
        png_files = list(ep_dir.glob("*.png"))
        txt_files = list(ep_dir.glob("*.txt"))
        csv_files = list(ep_dir.glob("*.csv"))
        print(f"   {ep_dir.name}: {len(gexf_files)} GEXF, {len(png_files)} PNG, {len(txt_files)} TXT, {len(csv_files)} CSV")
else:
    print("   ⚠️  Directory not found")

# 7. Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Network Types:
  1. Resin networks (vote-level): ✓ GEXF, PNG, basic metrics
  2. Resin MEP networks (MEP-level): ✓ GEXF, PNG, clusterability metrics, basic modularity
  3. Per-topic networks: ✓ Comprehensive metrics in topic_network_metrics.csv
  4. Global networks: ✓ GEXF files exist, modularity computed on-the-fly

Modularity Metrics Status:
  - Qmax: ✓
  - Qparty: ✓
  - Qcountry: ✓
  - Q_left_right: ✓
  - Q_extreme_centrist: ✓
  - Q_majority_opposition: ⚠️  In topic_network_metrics.csv but may need regeneration
  - Q_left_center_right: ⚠️  In topic_network_metrics.csv but may need regeneration

Missing/Needs Update:
  - modularity_results_EP*.csv files need q_majority_opposition and q_left_center_right
  - modularity_by_topic_EP*.png plots need to include new metrics
  - Global modularity CSV should be saved (currently computed on-the-fly)
""")


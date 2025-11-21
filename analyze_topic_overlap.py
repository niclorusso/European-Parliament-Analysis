"""
Analyze topic/policy area overlap across different European Parliament legislatures.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import shutil

# Schema for different EPs (matching resin_per_topic.ipynb)
SCHEMA = {
    6: {"vote_id": "Vote ID", "policy": "main_policy_name"},
    7: {"vote_id": "Vote ID", "policy": "De"},
    8: {"vote_id": "Vote ID", "policy": "De"},
    9: {"vote_id": "id", "policy": "policy_area"},
    10: {"vote_id": "id", "policy": "policy_area"},
}

ID_COLUMN_CANDIDATES = ["id", "Vote ID", "vote_id"]
TOPIC_COLUMN_CANDIDATES = ["committees", "main_policy_name", "policy_area", "policy", "topic", "De"]

# Harmonization mapping: variations -> canonical name
TOPIC_HARMONIZATION = {
    # Agriculture variations
    "agriculture": "Agriculture and Rural Development",
    "agriculture and rural development": "Agriculture and Rural Development",
    
    # Budget variations
    "budget": "Budgets",
    "budgets": "Budgets",
    
    # Constitutional variations
    "constitutional affairs": "Constitutional Affairs",
    "constitutional and inter-institutional affairs": "Constitutional Affairs",
    
    # Environment variations
    "environment and public health": "Environment, Climate and Food Safety",
    "environment, climate and food safety": "Environment, Climate and Food Safety",
    
    # Foreign Affairs variations
    "foreign affairs": "Foreign Affairs",
    "foreign and security policy": "Foreign Affairs",
    
    # Gender/Women's Rights variations
    "gender equality": "Women's Rights and Gender Equality",
    "women's rights and gender equality": "Women's Rights and Gender Equality",
    
    # Legal/Juridical variations
    "legal affairs": "Legal Affairs",
    "juridical affairs": "Legal Affairs",
    
    # Economic variations
    "economic and monetary affairs": "Economic and Monetary Affairs",
    "economics": "Economic and Monetary Affairs",
    
    # Regional Development variations (including typo)
    "regional development": "Regional Development",
    "regioanal development": "Regional Development",  # typo in EP8 data
}


def normalize_topic_name(topic_str):
    """Normalize topic names for comparison."""
    if pd.isna(topic_str):
        return None
    topic_str = str(topic_str).strip()
    
    # First normalize to lowercase for matching
    topic_lower = topic_str.lower()
    
    # Normalize spacing around special characters first
    topic_lower = topic_lower.replace(" & ", " and ")
    topic_lower = topic_lower.replace("&", " and ")
    topic_lower = topic_lower.replace("/", " ")
    topic_lower = " ".join(topic_lower.split())  # normalize whitespace
    
    # Check if we have a harmonization mapping
    if topic_lower in TOPIC_HARMONIZATION:
        return TOPIC_HARMONIZATION[topic_lower]
    
    # If no mapping, return title-cased version of normalized string
    # This preserves the original but standardizes formatting
    words = topic_lower.split()
    if words:
        # Title case each word
        return " ".join(word.capitalize() for word in words)
    
    return None


def harmonize_topic_cell(value):
    """Apply harmonization to a semicolon-separated topic cell."""
    if pd.isna(value):
        return value
    parts = [part.strip() for part in str(value).split(";") if part.strip()]
    harmonized = []
    for part in parts:
        normalized = normalize_topic_name(part)
        if normalized:
            harmonized.append(normalized)
    # Deduplicate while preserving order
    if harmonized:
        seen = set()
        ordered = []
        for topic in harmonized:
            if topic not in seen:
                seen.add(topic)
                ordered.append(topic)
        return "; ".join(ordered)
    return value


def harmonize_metadata_files(ep_numbers, backup=True):
    """Rewrite metadata CSVs with harmonized topic names."""
    updated_paths = []
    for ep in ep_numbers:
        meta_path = f"data/votewatch_csv/EP{ep}_Voted main docs.csv"
        if not os.path.exists(meta_path):
            print(f"⚠️  Cannot harmonize EP{ep}: {meta_path} missing.")
            continue

        try:
            meta = pd.read_csv(meta_path, low_memory=False)
        except Exception as exc:
            print(f"⚠️  Failed to read {meta_path}: {exc}")
            continue

        # Determine topic/policy column
        if ep in SCHEMA:
            policy_col = SCHEMA[ep]["policy"]
        else:
            policy_col = next((c for c in TOPIC_COLUMN_CANDIDATES if c in meta.columns), None)

        if policy_col not in meta.columns:
            print(f"⚠️  EP{ep}: Policy column '{policy_col}' not found; skipping.")
            continue

        before = meta[policy_col].copy()
        meta[policy_col] = meta[policy_col].apply(harmonize_topic_cell)

        if before.equals(meta[policy_col]):
            print(f"EP{ep}: No harmonization changes needed.")
            continue

        if backup:
            backup_path = meta_path + ".backup"
            if not os.path.exists(backup_path):
                shutil.copy(meta_path, backup_path)
                print(f"EP{ep}: Backup saved to {backup_path}")

        meta.to_csv(meta_path, index=False)
        updated_paths.append(meta_path)
        print(f"✅ EP{ep}: Harmonized topics written to {meta_path}")

    return updated_paths


def load_topics_for_ep(ep_number):
    """Load and extract unique topics for a given EP."""
    meta_path = f"data/votewatch_csv/EP{ep_number}_Voted main docs.csv"
    
    if not os.path.exists(meta_path):
        print(f"⚠️  Missing metadata for EP{ep_number}, skipping.")
        return None, None
    
    try:
        meta = pd.read_csv(meta_path, low_memory=False)
    except Exception as e:
        print(f"⚠️  Error loading EP{ep_number}: {e}")
        return None, None
    
    # Detect columns
    id_col = next((c for c in ID_COLUMN_CANDIDATES if c in meta.columns), None)
    if id_col is None:
        print(f"⚠️  EP{ep_number}: Could not find ID column")
        return None, None
    
    # Use schema if available, otherwise try to detect
    if ep_number in SCHEMA:
        policy_col = SCHEMA[ep_number]["policy"]
    else:
        policy_col = next((c for c in TOPIC_COLUMN_CANDIDATES if c in meta.columns), None)
    
    if policy_col is None or policy_col not in meta.columns:
        print(f"⚠️  EP{ep_number}: Could not find policy/topic column")
        return None, None
    
    # Extract topics (handle semicolon-separated values)
    topics_raw = set()
    topics_normalized = {}
    
    for _, row in meta.iterrows():
        topic_raw = row.get(policy_col)
        if pd.isna(topic_raw):
            continue
        
        # Handle semicolon-separated topics
        topic_list = [t.strip() for t in str(topic_raw).split(";") if t.strip()]
        for topic in topic_list:
            if topic and topic.lower() != "nan":
                topics_raw.add(topic)
                normalized = normalize_topic_name(topic)
                if normalized:
                    # Store mapping: normalized -> original (keep first occurrence)
                    if normalized not in topics_normalized:
                        topics_normalized[normalized] = topic
    
    print(f"EP{ep_number}: Found {len(topics_raw)} unique topics (raw), {len(topics_normalized)} normalized")
    
    return topics_normalized, topics_raw


def find_topic_matches(topic_dicts):
    """Find matching topics across EPs using normalized names."""
    # Build reverse mapping: normalized -> list of (EP, original_name)
    normalized_to_eps = defaultdict(list)
    
    for ep_num, (normalized_dict, raw_set) in topic_dicts.items():
        for normalized, original in normalized_dict.items():
            normalized_to_eps[normalized].append((ep_num, original))
    
    return normalized_to_eps


def create_overlap_matrix(topic_dicts):
    """Create a matrix showing which topics appear in which EPs."""
    normalized_to_eps = find_topic_matches(topic_dicts)
    
    # Get all EPs
    ep_numbers = sorted(topic_dicts.keys())
    
    # Build matrix: rows = topics, columns = EPs
    matrix_data = []
    topic_names = []
    
    for normalized, ep_list in sorted(normalized_to_eps.items()):
        # Use the harmonized canonical name (normalized is already the canonical name)
        topic_names.append(normalized)
        
        row = [0] * len(ep_numbers)
        for ep_num, _ in ep_list:
            if ep_num in ep_numbers:
                idx = ep_numbers.index(ep_num)
                row[idx] = 1
        matrix_data.append(row)
    
    df_matrix = pd.DataFrame(
        matrix_data,
        index=topic_names,
        columns=[f"EP{ep}" for ep in ep_numbers]
    )
    
    return df_matrix, normalized_to_eps


def analyze_topic_overlap():
    """Main analysis function."""
    print("=" * 80)
    print("TOPIC OVERLAP ANALYSIS ACROSS EUROPEAN PARLIAMENT LEGISLATURES")
    print("=" * 80)
    
    # Load topics for each EP
    topic_dicts = {}
    ep_numbers = [6, 7, 8, 9, 10]

    print("\n🔄 Harmonizing metadata files before analysis...")
    updated_paths = harmonize_metadata_files(ep_numbers)
    if updated_paths:
        print(f"   Harmonized files: {len(updated_paths)}")
    else:
        print("   No metadata files required changes.")
    
    for ep in ep_numbers:
        normalized_dict, raw_set = load_topics_for_ep(ep)
        if normalized_dict is not None:
            topic_dicts[ep] = (normalized_dict, raw_set)
    
    if not topic_dicts:
        print("❌ No topic data found for any EP!")
        return
    
    print(f"\n✅ Loaded topics for {len(topic_dicts)} legislatures: {sorted(topic_dicts.keys())}")
    
    # Create overlap matrix
    df_matrix, normalized_to_eps = create_overlap_matrix(topic_dicts)
    
    print(f"\n📊 Found {len(df_matrix)} unique topics across all EPs")
    
    # Calculate statistics
    ep_cols = [f"EP{ep}" for ep in sorted(topic_dicts.keys())]
    topic_counts = df_matrix[ep_cols].sum(axis=0)
    overlap_counts = df_matrix[ep_cols].sum(axis=1)
    
    print("\n" + "=" * 80)
    print("TOPIC COUNTS PER LEGISLATURE:")
    print("=" * 80)
    for col in ep_cols:
        print(f"  {col}: {int(topic_counts[col])} topics")
    
    print("\n" + "=" * 80)
    print("TOPIC OVERLAP STATISTICS:")
    print("=" * 80)
    for n_eps in range(1, len(ep_cols) + 1):
        count = (overlap_counts == n_eps).sum()
        print(f"  Topics appearing in exactly {n_eps} legislature(s): {count}")
    
    # Topics in all EPs
    topics_in_all = df_matrix[df_matrix[ep_cols].sum(axis=1) == len(ep_cols)]
    print(f"\n  Topics in ALL legislatures ({len(ep_cols)}): {len(topics_in_all)}")
    if len(topics_in_all) > 0:
        print("    " + ", ".join(topics_in_all.index.tolist()[:10]))
        if len(topics_in_all) > 10:
            print(f"    ... and {len(topics_in_all) - 10} more")
    
    # Topics unique to each EP
    print("\n" + "=" * 80)
    print("TOPICS UNIQUE TO EACH LEGISLATURE:")
    print("=" * 80)
    for col in ep_cols:
        unique = df_matrix[(df_matrix[col] == 1) & (df_matrix[ep_cols].sum(axis=1) == 1)]
        print(f"\n  {col}: {len(unique)} unique topics")
        if len(unique) > 0:
            print("    " + ", ".join(unique.index.tolist()[:5]))
            if len(unique) > 5:
                print(f"    ... and {len(unique) - 5} more")
    
    # Save matrix to CSV
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "topic_overlap_matrix.csv")
    df_matrix.to_csv(csv_path)
    print(f"\n💾 Saved overlap matrix to: {csv_path}")
    
    # Create visualization
    plot_path = os.path.join(output_dir, "topic_overlap_heatmap.png")
    create_overlap_heatmap(df_matrix, plot_path)
    print(f"📊 Saved heatmap to: {plot_path}")
    
    # Create detailed report
    report_path = os.path.join(output_dir, "topic_overlap_report.txt")
    create_detailed_report(df_matrix, normalized_to_eps, topic_dicts, report_path)
    print(f"📄 Saved detailed report to: {report_path}")
    
    return df_matrix, normalized_to_eps


def create_overlap_heatmap(df_matrix, output_path):
    """Create a heatmap visualization of topic overlap."""
    # Sort by total overlap (topics appearing in most EPs first)
    ep_cols = [c for c in df_matrix.columns if c.startswith("EP")]
    # Calculate total overlap for each topic and sort
    total_overlap = df_matrix[ep_cols].sum(axis=1)
    sorted_indices = total_overlap.sort_values(ascending=False).index
    df_sorted = df_matrix.loc[sorted_indices]
    
    # Limit to top 50 topics for readability
    if len(df_sorted) > 50:
        df_plot = df_sorted.head(50)
        title_suffix = " (top 50 by overlap)"
    else:
        df_plot = df_sorted
        title_suffix = ""
    
    plt.figure(figsize=(max(8, len(ep_cols) * 2), max(6, len(df_plot) * 0.3)))
    
    # Create heatmap
    sns.heatmap(
        df_plot[ep_cols],
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        cbar_kws={"label": "Present (1) / Absent (0)"},
        yticklabels=True,
        xticklabels=True
    )
    
    plt.title(f"Topic Overlap Across Legislatures{title_suffix}", fontsize=14, fontweight="bold")
    plt.ylabel("Topics", fontsize=12)
    plt.xlabel("Legislature", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def create_detailed_report(df_matrix, normalized_to_eps, topic_dicts, output_path):
    """Create a detailed text report of topic overlap."""
    ep_cols = [c for c in df_matrix.columns if c.startswith("EP")]
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TOPIC OVERLAP ANALYSIS - DETAILED REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        topic_counts = df_matrix[ep_cols].sum(axis=0)
        for col in ep_cols:
            f.write(f"{col}: {int(topic_counts[col])} topics\n")
        f.write(f"\nTotal unique topics: {len(df_matrix)}\n\n")
        
        # Topics by overlap count
        f.write("TOPICS BY OVERLAP COUNT\n")
        f.write("-" * 80 + "\n")
        overlap_counts = df_matrix[ep_cols].sum(axis=1)
        for n_eps in range(1, len(ep_cols) + 1):
            count = (overlap_counts == n_eps).sum()
            topics = df_matrix[overlap_counts == n_eps].index.tolist()
            f.write(f"\nAppearing in {n_eps} legislature(s): {count} topics\n")
            for topic in sorted(topics):
                eps_with_topic = [col for col in ep_cols if df_matrix.loc[topic, col] == 1]
                f.write(f"  - {topic} (in {', '.join(eps_with_topic)})\n")
        
        # Topics in all EPs
        f.write("\n\nTOPICS APPEARING IN ALL LEGISLATURES\n")
        f.write("-" * 80 + "\n")
        topics_in_all = df_matrix[df_matrix[ep_cols].sum(axis=1) == len(ep_cols)]
        for topic in sorted(topics_in_all.index):
            f.write(f"  - {topic}\n")
        
        # Topics unique to each EP
        f.write("\n\nTOPICS UNIQUE TO EACH LEGISLATURE\n")
        f.write("-" * 80 + "\n")
        for col in ep_cols:
            unique = df_matrix[(df_matrix[col] == 1) & (df_matrix[ep_cols].sum(axis=1) == 1)]
            f.write(f"\n{col} ({len(unique)} unique topics):\n")
            for topic in sorted(unique.index):
                f.write(f"  - {topic}\n")


if __name__ == "__main__":
    df_matrix, normalized_to_eps = analyze_topic_overlap()
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)


#!/usr/bin/env python3
"""List unique topics per EP legislature from harmonized metadata files."""

import pandas as pd
from pathlib import Path
from collections import OrderedDict

DATA_DIR = Path("data/votewatch_csv")
EP_NUMBERS = [6, 7, 8, 9, 10]

# Schema for policy/topic column per EP
SCHEMA = {
    6: "main_policy_name",
    7: "De",
    8: "De",
    9: "policy_area",
    10: "policy_area",
}


def load_unique_topics(ep):
    """Load and return unique topics for a given EP."""
    meta_path = DATA_DIR / f"EP{ep}_Voted main docs.csv"
    if not meta_path.exists():
        print(f"⚠️  Missing file: {meta_path}")
        return []
    
    try:
        df = pd.read_csv(meta_path, low_memory=False)
    except Exception as e:
        print(f"⚠️  Error loading EP{ep}: {e}")
        return []
    
    policy_col = SCHEMA.get(ep)
    if policy_col not in df.columns:
        print(f"⚠️  EP{ep}: column '{policy_col}' not found")
        return []
    
    topics = []
    for value in df[policy_col].dropna():
        # Handle semicolon-separated topics
        parts = [p.strip() for p in str(value).split(";") if p.strip() and p.strip().lower() != "nan"]
        topics.extend(parts)
    
    # Remove duplicates while preserving order
    unique_topics = list(OrderedDict.fromkeys(topics))
    return sorted(unique_topics)


def main():
    """Print topics per EP."""
    print("=" * 80)
    print("TOPICS PER LEGISLATURE (EP6-EP10)")
    print("=" * 80)
    
    topic_map = {}
    for ep in EP_NUMBERS:
        topics = load_unique_topics(ep)
        topic_map[ep] = topics
    
    # Print summary
    print("\n📊 SUMMARY:")
    for ep in EP_NUMBERS:
        topics = topic_map.get(ep, [])
        print(f"  EP{ep}: {len(topics)} topics")
    
    # Print detailed list
    print("\n" + "=" * 80)
    print("DETAILED LIST:")
    print("=" * 80)
    
    for ep in EP_NUMBERS:
        topics = topic_map.get(ep, [])
        print(f"\nEP{ep} — {len(topics)} topics")
        print("-" * 60)
        for i, topic in enumerate(topics, 1):
            print(f"  {i:2d}. {topic}")
    
    # Print overlap analysis
    print("\n" + "=" * 80)
    print("TOPIC OVERLAP ANALYSIS:")
    print("=" * 80)
    
    all_topics = set()
    for topics in topic_map.values():
        all_topics.update(topics)
    
    print(f"\nTotal unique topics across all EPs: {len(all_topics)}")
    
    # Topics present in all EPs
    topics_in_all = set(topic_map[EP_NUMBERS[0]])
    for ep in EP_NUMBERS[1:]:
        topics_in_all &= set(topic_map[ep])
    
    print(f"Topics present in all EPs: {len(topics_in_all)}")
    if topics_in_all:
        print("  " + ", ".join(sorted(topics_in_all)))
    
    # Topics unique to each EP
    print("\nTopics unique to each EP:")
    for ep in EP_NUMBERS:
        other_eps = [e for e in EP_NUMBERS if e != ep]
        other_topics = set()
        for other_ep in other_eps:
            other_topics.update(topic_map[other_ep])
        unique_to_ep = set(topic_map[ep]) - other_topics
        if unique_to_ep:
            print(f"  EP{ep} ({len(unique_to_ep)} unique): {', '.join(sorted(unique_to_ep))}")
        else:
            print(f"  EP{ep}: No unique topics")


if __name__ == "__main__":
    main()


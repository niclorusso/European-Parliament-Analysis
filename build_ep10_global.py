"""Build EP10 global MEP network from all votes."""
import pandas as pd
import networkx as nx
from pathlib import Path
from resin_mep import similarity_matrix, graph_from_similarity

EP = 10
OUTPUT_PATH = Path("results/resin_mep/EP10_global_20251118.gexf")

print(f"Building EP{EP} global network...")

# Load vote data
csv_path = f"data/all_votes_main_EP{EP}.csv"
if not Path(csv_path).exists():
    print(f"Error: {csv_path} not found")
    exit(1)

print(f"Loading vote data from {csv_path}...")
df = pd.read_csv(csv_path, low_memory=False)

# Get vote columns
vote_cols = [c for c in df.columns if c.isdigit() or c.replace(".0", "").isdigit()]
print(f"Found {len(vote_cols)} vote columns")

# Filter MEPs with reasonable participation
df = df[df.isna().sum(axis=1) < 0.5 * df.shape[1]]
print(f"After filtering: {len(df)} MEPs")

# Separate metadata and votes
meta_cols = [c for c in df.columns if not c.isdigit()]
df_meta = df[meta_cols].copy()
df_votes = df[vote_cols].copy()
mep_ids = df_meta['member.id'].astype(str).tolist()

# Filter MEPs with sufficient participation in votes
valid_mask = df_votes.isna().sum(axis=1) < 0.7 * len(vote_cols)
df_votes = df_votes.loc[valid_mask]
df_meta = df_meta.loc[valid_mask]
mep_ids = [mep_ids[i] for i in range(len(mep_ids)) if valid_mask.iloc[i]]

print(f"After vote participation filter: {len(df_votes)} MEPs")

if len(df_votes) < 10:
    print("Error: Too few MEPs")
    exit(1)

# Compute similarity matrix
print("Computing similarity matrix...")
A = similarity_matrix(df_votes)

# Build graph
print("Building network graph...")
G = graph_from_similarity(A, mep_ids, df_meta, min_weight=0.0)

# Remove isolated nodes
isolated = list(nx.isolates(G))
if isolated:
    G.remove_nodes_from(isolated)
    print(f"Removed {len(isolated)} isolated nodes")

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Save GEXF
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nx.write_gexf(G, OUTPUT_PATH)
print(f"✅ Saved global network to {OUTPUT_PATH}")


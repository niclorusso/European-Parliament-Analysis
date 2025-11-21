import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import datetime
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import random
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("Warning: python-louvain not installed. Community detection will be limited.")
try:
    import igraph as ig
    import leidenalg
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False
    print("Warning: igraph/leidenalg not installed. Leiden community detection disabled.")

# Configuration
OUTFOLDER = 'results/resin_mep'
os.makedirs(OUTFOLDER, exist_ok=True)
TOPIC_OUTFOLDER = os.path.join(OUTFOLDER, "topics")
os.makedirs(TOPIC_OUTFOLDER, exist_ok=True)
CLUSTERABILITY_METRICS = 0

# enable topic analysis is 1 or 0, boolean
ENABLE_TOPIC_ANALYSIS = int(os.getenv("RESIN_MEP_ENABLE_TOPICS", "0"))
TOPIC_MIN_VOTES = int(os.getenv("RESIN_MEP_TOPIC_MIN_VOTES", "0"))
_topic_whitelist_env = os.getenv("RESIN_MEP_TOPIC_WHITELIST", "")
TOPIC_WHITELIST = [t.strip().lower() for t in _topic_whitelist_env.split(",") if t.strip()] or None
TOPIC_COLUMN_OVERRIDE = os.getenv("RESIN_MEP_TOPIC_COLUMN")
EP_NUMBERS = os.getenv("RESIN_MEP_EP_NUMBERS", "").split(",")

ID_COLUMN_CANDIDATES = ["id", "Vote ID", "vote_id"]
TOPIC_COLUMN_CANDIDATES = ["committees", "main_policy_name", "policy_area", "policy", "topic", "De"]


def similarity_matrix(vote_df):
    """Compute similarity matrix from vote dataframe (MEPs x Votes)."""
    vote_cols = [c for c in vote_df.columns if c.isdigit()]
    
    mapping = {
        "FOR": 1,
        "AGAINST": -1,
        "ABSTENTION": 0,
        "DID_NOT_VOTE": np.nan,
        "DID_NO": np.nan,
        "DID_NOT_VOTE ": np.nan
    }
    
    V = vote_df[vote_cols].replace(mapping).to_numpy(dtype=float)
    n = V.shape[0]
    
    yes  = (V == 1).astype(float)
    no   = (V == -1).astype(float)
    abst = (V == 0).astype(float)
    
    S = np.zeros((n, n))
    counts = np.zeros((n, n))
    
    # AGREEMENT (YES-YES or NO-NO) → +1
    S += yes @ yes.T + no @ no.T
    counts += yes @ yes.T + no @ no.T
    
    # DISAGREEMENT (YES-NO / NO-YES) → -1
    neg_pairs = yes @ no.T + no @ yes.T
    S -= neg_pairs
    counts += neg_pairs
    
    # MIXED WITH ABSTENTION → -0.25
    mixed = yes @ abst.T + abst @ yes.T + no @ abst.T + abst @ no.T
    S += -0.25 * mixed
    counts += mixed
    
    # ABST-ABST → +0.25
    abst_pairs = abst @ abst.T
    S += 0.25 * abst_pairs
    counts += abst_pairs
    
    # Normalize S (similarity)
    S = np.divide(S, counts, out=np.zeros_like(S), where=counts > 0)
    
    # Convert [-1,1] → [0,1]
    A = 0.5 * S + 0.5
    
    # diagonal = 1
    np.fill_diagonal(A, 1.0)

    A = pow(A, 2) #### if we want to give more importance to the agreements
    
    return A


def graph_from_similarity(A, mep_ids, df_meta, min_weight=0.01):
    """Convert similarity matrix to NetworkX graph with MEP attributes."""
    G = nx.Graph()
    
    # Add nodes with metadata
    # Build lookup for faster access
    mep_id_to_idx = {}
    if df_meta is not None and 'member.id' in df_meta.columns:
        for idx, row in df_meta.iterrows():
            mep_id_to_idx[str(row['member.id'])] = idx
    
    for i, mep_id in enumerate(mep_ids):
        mep_id_str = str(mep_id)
        G.add_node(mep_id_str)
        
        # Add metadata if available
        if df_meta is not None and mep_id_str in mep_id_to_idx:
            idx = mep_id_to_idx[mep_id_str]
            row = df_meta.loc[idx]
            if 'member.group.short_label' in row:
                G.nodes[mep_id_str]['party'] = str(row['member.group.short_label'])
            if 'member.country.label' in row:
                G.nodes[mep_id_str]['country'] = str(row['member.country.label'])
            if 'member.country.code' in row:
                G.nodes[mep_id_str]['country_code'] = str(row['member.country.code'])
    
    # Add edges
    n = len(mep_ids)
    for i in range(n):
        for j in range(i + 1, n):
            w = A[i, j]
            if not np.isnan(w) and w >= min_weight:
                G.add_edge(str(mep_ids[i]), str(mep_ids[j]), weight=float(w))
    
    return G


def analyze_communities(G, df_meta_filtered, ep_number, label):
    """Detect communities and analyze party distribution."""
    results = {}
    
    # Louvain community detection
    if HAS_LOUVAIN:
        try:
            partition_louvain = community_louvain.best_partition(G, weight='weight', random_state=42)
            modularity_louvain = community_louvain.modularity(partition_louvain, G, weight='weight')
            nx.set_node_attributes(G, partition_louvain, 'louvain_community')
            results['louvain'] = {
                'partition': partition_louvain,
                'modularity': modularity_louvain,
                'n_communities': len(set(partition_louvain.values()))
            }
        except Exception as e:
            print(f"[EP{ep_number}] Louvain detection failed: {e}")
            results['louvain'] = None
    
    # Leiden community detection
    if HAS_LEIDEN:
        try:
            # Convert to igraph
            edge_list = [(str(u), str(v), data.get('weight', 1.0)) for u, v, data in G.edges(data=True)]
            gi = ig.Graph.TupleList(edge_list, directed=False, edge_attrs=['weight'])
            
            # Leiden algorithm
            leiden_partition = leidenalg.find_partition(
                gi, 
                leidenalg.ModularityVertexPartition,
                weights='weight',
                seed=42
            )
            
            # Convert back to dict
            leiden_dict = {gi.vs[i]['name']: c for i, c in enumerate(leiden_partition.membership)}
            modularity_leiden = leiden_partition.modularity
            
            nx.set_node_attributes(G, leiden_dict, 'leiden_community')
            results['leiden'] = {
                'partition': leiden_dict,
                'modularity': modularity_leiden,
                'n_communities': len(set(leiden_dict.values()))
            }
        except Exception as e:
            print(f"[EP{ep_number}] Leiden detection failed: {e}")
            results['leiden'] = None
    
    # Analyze party distribution per community
    party_col = 'member.group.short_label'
    if df_meta_filtered is not None and party_col in df_meta_filtered.columns:
        # Build fast lookup: node_id -> party
        node_to_party = {}
        for idx, row in df_meta_filtered.iterrows():
            node_id = str(row.get('member.id', ''))
            if node_id:
                node_to_party[node_id] = str(row[party_col])
        
        for method, data in results.items():
            if data is None:
                continue
            
            partition = data['partition']
            print(f"\n[EP{ep_number}] {label} - {method.upper()} Communities (Modularity: {data['modularity']:.4f}, N={data['n_communities']}):")
            
            # Build party distribution per community
            community_parties = defaultdict(Counter)
            for node_id, comm_id in partition.items():
                party = node_to_party.get(str(node_id), 'Unknown')
                community_parties[comm_id][party] += 1
            
            # Print statistics
            for comm_id in sorted(community_parties.keys()):
                party_counts = community_parties[comm_id]
                total = sum(party_counts.values())
                print(f"  Community {comm_id} ({total} MEPs):")
                for party, count in party_counts.most_common():
                    pct = 100 * count / total
                    print(f"    {party}: {count} ({pct:.1f}%)")
    
    return results


def compute_clusterability_metrics(G):
    """Compute clusterability metrics (transitivity, clustering coefficient, etc.)."""
    metrics = {}
    try:
        # Global clustering coefficient (transitivity)
        metrics['transitivity'] = nx.transitivity(G)
        
        # Average clustering coefficient
        metrics['avg_clustering'] = nx.average_clustering(G, weight='weight')
        
        # Density
        n = G.number_of_nodes()
        m = G.number_of_edges()
        metrics['density'] = 2 * m / (n * (n - 1)) if n > 1 else 0
        
        # Average path length (if connected)
        if nx.is_connected(G):
            metrics['avg_path_length'] = nx.average_shortest_path_length(G, weight='weight')
        else:
            # For disconnected graphs, compute for largest component
            largest_cc = max(nx.connected_components(G), key=len)
            G_sub = G.subgraph(largest_cc)
            if len(largest_cc) > 1:
                metrics['avg_path_length'] = nx.average_shortest_path_length(G_sub, weight='weight')
            else:
                metrics['avg_path_length'] = np.nan
        
        # Assortativity (by party if available)
        if 'party' in next(iter(G.nodes(data=True)), [None, {}])[1]:
            try:
                metrics['assortativity_party'] = nx.attribute_assortativity_coefficient(G, 'party')
            except:
                metrics['assortativity_party'] = np.nan
    except Exception as e:
        print(f"Error computing clusterability metrics: {e}")
    
    return metrics


def plot_network(G, title, output_path, node_size=20, edge_width=0.1):
    """Plot and save network visualization."""
    nodes_list = list(G.nodes())
    n_nodes = len(nodes_list)
    if n_nodes > 200:
        G = G.subgraph(random.sample(nodes_list, 200))
        title = title + f" (subsampled to {len(G.nodes())} nodes)"
    print(f"Plotting network with {len(G.nodes())} nodes and {len(G.edges())} edges...")
    pos = nx.spring_layout(G, iterations=3000, seed=42)
    print(f"Layout done.")
    
    plt.figure(figsize=(12, 12))
    
    nx.draw(G, pos, node_size=node_size, edge_color='grey', width=edge_width)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def load_topic_metadata(ep_number):
    """Load vote metadata with topic information."""
    candidates = [
        f"data/votewatch_csv/EP{ep_number}_Voted main docs.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path, low_memory=False)
    return None


def detect_columns(meta_df):
    """Detect ID and topic columns in metadata."""
    id_col = next((c for c in ID_COLUMN_CANDIDATES if c in meta_df.columns), None)
    if id_col is None:
        return None, None
    
    topic_col = None
    if TOPIC_COLUMN_OVERRIDE and TOPIC_COLUMN_OVERRIDE in meta_df.columns:
        topic_col = TOPIC_COLUMN_OVERRIDE
    else:
        topic_col = next((c for c in TOPIC_COLUMN_CANDIDATES if c in meta_df.columns), None)
    
    return id_col, topic_col


def build_topic_vote_map(meta_df, id_col, topic_col):
    """Build mapping from topic to set of vote IDs."""
    topic_map = defaultdict(set)
    for _, row in meta_df.iterrows():
        topic_raw = row.get(topic_col)
        if pd.isna(topic_raw):
            continue
        topics = [t.strip().lower() for t in str(topic_raw).split(";") if t.strip()]
        if not topics:
            continue
        vote_id = str(row[id_col])
        for topic in topics:
            topic_map[topic].add(vote_id)
    
    # Preserve original case for display
    topic_labels = {}
    for _, row in meta_df.iterrows():
        topic_raw = row.get(topic_col)
        if pd.isna(topic_raw):
            continue
        topics = [t.strip() for t in str(topic_raw).split(";") if t.strip()]
        for topic in topics:
            topic_lower = topic.lower()
            if topic_lower not in topic_labels:
                topic_labels[topic_lower] = topic
    
    return topic_map, topic_labels


def process_mep_network(ep_number, vote_cols, df_full, mep_ids, label, output_dir, today):
    """Build and save MEP network from vote columns."""
    if len(vote_cols) < TOPIC_MIN_VOTES:
        return None
    
    # Extract only the vote columns we need
    df_votes = df_full[vote_cols].copy()
    
    # Filter MEPs with sufficient participation
    valid_mask = df_votes.isna().sum(axis=1) < 0.7 * len(vote_cols)
    df_votes = df_votes.loc[valid_mask]
    
    # Get metadata for filtered MEPs
    meta_cols = [c for c in df_full.columns if not c.isdigit()]
    df_meta_filtered = df_full.loc[valid_mask, meta_cols].copy()
    mep_ids_filtered = [mep_ids[i] for i in range(len(mep_ids)) if valid_mask.iloc[i]]
    
    if len(df_votes) < 10:
        return None
    
    print(f"[EP{ep_number}] Computing similarity matrix for {label} ({len(vote_cols)} votes, {len(mep_ids_filtered)} MEPs)...")
    A = similarity_matrix(df_votes)
    
    print(f"[EP{ep_number}] Building network graph for {label}...")
    G = graph_from_similarity(A, mep_ids_filtered, df_meta_filtered, min_weight=0.0)
    
    if G.number_of_edges() == 0:
        print(f"[EP{ep_number}] Empty graph for {label}, skipping.")
        return None
    
    # Initialize analysis results
    clusterability = {}
    community_results = {}
    
    if CLUSTERABILITY_METRICS:
        # Compute clusterability metrics
        print(f"[EP{ep_number}] Computing clusterability metrics for {label}...")
        clusterability = compute_clusterability_metrics(G)
        print(f"[EP{ep_number}] {label} metrics: Transitivity={clusterability.get('transitivity', 0):.4f}, "
            f"Avg Clustering={clusterability.get('avg_clustering', 0):.4f}, "
            f"Density={clusterability.get('density', 0):.4f}")
        
        # Detect communities and analyze party distribution
        print(f"[EP{ep_number}] Detecting communities for {label}...")
        community_results = analyze_communities(G, df_meta_filtered, ep_number, label)
    
    os.makedirs(output_dir, exist_ok=True)

    # Save GEXF
    safe_label = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(label))[:80]
    filename = f"EP{ep_number}_{safe_label}_{today}"
    gexf_path = os.path.join(output_dir, f"{filename}.gexf")
    nx.write_gexf(G, gexf_path)
    print(f"[EP{ep_number}] Saved {label} network: {gexf_path}")
    
    # Save PNG plot
    plot_path = os.path.join(output_dir, f"{filename}.png")
    plot_title = f"EP{ep_number} - {label} (MEPs: {len(mep_ids_filtered)}, Edges: {G.number_of_edges()})"
    plot_network(G, plot_title, plot_path)
    print(f"[EP{ep_number}] Saved {label} plot: {plot_path}")
    
    # Save analysis results
    analysis_path = os.path.join(output_dir, f"{filename}_analysis.txt")
    with open(analysis_path, 'w') as f:
        f.write(f"Network Analysis: EP{ep_number} - {label}\n")
        f.write("=" * 80 + "\n\n")
        if CLUSTERABILITY_METRICS:
            f.write("CLUSTERABILITY METRICS:\n")
            for key, value in clusterability.items():
                if not np.isnan(value):
                    f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\nCOMMUNITY DETECTION:\n")
            for method, data in community_results.items():
                if data is not None:
                    f.write(f"\n{method.upper()}:\n")
                    f.write(f"  Modularity: {data['modularity']:.4f}\n")
                    f.write(f"  Number of communities: {data['n_communities']}\n")
    
    print(f"[EP{ep_number}] Saved analysis: {analysis_path}")
    
    # Prepare summary
    summary = {
        "ep": ep_number,
        "label": label,
        "vote_count": len(vote_cols),
        "mep_count": len(mep_ids_filtered),
        "edge_count": G.number_of_edges(),
        "gexf_path": gexf_path,
        "plot_path": plot_path,
        "analysis_path": analysis_path,
    }
    
    if CLUSTERABILITY_METRICS:
        # Add clusterability metrics
        for key, value in clusterability.items():
            if not np.isnan(value):
                summary[f"metric_{key}"] = value
        
        # Add modularity scores
        for method, data in community_results.items():
            if data is not None:
                summary[f"modularity_{method}"] = data['modularity']
                summary[f"n_communities_{method}"] = data['n_communities']
    
    return summary


def process_legislature(ep_number):
    """Process one legislature: global and per-topic MEP networks."""
    today = datetime.datetime.now().strftime("%Y%m%d")
    print(f"\n{'='*80}\nProcessing EP{ep_number} (Topic Analysis: {'ON' if ENABLE_TOPIC_ANALYSIS else 'OFF'})\n{'='*80}")
    
    # Load vote data (always needed)
    csv_path = f"data/all_votes_main_EP{ep_number}.csv"
    if not os.path.exists(csv_path):
        print(f"[EP{ep_number}] File not found: {csv_path}")
        return None
    
    print(f"[EP{ep_number}] Loading vote data...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Filter MEPs with reasonable participation
    vote_cols = [c for c in df.columns if c.isdigit()]
    df = df[df.isna().sum(axis=1) < 0.5 * df.shape[1]]
    
    # Separate metadata and votes
    meta_cols = [c for c in df.columns if not c.isdigit()]
    df_meta = df[meta_cols].copy()
    mep_ids = df_meta['member.id'].astype(str).tolist()
    
    summaries = []
    
    # Global network (all votes) - only if topic analysis is OFF
    if not ENABLE_TOPIC_ANALYSIS:
        print(f"[EP{ep_number}] Processing global network...")
        result = process_mep_network(
            ep_number, vote_cols, df, mep_ids,
            "global", OUTFOLDER, today
        )
        if result:
            summaries.append(result)
    
    # Per-topic networks - only if topic analysis is ON
    if ENABLE_TOPIC_ANALYSIS:
        print(f"[EP{ep_number}] Processing per-topic networks...")
        meta_df = load_topic_metadata(ep_number)
        if meta_df is not None:
            id_col, topic_col = detect_columns(meta_df)
            if id_col and topic_col:
                meta_df[id_col] = meta_df[id_col].astype(str)
                topic_map, topic_labels = build_topic_vote_map(meta_df, id_col, topic_col)
                
                if TOPIC_WHITELIST:
                    topic_map = {k: v for k, v in topic_map.items() if k in TOPIC_WHITELIST}
                
                print(f"[EP{ep_number}] Found {len(topic_map)} topics: {list(topic_map.keys())[:5]}..." if len(topic_map) > 5 else f"[EP{ep_number}] Found {len(topic_map)} topics: {list(topic_map.keys())}")
                
                ep_topic_dir = os.path.join(TOPIC_OUTFOLDER, f"EP{ep_number}")
                os.makedirs(ep_topic_dir, exist_ok=True)

                for topic_key, vote_ids in sorted(topic_map.items()):
                    topic_label = topic_labels.get(topic_key, topic_key)
                    vote_ids_set = set(vote_ids)
                    topic_vote_cols = [c for c in vote_cols if c in vote_ids_set]
                    
                    if len(topic_vote_cols) < TOPIC_MIN_VOTES:
                        continue
                    
                    result = process_mep_network(
                        ep_number, topic_vote_cols, df, mep_ids,
                        topic_label, ep_topic_dir, today
                    )
                    if result:
                        summaries.append(result)
            else:
                print(f"[EP{ep_number}] Could not detect topic columns, skipping topic analysis.")
        else:
            print(f"[EP{ep_number}] No topic metadata found, skipping topic analysis.")
    
    # Save summary
    if summaries:
        summary_df = pd.DataFrame(summaries)
        if ENABLE_TOPIC_ANALYSIS:
            output_dir = os.path.join(TOPIC_OUTFOLDER, f"EP{ep_number}")
        else:
            output_dir = OUTFOLDER
        summary_path = os.path.join(output_dir, f"mep_network_summary_EP{ep_number}_{today}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"[EP{ep_number}] Summary saved: {summary_path}")
    
    print(f"[EP{ep_number}] Complete.")
    return ep_number


if __name__ == "__main__":
    if EP_NUMBERS and EP_NUMBERS[0]:
        EPs = [int(ep.strip()) for ep in EP_NUMBERS if ep.strip()]
    else:
        EPs = list(range(6, 11))
    
    if not EPs:
        print("No EPs specified. Use RESIN_MEP_EP_NUMBERS environment variable or default to 6-10.")
        EPs = list(range(6, 11))
    
    # Parallel processing across EPs
    max_workers = max(1, min(len(EPs), multiprocessing.cpu_count() - 1))
    print(f"Processing {len(EPs)} legislatures with {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_legislature, EPs))
    
    print(f"\n{'='*80}\nCompleted: {results}\n{'='*80}")

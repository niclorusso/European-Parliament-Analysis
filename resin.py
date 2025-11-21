import pandas as pd
import numpy as np
import networkx as nx
import scipy.stats as stt
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from networkx.algorithms import community as nx_community
import community as community_louvain
from collections import Counter, defaultdict
import igraph as ig
import leidenalg
import datetime
import concurrent.futures

OUTFOLDER = 'results/resin'
os.makedirs(OUTFOLDER, exist_ok=True)
TOPIC_OUTFOLDER = os.path.join(OUTFOLDER, "topics")
os.makedirs(TOPIC_OUTFOLDER, exist_ok=True)

ENABLE_TOPIC_ANALYSIS = os.getenv("RESIN_ENABLE_TOPIC_ANALYSIS", "0")
TOPIC_MIN_VOTES = int(os.getenv("RESIN_TOPIC_MIN_VOTES", "0"))
_topic_whitelist_env = os.getenv("RESIN_TOPIC_WHITELIST", "")
TOPIC_WHITELIST = [t.strip() for t in _topic_whitelist_env.split(",") if t.strip()] or None
TOPIC_COLUMN_OVERRIDE = os.getenv("RESIN_TOPIC_COLUMN")
VALID_VOTE_VALUES = ("FOR", "AGAINST", "ABSTENTION")
EP_NUMBERS = os.getenv("RESIN_EP_NUMBERS", "").split(",")

def slugify_label(value: str, fallback: str = "network") -> str:
    """Create filesystem-safe slug."""
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    safe = "_".join(filter(None, safe.split("_")))
    return (safe[:80] if safe else fallback)


# --------------------------
# EDGE WEIGHT ANALYSIS
# --------------------------
def analyze_edge_weights(G, title="Network", output_dir=None):
    edges = []
    for u, v, data in G.edges(data=True):
        w = data.get("weight", None)
        if w is not None and np.isfinite(w):
            edges.append((u, v, float(w)))

    if not edges:
        print(f"[{title}] No finite weights found.")
        return None

    dfw = pd.DataFrame(edges, columns=["u", "v", "weight"])

    plt.figure(figsize=(8, 4))
    plt.hist(dfw["weight"], bins=50, edgecolor="white")
    plt.title(f"{title} - Edge Weight Distribution")
    plt.tight_layout()
    fname = slugify_label(title)
    if output_dir is None:
        if not ENABLE_TOPIC_ANALYSIS:
            output_dir = OUTFOLDER
        else:
            output_dir = TOPIC_OUTFOLDER
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{fname}_weights.png"), dpi=200)
    plt.close()

    return dfw


# --------------------------
# DUMMY CODING
# --------------------------
def make_dummy_coded_df(df_nodes, allowed_values=None):
    dummy_dict = {}
    allowed_set = set(allowed_values) if allowed_values else None
    for col in df_nodes.columns:
        if allowed_set:
            values = allowed_set
        else:
            values = [v for v in df_nodes[col].unique() if not pd.isna(v)]
        for value in values:
            if pd.isna(value):
                continue
            series = (df_nodes[col] == value).astype(int)
            if not series.any():
                continue
            name = f"{col}:{value}"
            dummy_dict[name] = series
    if not dummy_dict:
        raise ValueError("Dummy coding produced an empty dictionary; check allowed_values or input data.")
    return pd.concat(dummy_dict, axis=1)


# --------------------------
# PHI CORRELATION
# --------------------------
def phi_(n11, n00, n10, n01):
    n1p = n11 + n10
    n0p = n01 + n00
    np1 = n01 + n11
    np0 = n10 + n00
    den = n1p * n0p * np0 * np1
    if den == 0:
        return np.nan
    return (n11 * n00 - n10 * n01) / np.sqrt(den)


def phi(x, y, get_p=False):
    eq = x == y
    diff = ~eq
    n11 = np.sum(x[eq] == 1)
    n00 = np.sum(x[eq] == 0)
    n10 = np.sum(x[diff] == 1)
    n01 = np.sum(y[diff] == 1)

    r = phi_(n11, n00, n10, n01)

    if not get_p:
        return r

    if np.isnan(r):
        return r, 1.0

    t = r * np.sqrt((len(x) - 2) / (1 - r**2))
    p = stt.t.sf(abs(t), df=len(x) - 2) * 2
    return r, p


# --------------------------
# GRAPH CONSTRUCTION
# --------------------------
def make_graph_(df, list_of_nodes, alpha=0.05, get_p=True):
    G = nx.Graph()

    for i, ni in enumerate(list_of_nodes):
        for j, nj in enumerate(list_of_nodes):
            if j <= i:
                continue

            c1, c2 = df[ni], df[nj]

            if get_p:
                r, p = phi(c1, c2, True)
                if r > 0:
                    G.add_edge(ni, nj, weight=float(r), p=float(p))
            else:
                r = phi(c1, c2, False)
                if r > 0:
                    G.add_edge(ni, nj, weight=float(r))

    return G


# --------------------------
# IDEOLOGY HEAT
# --------------------------
def make_thermo_rep(df0, G):
    party_col = "member.group.short_label"
    assert party_col in df0.columns

    party_axis = {
        "GUE/NGL": -1, "Greens/EFA": -0.8, "S&D": -0.6,
        "Renew": -0.2, "EPP": 0.4, "ECR": 0.7, "ID": 1.0,
        "NI": 0.0
    }

    parties = df0[party_col].map(party_axis).fillna(0).astype(float)
    node_scores = {}

    for node in G.nodes():
        col = node.split(":")[0]
        vals = df0[col].astype(float)
        r, p = stt.pearsonr(vals, parties)
        node_scores[node] = r

    nx.set_node_attributes(G, node_scores, "ThermoRep_mean")


# --------------------------
# COORDINATES
# --------------------------
def get_x_y_coordinates(G, file_stub=None):
    pos = nx.spring_layout(G, iterations=5000)
    if file_stub:
        np.save(f"{OUTFOLDER}/{file_stub}_positions.npy", pos)
    arr = np.array([pos[n] for n in G.nodes()])
    xx, yy = arr[:, 0], arr[:, 1]
    return xx, yy


# --------------------------
# PLOT NETWORK
# --------------------------
def plot_network(G, title, output_path=None):
    print(f"Plotting network with {len(G.nodes())} nodes and {len(G.edges())} edges...")
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Node degree coloring
    degrees = dict(G.degree())
    node_colors = [degrees[n] for n in G.nodes()]
    
    # Build figure
    plt.figure(figsize=(10, 10))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.6)
    
    # Draw nodes with degree-based coloring
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        node_size=40,
        alpha=0.9
    )
    
    plt.title(title, fontsize=14)
    plt.axis("off")
    
    # Add degree colorbar
    if nodes is not None:
        plt.colorbar(nodes, label="Node degree", shrink=0.7)
    
    plt.tight_layout()
    
    if output_path is None:
        fname = slugify_label(title)
        output_path = os.path.join(OUTFOLDER, f"{fname}.png")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


# --------------------------
# CLUSTERING
# --------------------------
def cluster_analysis(G, label, output_file=None):
    part = community_louvain.best_partition(G)
    nx.set_node_attributes(G, part, "louvain_comm")

    gi = ig.Graph.TupleList(G.edges(), directed=False)
    leiden = leidenalg.find_partition(gi, leidenalg.ModularityVertexPartition)
    leiden_part = {gi.vs[i]["name"]: c for i, c in enumerate(leiden.membership)}
    nx.set_node_attributes(G, leiden_part, "leiden_comm")
    modularity = community_louvain.modularity(part, G)
    leiden_modularity = leiden.modularity

    out_path = output_file or f"{OUTFOLDER}/cluster_analysis.txt"
    with open(out_path, "a") as f:
        f.write(f"\n{label}\n")
        f.write(f"Louvain: {Counter(part.values())}\n")
        f.write(f"Leiden: {Counter(leiden.membership)}\n")
        f.write(f"Modularity (Louvain): {modularity:.3f}\n")
        f.write(f"Modularity (Leiden): {leiden_modularity:.3f}\n")


# --------------------------
# TOPIC HELPERS
# --------------------------
ID_COLUMN_CANDIDATES = ["id", "Vote ID", "vote_id"]
TOPIC_COLUMN_CANDIDATES = ["committees", "main_policy_name", "policy_area", "policy", "topic", "De"]

def load_topic_metadata(ep_number):
    candidates = [
        f"data/votewatch_csv/EP{ep_number}_Voted main docs.csv",
        #f"data/votewatch_csv/EP{ep_number}_Voted docs.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path, low_memory=False)
    raise FileNotFoundError(f"No vote metadata file found for EP{ep_number}")


def detect_columns(meta_df):
    id_col = next((c for c in ID_COLUMN_CANDIDATES if c in meta_df.columns), None)
    if id_col is None:
        raise KeyError("Could not find a vote ID column in metadata.")

    topic_col = None
    if TOPIC_COLUMN_OVERRIDE and TOPIC_COLUMN_OVERRIDE in meta_df.columns:
        topic_col = TOPIC_COLUMN_OVERRIDE
    else:
        topic_col = next((c for c in TOPIC_COLUMN_CANDIDATES if c in meta_df.columns), None)

    if topic_col is None:
        raise KeyError("Could not find a topic/committee column in metadata.")

    return id_col, topic_col


def build_topic_vote_map(meta_df, id_col, topic_col):
    topic_map = {}
    for _, row in meta_df.iterrows():
        topic_raw = row.get(topic_col)
        if pd.isna(topic_raw):
            continue
        topics = [t.strip() for t in str(topic_raw).split(";") if t.strip()]
        if not topics:
            continue
        vote_id = str(row[id_col])
        for topic in topics:
            normalized = topic.lower()
            if normalized not in topic_map:
                topic_map[normalized] = {"label": topic, "votes": set()}
            topic_map[normalized]["votes"].add(vote_id)
    return topic_map


def run_topic_networks(ep_number, df_votes, today):
    try:
        meta_df = load_topic_metadata(ep_number)
    except FileNotFoundError as exc:
        print(f"[EP{ep_number}] {exc}. Skipping topic analysis.")
        return

    try:
        id_col, topic_col = detect_columns(meta_df)
    except KeyError as exc:
        print(f"[EP{ep_number}] {exc}. Skipping topic analysis.")
        return

    meta_df[id_col] = meta_df[id_col].astype(str)
    topic_map = build_topic_vote_map(meta_df, id_col, topic_col)
    print(f"[EP{ep_number}] Found {len(topic_map)} topics.")
    if TOPIC_WHITELIST:
        whitelist_norm = {t.lower() for t in TOPIC_WHITELIST}
        topic_map = {k: v for k, v in topic_map.items() if k in whitelist_norm}

    if not topic_map:
        print(f"[EP{ep_number}] No topics found after filtering.")
        return

    summaries = []
    ep_topic_dir = os.path.join(TOPIC_OUTFOLDER, f"EP{ep_number}")
    os.makedirs(ep_topic_dir, exist_ok=True)
    for topic_key, payload in sorted(topic_map.items()):
        vote_ids = set(payload["votes"])
        topic_label = payload["label"]
        topic_cols = [c for c in df_votes.columns if c in vote_ids]
        if len(topic_cols) < TOPIC_MIN_VOTES:
            continue

        df_topic_raw = df_votes[topic_cols]
        if df_topic_raw.empty:
            continue

        print(f"[EP{ep_number}] Building topic network '{topic_label}' with {len(topic_cols)} votes...")
        df_topic_dummy = make_dummy_coded_df(df_topic_raw, allowed_values=VALID_VOTE_VALUES)
        G_topic = make_graph_(df_topic_dummy, df_topic_dummy.columns)
        if G_topic.number_of_edges() == 0:
            continue
        print(f"Check 1")

        slug = slugify_label(f"EP{ep_number}_{topic_label}")
        file_stub = f"{slug}_{today}"
        gexf_path = os.path.join(ep_topic_dir, f"{file_stub}.gexf")
        nx.write_gexf(G_topic, gexf_path)
        print(f"Check 2")
        plot_path = os.path.join(ep_topic_dir, f"{file_stub}.png")
        plot_network(G_topic, f"EP{ep_number} - {topic_label}", output_path=plot_path)
        analyze_edge_weights(G_topic, title=f"EP{ep_number}_{topic_label}", output_dir=ep_topic_dir)
        cluster_analysis(
            G_topic,
            label=f"EP{ep_number} Topic: {topic_label}",
            output_file=os.path.join(ep_topic_dir, "cluster_analysis_topics.txt"),
        )
        print(f"Check 3")
        summaries.append(
            {
                "ep": ep_number,
                "topic": topic_label,
                "vote_count": len(topic_cols),
                "node_count": G_topic.number_of_nodes(),
                "edge_count": G_topic.number_of_edges(),
                "gexf_path": gexf_path,
            }
        )

    summary_path = None
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = os.path.join(ep_topic_dir, f"topic_summary_EP{ep_number}_{today}.csv")
        summary_df.to_csv(summary_path, index=False)
    print(f"EP{ep_number} topic analysis complete.")
    if summary_path:
        print(f"Topic summary saved to {summary_path}")



# --------------------------
# MAIN PROCESS
# --------------------------
def process_legislature(i):
    today = datetime.datetime.now().strftime("%Y%m%d")
    df0 = pd.read_csv(f"data/all_votes_main_EP{i}.csv")
    df0 = df0[df0.isna().sum(axis=1) < 0.5 * df0.shape[1]] # remove rows with more than p% missing values
    nodes = [c for c in df0.columns if c.isdigit()]

    if ENABLE_TOPIC_ANALYSIS:
        df_votes = df0[nodes]
        run_topic_networks(i, df_votes, today)
    else:
        print(f"Starting EP{i}...")
        nodes = random.sample(nodes, 100)
        df_votes = df0[nodes]
        df_dummy = make_dummy_coded_df(df_votes, allowed_values=VALID_VOTE_VALUES)
        G = make_graph_(df_dummy, df_dummy.columns)

        #make_thermo_rep(df0, G)

        filename = f"EP{i}_{today}"

        nx.write_gexf(G, f"{OUTFOLDER}/{filename}.gexf")

        file_stub = f"EP{i}_{today}"
        xx, yy = get_x_y_coordinates(G, file_stub=file_stub)
        plt.scatter(xx, yy, s=5)
        plt.savefig(f"{OUTFOLDER}/{filename}_XY.png")
        plt.close()

        plot_network(G, f"Full_network_EP{i}")
        analyze_edge_weights(G, f"Full_network_EP{i}")
        cluster_analysis(G, f"EP {i}")

    print(f"EP{i} done.")
    return f"EP{i}"


# --------------------------
# PARALLEL EXECUTION
# --------------------------
if __name__ == "__main__":
    EPs = [int(ep) for ep in EP_NUMBERS]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(process_legislature, EPs))
    print(results)
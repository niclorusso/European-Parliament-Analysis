import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain
from collections import Counter, defaultdict
import warnings
from tqdm import tqdm
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')

# -----------------------------------------------------------------------------
# Party normalization + ideological bins
# -----------------------------------------------------------------------------

PARTY_NORMALIZATION = {
    # Core abbreviations / canonical labels
    'gue_ngl': 'GUE_NGL',
    'green_efa': 'GREEN_EFA',
    'greens/efa': 'GREEN_EFA',
    'greens-efa': 'GREEN_EFA',
    'green efa': 'GREEN_EFA',
    'sd': 'SD',
    's&d': 'SD',
    'progressive alliance of socialists and democrats': 'SD',
    'group of the progressive alliance of socialists and democrats in the european parliament': 'SD',
    'socialist group in the european parliament': 'SD',
    'confederal group of the european united left - nordic green left': 'GUE_NGL',
    'group of the greens/european free alliance': 'GREEN_EFA',
    'greens/european free alliance': 'GREEN_EFA',

    'epp': 'EPP',
    "group of the european people's party (christian democrats)": 'EPP',
    "group of the european people's party (christian democrats) and european democrats": 'EPP',
    'ppe-de': 'EPP',

    'ecr': 'ECR',
    'european conservatives and reformists group': 'ECR',

    'renew': 'RENEW',
    'renew europe': 'RENEW',
    'renew europe group': 'RENEW',
    'alde': 'ALDE',
    'group of the alliance of liberals and democrats for europe': 'ALDE',
    'alliance of liberals and democrats for europe': 'ALDE',

    'ni': 'NI',
    'non-attached members': 'NI',

    'id': 'ID',
    'identity and democracy group': 'ID',

    'pfe': 'PFE',
    'esn': 'ESN',

    'enf': 'ENF',
    'europe of nations and freedom group': 'ENF',

    'efdd': 'EFDD',
    'efd': 'EFDD',
    'europe of freedom and direct democracy group': 'EFDD',
    'europe of freedom and democracy group': 'EFDD',

    'uen': 'UEN',
    'union for europe of the nations group': 'UEN',

    'ind/dem': 'IND/DEM',
    'independence/democracy group': 'IND/DEM',
}

LEFT_CANONICAL = {
    'GUE_NGL',
    'GREEN_EFA',
    'SD',
}

RIGHT_CANONICAL = {
    'EPP',
    'ECR',
    'PFE',
    'RENEW',
    'ALDE',
    'NI',
    'ESN',
    'ID',
    'ENF',
    'EFDD',
    'UEN',
    'IND/DEM',
}

EXTREME_CANONICAL = {
    'GUE_NGL',
    'GREEN_EFA',
    'PFE',
    'NI',
    'ID',
    'ESN',
    'ENF',
    'EFDD',
    'UEN',
    'IND/DEM',
}

CENTRIST_CANONICAL = {
    'SD',
    'RENEW',
    'ALDE',
    'EPP',
    'ECR',
    'GREEN_EFA',
}

MAJORITY_COALITIONS = {
    6: {'EPP', 'SD', 'ALDE'},
    7: {'EPP', 'SD', 'ALDE'},
    8: {'EPP', 'SD', 'ALDE'},
    9: {'EPP', 'SD', 'RENEW'},
    10: {'EPP', 'SD', 'RENEW'},
}


def normalize_party_label(party: str | None) -> str | None:
    """Return canonical party code from raw labels."""
    if party is None or (isinstance(party, float) and np.isnan(party)):
        return None
    party_str = str(party).strip()
    if not party_str:
        return None
    key = party_str.lower()
    return PARTY_NORMALIZATION.get(key, party_str)

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def similarity_matrix(vote_df):
    """Compute similarity matrix from vote dataframe."""
    vote_cols = [c for c in vote_df.columns if c.isdigit()]
    
    mapping = {'FOR': 1, 'AGAINST': -1, 'ABSTENTION': 0, 'DID_NOT_VOTE': np.nan}
    V = vote_df[vote_cols].replace(mapping).to_numpy(dtype=float)
    n = V.shape[0]
    
    yes = (V == 1).astype(float)
    no = (V == -1).astype(float)
    abst = (V == 0).astype(float)
    
    S = np.zeros((n, n))
    counts = np.zeros((n, n))
    
    S += yes @ yes.T + no @ no.T
    counts += yes @ yes.T + no @ no.T
    
    neg_pairs = yes @ no.T + no @ yes.T
    S -= neg_pairs
    counts += neg_pairs
    
    mixed = (yes @ abst.T) + (abst @ yes.T) + (no @ abst.T) + (abst @ no.T)
    S += -0.25 * mixed
    counts += mixed
    
    abst_pairs = abst @ abst.T
    S += 0.25 * abst_pairs
    counts += abst_pairs
    
    S = np.divide(S, counts, out=np.zeros_like(S), where=counts > 0)
    A = 0.5 * S + 0.5
    np.fill_diagonal(A, 1.0)
    
    return A


def graph_from_similarity(A, meps):
    """Convert similarity matrix to NetworkX graph."""
    G = nx.Graph()
    for m in meps:
        G.add_node(m)
    for i in range(len(meps)):
        for j in range(i + 1, len(meps)):
            w = A[i, j]
            if not np.isnan(w):
                G.add_edge(meps[i], meps[j], weight=w)
    return G


def compute_qmax(G, R=10, gamma=1.0, random_state=42, return_max=False):
    """Compute Qmax over R runs.
    
    Args:
        G: NetworkX graph
        R: Number of runs
        gamma: Resolution parameter
        random_state: Random seed
        return_max: If True, return max Qmax across runs; if False, return mean (default: False)
    
    Returns:
        (Q_value, Q_std) where Q_value is max or mean depending on return_max
    """
    Q_values = []
    for r in range(R):
        print("Run", r)
        partition = community_louvain.best_partition(
            G, weight='weight', resolution=gamma, random_state=random_state + r
        )
        Q = community_louvain.modularity(partition, G, weight='weight')
        Q_values.append(Q)
    
    if return_max:
        return np.max(Q_values), np.std(Q_values)
    else:
        return np.mean(Q_values), np.std(Q_values)


def compute_partition_modularity(G, partition_dict, weight='weight'):
    """Compute modularity for a given partition."""
    nodes_in_both = [n for n in G.nodes() if str(n) in partition_dict]
    if len(nodes_in_both) == 0:
        return np.nan
    
    G_filtered = G.subgraph(nodes_in_both).copy()
    mapping = {n: partition_dict[str(n)] for n in nodes_in_both}
    
    return community_louvain.modularity(mapping, G_filtered, weight=weight)

def create_left_right_partition(df):
    """Map parties to Left / Right / Other bins."""
    partition = {}
    if 'member.group.short_label' in df.columns and 'member.id' in df.columns:
        for idx in df.index:
            party = normalize_party_label(df.loc[idx, 'member.group.short_label'])
            member_id = str(df.loc[idx, 'member.id'])
            if party in LEFT_CANONICAL:
                partition[member_id] = 'Left'
            elif party in RIGHT_CANONICAL:
                partition[member_id] = 'Right'
            else:
                partition[member_id] = 'Other'
    return partition


def create_extreme_centrist_partition(df):
    """Extreme: (radical left/right) vs Centrist (pro-EU mainstream)."""
    partition = {}
    if 'member.group.short_label' in df.columns and 'member.id' in df.columns:
        for idx in df.index:
            party = normalize_party_label(df.loc[idx, 'member.group.short_label'])
            member_id = str(df.loc[idx, 'member.id'])
            if party in EXTREME_CANONICAL:
                partition[member_id] = 'Extreme'
            elif party in CENTRIST_CANONICAL:
                partition[member_id] = 'Centrist'
            else:
                partition[member_id] = 'Other'
    return partition


def create_left_center_right_partition(df):
    """Split parties into Left / Center / Right bins."""
    partition = {}
    if 'member.group.short_label' in df.columns and 'member.id' in df.columns:
        for idx in df.index:
            party = normalize_party_label(df.loc[idx, 'member.group.short_label'])
            member_id = str(df.loc[idx, 'member.id'])
            if party in LEFT_CANONICAL:
                partition[member_id] = 'Left'
            elif party in CENTRIST_CANONICAL:
                partition[member_id] = 'Center'
            elif party in RIGHT_CANONICAL:
                partition[member_id] = 'Right'
            else:
                partition[member_id] = 'Other'
    return partition


def create_coalition_partition(df, ep):
    """Divide MEPs into majority coalition vs opposition for a given EP."""
    majority_parties = MAJORITY_COALITIONS.get(ep)
    if not majority_parties:
        return {}

    partition = {}
    if 'member.group.short_label' in df.columns and 'member.id' in df.columns:
        for idx in df.index:
            party = normalize_party_label(df.loc[idx, 'member.group.short_label'])
            member_id = str(df.loc[idx, 'member.id'])
            if party in majority_parties:
                partition[member_id] = 'Majority'
            else:
                partition[member_id] = 'Opposition'
    return partition


def mean_num_communities_vs_gamma(G, gammas=None, R=10, seed=42, label="Agriculture"):
    """
    Computes mean number of communities (q) detected by Louvain for different γ.
    """
    if gammas is None:
        gammas = np.arange(0.9, 1.101, 0.01)
        
    results = []
    for gamma in gammas:
        q_list = []
        for r in range(R):
            print(f"Running {r} of {R} for gamma={gamma}")
            part = community_louvain.best_partition(
                G, resolution=gamma, weight='weight', random_state=seed + r
            )
            q_list.append(len(set(part.values())))
        q_mean = np.mean(q_list)
        results.append((gamma, q_mean))
        print(f"γ={gamma:.3f} → mean q={q_mean:.2f}")
    
    # ---- Plot ----
    gammas, q_means = zip(*results)
    plt.figure(figsize=(8, 5))
    plt.plot(gammas, q_means, 'o--', color='crimson', lw=1.5)
    plt.xlabel("Resolution parameter (γ)", fontsize=12)
    plt.ylabel("Mean number of communities (q)", fontsize=12)
    plt.title(f"Mean number q of communities for RCVs related to {label}, R = {R}",
              fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figs/q_vs_gamma_{label.replace(' ', '_')}.png", dpi=300)
    plt.show()
    print(f"✅ Saved q_vs_gamma_{label.replace(' ', '_')}.png")


def plot_party_fraction_per_community(G, df_policy, gamma=1.02, label="Gender equality"):
    """
    Plot fraction of each party's MEPs that belong to each community (Fig. 8.18 style).

    Parameters
    ----------
    G : networkx.Graph
        Similarity or adjacency network of MEPs
    df_policy : pd.DataFrame
        Must contain 'member.id' and 'member.group.short_label'
    gamma : float
        Louvain resolution parameter
    label : str
        Policy area name for title and filename
    """

    # --- Louvain community detection ---
    partition = community_louvain.best_partition(G, resolution=gamma, weight='weight', random_state=42)
    df_policy = df_policy.copy()
    df_policy["community"] = df_policy["member.id"].astype(str).map(partition)

    # --- Count MEPs per party & community ---
    counts = pd.crosstab(df_policy["member.group.short_label"], df_policy["community"])

    # --- Normalize by total MEPs in each PARTY (row-wise) ---
    fractions = counts.div(counts.sum(axis=1), axis=0).fillna(0)

    # --- Plot ---
    plt.figure(figsize=(9, 6))
    colors = sns.color_palette("tab10", n_colors=len(fractions.index))
    fractions.T.plot(kind="bar", color=colors, width=0.8)

    plt.xlabel("Community")
    plt.ylabel("Fraction of MEPs in community (per party)")
    plt.title(
        f"Fraction of MEPs in each community for RCVs related to {label}, γ = {gamma}",
        fontsize=12,
        fontweight="bold"
    )
    plt.legend(
        title="Party",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
        frameon=False
    )
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"figs/party_fraction_per_community_intra_party_{label.replace(' ', '_')}.png", dpi=300)
    plt.show()

    print(f"✅ Saved party_fraction_per_community_intra_party_{label.replace(' ', '_')}.png")


def process_policy_area(args):
    """Runs modularity computation for one policy area (for parallel execution)."""
    ep_number, policy_area, vote_ids, dff, schema, R, gamma = args
    try:
        vote_cols = [c for c in dff.columns if c.isdigit()]
        policy_vote_cols = [c for c in vote_cols if c in vote_ids]
        if len(policy_vote_cols) < 10:###############################################
            return None

        votes_policy = dff[policy_vote_cols].copy()
        valid_rows = votes_policy.isna().sum(axis=1) < 0.7 * len(votes_policy.columns)
        votes_policy = votes_policy.loc[valid_rows]
        df_policy = dff.loc[valid_rows].copy()

        if len(votes_policy) < 10:#################################################
            return None

        A = similarity_matrix(votes_policy)
        meps = df_policy[schema["member_id"]].astype(str).tolist()
        G = graph_from_similarity(A, meps)

        isolated = list(nx.isolates(G))
        if isolated:
            G.remove_nodes_from(isolated)
            df_policy = df_policy[~df_policy[schema["member_id"]].astype(str).isin(isolated)].copy()

        if G.number_of_nodes() < 10:
            return None

        # --- Modularities ---
        Qmax_mean, Qmax_std = compute_qmax(G, R=R, gamma=gamma)
        party_partition = {str(df_policy.loc[i, schema["member_id"]]): str(df_policy.loc[i, "member.group.short_label"])
                           for i in df_policy.index if "member.group.short_label" in df_policy.columns}
        Qparty = compute_partition_modularity(G, party_partition)

        country_partition = {str(df_policy.loc[i, schema["member_id"]]): str(df_policy.loc[i, schema["country"]])
                             for i in df_policy.index}
        Qcountry = compute_partition_modularity(G, country_partition)

        lr_partition = create_left_right_partition(df_policy)
        Q_left_right = compute_partition_modularity(G, lr_partition)

        lcr_partition = create_left_center_right_partition(df_policy)
        Q_left_center_right = compute_partition_modularity(G, lcr_partition)

        ec_partition = create_extreme_centrist_partition(df_policy)
        Q_extreme_centrist = compute_partition_modularity(G, ec_partition)

        coalition_partition = create_coalition_partition(df_policy, ep_number)
        Q_coalition = compute_partition_modularity(G, coalition_partition) if coalition_partition else np.nan

        rand_partition = create_random_partition(G)
        Q_random = compute_partition_modularity(G, rand_partition)

        return {
            "Policy Area": policy_area,
            "Qmax": Qmax_mean,
            "Qmax_std": Qmax_std,
            "Qparty": Qparty,
            "Qcountry": Qcountry,
            "Q_left_right": Q_left_right,
            "Q_left_center_right": Q_left_center_right,
            "Q_extreme_centrist": Q_extreme_centrist,
            "Q_majority_opposition": Q_coalition,
            "Q_random": Q_random,
            "Qparty/Qmax": Qparty / Qmax_mean if Qmax_mean > 0 else np.nan,
            "Qcountry/Qmax": Qcountry / Qmax_mean if Qmax_mean > 0 else np.nan,
            "Q_left_right/Qmax": Q_left_right / Qmax_mean if Qmax_mean > 0 else np.nan,
            "Q_left_center_right/Qmax": Q_left_center_right / Qmax_mean if Qmax_mean > 0 else np.nan,
            "Q_majority_opposition/Qmax": Q_coalition / Qmax_mean if Qmax_mean and not np.isnan(Q_coalition) else np.nan,
            "Q_random/Qmax": Q_random / Qmax_mean if Qmax_mean > 0 else np.nan,
            "N_votes": len(policy_vote_cols),
            "N_MEPs": G.number_of_nodes()
        }
    except Exception as e:
        print(f"⚠️ Error in {policy_area}: {e}")
        return None

def create_random_partition(G, seed=42):
    """
    Create a random partition assigning nodes to random communities.
    The number of communities equals the number of parties in the current graph.
    """
    np.random.seed(seed)
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    # Pick a small random number of groups (2–10)
    n_groups = np.random.randint(2, min(10, n_nodes))
    random_labels = np.random.choice(range(n_groups), size=n_nodes)
    return {node: str(label) for node, label in zip(nodes, random_labels)}

# ======================================================================
# CONFIG
# ======================================================================

EP_LIST = [6, 7, 8, 9, 10]
R = 10
gamma = 1.0

SCHEMA = {
    6: {"vote_id": "euro_act_id", "policy": "main_policy_name", "member_id": "member.id", "country": "member.country.label"},
    7: {"vote_id": "Vote ID", "policy": "De", "member_id": "member.id", "country": "member.country.label"},
    8: {"vote_id": "Vote ID", "policy": "De", "member_id": "member.id", "country": "member.country.label"},
    9: {"vote_id": "id", "policy": "committees", "member_id": "member.id", "country": "member.country.code"},
    10: {"vote_id": "id", "policy": "committees", "member_id": "member.id", "country": "member.country.code"},
}

# ======================================================================
# UTILS
# ======================================================================

def create_random_partition(G, seed=42):
    np.random.seed(seed)
    return {str(n): np.random.choice(["A", "B"]) for n in G.nodes()}

# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    EP_LIST = [6, 7, 8, 9, 10]  # ← or whichever EPs you want to process

    for EP_NUMBER in EP_LIST:
        print(f"\n{'='*80}\n🧩 Running modularity analysis for EP{EP_NUMBER}\n{'='*80}")

        schema = SCHEMA[EP_NUMBER]
        DATASET_PATH = f"data/all_votes_main_EP{EP_NUMBER}.csv"
        VOTE_SUBJECTS_PATH = f"data/votewatch_csv/EP{EP_NUMBER}_Voted main docs.csv"

        try:
            vote_subjects = pd.read_csv(VOTE_SUBJECTS_PATH)
            dff = pd.read_csv(DATASET_PATH, low_memory=False)
        except FileNotFoundError:
            print(f"⚠️ Missing data for EP{EP_NUMBER}, skipping.")
            continue

        dff = dff[dff.isna().sum(axis=1) < 0.5 * len(dff.columns)]

        # --- Extract policy areas ---
        policy_area_map = defaultdict(list)
        for _, row in vote_subjects.iterrows():
            vote_id = str(row.get(schema["vote_id"], ""))
            subjects = str(row.get(schema["policy"], ""))
            if subjects and subjects != "nan":
                for area in [s.strip() for s in subjects.split(";") if s.strip()]:
                    policy_area_map[area].append(vote_id)

        print(f"Found {len(policy_area_map)} policy areas for EP{EP_NUMBER}")

        vote_cols = [c for c in dff.columns if c.isdigit()]
        if not vote_cols:
            print(f"⚠️ No numeric vote columns in EP{EP_NUMBER}, skipping.")
            continue

        min_votes = len(vote_cols) / 3000#################################################
        policy_areas_filtered = {
            a: v for a, v in policy_area_map.items() if len(v) >= min_votes
        }

        print(f"Analyzing {len(policy_areas_filtered)} policy areas (≥{min_votes:.0f} votes)\n")

        # --- Prepare tasks ---
        tasks = [
            (EP_NUMBER, policy_area, vote_ids, dff, schema, R, gamma)
            for policy_area, vote_ids in policy_areas_filtered.items()
        ]

        num_workers = max(1, multiprocessing.cpu_count() - 1)
        print(f"🚀 Launching pool with {num_workers} workers...")
        results = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_policy_area, t) for t in tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"EP{EP_NUMBER} areas"):
                res = f.result()
                if res:
                    results.append(res)

        if not results:
            print(f"⚠️ No valid results for EP{EP_NUMBER}. Skipping plot.")
            continue

        df_results = pd.DataFrame(results).sort_values("Qmax", ascending=False)
        df_results.to_csv(f"results/modularity_results_EP{EP_NUMBER}.csv", index=False)
        print(f"✅ Results saved to results/modularity_results_EP{EP_NUMBER}.csv")

        # # --- Plot ---
        # df_plot = df_results.copy()
        # x = np.arange(len(df_plot))
        # width = 0.12

        # fig, ax = plt.subplots(figsize=(16, 6))
        # ax.bar(x - 2.5*width, df_plot["Qmax"], width, color="#fee090", label="Maximum modularity")
        # ax.bar(x - 1.5*width, df_plot["Q_extreme_centrist"], width, color="#fc8d59", label="Extreme–Centrist modularity")
        # ax.bar(x - 0.5*width, df_plot["Qparty"], width, color="#d73027", label="Party modularity")
        # ax.bar(x + 0.5*width, df_plot["Q_left_right"], width, color="#91bfdb", label="Left–Right modularity")
        # ax.bar(x + 1.5*width, df_plot["Qcountry"], width, color="#4575b4", label="Country modularity")
        # ax.bar(x + 2.5*width, df_plot["Q_random"], width, color="#999999", label="Random modularity")

        # ax.set_xticks(x)
        # ax.set_xticklabels(df_plot["Policy Area"], rotation=45, ha="right", fontsize=9)
        # ax.set_ylabel("Q", fontsize=12, fontweight="bold")
        # ax.set_title(f"Modularity values by policy area: EP{EP_NUMBER}", fontsize=14, fontweight="bold")
        # ax.legend(frameon=False, loc="upper right", fontsize=9)
        # ax.grid(axis="y", alpha=0.3)

        # plt.tight_layout()
        # plt.savefig(f"figs/modularity_by_policy_area_EP{EP_NUMBER}.png", dpi=300, bbox_inches="tight")
        # plt.close()

        # print(f"✅ Plot saved as figs/modularity_by_policy_area_EP{EP_NUMBER}.png")

    print("\n🎯 All legislatures processed successfully.")
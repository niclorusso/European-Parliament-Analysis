"""
Modularity computation functions.
Standalone module for computing various modularity metrics.
"""

import numpy as np
import pandas as pd
import networkx as nx

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("⚠️  python-louvain not installed. Modularity computation will fail.")


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


def compute_qmax(G: nx.Graph, R: int = 10, gamma: float = 1.0, random_state: int = 42, return_max: bool = False) -> tuple[float, float]:
    """
    Compute Qmax over R runs.
    
    Args:
        G: NetworkX graph
        R: Number of runs
        gamma: Resolution parameter
        random_state: Random seed
        return_max: If True, return max Qmax across runs; if False, return mean (default: False)
    
    Returns:
        (Q_value, Q_std) where Q_value is max or mean depending on return_max
    """
    if not HAS_LOUVAIN:
        raise ImportError("python-louvain is required for modularity computation")
    
    Q_values = []
    for r in range(R):
        partition = community_louvain.best_partition(
            G, weight='weight', resolution=gamma, random_state=random_state + r
        )
        Q = community_louvain.modularity(partition, G, weight='weight')
        Q_values.append(Q)
    
    if return_max:
        return np.max(Q_values), np.std(Q_values)
    else:
        return np.mean(Q_values), np.std(Q_values)


def compute_partition_modularity(G: nx.Graph, partition_dict: dict, weight: str = 'weight') -> float:
    """Compute modularity for a given partition."""
    nodes_in_both = [n for n in G.nodes() if str(n) in partition_dict]
    if len(nodes_in_both) == 0:
        return np.nan
    
    G_filtered = G.subgraph(nodes_in_both).copy()
    mapping = {n: partition_dict[str(n)] for n in nodes_in_both}
    
    return community_louvain.modularity(mapping, G_filtered, weight=weight)


def create_left_right_partition(df: pd.DataFrame) -> dict:
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


def create_extreme_centrist_partition(df: pd.DataFrame) -> dict:
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


def create_left_center_right_partition(df: pd.DataFrame) -> dict:
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


def create_coalition_partition(df: pd.DataFrame, ep: int) -> dict:
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


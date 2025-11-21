"""
Utility functions for modularity analysis.
Computes similarity matrices and builds graphs from raw vote data.
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


# Schema for different EP data formats
SCHEMA = {
    6: {
        "vote_id": "Vote ID",
        "policy": "main_policy_name",
        "member_id": "member.id",
        "country": "member.country.label",
        "party": "member.group.short_label",
    },
    7: {
        "vote_id": "Vote ID",
        "policy": "De",
        "member_id": "member.id",
        "country": "member.country.label",
        "party": "member.group.short_label",
    },
    8: {
        "vote_id": "Vote ID",
        "policy": "De",
        "member_id": "member.id",
        "country": "member.country.label",
        "party": "member.group.short_label",
    },
    9: {
        "vote_id": "id",
        "policy": "committees",
        "member_id": "member.id",
        "country": "member.country.code",
        "party": "member.group.short_label",
    },
    10: {
        "vote_id": "id",
        "policy": "committees",
        "member_id": "member.id",
        "country": "member.country.code",
        "party": "member.group.short_label",
    },
}


def similarity_matrix(vote_df: pd.DataFrame) -> np.ndarray:
    """
    Compute similarity matrix from vote dataframe (MEPs x Votes).
    
    Agreement scoring:
    - FOR-FOR or AGAINST-AGAINST: +1
    - FOR-AGAINST or AGAINST-FOR: -1
    - Mixed with ABSTENTION: -0.25
    - ABSTENTION-ABSTENTION: +0.25
    
    Returns normalized similarity matrix [0, 1] where 1 = perfect agreement.
    """
    vote_cols = [c for c in vote_df.columns if c.isdigit() or c.replace(".0", "").isdigit()]
    
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
    
    yes = (V == 1).astype(float)
    no = (V == -1).astype(float)
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
    
    # Diagonal = 1
    np.fill_diagonal(A, 1.0)
    
    # Square to emphasize agreements
    A = pow(A, 2)
    
    return A


def graph_from_similarity(A: np.ndarray, mep_ids: List[str], min_weight: float = 0.01) -> nx.Graph:
    """
    Build NetworkX graph from similarity matrix.
    
    Args:
        A: Similarity matrix (n x n)
        mep_ids: List of MEP IDs (length n)
        min_weight: Minimum edge weight threshold
    
    Returns:
        NetworkX graph with nodes as MEPs and edges weighted by similarity
    """
    G = nx.Graph()
    
    # Add nodes
    for mep_id in mep_ids:
        G.add_node(str(mep_id))
    
    # Add edges above threshold
    n = len(mep_ids)
    for i in range(n):
        for j in range(i + 1, n):
            weight = A[i, j]
            if weight >= min_weight:
                G.add_edge(str(mep_ids[i]), str(mep_ids[j]), weight=weight)
    
    return G


def load_vote_data(ep: int) -> Optional[pd.DataFrame]:
    """Load vote data for a specific EP from data folder."""
    vote_matrix_path = Path(f"data/all_votes_main_EP{ep}.csv")
    
    if not vote_matrix_path.exists():
        print(f"  ⚠️  Vote matrix not found: {vote_matrix_path}")
        return None
    
    try:
        df = pd.read_csv(vote_matrix_path, low_memory=False)
        return df
    except Exception as e:
        print(f"  ⚠️  Error loading vote data: {e}")
        return None


def load_topic_metadata(ep: int) -> Optional[pd.DataFrame]:
    """Load topic metadata for a specific EP from data folder."""
    metadata_path = Path(f"data/votewatch_csv/EP{ep}_Voted main docs.csv")
    
    if not metadata_path.exists():
        print(f"  ⚠️  Topic metadata not found: {metadata_path}")
        return None
    
    try:
        df = pd.read_csv(metadata_path, low_memory=False)
        return df
    except Exception as e:
        print(f"  ⚠️  Error loading topic metadata: {e}")
        return None


def get_topic_vote_map(ep: int) -> Dict[str, List[str]]:
    """
    Build a map from topic names to vote IDs for a specific EP.
    
    Returns:
        Dictionary mapping topic names (normalized) to lists of vote IDs
    """
    df_meta = load_topic_metadata(ep)
    if df_meta is None:
        return {}
    
    schema = SCHEMA[ep]
    vote_id_col = schema["vote_id"]
    policy_col = schema["policy"]
    
    # For EP9/EP10, try both "committees" and "policy_area" (harmonized files use policy_area)
    if policy_col not in df_meta.columns:
        if ep in (9, 10) and "policy_area" in df_meta.columns:
            policy_col = "policy_area"
        else:
            return {}
    
    if vote_id_col not in df_meta.columns:
        return {}
    
    topic_vote_map = {}
    
    for _, row in df_meta.iterrows():
        vote_id = str(row[vote_id_col])
        policy_str = str(row[policy_col]) if pd.notna(row[policy_col]) else ""
        
        # Handle semicolon-separated topics
        topics = [t.strip() for t in policy_str.split(";") if t.strip()]
        
        for topic in topics:
            # Normalize topic name (lowercase, replace spaces/special chars)
            topic_normalized = topic.lower().strip()
            if topic_normalized:
                if topic_normalized not in topic_vote_map:
                    topic_vote_map[topic_normalized] = []
                topic_vote_map[topic_normalized].append(vote_id)
    
    return topic_vote_map


def filter_votes_by_topic(df_votes: pd.DataFrame, vote_ids: List[str], ep: int) -> pd.DataFrame:
    """
    Filter vote dataframe to only include votes for a specific topic.
    
    Note: In the vote matrix, votes are COLUMNS (numeric column names), not rows.
    So we filter by selecting only the vote columns that match the vote IDs,
    while keeping all metadata columns.
    """
    # Convert vote_ids to strings and create a set for fast lookup
    vote_ids_set = set(str(vid) for vid in vote_ids)
    # Also add normalized versions (remove .0 suffix)
    vote_ids_set_normalized = set(str(vid).replace(".0", "") for vid in vote_ids)
    vote_ids_set.update(vote_ids_set_normalized)
    
    # Get all columns
    all_cols = df_votes.columns.tolist()
    
    # Separate metadata columns from vote columns
    # Vote columns are numeric (as strings) or can be converted to numeric
    metadata_cols = []
    vote_cols = []
    
    for col in all_cols:
        col_str = str(col)
        # Check if it's a vote column (numeric)
        is_vote_col = (col_str.isdigit() or 
                      col_str.replace(".0", "").isdigit() or
                      (col_str.replace(".", "").isdigit() and "." in col_str))
        
        if is_vote_col:
            # Check if this vote column matches a topic vote ID
            col_normalized = col_str.replace(".0", "")
            if col_str in vote_ids_set or col_normalized in vote_ids_set:
                vote_cols.append(col)
        else:
            # It's a metadata column, keep it
            metadata_cols.append(col)
    
    if not vote_cols:
        return pd.DataFrame()
    
    # Return dataframe with metadata columns + filtered vote columns
    df_filtered = df_votes[metadata_cols + vote_cols].copy()
    
    return df_filtered


def extract_mep_metadata(df_votes: pd.DataFrame, ep: int) -> pd.DataFrame:
    """Extract MEP metadata from vote dataframe."""
    schema = SCHEMA[ep]
    
    # Get vote columns (numeric)
    vote_cols = [c for c in df_votes.columns if c.isdigit() or c.replace(".0", "").isdigit()]
    
    # Get MEP metadata columns
    metadata_cols = [schema["member_id"]]
    if schema.get("party") and schema["party"] in df_votes.columns:
        metadata_cols.append(schema["party"])
    if schema.get("country") and schema["country"] in df_votes.columns:
        metadata_cols.append(schema["country"])
    
    # Create metadata dataframe (one row per MEP)
    # MEPs are rows in the vote dataframe
    df_meta = df_votes[metadata_cols].copy()
    df_meta = df_meta.rename(columns={
        schema["member_id"]: "member.id"
    })
    
    # Add party and country columns with standardized names
    if schema.get("party") and schema["party"] in df_votes.columns:
        df_meta["member.group.short_label"] = df_votes[schema["party"]]
    if schema.get("country") and schema["country"] in df_votes.columns:
        df_meta["member.country.label"] = df_votes[schema["country"]]
        df_meta["member.country.code"] = df_votes[schema["country"]]
    
    return df_meta


def normalize_topic_name(topic: str) -> str:
    """Normalize topic name for matching (case-insensitive, handle variations)."""
    # Convert to lowercase
    topic = topic.lower().strip()
    
    # Replace common variations
    topic = topic.replace("&", "and")
    topic = topic.replace("'", "'")  # Normalize apostrophes
    topic = re.sub(r'\s+', ' ', topic)  # Normalize whitespace
    
    return topic


def match_topic(topic_query: str, available_topics: List[str]) -> List[str]:
    """
    Find matching topics (case-insensitive, partial match).
    
    Returns list of matching topic names from available_topics.
    """
    query_normalized = normalize_topic_name(topic_query)
    matches = []
    
    for topic in available_topics:
        topic_normalized = normalize_topic_name(topic)
        if query_normalized in topic_normalized or topic_normalized in query_normalized:
            matches.append(topic)
    
    return matches


import streamlit as st
import pandas as pd
from itertools import combinations
import random # Used for mock data generation

# Global Constants/Setup (Mocked for demonstration)
CLUSTER_COL = 'cluster_id' 
ENTITY_COL = 'entity_name'
FRAME_COL = 'frames' 

st.set_page_config(layout="wide", page_title="Media Intelligence Divergence Fix")

# --- Utility Functions (Mocked for demonstration) ---

def generate_mock_data(n_rows=50):
    """Generates mock data resembling media framing analysis."""
    data = {
        ENTITY_COL: [f'Entity_{i % 10}' for i in range(n_rows)],
        CLUSTER_COL: [f'C{i % 3}' for i in range(n_rows)],
        FRAME_COL: [random.sample(['economic', 'social', 'political', 'health', 'environmental'], k=random.randint(1, 3)) for _ in range(n_rows)],
        'article_id': [i for i in range(n_rows)],
    }
    return pd.DataFrame(data)

# --- CORRECTED FUNCTION (Relevant block for line 390) ---

def framing_divergence(df: pd.DataFrame, sel_cid2: str) -> pd.DataFrame:
    """
    Calculates the Jaccard divergence (distance) between frame sets
    of different entities within a selected cluster.

    The original code failed because when 'pairs' was empty (e.g., if sel_cid2 
    filtered out all data), pd.DataFrame(pairs) had no columns, leading to a 
    KeyError when attempting to sort by the 'jaccard' column.
    
    The fix is to check if 'pairs' is empty and, if so, return a blank 
    DataFrame with the expected columns defined.
    """
    st.subheader(f"Framing Divergence for Cluster: {sel_cid2}")

    # 1. Filter data based on selected cluster ID
    filtered_df = df[df[CLUSTER_COL] == sel_cid2]

    if filtered_df.empty:
        st.info(f"Selected cluster '{sel_cid2}' has no data to analyze.")
        # Return an empty DataFrame with defined columns to prevent errors downstream
        return pd.DataFrame([], columns=["entity_a", "entity_b", "jaccard"])

    # 2. Aggregate unique frames for each entity in the cluster
    entity_frames = filtered_df.groupby(ENTITY_COL)[FRAME_COL].apply(
        lambda x: set(frame for sublist in x for frame in sublist)
    ).to_dict()

    entity_names = list(entity_frames.keys())
    pairs = []

    # 3. Calculate Jaccard similarity for all unique pairs of entities
    for entity_a, entity_b in combinations(entity_names, 2):
        set_a = entity_frames[entity_a]
        set_b = entity_frames[entity_b]
        
        intersection_size = len(set_a.intersection(set_b))
        union_size = len(set_a.union(set_b))
        
        # Calculate Jaccard Similarity (1 is max similarity, 0 is min)
        jaccard_similarity = intersection_size / union_size if union_size > 0 else 0
        
        pairs.append({
            "entity_a": entity_a,
            "entity_b": entity_b,
            "jaccard": jaccard_similarity, # <-- CRITICAL: Key must match sorting name
        })

    # --- Line 390 in the original context (now corrected logic) ---
    if not pairs:
        # This check is vital. It handles the case where there is only one 
        # entity in the cluster, so no combinations were created.
        st.info("Less than two entities found in this cluster. Cannot calculate divergence.")
        return pd.DataFrame([], columns=["entity_a", "entity_b", "jaccard"]) 

    # CORRECTED LINE 390 (The sort_values call now works reliably)
    return pd.DataFrame(pairs).sort_values("jaccard", ascending=False)
# --- End of CORRECTED FUNCTION ---


# --- Streamlit Main App Logic ---

# 1. Load Data
df = generate_mock_data()
available_clusters = df[CLUSTER_COL].unique()

st.title("📊 Media Intelligence: Framing Divergence Analysis")
st.markdown("This app calculates the Jaccard similarity between the unique frames used by different entities within a selected cluster. High 'jaccard' score means low divergence.")

# 2. Sidebar/Selection
st.sidebar.header("Controls")
sel_cid2 = st.sidebar.selectbox(
    "Select Cluster ID to analyze:",
    options=available_clusters,
    index=0
)

# 3. Run Analysis (This is where line 660 occurs)
try:
    # This calls framing_divergence, which now returns a safe DataFrame
    fd = framing_divergence(df, sel_cid2) 
    
    st.markdown("---")
    st.subheader("Results: Entity Divergence (by Jaccard Similarity)")
    st.dataframe(fd, use_container_width=True)

    if not fd.empty:
        # Display the most similar pair (Highest Jaccard score)
        most_similar = fd.iloc[0]
        st.metric(
            "Most Similar Entities (Lowest Divergence)", 
            f"{most_similar['entity_a']} & {most_similar['entity_b']}", 
            f"Jaccard: {most_similar['jaccard']:.3f}"
        )
    
except KeyError as e:
    # Fallback error handling for any unexpected KeyErrors
    st.error(f"An internal error occurred: Missing column {e}. The divergence calculation failed.")
    st.warning("Please ensure your data contains the expected columns for grouping and frames.")

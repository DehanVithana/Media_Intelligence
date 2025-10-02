import streamlit as st
import pandas as pd
import altair as alt

# --- Configuration ---
st.set_page_config(
    page_title="LK News Data Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    """Loads and preprocesses the news data."""
    try:
        # NOTE: Replace 'lk_news.csv' with the actual file name if it differs
        df = pd.read_csv('lk_news.csv')

        # Convert 'time' to datetime, assuming it's the primary date column
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df.dropna(subset=['time'], inplace=True)
        else:
            st.error("Data missing required 'time' column.")
            return pd.DataFrame() # Return empty DataFrame on failure

        # Basic cleanup for 'source' and 'topic'
        if 'source' not in df.columns:
            # Placeholder for 'source' if missing, but it's a critical feature
            df['source'] = 'Unknown Source'
        if 'topic' not in df.columns:
            # Placeholder for 'topic' if missing (real implementation would use NLP)
            df['topic'] = 'Uncategorized'

        return df
    except FileNotFoundError:
        st.error("Error: 'lk_news.csv' not found. Please ensure the data file is in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return pd.DataFrame()

df = load_data()

# --- Streamlit Dashboard Layout ---
st.title("🇱🇰 Sri Lanka News Dataset Analysis")

if not df.empty:
    st.sidebar.header("Filter & Settings")

    # Date Range Filter
    min_date = df['time'].min().date()
    max_date = df['time'].max().date()
    date_range = st.sidebar.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    df_filtered = df[
        (df['time'].dt.date >= date_range[0]) &
        (df['time'].dt.date <= date_range[1])
    ]

    st.markdown(f"**Total Articles Analyzed:** {len(df_filtered):,}")
    st.markdown("---")

    col1, col2 = st.columns(2)

    # =========================================================================
    # METRIC 1: News Volume Over Time
    # =========================================================================
    with col1:
        st.subheader("📰 News Volume Over Time")
        
        # Aggregate by day
        volume_df = df_filtered.set_index('time').resample('D').size().reset_index(name='Article_Count')

        chart_volume = alt.Chart(volume_df).mark_line(point=True).encode(
            x=alt.X('time', title='Date'),
            y=alt.Y('Article_Count', title='Article Count'),
            tooltip=[alt.Tooltip('time', format='%Y-%m-%d', title='Date'), 'Article_Count']
        ).properties(
            title="Daily Article Count"
        ).interactive()

        st.altair_chart(chart_volume, use_container_width=True)


    # =========================================================================
    # METRIC 2: Source Coverage Distribution
    # =========================================================================
    with col2:
        st.subheader("🗞️ Article Distribution by Source")

        # Aggregate by source
        source_df = df_filtered['source'].value_counts().reset_index(name='Article_Count')
        source_df.columns = ['Source', 'Article_Count']
        
        # Allow user to select top N sources
        top_n_sources = st.sidebar.slider("Top N Sources to Display", 5, 20, 10)
        source_df = source_df.head(top_n_sources)
        
        chart_source = alt.Chart(source_df).mark_bar().encode(
            x=alt.X('Article_Count', title='Article Count'),
            y=alt.Y('Source', sort='-x', title='News Source'),
            tooltip=['Source', 'Article_Count']
        ).properties(
            title=f"Top {top_n_sources} News Sources by Volume"
        )
        
        st.altair_chart(chart_source, use_container_width=True)
    
    st.markdown("---")

    # =========================================================================
    # METRIC 3: Top Topic Distribution (requires 'topic' column)
    # =========================================================================
    st.subheader("🔥 Top Topic Distribution")
    
    # Aggregate by topic
    topic_df = df_filtered['topic'].value_counts().reset_index(name='Article_Count')
    topic_df.columns = ['Topic', 'Article_Count']
    
    top_n_topics = st.slider("Top N Topics to Display", 5, 20, 10)
    topic_df = topic_df.head(top_n_topics)

    chart_topic = alt.Chart(topic_df).mark_bar().encode(
        x=alt.X('Article_Count', title='Article Count'),
        y=alt.Y('Topic', sort='-x', title='Topic'),
        tooltip=['Topic', 'Article_Count'],
        color=alt.Color('Topic', legend=None)
    ).properties(
        title=f"Top {top_n_topics} Topics by Article Count"
    )

    st.altair_chart(chart_topic, use_container_width=True)

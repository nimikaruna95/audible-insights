import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from processing_data.recommendation_system import (
        recommend_books, genre_based, get_hidden_gems,
        science_fiction_books
    )
except Exception as e:
    st.error("Unable to load the recommendation system.")
    st.exception(e)
    st.stop()

st.set_page_config(
    page_title="Audible Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-title {font-size:42px;font-weight:700;margin-bottom:0}
.subtitle {font-size:18px;color:#666;margin-top:0;margin-bottom:25px}
.recommendation-card {padding:15px;border-radius:10px;border:1px solid #ddd;margin-bottom:12px}
.section-title {margin-top:10px}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">📚 Audible Insights</div>',
    unsafe_allow_html=True
)
st.markdown("""
<div class="subtitle">
Intelligent Book Recommendation System using NLP,
TF-IDF, K-Means Clustering and Hybrid Recommendation.
</div>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join(BASE_DIR, "data", "clustered_books.csv")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found:\n{DATA_PATH}")
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
except Exception as e:
    st.error("Unable to load clustered dataset.")
    st.exception(e)
    st.stop()

required_columns = [
    "Book Name", "Author", "Genre", "Rating", "Number of Reviews"
]
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error("Missing required columns: " + ", ".join(missing_columns))
    st.stop()

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df["Number of Reviews"] = pd.to_numeric(
    df["Number of Reviews"], errors="coerce"
)

if "Weighted_Rating" in df.columns:
    df["Weighted_Rating"] = pd.to_numeric(
        df["Weighted_Rating"], errors="coerce"
    )

for column in ["Author", "Genre", "Book Name"]:
    df[column] = df[column].fillna("Unknown").astype(str)

st.sidebar.title("Dashboard Filters")
st.sidebar.markdown("Use the filters below to explore the book dataset.")

min_rating = st.sidebar.slider(
    "Minimum Rating", 0.0, 5.0, 4.0, 0.1
)

genre_values = {
    genre.strip()
    for value in df["Genre"].dropna()
    for genre in str(value).split("|")
    if genre.strip()
}
genres = sorted(genre_values)

selected_genre = st.sidebar.selectbox(
    "Genre", ["All"] + genres
)

if "Cluster" in df.columns:
    cluster_values = sorted(df["Cluster"].dropna().unique())
    selected_cluster = st.sidebar.selectbox(
        "Cluster", ["All"] + list(cluster_values)
    )
else:
    selected_cluster = "All"

filtered_df = df[df["Rating"] >= min_rating].copy()

if selected_genre != "All":
    filtered_df = filtered_df[
        filtered_df["Genre"].str.contains(
            selected_genre, case=False, na=False, regex=False
        )
    ]

if selected_cluster != "All" and "Cluster" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["Cluster"] == selected_cluster
    ]

if filtered_df.empty:
    st.warning("No books match the selected filters.")
    st.stop()

st.subheader("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Books", f"{len(filtered_df):,}")
col2.metric("Authors", f"{filtered_df['Author'].nunique():,}")

filtered_genres = {
    genre.strip()
    for value in filtered_df["Genre"].dropna()
    for genre in str(value).split("|")
    if genre.strip()
}
col3.metric("Genres", f"{len(filtered_genres):,}")
col4.metric("Average Rating", round(filtered_df["Rating"].mean(), 2))

st.divider()
st.subheader("Dataset Analysis")

left, right = st.columns(2)

with left:
    fig = px.histogram(
        filtered_df, x="Rating", nbins=20,
        title="Rating Distribution", labels={"Rating": "Rating"}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with right:
    genre_count = {}
    for value in filtered_df["Genre"].dropna():
        for genre in str(value).split("|"):
            genre = genre.strip()
            if genre:
                genre_count[genre] = genre_count.get(genre, 0) + 1

    genre_count = pd.Series(genre_count).sort_values(
        ascending=False
    ).head(10)

    fig = px.bar(
        x=genre_count.index,
        y=genre_count.values,
        labels={"x": "Genre", "y": "Books"},
        title="Top 10 Genres"
    )
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Rating vs Review Count")

fig = px.scatter(
    filtered_df,
    x="Number of Reviews",
    y="Rating",
    hover_name="Book Name",
    hover_data=["Author", "Genre"],
    title="Ratings vs Number of Reviews"
)
st.plotly_chart(fig, use_container_width=True)

if "Cluster" in filtered_df.columns:
    st.subheader("Cluster Distribution")
    cluster_counts = (
        filtered_df["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    cluster_counts.columns = ["Cluster", "Books"]

    fig = px.pie(
        cluster_counts,
        names="Cluster",
        values="Books",
        title="Books by K-Means Cluster"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Top Authors")

top_authors = (
    filtered_df.groupby("Author")
    .agg(
        Average_Rating=("Rating", "mean"),
        Books=("Book Name", "count")
    )
    .sort_values("Average_Rating", ascending=False)
    .head(10)
    .reset_index()
)

fig = px.bar(
    top_authors,
    x="Author",
    y="Average_Rating",
    hover_data=["Books"],
    title="Top Rated Authors"
)
fig.update_xaxes(tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Hybrid Book Recommendation")

st.write("""
Select a book and the recommendation engine will combine
content similarity, cluster membership and weighted rating
to produce recommendations.
""")

book_list = sorted(df["Book Name"].dropna().unique())
selected_book = st.selectbox("Choose a Book", book_list)

if st.button("Recommend Similar Books", type="primary"):
    with st.spinner("Generating recommendations..."):
        try:
            recommendations = recommend_books(selected_book, top_n=5)
        except Exception as e:
            st.error("Unable to generate recommendations.")
            st.exception(e)
            recommendations = pd.DataFrame()

    if recommendations.empty:
        st.warning("No recommendations found.")
    else:
        st.success(f"Recommendations for: {selected_book}")

        for rank, (_, row) in enumerate(
            recommendations.iterrows(), start=1
        ):
            book_name = row.get("Book Name", "Unknown")
            author = row.get("Author", "Unknown")
            genre = row.get("Genre", "Unknown")
            rating = row.get("Rating", 0)
            similarity = row.get(
                "Similarity", row.get("Content_Score", 0)
            )
            weighted_rating = row.get("Weighted_Rating", rating)
            hybrid_score = row.get("Hybrid_Score", 0)

            st.markdown(f"""
### {rank}.  {book_name}

**Author:** {author}  
**Genre:** {genre}  
**Rating:** {rating:.2f}  
**Weighted Rating:** {weighted_rating:.4f}  
**Content Similarity:** {similarity:.4f}  
**Hybrid Score:** **{hybrid_score:.4f}**
""")
            st.divider()

        display_columns = [
            "Book Name", "Author", "Genre", "Rating",
            "Number of Reviews", "Similarity",
            "Weighted_Rating", "Hybrid_Score"
        ]
        display_columns = [
            c for c in display_columns if c in recommendations.columns
        ]

        st.dataframe(
            recommendations[display_columns],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            " Download Recommendations CSV",
            data=recommendations.to_csv(index=False),
            file_name="recommendations.csv",
            mime="text/csv"
        )

st.divider()
st.subheader("Genre-Based Recommendation")

st.write("""
Select a genre to find highly rated books from that category.
""")

genre_choice = st.selectbox(
    "Choose a Genre", genres, key="genre_recommendation"
)

if st.button("Recommend by Genre"):
    with st.spinner("Finding books..."):
        try:
            recommendations = genre_based(genre_choice, top_n=5)
        except Exception as e:
            st.error("Unable to generate genre recommendations.")
            st.exception(e)
            recommendations = pd.DataFrame()

    if recommendations.empty:
        st.warning("No books found for this genre.")
    else:
        st.success(f"Top books for: {genre_choice}")
        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True
        )
        st.download_button(
            " Download Genre Recommendations",
            data=recommendations.to_csv(index=False),
            file_name="genre_recommendations.csv",
            mime="text/csv"
        )

st.divider()
st.subheader(" Scenario: Science Fiction Reader")

st.write("""
A new user likes Science Fiction. The system searches the dataset
for Science Fiction-related genres and ranks books using weighted ratings.
""")

if st.button(" Recommend Top 5 Science Fiction Books"):
    with st.spinner("Finding Science Fiction books..."):
        try:
            sci_fi = science_fiction_books(top_n=5)
        except Exception as e:
            st.error("Unable to generate Science Fiction recommendations.")
            st.exception(e)
            sci_fi = pd.DataFrame()

    if sci_fi.empty:
        st.info("No Science Fiction books are available in the current dataset.")
    else:
        st.success("Top Science Fiction Recommendations")
        st.dataframe(sci_fi, use_container_width=True, hide_index=True)
        st.download_button(
            " Download Science Fiction Recommendations",
            data=sci_fi.to_csv(index=False),
            file_name="science_fiction_recommendations.csv",
            mime="text/csv"
        )

st.divider()
st.subheader(" Hidden Gems")

st.write("""
Hidden gems are books with high ratings but comparatively fewer reviews.
""")

with st.spinner("Finding hidden gems..."):
    try:
        hidden_gems = get_hidden_gems(top_n=10)
    except Exception as e:
        st.error("Unable to load hidden gems.")
        st.exception(e)
        hidden_gems = pd.DataFrame()

if hidden_gems.empty:
    st.info("No hidden gems found.")
else:
    st.dataframe(
        hidden_gems,
        use_container_width=True,
        hide_index=True
    )
    st.download_button(
        " Download Hidden Gems",
        data=hidden_gems.to_csv(index=False),
        file_name="hidden_gems.csv",
        mime="text/csv"
    )

st.divider()
st.subheader(" Top Rated Books")

if "Weighted_Rating" in filtered_df.columns:
    top_books = filtered_df.sort_values(
        "Weighted_Rating", ascending=False
    ).head(10)
else:
    top_books = filtered_df.sort_values(
        "Rating", ascending=False
    ).head(10)

display_columns = [
    "Book Name", "Author", "Genre", "Rating",
    "Number of Reviews", "Weighted_Rating"
]
display_columns = [c for c in display_columns if c in top_books.columns]

st.dataframe(
    top_books[display_columns],
    use_container_width=True,
    hide_index=True
)

st.divider()
st.subheader(" Recommendation System Evaluation")

st.write("""
Evaluation results are generated using the recommendation evaluation pipeline.
""")

evaluation_path = os.path.join(
    REPORT_DIR, "recommendation_evaluation.csv"
)
content_evaluation_path = os.path.join(
    REPORT_DIR, "content_similarity_evaluation.csv"
)
cluster_evaluation_path = os.path.join(
    REPORT_DIR, "cluster_recommendation_evaluation.csv"
)

if os.path.exists(evaluation_path):
    try:
        evaluation_df = pd.read_csv(evaluation_path)
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        precision_value = (
            evaluation_df["Precision@5"].mean()
            if "Precision@5" in evaluation_df.columns else 0
        )
        recall_value = (
            evaluation_df["Recall@5"].mean()
            if "Recall@5" in evaluation_df.columns else 0
        )
        f1_value = (
            evaluation_df["F1@5"].mean()
            if "F1@5" in evaluation_df.columns else 0
        )

        metric_col1.metric("Precision@5", f"{precision_value:.4f}")
        metric_col2.metric("Recall@5", f"{recall_value:.4f}")
        metric_col3.metric("F1@5", f"{f1_value:.4f}")

        st.markdown("### Detailed Evaluation")
        st.dataframe(
            evaluation_df,
            use_container_width=True,
            hide_index=True
        )
        st.download_button(
            "Download Evaluation Results",
            data=evaluation_df.to_csv(index=False),
            file_name="recommendation_evaluation.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error("Unable to read recommendation evaluation.")
        st.exception(e)
else:
    st.info("Recommendation evaluation report not found. Run evaluation.py first.")

if os.path.exists(content_evaluation_path):
    st.markdown("###  Content Similarity Evaluation")
    try:
        content_eval = pd.read_csv(content_evaluation_path)
        similarity_col1, similarity_col2 = st.columns(2)

        average_similarity = (
            content_eval["Average_Similarity"].mean()
            if "Average_Similarity" in content_eval.columns else 0
        )
        maximum_similarity = (
            content_eval["Maximum_Similarity"].mean()
            if "Maximum_Similarity" in content_eval.columns else 0
        )

        similarity_col1.metric(
            "Average Content Similarity",
            f"{average_similarity:.4f}"
        )
        similarity_col2.metric(
            "Average Maximum Similarity",
            f"{maximum_similarity:.4f}"
        )
        st.dataframe(
            content_eval,
            use_container_width=True,
            hide_index=True
        )
    except Exception:
        st.warning("Unable to load content similarity evaluation.")

if os.path.exists(cluster_evaluation_path):
    st.markdown("###  Cluster Recommendation Evaluation")
    try:
        cluster_eval = pd.read_csv(cluster_evaluation_path)
        st.dataframe(
            cluster_eval,
            use_container_width=True,
            hide_index=True
        )
    except Exception:
        st.warning("Unable to load cluster evaluation report.")

st.markdown("###  Evaluation Interpretation")

st.info("""
**Precision@5** measures how many of the top-5 recommendations are considered relevant.

**Recall@5** measures how many relevant books are retrieved within the top-5 recommendations.

**F1@5** combines precision and recall into a single metric.

Content similarity measures how similar the recommended books are to the selected book based on TF-IDF representations.
""")

st.divider()

with st.expander(" View Filtered Dataset"):
    st.write(f"Showing {len(filtered_df):,} books.")
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )
    st.download_button(
        " Download Filtered Dataset",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_books.csv",
        mime="text/csv"
    )

st.divider()
st.subheader("🤖 About the Recommendation System")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("""
###  NLP

**TF-IDF** converts book information into numerical feature vectors.

Cosine similarity is used to identify contentually similar books.
""")

with info_col2:
    st.markdown("""
###  Clustering

**K-Means clustering** groups books according to their feature representations.

The current pipeline selected **K = 4**.
""")

with info_col3:
    st.markdown("""
###  Hybrid Model

Recommendations combine:

- Content similarity
- Cluster membership
- Weighted rating
""")

st.divider()

st.caption(
    "Audible Insights | Python • Pandas • NLP • TF-IDF • "
    "K-Means • Recommendation Systems • Streamlit • Plotly"
)
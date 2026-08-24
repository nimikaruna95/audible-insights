# st.py
import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from processing_data.recommendation_system import (
    recommend_books,
    genre_based,
    get_hidden_gems,
    science_fiction_books,
    get_top_books)

st.set_page_config(
    page_title="Audible Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load clustered dataset
DATA_PATH = os.path.join(BASE_DIR,"processed_data","clustered_books.csv")
REPORT_DIR = os.path.join(BASE_DIR,"outputs","reports")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found:\n{DATA_PATH}")

    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
except Exception as error:
    st.error("Unable to load dataset.")
    st.exception(error)
    st.stop()

# Validate important columns
required_columns = [
    "Book Name",
    "Author",
    "Genre",
    "Rating",
    "Number of Reviews",
    "Weighted_Rating"]

missing = [
    c for c in required_columns
    if c not in df.columns]

if missing:
    st.error("Missing required columns: " +", ".join(missing))
    st.stop()

# Prepare data types
for column in ["Book Name", "Author", "Genre"]:
    df[column] = (df[column].fillna("Unknown").astype(str))

for column in ["Rating","Number of Reviews","Weighted_Rating"]:
    df[column] = pd.to_numeric(df[column],errors="coerce")

if "Cluster" in df.columns:
    df["Cluster"] = pd.to_numeric(
        df["Cluster"],
        errors="coerce"
    )


# Page title
st.markdown(
    "<h1>Audible Insights</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    **Intelligent Book Recommendation System**

    Uses NLP, TF-IDF, cosine similarity, K-Means clustering
    and a hybrid recommendation model.
    """
)

st.divider()


# Sidebar filters
st.sidebar.title("Dashboard Filters")

min_rating = st.sidebar.slider(
    "Minimum Rating",
    0.0,
    5.0,
    4.0,
    0.1
)

genre_values = set()

for value in df["Genre"].dropna():
    for genre in str(value).split("|"):
        genre = genre.strip()
        if genre:
            genre_values.add(genre)

genres = sorted(genre_values)

selected_genre = st.sidebar.selectbox(
    "Genre",
    ["All"] + genres
)

if "Cluster" in df.columns:
    clusters = sorted(
        df["Cluster"].dropna().unique()
    )

    selected_cluster = st.sidebar.selectbox(
        "Cluster",
        ["All"] + list(clusters)
    )
else:
    selected_cluster = "All"


# Apply filters
filtered_df = df[
    df["Rating"].fillna(0) >= min_rating
].copy()

if selected_genre != "All":
    filtered_df = filtered_df[
        filtered_df["Genre"].str.contains(
            selected_genre,
            case=False,
            na=False,
            regex=False
        )
    ]

if selected_cluster != "All":
    filtered_df = filtered_df[
        filtered_df["Cluster"] == selected_cluster
    ]


if filtered_df.empty:
    st.warning(
        "No books match the selected filters."
    )
    st.stop()


# Dataset overview
st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Books",
    f"{len(filtered_df):,}"
)

col2.metric(
    "Authors",
    f"{filtered_df['Author'].nunique():,}"
)

col3.metric(
    "Genres",
    f"{filtered_df['Genre'].nunique():,}"
)

col4.metric(
    "Average Rating",
    f"{filtered_df['Rating'].mean():.2f}"
)


# Rating distribution
st.subheader("Dataset Analysis")

left, right = st.columns(2)

with left:
    fig = px.histogram(
        filtered_df,
        x="Rating",
        nbins=20,
        title="Rating Distribution"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


with right:
    genre_count = {}

    for value in filtered_df["Genre"].dropna():
        for genre in str(value).split("|"):
            genre = genre.strip()

            if genre:
                genre_count[genre] = (
                    genre_count.get(genre, 0) + 1
                )

    genre_series = pd.Series(
        genre_count
    ).sort_values(
        ascending=False
    ).head(10)

    fig = px.bar(
        x=genre_series.index,
        y=genre_series.values,
        labels={
            "x": "Genre",
            "y": "Books"
        },
        title="Top 10 Genres"
    )

    fig.update_xaxes(tickangle=-45)

    st.plotly_chart(
        fig,
        width="stretch"
    )


# Rating versus reviews
st.subheader("Rating vs Review Count")

fig = px.scatter(
    filtered_df,
    x="Number of Reviews",
    y="Rating",
    hover_name="Book Name",
    hover_data=["Author", "Genre"],
    title="Ratings vs Number of Reviews"
)

st.plotly_chart(
    fig,
    width="stretch"
)


# Cluster distribution
if "Cluster" in filtered_df.columns:
    st.subheader("Cluster Distribution")

    cluster_counts = (
        filtered_df["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Books"
    ]

    fig = px.pie(
        cluster_counts,
        names="Cluster",
        values="Books",
        title="Books by K-Means Cluster"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# Top authors
st.subheader("Top Authors")

top_authors = (
    filtered_df.groupby("Author")
    .agg(
        Average_Rating=("Rating", "mean"),
        Books=("Book Name", "count")
    )
    .sort_values(
        "Average_Rating",
        ascending=False
    )
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

st.plotly_chart(
    fig,
    width="stretch"
)


# Hybrid recommendations
st.divider()
st.subheader("Hybrid Book Recommendation")

st.write(
    "Select a book to receive recommendations based on "
    "content similarity, genre, cluster, rating, popularity "
    "and author similarity."
)

book_list = sorted(
    df["Book Name"].unique()
)

selected_book = st.selectbox(
    "Choose a Book",
    book_list
)

if st.button(
    "Recommend Similar Books",
    type="primary"
):
    recommendations = recommend_books(
        selected_book,
        top_n=5
    )

    if recommendations.empty:
        st.warning(
            "No recommendations found."
        )
    else:
        st.success(
            f"Recommendations for: {selected_book}"
        )

        for rank, (_, row) in enumerate(
            recommendations.iterrows(),
            start=1
        ):
            st.markdown(
                f"""
                ### {rank}. {row['Book Name']}

                **Author:** {row['Author']}  
                **Genre:** {row['Genre']}  
                **Rating:** {row['Rating']:.2f}  
                **Weighted Rating:** {row['Weighted_Rating']:.4f}  
                **Content Score:** {row['Content_Score']:.4f}  
                **Hybrid Score:** **{row['Hybrid_Score']:.4f}**
                """
            )

        st.dataframe(
            recommendations,
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "Download Recommendations CSV",
            recommendations.to_csv(index=False),
            "recommendations.csv",
            "text/csv"
        )


# Genre recommendation
st.divider()
st.subheader("Genre-Based Recommendation")

genre_choice = st.selectbox(
    "Choose a Genre",
    genres,
    key="genre_recommendation"
)

if st.button("Recommend by Genre"):
    recommendations = genre_based(
        genre_choice,
        top_n=5
    )

    if recommendations.empty:
        st.info(
            "No books found for this genre."
        )
    else:
        st.dataframe(
            recommendations,
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "Download Genre Recommendations",
            recommendations.to_csv(index=False),
            "genre_recommendations.csv",
            "text/csv"
        )


# Science fiction scenario
st.divider()
st.subheader("Science Fiction Scenario")

st.write(
    "Example scenario: a new user wants highly rated "
    "Science Fiction books."
)

if st.button(
    "Recommend Top 5 Science Fiction Books"
):
    sci_fi = science_fiction_books(
        top_n=5
    )

    if sci_fi.empty:
        st.info(
            "No Science Fiction books found."
        )
    else:
        st.dataframe(
            sci_fi,
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "Download Science Fiction Recommendations",
            sci_fi.to_csv(index=False),
            "science_fiction_recommendations.csv",
            "text/csv"
        )


# Hidden gems
st.divider()
st.subheader("Hidden Gems")

hidden_gems = get_hidden_gems(
    top_n=10
)

if hidden_gems.empty:
    st.info(
        "No hidden gems found."
    )
else:
    st.dataframe(
        hidden_gems,
        width="stretch",
        hide_index=True
    )

    st.download_button(
        "Download Hidden Gems",
        hidden_gems.to_csv(index=False),
        "hidden_gems.csv",
        "text/csv"
    )


# Top rated books
st.divider()
st.subheader("Top Rated Books")

top_books = get_top_books(
    top_n=10
)

st.dataframe(
    top_books,
    width="stretch",
    hide_index=True
)


# Evaluation reports
st.divider()
st.subheader("Recommendation Evaluation")

evaluation_path = os.path.join(
    REPORT_DIR,
    "evaluation_summary.csv"
)

if os.path.exists(evaluation_path):
    evaluation = pd.read_csv(
        evaluation_path
    )

    st.dataframe(
        evaluation,
        width="stretch",
        hide_index=True
    )

    st.download_button("Download Evaluation Summary",evaluation.to_csv(index=False),"evaluation_summary.csv","text/csv")
    st.info(
        "Precision, Recall and F1 are proxy metrics because the dataset does not contain user-item interaction history.")
else:
    st.info("Evaluation report not found, Run evaluation.py if evaluation results are required.")


# Filtered dataset
st.divider()

with st.expander("View Filtered Dataset"):
    st.write(f"Showing {len(filtered_df):,} books.")
    st.dataframe(filtered_df,width="stretch",hide_index=True)
    st.download_button("Download Filtered Dataset",filtered_df.to_csv(index=False),"filtered_books.csv","text/csv")


# Project explanation
st.divider()
st.subheader("About the Recommendation System")

info1, info2, info3 = st.columns(3)
with info1:
    st.markdown("""
        ### NLP

        TF-IDF converts book text into numerical vectors.
        Cosine similarity measures content similarity.
        """)

with info2:
    st.markdown("""
        ### Clustering

        K-Means groups books based on their TF-IDF/SVD
        representations. The current pipeline selected K=3.
        """ )

with info3:
    st.markdown("""
        ### Hybrid Model

        Recommendations combine content similarity,
        genre, cluster, rating, popularity and author similarity.
        """ )

st.caption("Audible Insights | Python | Pandas | NLP | TF-IDF | "
            "K-Means | Recommendation Systems | Streamlit | Plotly")

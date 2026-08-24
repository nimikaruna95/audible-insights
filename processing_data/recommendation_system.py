# recommendation.py
import os
import pickle
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(REPORT_DIR, exist_ok=True)

DATA_PATH = os.path.join(PROCESSED_DIR, "clustered_books.csv")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")

# Load recommendation dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found:\n{DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required = ["Book Name", "Author", "Genre", "Rating",
    "Number of Reviews", "Weighted_Rating",
    "combined_features", "Cluster"]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Prepare columns
for col in ["Book Name", "Author", "Genre", "combined_features"]:
    df[col] = df[col].fillna("").astype(str).str.strip()

for col in ["Rating", "Number of Reviews", "Weighted_Rating"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["Cluster"] = pd.to_numeric(df["Cluster"], errors="coerce").fillna(-1).astype(int)

df.reset_index(drop=True, inplace=True)

# Load trained models
if not os.path.exists(TFIDF_PATH):
    raise FileNotFoundError(f"TF-IDF vectorizer not found")

if not os.path.exists(MATRIX_PATH):
    raise FileNotFoundError(f"TF-IDF matrix not found")

if not os.path.exists(KMEANS_PATH):
    raise FileNotFoundError(f"K-Means model not found")

tfidf = joblib.load(TFIDF_PATH)
tfidf_matrix = joblib.load(MATRIX_PATH)
kmeans = joblib.load(KMEANS_PATH)

if len(df) != tfidf_matrix.shape[0]:
    raise ValueError("Dataset and TF-IDF matrix are not aligned.")

# Similarity is calculated only when needed
def get_similarity(index):
    return cosine_similarity(tfidf_matrix[index],tfidf_matrix).flatten()

def find_book_index(book_name):
    if not book_name:
        return None

    name = str(book_name).strip().lower()
    exact = df[df["Book Name"].str.lower() == name]

    if not exact.empty:
        return exact.index[0]

    partial = df[df["Book Name"].str.contains(name, case=False, na=False, regex=False)]

    return partial.index[0] if not partial.empty else None

def content_based(book_name, top_n=10):
    index = find_book_index(book_name)

    if index is None:
        return pd.DataFrame()

    scores = get_similarity(index)
    indices = [i for i in scores.argsort()[::-1] if i != index][:top_n]

    result = df.loc[indices,["Book Name", "Author", "Genre",
            "Rating", "Number of Reviews",
            "Weighted_Rating"
        ]].copy()
    result["Similarity_Score"] = [
        round(float(scores[i]), 4) for i in indices]

    return result.reset_index(drop=True)

def cluster_based(book_name, top_n=10):
    index = find_book_index(book_name)

    if index is None:
        return pd.DataFrame()

    cluster = df.loc[index, "Cluster"]
    scores = get_similarity(index)

    books = df[(df["Cluster"] == cluster) &(df.index != index)].copy()

    books["Similarity_Score"] = [scores[i] for i in books.index]

    books["Cluster_Score"] = (0.7 * books["Similarity_Score"] + 0.3 * (books["Weighted_Rating"] / 5))

    return books.sort_values("Cluster_Score",ascending=False)[[
            "Book Name", "Author", "Genre",
            "Rating", "Number of Reviews",
            "Weighted_Rating", "Similarity_Score",
            "Cluster", "Cluster_Score"]].head(top_n)

def genre_based(genre, top_n=10):
    if not genre:
        return pd.DataFrame()

    books = df[df["Genre"].str.contains(str(genre),case=False,na=False,regex=False)].copy()

    return books.sort_values("Weighted_Rating",ascending=False)[[
            "Book Name", "Author", "Genre",
            "Rating", "Number of Reviews",
            "Weighted_Rating"]].head(top_n)

def author_based(author, top_n=10):
    if not author:
        return pd.DataFrame()

    books = df[df["Author"].str.contains(str(author),case=False,na=False,regex=False)].copy()

    return books.sort_values("Weighted_Rating",ascending=False)[[
            "Book Name", "Author", "Genre",
            "Rating", "Number of Reviews",
            "Weighted_Rating"]].head(top_n)

def recommend_books(book_name, top_n=10):
    index = find_book_index(book_name)

    if index is None:
        return pd.DataFrame()

    scores = get_similarity(index)
    candidates = df.drop(index).copy()

    candidates["Content_Score"] = [scores[i] for i in candidates.index]

    selected_cluster = df.loc[index, "Cluster"]
    candidates["Cluster_Score"] = (candidates["Cluster"] == selected_cluster).astype(float)

    candidates["Genre_Score"] = 0.0

    selected_genre = str(df.loc[index, "Genre"])

    if selected_genre and selected_genre != "Unknown":
        first_genre = selected_genre.split("|")[0].strip()

        candidates["Genre_Score"] = candidates["Genre"].str.contains(first_genre,case=False,
                                                                     na=False,regex=False).astype(float)

    candidates["Rating_Score"] = (candidates["Weighted_Rating"] / 5)
    popularity = np.log1p(candidates["Number of Reviews"].clip(lower=0))

    max_popularity = popularity.max()

    candidates["Popularity_Score"] = (
        popularity / max_popularity
        if max_popularity > 0 else 0)

    selected_author = str(df.loc[index, "Author"]).strip()
    candidates["Author_Score"] = (candidates["Author"].str.lower() == selected_author.lower()).astype(float)

    # Hybrid recommendation score
    candidates["Hybrid_Score"] = (
        0.40 * candidates["Content_Score"] +
        0.15 * candidates["Genre_Score"] +
        0.15 * candidates["Cluster_Score"] +
        0.15 * candidates["Rating_Score"] +
        0.10 * candidates["Popularity_Score"] +
        0.05 * candidates["Author_Score"])

    return candidates.sort_values("Hybrid_Score",ascending=False)[[
            "Book Name", "Author", "Genre","Rating", "Number of Reviews",
            "Weighted_Rating", "Content_Score","Genre_Score", "Cluster_Score",
            "Rating_Score", "Popularity_Score","Author_Score", "Hybrid_Score"]].head(top_n)

def science_fiction_books(top_n=5):
    terms = ["science fiction","sci fi",
        "sci-fi","science-fiction"]

    mask = pd.Series(False, index=df.index)

    for term in terms:
        mask |= df["Genre"].str.contains(term,case=False,na=False,regex=False)

    return df[mask].sort_values("Weighted_Rating",ascending=False)[[
            "Book Name", "Author", "Genre",
            "Rating", "Number of Reviews",
            "Weighted_Rating"]].head(top_n)

def get_hidden_gems(top_n=5):
    books = df[(df["Rating"] >= 4.5) &(df["Number of Reviews"] <= 500)]

    return books.sort_values("Weighted_Rating",ascending=False)[[
            "Book Name", "Author", "Genre",
            "Rating", "Number of Reviews",
            "Weighted_Rating"]].head(top_n)

def get_top_books(top_n=10):
    return df.sort_values("Weighted_Rating",ascending=False)[[
            "Book Name", "Author", "Genre",
            "Rating", "Number of Reviews",
            "Weighted_Rating"]].head(top_n)

if __name__ == "__main__":
    sample_book = df.iloc[0]["Book Name"]

    print("Sample Book:", sample_book)
    print("\nHybrid Recommendations:")
    print(recommend_books(sample_book,top_n=10).to_string(index=False))
    print("\nRecommendation system completed successfully.")

# recommendation_system.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "clustered_books.csv")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")

print("=" * 70)
print("AUDIBLE INSIGHTS - RECOMMENDATION SYSTEM")
print("=" * 70)
print("\nLoading clustered dataset...")

df = pd.read_csv(DATA_PATH)
print("Dataset Shape:", df.shape)

required_columns = [
    "Book Name", "Author", "Genre", "Rating", "Number of Reviews",
    "Weighted_Rating", "combined_features", "Cluster"
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

text_columns = ["Book Name", "Author", "Genre", "combined_features"]
for col in text_columns:
    df[col] = df[col].fillna("").astype(str).str.strip()

numeric_columns = ["Rating", "Number of Reviews", "Weighted_Rating", "Cluster"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Rating"] = df["Rating"].fillna(0)
df["Number of Reviews"] = df["Number of Reviews"].fillna(0)
df["Weighted_Rating"] = df["Weighted_Rating"].fillna(df["Rating"])
df["Cluster"] = df["Cluster"].fillna(-1).astype(int)
df.reset_index(drop=True, inplace=True)

print("\nLoading trained models...")

if not os.path.exists(TFIDF_PATH):
    raise FileNotFoundError(f"TF-IDF model not found:\n{TFIDF_PATH}")

if not os.path.exists(KMEANS_PATH):
    raise FileNotFoundError(f"K-Means model not found:\n{KMEANS_PATH}")

tfidf = joblib.load(TFIDF_PATH)
kmeans = joblib.load(KMEANS_PATH)

print("TF-IDF vectorizer loaded.")
print("K-Means model loaded.")

print("\nGenerating TF-IDF matrix...")

tfidf_matrix = tfidf.transform(df["combined_features"])
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

tfidf_matrix_path = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")
joblib.dump(tfidf_matrix, tfidf_matrix_path)
print("TF-IDF matrix saved to:", tfidf_matrix_path)

print("\nCalculating cosine similarity...")

cosine_sim = cosine_similarity(tfidf_matrix)
print("Cosine Similarity Matrix Shape:", cosine_sim.shape)


def find_book_index(book_name):
    if not book_name:
        return None

    book_name = str(book_name).strip()

    exact_match = df[df["Book Name"].str.lower() == book_name.lower()]
    if not exact_match.empty:
        return exact_match.index[0]

    partial_match = df[df["Book Name"].str.contains(
        book_name, case=False, na=False, regex=False
    )]

    return partial_match.index[0] if not partial_match.empty else None


def content_based(book_name, top_n=10):
    idx = find_book_index(book_name)
    if idx is None:
        return pd.DataFrame()

    scores = sorted(
        enumerate(cosine_sim[idx]),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i, score in scores:
        if i == idx:
            continue

        recommendations.append({
            "Book Name": df.iloc[i]["Book Name"],
            "Author": df.iloc[i]["Author"],
            "Genre": df.iloc[i]["Genre"],
            "Rating": df.iloc[i]["Rating"],
            "Number of Reviews": df.iloc[i]["Number of Reviews"],
            "Weighted_Rating": df.iloc[i]["Weighted_Rating"],
            "Similarity": round(float(score), 4)
        })

        if len(recommendations) >= top_n:
            break

    return pd.DataFrame(recommendations)


def cluster_based(book_name, top_n=10):
    idx = find_book_index(book_name)
    if idx is None:
        return pd.DataFrame()

    cluster = df.loc[idx, "Cluster"]
    books = df[(df["Cluster"] == cluster) & (df.index != idx)].copy()

    books["Similarity"] = [cosine_sim[idx, i] for i in books.index]
    books["Cluster_Score"] = (
        0.70 * books["Similarity"] +
        0.30 * (books["Weighted_Rating"] / 5.0)
    )

    books = books.sort_values("Cluster_Score", ascending=False)

    return books[
        [
            "Book Name", "Author", "Genre", "Rating",
            "Number of Reviews", "Weighted_Rating",
            "Similarity", "Cluster", "Cluster_Score"
        ]
    ].head(top_n)


def genre_based(genre, top_n=10):
    if not genre:
        return pd.DataFrame()

    books = df[df["Genre"].str.contains(
        str(genre), case=False, na=False, regex=False
    )].copy()

    books = books.sort_values("Weighted_Rating", ascending=False)

    return books[
        [
            "Book Name", "Author", "Genre", "Rating",
            "Number of Reviews", "Weighted_Rating"
        ]
    ].head(top_n)


def author_based(author, top_n=10):
    if not author:
        return pd.DataFrame()

    books = df[df["Author"].str.contains(
        str(author), case=False, na=False, regex=False
    )].copy()

    books = books.sort_values("Weighted_Rating", ascending=False)

    return books[
        [
            "Book Name", "Author", "Genre", "Rating",
            "Number of Reviews", "Weighted_Rating"
        ]
    ].head(top_n)


def recommend_books(book_name, top_n=10):
    idx = find_book_index(book_name)
    if idx is None:
        return pd.DataFrame()

    candidates = df[df.index != idx].copy()
    candidates["Content_Score"] = [
        cosine_sim[idx, i] for i in candidates.index
    ]

    selected_cluster = df.loc[idx, "Cluster"]
    candidates["Cluster_Score"] = np.where(
        candidates["Cluster"] == selected_cluster, 1.0, 0.0
    )

    candidates["Rating_Score"] = candidates["Weighted_Rating"] / 5.0
    candidates["Popularity_Score"] = np.log1p(
        candidates["Number of Reviews"]
    )

    max_popularity = candidates["Popularity_Score"].max()

    if max_popularity > 0:
        candidates["Popularity_Score"] /= max_popularity
    else:
        candidates["Popularity_Score"] = 0

    candidates["Hybrid_Score"] = (
        0.50 * candidates["Content_Score"] +
        0.20 * candidates["Cluster_Score"] +
        0.20 * candidates["Rating_Score"] +
        0.10 * candidates["Popularity_Score"]
    )

    candidates = candidates.sort_values("Hybrid_Score", ascending=False)

    output_columns = [
        "Book Name", "Author", "Genre", "Rating",
        "Number of Reviews", "Weighted_Rating",
        "Content_Score", "Cluster_Score",
        "Rating_Score", "Popularity_Score", "Hybrid_Score"
    ]

    return candidates[output_columns].head(top_n)


def science_fiction_books(top_n=5):
    possible_terms = [
        "science fiction", "sci fi", "sci-fi", "science-fiction"
    ]

    mask = pd.Series(False, index=df.index)

    for term in possible_terms:
        mask |= df["Genre"].str.contains(
            term, case=False, na=False, regex=False
        )

    books = df[mask].sort_values(
        "Weighted_Rating", ascending=False
    )

    return books[
        [
            "Book Name", "Author", "Genre", "Rating",
            "Number of Reviews", "Weighted_Rating"
        ]
    ].head(top_n)


def get_hidden_gems(top_n=5):
    gems = df[
        (df["Rating"] >= 4.5) &
        (df["Number of Reviews"] <= 500)
    ].copy()

    gems = gems.sort_values("Weighted_Rating", ascending=False)

    return gems[
        [
            "Book Name", "Author", "Genre", "Rating",
            "Number of Reviews", "Weighted_Rating"
        ]
    ].head(top_n)


def get_top_books(top_n=10):
    books = df.sort_values(
        "Weighted_Rating", ascending=False
    ).head(top_n)

    return books[
        [
            "Book Name", "Author", "Genre", "Rating",
            "Number of Reviews", "Weighted_Rating"
        ]
    ]


def precision_at_k(recommended, relevant, k=5):
    if k <= 0:
        return 0.0

    recommended = list(recommended)[:k]
    relevant = set(relevant)

    if not recommended:
        return 0.0

    hits = len(set(recommended).intersection(relevant))
    return hits / len(recommended)


def recall_at_k(recommended, relevant, k=5):
    recommended = list(recommended)[:k]
    relevant = set(relevant)

    if len(relevant) == 0:
        return 0.0

    hits = len(set(recommended).intersection(relevant))
    return hits / len(relevant)


def get_relevant_books(book_index, similarity_threshold=0.20):
    similarities = cosine_sim[book_index]

    relevant_indices = np.where(
        similarities >= similarity_threshold
    )[0]

    relevant_indices = [
        i for i in relevant_indices if i != book_index
    ]

    return [df.iloc[i]["Book Name"] for i in relevant_indices]


def evaluate_system(sample_size=20, k=5):
    print("\n" + "=" * 70)
    print("RECOMMENDATION SYSTEM EVALUATION")
    print("=" * 70)

    sample_size = min(sample_size, len(df))
    precision_scores, recall_scores = [], []

    sample_indices = np.linspace(
        0, len(df) - 1, sample_size, dtype=int
    )

    for idx in sample_indices:
        book_name = df.iloc[idx]["Book Name"]

        recommendations = recommend_books(book_name, top_n=k)

        if recommendations.empty:
            continue

        recommended = list(recommendations["Book Name"])
        relevant = get_relevant_books(
            idx, similarity_threshold=0.20
        )

        precision_scores.append(
            precision_at_k(recommended, relevant, k)
        )

        recall_scores.append(
            recall_at_k(recommended, relevant, k)
        )

    if not precision_scores:
        print("No valid evaluation results.")
        return

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)

    print(f"Books Evaluated: {len(precision_scores)}")
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")

    evaluation_df = pd.DataFrame({
        "Metric": [f"Precision@{k}", f"Recall@{k}"],
        "Score": [avg_precision, avg_recall]
    })

    evaluation_path = os.path.join(
        REPORT_DIR, "recommendation_evaluation.csv"
    )

    evaluation_df.to_csv(evaluation_path, index=False)

    print("\nEvaluation report saved to:")
    print(evaluation_path)
    print("=" * 70)

    return evaluation_df


def test_recommendation():
    sample_book = df.iloc[0]["Book Name"]

    print("\n" + "=" * 70)
    print("SAMPLE RECOMMENDATION")
    print("=" * 70)

    print("\nSelected Book:")
    print(sample_book)

    print("\nContent-Based Recommendations:")
    content = content_based(sample_book, top_n=5)
    print(content.to_string(index=False))

    print("\nCluster-Based Recommendations:")
    cluster = cluster_based(sample_book, top_n=5)
    print(cluster.to_string(index=False))

    print("\nHybrid Recommendations:")
    hybrid = recommend_books(sample_book, top_n=5)
    print(hybrid.to_string(index=False))


if __name__ == "__main__":
    test_recommendation()

    print("\n" + "=" * 70)
    print("HIDDEN GEMS")
    print("=" * 70)
    print(get_hidden_gems(top_n=5).to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP BOOKS")
    print("=" * 70)
    print(get_top_books(top_n=5).to_string(index=False))

    evaluate_system(sample_size=20, k=5)

    print("\n" + "=" * 70)
    print("RECOMMENDATION SYSTEM COMPLETED SUCCESSFULLY")
    print("=" * 70)
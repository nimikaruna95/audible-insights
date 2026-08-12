import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "clustered_books.csv")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")

TOP_K = 5
EVALUATION_BOOKS = 20

print("=" * 70)
print("AUDIBLE INSIGHTS - RECOMMENDATION SYSTEM EVALUATION")
print("=" * 70)
print("\nLoading clustered dataset...")

df = pd.read_csv(DATA_PATH)
print(f"Dataset Shape: {df.shape}")

required_columns = [
    "Book Name", "Author", "Rating", "Number of Reviews",
    "Weighted_Rating", "Cluster", "combined_features"
]

missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df["combined_features"] = df["combined_features"].fillna("").astype(str)
df["Book Name"] = df["Book Name"].fillna("").astype(str)

print("\nLoading trained models...")
tfidf = joblib.load(TFIDF_PATH)
kmeans = joblib.load(KMEANS_PATH)
print("TF-IDF vectorizer loaded.")
print("K-Means model loaded.")

print("\nGenerating TF-IDF matrix...")
tfidf_matrix = tfidf.transform(df["combined_features"])
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

print("\nCalculating cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix)
print("Cosine Similarity Matrix Shape:", cosine_sim.shape)


def find_book_index(book_name):
    matches = df["Book Name"].str.contains(
        str(book_name), case=False, na=False, regex=False
    )
    return df.index[matches][0] if matches.any() else None


def content_recommendations(book_name, top_n=5):
    idx = find_book_index(book_name)
    if idx is None:
        return pd.DataFrame()

    scores = cosine_sim[idx]
    indices = np.argsort(scores)[::-1]
    recommendations = []

    for i in indices:
        if i == idx:
            continue

        recommendations.append({
            "Book Name": df.iloc[i]["Book Name"],
            "Author": df.iloc[i]["Author"],
            "Rating": df.iloc[i]["Rating"],
            "Number of Reviews": df.iloc[i]["Number of Reviews"],
            "Weighted_Rating": df.iloc[i]["Weighted_Rating"],
            "Similarity": float(scores[i])
        })

        if len(recommendations) >= top_n:
            break

    return pd.DataFrame(recommendations)


def cluster_recommendations(book_name, top_n=5):
    idx = find_book_index(book_name)
    if idx is None:
        return pd.DataFrame()

    cluster_id = df.loc[idx, "Cluster"]
    books = df[(df["Cluster"] == cluster_id) & (df.index != idx)].copy()
    books = books.sort_values("Weighted_Rating", ascending=False)

    columns = [
        "Book Name", "Author", "Rating", "Number of Reviews",
        "Weighted_Rating", "Cluster"
    ]
    return books[columns].head(top_n)


def hybrid_recommendations(book_name, top_n=5):
    idx = find_book_index(book_name)
    if idx is None:
        return pd.DataFrame()

    cluster_id = df.loc[idx, "Cluster"]
    candidates = df[(df["Cluster"] == cluster_id) & (df.index != idx)].copy()

    if candidates.empty:
        return pd.DataFrame()

    candidates["Content_Score"] = [
        cosine_sim[idx, i] for i in candidates.index
    ]
    candidates["Cluster_Score"] = 1.0
    candidates["Rating_Score"] = (
        candidates["Weighted_Rating"] / 5.0
    ).clip(0, 1)

    review_log = np.log1p(
        candidates["Number of Reviews"].clip(lower=0)
    )
    max_review_log = review_log.max()

    candidates["Popularity_Score"] = (
        review_log / max_review_log if max_review_log > 0 else 0.0
    )

    candidates["Hybrid_Score"] = (
        0.50 * candidates["Content_Score"]
        + 0.20 * candidates["Cluster_Score"]
        + 0.20 * candidates["Rating_Score"]
        + 0.10 * candidates["Popularity_Score"]
    )

    columns = [
        "Book Name", "Author", "Rating", "Number of Reviews",
        "Weighted_Rating", "Content_Score", "Cluster_Score",
        "Rating_Score", "Popularity_Score", "Hybrid_Score"
    ]

    return candidates.sort_values(
        "Hybrid_Score", ascending=False
    )[columns].head(top_n)


def precision_at_k(recommended, relevant, k=5):
    recommended = list(recommended)[:k]
    if not recommended or k == 0:
        return 0.0
    return len(set(recommended) & set(relevant)) / k


def recall_at_k(recommended, relevant, k=5):
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(set(list(recommended)[:k]) & relevant) / len(relevant)


def f1_at_k(precision, recall):
    return 0.0 if precision + recall == 0 else (
        2 * precision * recall / (precision + recall)
    )


def evaluate_content_similarity(book_name, top_n=5):
    recommendations = content_recommendations(book_name, top_n)
    if recommendations.empty:
        return None

    values = recommendations["Similarity"].astype(float).values
    return {
        "Book Name": book_name,
        "Average_Similarity": float(np.mean(values)),
        "Maximum_Similarity": float(np.max(values)),
        "Minimum_Similarity": float(np.min(values))
    }


def evaluate_book(book_name, top_k=5):
    idx = find_book_index(book_name)
    if idx is None:
        return None

    print(f"Evaluating: {book_name}")
    target_cluster = df.loc[idx, "Cluster"]

    relevant_df = df[
        (df["Cluster"] == target_cluster) & (df.index != idx)
    ]
    relevant_books = set(relevant_df["Book Name"])

    recommendations = hybrid_recommendations(book_name, top_k)
    if recommendations.empty:
        return None

    recommended_books = list(recommendations["Book Name"])
    precision = precision_at_k(recommended_books, relevant_books, top_k)
    recall = recall_at_k(recommended_books, relevant_books, top_k)
    f1 = f1_at_k(precision, recall)

    content_eval = evaluate_content_similarity(book_name, top_k)
    if content_eval:
        avg_similarity = content_eval["Average_Similarity"]
        max_similarity = content_eval["Maximum_Similarity"]
        min_similarity = content_eval["Minimum_Similarity"]
    else:
        avg_similarity = max_similarity = min_similarity = 0.0

    cluster_df = df[df["Cluster"] == target_cluster]

    return {
        "Book Name": book_name,
        "Recommended_Count": len(recommended_books),
        "Relevant_Count": len(relevant_books),
        "Precision@5": precision,
        "Recall@5": recall,
        "F1@5": f1,
        "Average_Similarity": avg_similarity,
        "Maximum_Similarity": max_similarity,
        "Minimum_Similarity": min_similarity,
        "Cluster": target_cluster,
        "Cluster_Average_Rating": cluster_df["Rating"].mean(),
        "Cluster_Average_Reviews": cluster_df["Number of Reviews"].mean(),
        "Cluster_Average_Weighted_Rating": cluster_df["Weighted_Rating"].mean()
    }


def select_evaluation_books(n=20):
    return (
        df.sort_values("Number of Reviews", ascending=False)["Book Name"]
        .drop_duplicates()
        .head(n)
        .tolist()
    )


def get_hidden_gems(top_n=10):
    gems = df[
        (df["Rating"] >= 4.5) &
        (df["Number of Reviews"] <= 500)
    ].sort_values("Weighted_Rating", ascending=False)

    columns = [
        "Book Name", "Author", "Rating",
        "Number of Reviews", "Weighted_Rating"
    ]
    return gems[columns].head(top_n)


def main():
    print("\nSelecting evaluation books...")
    evaluation_books = select_evaluation_books(EVALUATION_BOOKS)
    print(f"Books selected for evaluation: {len(evaluation_books)}")
    print("\nRunning evaluation...")

    results = []
    content_results = []

    for book_name in evaluation_books:
        result = evaluate_book(book_name, TOP_K)
        if result is not None:
            results.append(result)

        content_result = evaluate_content_similarity(book_name, TOP_K)
        if content_result is not None:
            content_results.append(content_result)

    evaluation_df = pd.DataFrame(results)
    content_df = pd.DataFrame(content_results)

    if evaluation_df.empty:
        print("\nNo evaluation results generated.")
        return

    numeric_columns = [
        "Precision@5", "Recall@5", "F1@5",
        "Average_Similarity", "Maximum_Similarity",
        "Minimum_Similarity", "Cluster_Average_Rating",
        "Cluster_Average_Reviews", "Cluster_Average_Weighted_Rating"
    ]

    for column in numeric_columns:
        if column in evaluation_df:
            evaluation_df[column] = evaluation_df[column].astype(float).round(4)

    if not content_df.empty:
        for column in [
            "Average_Similarity", "Maximum_Similarity",
            "Minimum_Similarity"
        ]:
            content_df[column] = content_df[column].astype(float).round(4)

    precision_mean = evaluation_df["Precision@5"].mean()
    recall_mean = evaluation_df["Recall@5"].mean()
    f1_mean = evaluation_df["F1@5"].mean()
    similarity_mean = evaluation_df["Average_Similarity"].mean()
    max_similarity_mean = evaluation_df["Maximum_Similarity"].mean()

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    display_columns = [
        "Book Name", "Recommended_Count", "Relevant_Count",
        "Precision@5", "Recall@5", "F1@5",
        "Average_Similarity", "Maximum_Similarity"
    ]
    print(evaluation_df[display_columns].to_string(index=False))

    print("\n" + "=" * 70)
    print("OVERALL EVALUATION")
    print("=" * 70)
    print(f"Books Evaluated: {len(evaluation_df)}")
    print(f"Precision@5: {precision_mean:.4f}")
    print(f"Recall@5:    {recall_mean:.4f}")
    print(f"F1@5:        {f1_mean:.4f}")
    print(f"Average Content Similarity: {similarity_mean:.4f}")
    print(f"Average Maximum Similarity: {max_similarity_mean:.4f}")

    print("\n" + "=" * 70)
    print("CONTENT SIMILARITY EVALUATION")
    print("=" * 70)

    if not content_df.empty:
        print(content_df.to_string(index=False))
        print(
            "\nOverall Average Content Similarity: "
            f"{content_df['Average_Similarity'].mean():.4f}"
        )

    cluster_columns = [
        "Book Name", "Cluster", "Cluster_Average_Rating",
        "Cluster_Average_Reviews", "Cluster_Average_Weighted_Rating"
    ]

    cluster_df = evaluation_df[cluster_columns].rename(columns={
        "Cluster_Average_Rating": "Average_Rating",
        "Cluster_Average_Reviews": "Average_Reviews",
        "Cluster_Average_Weighted_Rating": "Average_Weighted_Rating"
    })

    print("\n" + "=" * 70)
    print("CLUSTER EVALUATION")
    print("=" * 70)
    print(cluster_df.to_string(index=False))

    hidden_gems = get_hidden_gems(10)

    print("\n" + "=" * 70)
    print("TOP HIDDEN GEMS")
    print("=" * 70)
    print(hidden_gems.to_string(index=False))

    recommendation_path = os.path.join(
        REPORT_DIR, "recommendation_evaluation.csv"
    )
    content_path = os.path.join(
        REPORT_DIR, "content_similarity_evaluation.csv"
    )
    cluster_path = os.path.join(
        REPORT_DIR, "cluster_recommendation_evaluation.csv"
    )
    hidden_gems_path = os.path.join(
        REPORT_DIR, "hidden_gems_evaluation.csv"
    )

    evaluation_df.to_csv(recommendation_path, index=False)
    content_df.to_csv(content_path, index=False)
    cluster_df.to_csv(cluster_path, index=False)
    hidden_gems.to_csv(hidden_gems_path, index=False)

    summary_path = os.path.join(
        REPORT_DIR, "evaluation_summary.txt"
    )

    summary = [
        "=" * 70,
        "AUDIBLE INSIGHTS - EVALUATION SUMMARY",
        "=" * 70,
        f"Books Evaluated: {len(evaluation_df)}",
        f"Precision@5: {precision_mean:.4f}",
        f"Recall@5: {recall_mean:.4f}",
        f"F1@5: {f1_mean:.4f}",
        f"Average Content Similarity: {similarity_mean:.4f}",
        f"Average Maximum Similarity: {max_similarity_mean:.4f}",
        "",
        "Interpretation:",
        "- Precision@5 measures the proportion of recommended books",
        "  belonging to the same cluster as the input book.",
        "- Recall@5 measures how many relevant same-cluster books",
        "  were retrieved among all relevant books.",
        "- F1@5 is the harmonic mean of Precision@5 and Recall@5.",
        "- Content similarity measures TF-IDF cosine similarity.",
        "- Weighted Rating combines rating and review count.",
        "- Hidden gems are books with Rating >= 4.5 and",
        "  Number of Reviews <= 500."
    ]

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("\n".join(summary))

    print("\n" + "=" * 70)
    print("EVALUATION REPORTS SAVED")
    print("=" * 70)
    print(f"\nRecommendation evaluation:\n{recommendation_path}")
    print(f"\nContent similarity evaluation:\n{content_path}")
    print(f"\nCluster evaluation:\n{cluster_path}")
    print(f"\nHidden gems evaluation:\n{hidden_gems_path}")
    print(f"\nEvaluation summary:\n{summary_path}")
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
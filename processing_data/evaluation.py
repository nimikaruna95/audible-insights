#evaluation.py
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(REPORT_DIR, exist_ok=True)

DATA_PATH = os.path.join(PROCESSED_DIR, "clustered_books.csv")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
SVD_PATH = os.path.join(MODEL_DIR, "svd_model.pkl")

# Load data and trained models
def load_resources():
    df = pd.read_csv(DATA_PATH)
    tfidf_matrix = joblib.load(TFIDF_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    svd = joblib.load(SVD_PATH)

    if len(df) != tfidf_matrix.shape[0]:
        raise ValueError(
            "Dataset and TF-IDF matrix are not aligned.")
    return df, tfidf_matrix, kmeans, svd

def evaluate_content_similarity(df,tfidf_matrix,sample_size=100,top_n=5):
    sample_size = min(sample_size, len(df))
    indices = np.linspace(0,len(df) - 1,sample_size,dtype=int)

    top_scores = []
    maximum_scores = []

    for index in indices:
        scores = cosine_similarity(tfidf_matrix[index],tfidf_matrix).flatten()

        scores[index] = -1

        top = np.sort(scores)[::-1][:top_n]

        top_scores.append(np.mean(top))
        maximum_scores.append(top[0])

    result = pd.DataFrame({
        "Metric": [
            "Average Similarity",
            "Average Maximum Similarity",
            f"Average Top-{top_n} Similarity"
        ],

        "Score": [
            np.mean(top_scores),
            np.mean(maximum_scores),
            np.mean(top_scores)
        ]})
    
    result.to_csv(os.path.join(REPORT_DIR, "content_similarity_evaluation.csv"),index=False)
    return result

def evaluate_clusters(df, kmeans, svd):
    X = svd.transform(joblib.load(TFIDF_PATH))

    labels = kmeans.predict(X)
    score = silhouette_score(X, labels)

    report = df.groupby("Cluster").size().reset_index(name="Book_Count")
    report["Percentage"] = (report["Book_Count"] / len(df) * 100).round(2)

    report["Silhouette_Score"] = score
    report["Inertia"] = kmeans.inertia_

    report.to_csv(os.path.join(REPORT_DIR,"cluster_recommendation_evaluation.csv"),index=False)
    return score, report

def precision_at_k(recommended, relevant, k):
    recommended = list(recommended)[:k]
    relevant = set(relevant)

    if not recommended:
        return 0.0

    return len(set(recommended) & relevant) / len(recommended)

def recall_at_k(recommended, relevant, k):
    recommended = list(recommended)[:k]
    relevant = set(relevant)

    if not relevant:
        return 0.0

    return len(set(recommended) & relevant) / len(relevant)

def evaluate_recommendations(df,tfidf_matrix,sample_size=100,k=5,threshold=0.20):
    indices = np.linspace(0,len(df) - 1,min(sample_size, len(df)),dtype=int)

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for index in indices:
        scores = cosine_similarity(tfidf_matrix[index],tfidf_matrix).flatten()

        scores[index] = -1

        recommended_indices = np.argsort(scores)[::-1][:k]

        relevant_indices = np.where(scores >= threshold)[0]

        recommended = set(recommended_indices)
        relevant = set(relevant_indices)

        precision = (len(recommended & relevant) / k)
        recall = (len(recommended & relevant) / len(relevant) if relevant else 0)
        f1 = (2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    result = pd.DataFrame({
        "Metric": [
            f"Precision@{k}",
            f"Recall@{k}",
            f"F1@{k}"
        ],

        "Score": [
            np.mean(precision_scores),
            np.mean(recall_scores),
            np.mean(f1_scores)
        ]})
    result.to_csv(os.path.join(REPORT_DIR,"recommendation_proxy_evaluation.csv"),index=False)
    return result

def evaluate_coverage(df,tfidf_matrix,sample_size=100,k=5):
    indices = np.linspace(0,len(df) - 1,min(sample_size, len(df)),dtype=int)
    recommended_books = set()

    for index in indices:
        scores = cosine_similarity(tfidf_matrix[index],tfidf_matrix).flatten()
        scores[index] = -1
        top_indices = np.argsort(scores)[::-1][:k]

        recommended_books.update(top_indices)

    coverage = len(recommended_books) / len(df)
    result = pd.DataFrame({
        "Metric": ["Recommendation Coverage"],
        "Score": [coverage],
        "Unique_Recommended_Books": [len(recommended_books)],
        "Dataset_Books": [len(df)]})
    
    result.to_csv(os.path.join(REPORT_DIR,"recommendation_coverage.csv"),index=False)
    return result

def evaluate_diversity(df, tfidf_matrix, sample_size=100, k=5):
    indices = np.linspace(0,len(df) - 1,min(sample_size, len(df)),dtype=int)
    diversity_scores = []
    for index in indices:
        scores = cosine_similarity(tfidf_matrix[index],tfidf_matrix).flatten()

        scores[index] = -1
        top_indices = np.argsort(scores)[::-1][:k]

        if len(top_indices) < 2:
            continue

        pairwise = cosine_similarity(tfidf_matrix[top_indices])
        values = pairwise[np.triu_indices(len(top_indices),k=1)]

        diversity_scores.append(1 - np.mean(values))

    diversity = np.mean(diversity_scores)
    result = pd.DataFrame(
        {"Metric": ["Average Recommendation Diversity"],
         "Score": [diversity]})
    
    result.to_csv(os.path.join(REPORT_DIR,"recommendation_diversity.csv"),index=False)
    return result

def evaluate_popularity_bias(df,tfidf_matrix,sample_size=100,k=5):
    indices = np.linspace(0,len(df) - 1,min(sample_size, len(df)),dtype=int)

    input_reviews = []
    recommended_reviews = []

    for index in indices:
        scores = cosine_similarity(tfidf_matrix[index],tfidf_matrix).flatten()
        scores[index] = -1
        top_indices = np.argsort(scores)[::-1][:k]

        input_reviews.append(df.iloc[index]["Number of Reviews"])
        recommended_reviews.extend(df.iloc[top_indices]["Number of Reviews"])

    average_input = np.mean(input_reviews)
    average_recommended = np.mean(recommended_reviews)

    ratio = (average_recommended / average_input if average_input > 0 else 0)

    result = pd.DataFrame({
        "Metric": ["Popularity Bias Ratio"],
        "Score": [ratio],
        "Average_Input_Reviews": [average_input],
        "Average_Recommended_Reviews": [average_recommended]})
    result.to_csv(os.path.join(REPORT_DIR,"recommendation_bias.csv"),index=False)

    return result

if __name__ == "__main__":
    print("AUDIBLE INSIGHTS - EVALUATION")
    df, tfidf_matrix, kmeans, svd = load_resources()

    print("Dataset:", len(df))
    print("TF-IDF features:", tfidf_matrix.shape[1])
    print("Clusters:", kmeans.n_clusters)

    # Content similarity
    content = evaluate_content_similarity(df,tfidf_matrix)

    print("\nContent Similarity:")
    print(content.to_string(index=False))

    # Clustering
    cluster_score, cluster_report = evaluate_clusters(df,kmeans,svd)

    print("\nClustering Silhouette Score:")
    print(round(cluster_score, 4))

    # Proxy recommendation metrics
    proxy = evaluate_recommendations(df,tfidf_matrix,sample_size=100,k=5,threshold=0.20)

    print("\nProxy Recommendation Evaluation:")
    print(proxy.to_string(index=False))

    # Coverage
    coverage = evaluate_coverage(df,tfidf_matrix)

    # Diversity
    diversity = evaluate_diversity(df,tfidf_matrix)

    # Popularity bias
    bias = evaluate_popularity_bias(df,tfidf_matrix)

    # Final summary
    summary = pd.DataFrame({
        "Category": [
            "Content Similarity",
            "Content Similarity",
            "Clustering",
            "Recommendation Proxy",
            "Recommendation Proxy",
            "Recommendation Proxy",
            "Coverage",
            "Diversity",
            "Popularity"],

        "Metric": [
            "Average Similarity",
            "Maximum Similarity",
            "Silhouette Score",
            "Precision@5",
            "Recall@5",
            "F1@5",
            "Recommendation Coverage",
            "Average Recommendation Diversity",
            "Popularity Bias Ratio"],

        "Score": [
            content.iloc[0]["Score"],
            content.iloc[1]["Score"],
            cluster_score,
            proxy.iloc[0]["Score"],
            proxy.iloc[1]["Score"],
            proxy.iloc[2]["Score"],
            coverage.iloc[0]["Score"],
            diversity.iloc[0]["Score"],
            bias.iloc[0]["Score"]]})

    summary.to_csv(os.path.join(REPORT_DIR,"evaluation_summary.csv"),index=False)

    print("\nEvaluation summary saved to:")
    print(os.path.join(REPORT_DIR,"evaluation_summary.csv"))

    print("\nImportant:")
    print("Precision, Recall and F1 are proxy metrics because the dataset has no user-item interaction history.")
    print("RMSE is not required because this project does not predict user ratings.")
    print("\nEvaluation completed successfully.")
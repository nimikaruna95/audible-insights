# clustering.py
import os
import pickle
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")
FIGURE_DIR = os.path.join(BASE_DIR, "outputs", "figures")

for folder in [MODEL_DIR, REPORT_DIR, FIGURE_DIR]:
    os.makedirs(folder, exist_ok=True)

DATA_PATH = os.path.join(PROCESSED_DIR, "engineered_books.csv")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")

OUTPUT_PATH = os.path.join(PROCESSED_DIR, "clustered_books.csv")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
SVD_PATH = os.path.join(MODEL_DIR, "svd_model.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "clustering_config.pkl")

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found.")
    
    return pd.read_csv(DATA_PATH)

def load_tfidf():
    if not os.path.exists(TFIDF_PATH):
        raise FileNotFoundError(f"TF-IDF matrix not found")
    
    with open(TFIDF_PATH, "rb") as file:
        return pickle.load(file)

# Reduce sparse TF-IDF dimensions before K-Means
def reduce_dimensions(X, components=100):
    components = min(components, X.shape[1] - 1)

    svd = TruncatedSVD(n_components=components,random_state=42)
    reduced = svd.fit_transform(X)

    print("Original dimensions:", X.shape[1])
    print("Reduced dimensions:", reduced.shape[1])
    print("Explained variance:",round(svd.explained_variance_ratio_.sum(), 4))

    joblib.dump(svd, SVD_PATH)
    return reduced, svd

# Compare different K values
def evaluate_clusters(X, min_k=2, max_k=10):
    max_k = min(max_k, len(X) - 1)
    results = []

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k,
            random_state=42,
            n_init=10,
            max_iter=300)

        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)

        results.append({
            "K": k,
            "WCSS": model.inertia_,
            "Silhouette_Score": score})

        print(f"K={k} | WCSS={model.inertia_:.2f} | "f"Silhouette={score:.4f}")

    evaluation = pd.DataFrame(results)
    evaluation.to_csv(os.path.join(REPORT_DIR, "cluster_evaluation.csv"),index=False)

    best_row = evaluation.loc[evaluation["Silhouette_Score"].idxmax()]
    best_k = int(best_row["K"])

    # Elbow plot
    plt.figure(figsize=(8, 5))
    plt.plot(evaluation["K"], evaluation["WCSS"], marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "elbow_method.png"),dpi=150)
    plt.close()

    # Silhouette plot
    plt.figure(figsize=(8, 5))
    plt.plot(evaluation["K"],evaluation["Silhouette_Score"],marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score by K")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "silhouette_scores.png"),dpi=150)
    plt.close()

    return best_k, evaluation

def train_kmeans(X, k):
    model = KMeans(n_clusters=k,random_state=42,n_init=10,max_iter=300)

    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)

    print("Final K:", k)
    print("Inertia:", round(model.inertia_, 4))
    print("Silhouette Score:", round(score, 4))

    return model, labels, score

def create_cluster_report(df):
    report = df.groupby("Cluster").agg(
        Book_Count=("Book Name", "count"),
        Average_Rating=("Rating_Filled", "mean"),
        Average_Reviews=("Number of Reviews", "mean"),
        Total_Reviews=("Number of Reviews", "sum"),
        Average_Price=("Price", "mean"),
        Average_Weighted_Rating=("Weighted_Rating", "mean"),
        Average_Popularity=("Popularity_Score", "mean"),
        Average_Quality=("Quality_Score", "mean"),
        Average_Listening_Hours=("Listening_Time_Hours", "mean")).reset_index()

    report.to_csv(os.path.join(REPORT_DIR, "cluster_report.csv"),index=False)
    return report

def visualize_clusters(X, labels):
    svd = TruncatedSVD(n_components=2, random_state=42)
    reduced = svd.fit_transform(X)

    plt.figure(figsize=(9, 6))

    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        plt.scatter(reduced[mask, 0],reduced[mask, 1],
            label=f"Cluster {cluster}",alpha=0.6,s=25)

    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.title("Book Clusters")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "cluster_visualization.png"),dpi=150)
    plt.close()

if __name__ == "__main__":
    print("AUDIBLE INSIGHTS - CLUSTERING PIPELINE")

    df = load_data()
    tfidf_matrix = load_tfidf()

    if len(df) != tfidf_matrix.shape[0]:
        raise ValueError("Dataset and TF-IDF matrix rows do not match.")

    X, svd = reduce_dimensions(tfidf_matrix)

    best_k, evaluation = evaluate_clusters(X)
    model, labels, score = train_kmeans(X, best_k)

    df["Cluster"] = labels

    create_cluster_report(df)
    visualize_clusters(X, labels)

    joblib.dump(model, KMEANS_PATH)

    with open(CONFIG_PATH, "wb") as file:
        pickle.dump({"n_clusters": best_k, "silhouette_score": score, "svd_components": X.shape[1]},file)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Clustered dataset saved to:")
    print(OUTPUT_PATH)
    print("K-Means model saved to:")
    print(KMEANS_PATH)
    print("Clustering completed successfully.")


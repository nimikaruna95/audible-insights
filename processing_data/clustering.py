import os
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
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")
FIGURE_DIR = os.path.join(BASE_DIR, "outputs", "figures")

for folder in [MODEL_DIR, REPORT_DIR, FIGURE_DIR]:
    os.makedirs(folder, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "engineered_books.csv")
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")
OUTPUT_PATH = os.path.join(DATA_DIR, "clustered_books.csv")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
CLUSTER_INFO_PATH = os.path.join(REPORT_DIR, "cluster_report.csv")
ELBOW_PATH = os.path.join(FIGURE_DIR, "elbow_method.png")
SILHOUETTE_PATH = os.path.join(FIGURE_DIR, "silhouette_scores.png")
CLUSTER_VISUALIZATION_PATH = os.path.join(FIGURE_DIR, "cluster_visualization.png")


def load_data():
    print("AUDIBLE INSIGHTS - CLUSTERING PIPELINE")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_PATH}\n\n"
            "Run feature_engineering.py first."
        )
    df = pd.read_csv(DATA_PATH)
    print("Dataset Shape:", df.shape)
    return df


def load_tfidf_matrix():
    print("\nLoading TF-IDF matrix...")
    if not os.path.exists(TFIDF_MATRIX_PATH):
        raise FileNotFoundError(
            f"\nTF-IDF matrix not found:\n{TFIDF_MATRIX_PATH}\n\n"
            "Run nlp.py first."
        )

    import pickle
    with open(TFIDF_MATRIX_PATH, "rb") as file:
        tfidf_matrix = pickle.load(file)

    print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
    return tfidf_matrix


def evaluate_clusters(X, min_k=2, max_k=10):
    max_k = min(max_k, X.shape[0] - 1)
    if max_k < min_k:
        raise ValueError("Not enough records for clustering.")

    cluster_values = range(min_k, max_k + 1)
    wcss, silhouettes = [], []

    for k in cluster_values:
        print(f"Testing K = {k}")
        model = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = model.fit_predict(X)
        wcss.append(model.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    cluster_values = list(cluster_values)

    plt.figure(figsize=(9, 6))
    plt.plot(cluster_values, wcss, marker="o")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS / Inertia")
    plt.xticks(cluster_values)
    plt.tight_layout()
    plt.savefig(ELBOW_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(cluster_values, silhouettes, marker="o")
    plt.title("Silhouette Score by Number of Clusters")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.xticks(cluster_values)
    plt.tight_layout()
    plt.savefig(SILHOUETTE_PATH, dpi=150)
    plt.close()

    evaluation_df = pd.DataFrame({
        "K": cluster_values,
        "WCSS": wcss,
        "Silhouette_Score": silhouettes
    })
    evaluation_df.to_csv(
        os.path.join(REPORT_DIR, "cluster_evaluation.csv"),
        index=False
    )

    best_index = int(np.argmax(silhouettes))
    best_k = cluster_values[best_index]
    best_score = silhouettes[best_index]

    print("\nCluster Evaluation:")
    print(evaluation_df.to_string(index=False))
    print(f"\nBest K based on Silhouette Score: {best_k}")
    print(f"Best Silhouette Score: {best_score:.4f}")

    return best_k, evaluation_df


def train_kmeans(X, n_clusters):
    print("\nTraining final K-Means model...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)

    print("Final Model Inertia:", round(kmeans.inertia_, 4))
    print("Final Silhouette Score:", round(score, 4))
    return kmeans, labels, score


def add_cluster_labels(df, labels):
    df = df.copy()
    df["Cluster"] = labels
    print("\nCluster Distribution:")
    print(df["Cluster"].value_counts().sort_index())
    return df


def visualize_clusters(X, labels):
    if X.shape[1] < 2:
        print("Not enough dimensions for visualization.")
        return

    print("\nGenerating cluster visualization...")
    svd = TruncatedSVD(n_components=2, random_state=42)
    reduced = svd.fit_transform(X)

    plt.figure(figsize=(10, 7))
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            label=f"Cluster {cluster}",
            alpha=0.6,
            s=25
        )

    plt.title("Book Clusters using TF-IDF + K-Means")
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CLUSTER_VISUALIZATION_PATH, dpi=150)
    plt.close()

    print(
        "2D SVD Explained Variance:",
        round(svd.explained_variance_ratio_.sum(), 4)
    )


def create_cluster_report(df):
    print("\nCreating cluster profile...")

    report = (
        df.groupby("Cluster")
        .agg({
            "Book Name": "count",
            "Rating": "mean",
            "Number of Reviews": "mean",
            "Price": "mean"
        })
        .reset_index()
        .rename(columns={
            "Book Name": "Book_Count",
            "Rating": "Average_Rating",
            "Number of Reviews": "Average_Reviews",
            "Price": "Average_Price"
        })
    )

    report["Average_Rating"] = report["Average_Rating"].round(3)
    report["Average_Reviews"] = report["Average_Reviews"].round(2)
    report["Average_Price"] = report["Average_Price"].round(2)

    report.to_csv(CLUSTER_INFO_PATH, index=False)

    print("\nCluster Profile:")
    print(report.to_string(index=False))
    return report


def show_cluster_samples(df, samples_per_cluster=5):
    print("\nSAMPLE BOOKS FROM EACH CLUSTER")

    for cluster in sorted(df["Cluster"].unique()):
        print(f"\nCluster {cluster}:")

        subset = (
            df[df["Cluster"] == cluster]
            .sort_values("Weighted_Rating", ascending=False)
            .head(samples_per_cluster)
        )

        columns = [
            c for c in ["Book Name", "Author", "Rating"]
            if c in subset.columns
        ]
        print(subset[columns].to_string(index=False))


def save_outputs(kmeans, df):
    joblib.dump(kmeans, KMEANS_PATH)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nK-Means model saved to:", KMEANS_PATH)
    print("Clustered dataset saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    df = load_data()
    X = load_tfidf_matrix()

    if len(df) != X.shape[0]:
        raise ValueError(
            "Dataset and TF-IDF matrix row counts do not match.\n"
            f"Dataset rows: {len(df)}\n"
            f"TF-IDF rows: {X.shape[0]}"
        )

    best_k, evaluation_df = evaluate_clusters(X, min_k=2, max_k=10)
    kmeans, labels, final_score = train_kmeans(X, best_k)

    df = add_cluster_labels(df, labels)
    visualize_clusters(X, labels)
    create_cluster_report(df)
    show_cluster_samples(df, samples_per_cluster=5)
    save_outputs(kmeans, df)

    print("\nCLUSTERING COMPLETED SUCCESSFULLY")
    print(f"Optimal K: {best_k}")
    print(f"Silhouette Score: {final_score:.4f}")
    print(f"Clustered Dataset: {OUTPUT_PATH}")
    print(f"K-Means Model: {KMEANS_PATH}")
    print(f"Cluster Report: {CLUSTER_INFO_PATH}")
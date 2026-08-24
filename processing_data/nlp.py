# nlp.py
import os
import pickle
import warnings
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

INPUT_PATH = os.path.join(PROCESSED_DIR, "engineered_books.csv")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")
BOOK_DATA_PATH = os.path.join(MODEL_DIR, "nlp_books.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "nlp_config.pkl")

# Load engineered book data
def load_data():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Dataset not found.")

    df = pd.read_csv(INPUT_PATH)
    print("Dataset Shape:", df.shape)
    return df

# Prepare the text representation used by TF-IDF
def prepare_text(df):
    if "combined_features" not in df.columns:
        required = ["Book Name", "Author", "Genre", "Description"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["combined_features"] = (
            df["Book Name"].fillna("").astype(str) + " " +
            df["Author"].fillna("").astype(str) + " " +
            df["Genre"].fillna("").astype(str) + " " +
            df["Description"].fillna("").astype(str))

    df["NLP_Text"] = (
        df["combined_features"].fillna("").astype(str)
        .str.lower().str.replace(r"[^a-zA-Z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True).str.strip())

    df["NLP_Text"] = df["NLP_Text"].replace("", "unknown book")
    print("NLP text preparation completed.")

    return df

# Create TF-IDF vectors
def create_tfidf(df):
    vectorizer = TfidfVectorizer(
        lowercase=True,stop_words="english",
        ngram_range=(1, 2),min_df=2,
        max_features=30000,sublinear_tf=True)

    matrix = vectorizer.fit_transform(df["NLP_Text"])

    print("TF-IDF Matrix Shape:", matrix.shape)
    print("Vocabulary Size:", len(vectorizer.vocabulary_))

    return vectorizer, matrix

# Save NLP models and metadata
def save_models(vectorizer, matrix, df):
    with open(TFIDF_PATH, "wb") as file:
        pickle.dump(vectorizer, file)

    with open(MATRIX_PATH, "wb") as file:
        pickle.dump(matrix, file)

    columns = [
        "Book Name", "Author", "Genre", "Description",
        "Rating", "Number of Reviews", "Price",
        "Weighted_Rating", "NLP_Text"]

    book_data = df[[c for c in columns if c in df.columns]].copy()

    with open(BOOK_DATA_PATH, "wb") as file:
        pickle.dump(book_data, file)

    config = {
        "ngram_range": (1, 2),"min_df": 2,
        "max_features": 30000,"stop_words": "english"}

    with open(CONFIG_PATH, "wb") as file:
        pickle.dump(config, file)

    print("NLP models saved successfully.")

# Test content-based recommendations
def recommend_books(book_name, df, matrix, top_n=10):
    matches = df[df["Book Name"].str.lower().str.strip() == str(book_name).lower().strip()]

    if matches.empty:
        return pd.DataFrame()

    index = matches.index[0]
    scores = cosine_similarity(matrix[index], matrix).flatten()

    indices = [i for i in scores.argsort()[::-1] if i != index][:top_n]

    result = df.loc[indices,["Book Name", "Author", "Genre", "Rating", "Number of Reviews"]].copy()

    result["Similarity_Score"] = [scores[i] for i in indices]
    return result.reset_index(drop=True)

if __name__ == "__main__":
    print("AUDIBLE INSIGHTS - NLP PIPELINE")

    df = prepare_text(load_data())
    vectorizer, matrix = create_tfidf(df)
    save_models(vectorizer, matrix, df)

    # Test using the first book with useful text
    candidates = df[df["NLP_Text"].str.len() > 50]

    if not candidates.empty:
        test_book = candidates.iloc[0]["Book Name"]
        recommendations = recommend_books(test_book, df, matrix, top_n=10)

        print("\nSelected Book:", test_book)
        print("\nTop Recommendations:")
        print(recommendations.to_string(index=False))

        recommendations.to_csv(os.path.join(REPORT_DIR, "sample_nlp_recommendations.csv"),index=False)
    print("\nNLP pipeline completed successfully.")


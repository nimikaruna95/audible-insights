import os
import re
import pickle
import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

INPUT_PATH = os.path.join(DATA_DIR, "engineered_books.csv")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")
BOOK_DATA_PATH = os.path.join(MODEL_DIR, "nlp_books.pkl")


def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def load_data():
    print("AUDIBLE INSIGHTS - NLP PIPELINE")
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"\nDataset not found:\n{INPUT_PATH}\n\n"
            "Run feature_engineering.py first."
        )

    df = pd.read_csv(INPUT_PATH)
    print("Dataset Shape:", df.shape)
    return df


def prepare_text(df):
    required = ["Book Name", "Author", "Genre", "Description"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required:
        df[col] = df[col].fillna("").astype(str)

    for col, new_col in {
        "Book Name": "NLP_Title",
        "Author": "NLP_Author",
        "Genre": "NLP_Genre",
        "Description": "NLP_Description"
    }.items():
        df[new_col] = df[col].apply(preprocess_text)

    df["NLP_Text"] = (
        df["NLP_Title"] + " " + df["NLP_Title"] + " " +
        df["NLP_Author"] + " " + df["NLP_Author"] + " " +
        df["NLP_Genre"] + " " + df["NLP_Genre"] + " " +
        df["NLP_Description"]
    ).apply(preprocess_text).replace("", "unknown book")

    print("NLP text preparation completed.")
    return df


def create_tfidf(df):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform(df["NLP_Text"])

    print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
    print("Vocabulary Size:", len(vectorizer.vocabulary_))
    return vectorizer, tfidf_matrix


def save_models(vectorizer, tfidf_matrix, df):
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(MATRIX_PATH, "wb") as f:
        pickle.dump(tfidf_matrix, f)

    columns = [
        "Book Name", "Author", "Genre", "Description",
        "Rating", "Number of Reviews", "Price",
        "Popularity", "Weighted_Rating", "NLP_Text"
    ]

    book_data = df[[c for c in columns if c in df.columns]].copy()

    with open(BOOK_DATA_PATH, "wb") as f:
        pickle.dump(book_data, f)

    print("\nNLP models saved successfully.")


def recommend_books(book_name, df, tfidf_matrix, top_n=10):
    matches = df[
        df["Book Name"].str.lower().str.strip() ==
        str(book_name).lower().strip()
    ]

    if matches.empty:
        print(f"\nBook not found: {book_name}")
        return pd.DataFrame()

    book_index = matches.index[0]
    scores = cosine_similarity(
        tfidf_matrix[book_index], tfidf_matrix
    ).flatten()

    indices = [i for i in scores.argsort()[::-1] if i != book_index][:top_n]

    columns = [
        "Book Name", "Author", "Genre",
        "Rating", "Number of Reviews", "Price"
    ]

    result = df.loc[indices, columns].copy()
    result["Similarity_Score"] = [scores[i] for i in indices]
    return result.reset_index(drop=True)


def test_recommendation(df, tfidf_matrix):
    candidates = (
        df[df["NLP_Text"].str.len() > 50]["Book Name"]
        .drop_duplicates()
        .head(1)
        .tolist()
    )

    if not candidates:
        print("No suitable test book found.")
        return

    test_book = candidates[0]
    print(f"\nSelected Book: {test_book}")

    recommendations = recommend_books(
        test_book, df, tfidf_matrix, top_n=10
    )

    if recommendations.empty:
        return

    columns = [
        "Book Name", "Author", "Genre",
        "Rating", "Number of Reviews", "Similarity_Score"
    ]

    print("\nRecommended Books:")
    print(recommendations[columns].to_string(index=False))


if __name__ == "__main__":
    df = load_data()
    df = prepare_text(df)

    vectorizer, tfidf_matrix = create_tfidf(df)
    save_models(vectorizer, tfidf_matrix, df)

    test_recommendation(df, tfidf_matrix)

    print("\nNLP PIPELINE COMPLETED SUCCESSFULLY")
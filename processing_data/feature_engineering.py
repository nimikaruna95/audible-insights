import os
import re
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def extract_genres(text):
    if pd.isna(text) or str(text).strip().lower() in ["", "unknown", "-1"]:
        return "Unknown"

    genres = []
    for part in re.split(r",(?=#)", str(text).strip()):
        match = re.search(r"#\s*[\d,]+\s+in\s+(.+)", part.strip(), re.I)
        if match:
            genre = re.sub(
                r"\s*\(See Top 100.*?\)", "", match.group(1).strip(), flags=re.I
            ).strip()
            if genre:
                genres.append(genre)

    genres = list(dict.fromkeys(genres))
    return " | ".join(genres) if genres else "Unknown"


def safe_log1p(series):
    return np.log1p(
        pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
    )


def min_max_scale(series):
    minimum, maximum = series.min(), series.max()
    if maximum == minimum:
        return pd.Series(0, index=series.index)
    return (series - minimum) / (maximum - minimum)


def classify_length(length):
    return "Short" if length < 40 else "Medium" if length < 100 else "Long"


def rating_category(rating):
    return (
        "Excellent" if rating >= 4.5 else
        "Good" if rating >= 4.0 else
        "Average" if rating >= 3.0 else
        "Low"
    )


def feature_engineering(df):
    print("=" * 70)
    print("AUDIBLE INSIGHTS - FEATURE ENGINEERING")
    print("=" * 70)
    print("\nInitial Dataset Shape:", df.shape)

    numeric = ["Rating", "Number of Reviews", "Price"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    text_columns = [
        "Book Name", "Author", "Description",
        "Listening Time", "Ranks and Genre"
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    print("\nCreating Genre feature...")
    df["Genre"] = df["Ranks and Genre"].apply(extract_genres)
    df["Genre_Count"] = df["Genre"].apply(
        lambda x: 0 if x == "Unknown" else len(x.split("|"))
    )

    print("Cleaning text features...")
    df["Clean_Description"] = df["Description"].apply(clean_text)
    df["Clean_Title"] = df["Book Name"].apply(clean_text)
    df["Clean_Author"] = df["Author"].apply(clean_text)
    df["Clean_Genre"] = df["Genre"].apply(clean_text)

    print("Creating review features...")
    df["Log_Reviews"] = safe_log1p(df["Number of Reviews"])
    df["Popularity"] = df["Rating"] * df["Log_Reviews"]

    print("Creating weighted rating...")
    C = df["Rating"].mean()
    m = max(df["Number of Reviews"].quantile(0.75), 1)

    df["Weighted_Rating"] = (
        (df["Number of Reviews"] / (df["Number of Reviews"] + m)) * df["Rating"]
        + (m / (df["Number of Reviews"] + m)) * C
    )

    print("Creating author popularity...")
    author_popularity = df.groupby("Author")["Number of Reviews"].sum()
    df["Author_Popularity"] = df["Author"].map(author_popularity).fillna(0)
    df["Log_Author_Popularity"] = safe_log1p(df["Author_Popularity"])

    print("Creating description features...")
    df["Description_Length"] = df["Description"].apply(lambda x: len(str(x).split()))
    df["Description_Characters"] = df["Description"].apply(lambda x: len(str(x)))
    df["Book_Length"] = df["Description_Length"].apply(classify_length)
    df["Rating_Category"] = df["Rating"].apply(rating_category)

    price_median = df["Price"].median()
    df["Price_Category"] = df["Price"].apply(
        lambda x:
        "Budget" if x <= price_median * 0.75 else
        "Moderate" if x <= price_median * 1.5 else
        "Premium"
    )

    print("Creating normalized features...")
    df["Rating_Normalized"] = min_max_scale(df["Rating"])
    df["Reviews_Normalized"] = min_max_scale(df["Log_Reviews"])
    df["Price_Normalized"] = min_max_scale(df["Price"])
    df["Popularity_Normalized"] = min_max_scale(df["Popularity"])

    print("Creating combined NLP features...")
    df["combined_features"] = (
        df["Clean_Title"] + " " +
        df["Clean_Author"] + " " +
        df["Clean_Genre"] + " " +
        df["Clean_Description"]
    ).apply(clean_text)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    engineered_numeric = [
        "Popularity", "Weighted_Rating", "Author_Popularity",
        "Log_Author_Popularity", "Description_Length",
        "Description_Characters", "Genre_Count", "Log_Reviews",
        "Rating_Normalized", "Reviews_Normalized",
        "Price_Normalized", "Popularity_Normalized"
    ]

    for col in engineered_numeric:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df.reset_index(drop=True, inplace=True)

    print("\nFeature Engineering Completed.")
    print("Final Dataset Shape:", df.shape)

    return df


if __name__ == "__main__":
    input_path = os.path.join(DATA_DIR, "cleaned_books.csv")
    output_path = os.path.join(DATA_DIR, "engineered_books.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"\nInput dataset not found:\n{input_path}\n\n"
            "Run data_preprocessing.py first."
        )

    print("\nLoading cleaned dataset...")
    df = pd.read_csv(input_path)

    df = feature_engineering(df)
    df.to_csv(output_path, index=False)

    print("\nEngineered dataset saved to:")
    print(output_path)

    sample_columns = [
        "Book Name", "Author", "Genre", "Rating",
        "Number of Reviews", "Popularity",
        "Weighted_Rating", "Author_Popularity",
        "Description_Length"
    ]

    print("\nSample Engineered Data:")
    print(df[[c for c in sample_columns if c in df.columns]].head())

    print("\nFeature engineering completed successfully.")
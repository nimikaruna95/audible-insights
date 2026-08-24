# feature_engineering.py
import os
import re
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(PROCESSED_DIR, exist_ok=True)

INPUT_PATH = os.path.join(PROCESSED_DIR, "cleaned_books.csv")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "engineered_books.csv")

# Clean text for NLP features
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()

# Extract Audible genres from the rank field
def extract_genres(text):
    if pd.isna(text) or str(text).strip().lower() in {"", "unknown", "-1"}:
        return "Unknown"

    genres = []
    for part in re.split(r",(?=#)", str(text).strip()):
        match = re.search(r"#\s*[\d,]+\s+in\s+(.+)", part, re.I)
        if match:
            genre = re.sub(
                r"\s*\(See Top 100.*?\)", "", match.group(1).strip(), flags=re.I
            ).strip()
            if genre:
                genres.append(genre)

    genres = list(dict.fromkeys(genres))
    return " | ".join(genres) if genres else "Unknown"

def safe_log1p(series):
    values = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
    return np.log1p(values)

def min_max_scale(series):
    minimum, maximum = series.min(), series.max()
    if maximum == minimum:
        return pd.Series(0, index=series.index)
    return (series - minimum) / (maximum - minimum)

def classify_length(length):
    if length < 40:
        return "Short"
    if length < 100:
        return "Medium"
    return "Long"

def rating_category(rating):
    if rating >= 4.5:
        return "Excellent"
    if rating >= 4.0:
        return "Good"
    if rating >= 3.0:
        return "Average"
    return "Low"

def feature_engineering(df):
    print("AUDIBLE INSIGHTS - FEATURE ENGINEERING")
    print("Initial Dataset Shape:", df.shape)

    # Remove duplicated feature created during preprocessing
    if "Log Reviews" in df.columns:
        df.drop(columns=["Log Reviews"], inplace=True)
        print("Removed duplicate column: Log Reviews")

    # Prepare numeric columns
    numeric = ["Rating", "Number of Reviews", "Price"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Prepare text columns
    text_columns = [
        "Book Name", "Author", "Description",
        "Listening Time", "Ranks and Genre"
    ]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    # Genre features
    df["Genre"] = df["Ranks and Genre"].apply(extract_genres)
    df["Genre_Count"] = df["Genre"].apply(lambda x: 0 if x == "Unknown" else len(x.split("|")))
    df["Primary_Genre"] = df["Genre"].apply(lambda x: x.split("|")[0].strip() if x != "Unknown" else "Unknown")

    # Clean text features
    df["Clean_Title"] = df["Book Name"].apply(clean_text)
    df["Clean_Author"] = df["Author"].apply(clean_text)
    df["Clean_Genre"] = df["Genre"].apply(clean_text)
    df["Clean_Primary_Genre"] = df["Primary_Genre"].apply(clean_text)
    df["Clean_Description"] = df["Description"].apply(clean_text)

    # Review and popularity features
    df["Log_Reviews"] = safe_log1p(df["Number of Reviews"])
    df["Popularity_Score"] = min_max_scale(df["Log_Reviews"])

    # Weighted rating reduces the effect of books with very few reviews
    C = df["Rating"].replace(0, np.nan).mean()
    C = 4.5 if pd.isna(C) else C
    m = max(df["Number of Reviews"].quantile(0.75), 1)

    df["Rating_Filled"] = df["Rating"].replace(0, np.nan).fillna(C)
    df["Weighted_Rating"] = (
        (df["Number of Reviews"] / (df["Number of Reviews"] + m))
        * df["Rating_Filled"]
        + (m / (df["Number of Reviews"] + m)) * C)

    # Author-level features
    author_reviews = df.groupby("Author")["Number of Reviews"].sum()
    author_books = df.groupby("Author")["Book Name"].count()
    author_rating = df.groupby("Author")["Rating_Filled"].mean()

    df["Author_Popularity"] = df["Author"].map(author_reviews).fillna(0)
    df["Author_Book_Count"] = df["Author"].map(author_books).fillna(0)
    df["Author_Average_Rating"] = df["Author"].map(author_rating).fillna(C)
    df["Log_Author_Popularity"] = safe_log1p(df["Author_Popularity"])
    df["Author_Popularity_Normalized"] = min_max_scale(
        df["Log_Author_Popularity"])

    # Listening time features
    if "Listening Time Minutes" in df.columns:
        df["Listening Time Minutes"] = pd.to_numeric(
            df["Listening Time Minutes"], errors="coerce"
        ).fillna(0)
    else:
        df["Listening Time Minutes"] = 0

    df["Listening_Time_Hours"] = df["Listening Time Minutes"] / 60
    df["Listening_Time_Category"] = pd.cut(
        df["Listening_Time_Hours"],
        bins=[-1, 2, 6, np.inf],
        labels=["Short", "Medium", "Long"]).astype(str)

    # Audible rank features
    if "Audible Rank" in df.columns:
        rank = pd.to_numeric(df["Audible Rank"], errors="coerce")
        df["Has_Audible_Rank"] = rank.notna().astype(int)
        df["Audible_Rank_Score"] = 1 / (1 + rank.fillna(np.inf))
        df["Audible_Rank_Score"] = df["Audible_Rank_Score"].replace(
            [np.inf, -np.inf], 0)
    else:
        df["Has_Audible_Rank"] = 0
        df["Audible_Rank_Score"] = 0

    # Price features
    df["Log_Price"] = safe_log1p(df["Price"])
    df["Price_Normalized"] = min_max_scale(df["Price"])

    median_price = df["Price"].median()
    df["Price_Category"] = np.select(
        [df["Price"] <= median_price * 0.75,df["Price"] <= median_price * 1.5],
        ["Budget", "Moderate"],default="Premium")

    # Rating and quality features
    df["Rating_Normalized"] = min_max_scale(df["Rating_Filled"])
    df["Weighted_Rating_Normalized"] = min_max_scale(df["Weighted_Rating"])
    df["Rating_Category"] = df["Rating_Filled"].apply(rating_category)
    df["Quality_Score"] = (0.7 * df["Rating_Normalized"] +0.3 * df["Weighted_Rating_Normalized"])
    df["Popularity_Quality_Score"] = (0.6 * df["Popularity_Score"] +0.4 * df["Quality_Score"])

    # Description features
    df["Description_Length"] = df["Description"].apply(lambda x: len(str(x).split()))
    df["Description_Characters"] = df["Description"].apply(len)
    df["Description_Length_Category"] = df["Description_Length"].apply(classify_length)

    # Normalized review feature
    df["Reviews_Normalized"] = min_max_scale(df["Log_Reviews"])

    # Combined NLP representation
    df["combined_features"] = (df["Clean_Title"] + " " + df["Clean_Title"] + " " +
        df["Clean_Author"] + " " + df["Clean_Author"] + " " +
        df["Clean_Genre"] + " " +  df["Clean_Genre"] + " " +
        df["Clean_Description"]).apply(clean_text)

    df["metadata_features"] = (df["Clean_Title"] + " " +
        df["Clean_Author"] + " " + df["Clean_Primary_Genre"]).apply(clean_text)

    # Availability flags
    df["Has_Description"] = df["Description_Length"].gt(0).astype(int)
    df["Has_Genre"] = df["Genre"].ne("Unknown").astype(int)
    df["Has_Listening_Time"] = df["Listening_Time_Hours"].gt(0).astype(int)
    df["Has_Rating"] = df["Rating"].gt(0).astype(int)

    # Duplicate-safe keys
    df["Book_Name_Key"] = df["Book Name"].apply(clean_text)
    df["Author_Key"] = df["Author"].apply(clean_text)

    # Remove invalid numeric values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_features = [
        "Rating", "Number of Reviews", "Price", "Rating_Filled",
        "Log_Reviews", "Reviews_Normalized", "Log_Price",
        "Price_Normalized", "Genre_Count", "Listening_Time_Hours",
        "Audible_Rank_Score", "Author_Popularity",
        "Author_Book_Count", "Author_Average_Rating",
        "Log_Author_Popularity", "Author_Popularity_Normalized",
        "Weighted_Rating", "Rating_Normalized",
        "Weighted_Rating_Normalized", "Popularity_Score",
        "Quality_Score", "Popularity_Quality_Score",
        "Description_Length", "Description_Characters"]

    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Remove exact duplicate book-author combinations
    before = len(df)
    df.drop_duplicates(subset=["Book Name", "Author"], inplace=True)
    print("Duplicate Book + Author records removed:", before - len(df))

    df.reset_index(drop=True, inplace=True)

    print("Final Dataset Shape:", df.shape)
    print("Missing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    return df

if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input dataset not found.")

    df = pd.read_csv(INPUT_PATH)
    df = feature_engineering(df)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Engineered dataset saved to:")
    print(OUTPUT_PATH)
    print("\nFeature engineering completed successfully.")


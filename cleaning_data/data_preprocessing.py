# data_preprocessing.py
import os
import re
import unicodedata

import numpy as np
import pandas as pd

# PROJECT PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

os.makedirs(PROCESSED_DIR, exist_ok=True)

FILE_1 = os.path.join(DATA_DIR, "Audible_Catlog.csv")
FILE_2 = os.path.join(DATA_DIR, "Audible_Catlog_Advanced_Features.csv")

# LOAD DATA
def load_data():
    print("\nLOADING DATASETS")

    if not os.path.exists(FILE_1):
        raise FileNotFoundError(f"Dataset 1 not found:\n{FILE_1}")

    if not os.path.exists(FILE_2):
        raise FileNotFoundError(f"Dataset 2 not found:\n{FILE_2}")

    df1 = pd.read_csv(FILE_1)
    df2 = pd.read_csv(FILE_2)
    print(f"Dataset 1 Shape: {df1.shape}")
    print(f"Dataset 2 Shape: {df2.shape}")
    return df1, df2

# STANDARDIZE COLUMN NAMES
def standardize_column_names(df):
    df = df.copy()
    df.columns = (df.columns.str.replace("\ufeff", "", regex=False).str.strip())
    return df

# NORMALIZE TEXT
def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value)
    value = unicodedata.normalize("NFKC", value)
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    value = value.casefold()
    return value

# CREATE MERGE KEYS
def create_merge_keys(df):
    df = df.copy()
    df["Book_Name_Key"] = df["Book Name"].apply(normalize_text)
    df["Author_Key"] = df["Author"].apply(normalize_text)
    return df

# REPLACE INVALID -1 VALUES
def replace_invalid_values(df):
    df = df.copy()

    # Numeric columns
    numeric_columns = ["Rating","Number of Reviews","Price"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column],errors="coerce")
            df[column] = df[column].replace(-1, np.nan)

    # Listening Time
    if "Listening Time" in df.columns:
        df["Listening Time"] = (df["Listening Time"].replace(-1, np.nan).replace("-1", np.nan))

    # Ranks and Genre
    if "Ranks and Genre" in df.columns:
        df["Ranks and Genre"] = (df["Ranks and Genre"].replace(-1, np.nan).replace("-1", np.nan))
    return df

# CLEAN STRING COLUMNS
def clean_string_columns(df):
    df = df.copy()
    string_columns = ["Book Name","Author","Description","Listening Time","Ranks and Genre"]

    for column in string_columns:
        if column in df.columns:
            df[column] = (df[column].astype(object).where(df[column].notna(), np.nan)
                          .astype("string").str.strip())
    return df

# REMOVE EXACT DUPLICATES
def remove_exact_duplicates(df, dataset_name):
    before = len(df)
    df = df.drop_duplicates(keep="first").copy()
    removed = before - len(df)
    print(f"{dataset_name} - exact duplicate rows removed: "f"{removed}")
    return df

# FIRST VALID VALUE
def first_valid(series):
    series = series.dropna()
    if len(series) == 0:
        return np.nan
    for value in series:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                continue
        return value
    return np.nan

# AVERAGE VALID VALUES
def average_valid(series):
    values = pd.to_numeric(series,errors="coerce").dropna()

    if len(values) == 0:
        return np.nan
    return values.mean()

# MAXIMUM VALID VALUE
def maximum_valid(series):
    values = pd.to_numeric(series,errors="coerce").dropna()

    if len(values) == 0:
        return np.nan
    return values.max()

# CONSOLIDATE DUPLICATE BOOKS
def consolidate_dataset(df, dataset_name):
    print(f"\nCONSOLIDATING {dataset_name}")
    df = create_merge_keys(df)
    duplicate_count = (df.duplicated(subset=["Book_Name_Key", "Author_Key"]).sum())
    print(f"Duplicate Book + Author records: "f"{duplicate_count}")

    grouped_rows = []

    for _, group in df.groupby(["Book_Name_Key", "Author_Key"],sort=False):

        row = {}

        # Book Name
        row["Book Name"] = first_valid(group["Book Name"])

        # Author
        row["Author"] = first_valid(group["Author"])

        # Rating
        if "Rating" in group.columns:
            row["Rating"] = average_valid(group["Rating"])

        # Number of Reviews
        if "Number of Reviews" in group.columns:
            row["Number of Reviews"] = maximum_valid(
                group["Number of Reviews"])

        # Price
        if "Price" in group.columns:
            row["Price"] = first_valid(group["Price"])

        # Description
        if "Description" in group.columns:
            row["Description"] = first_valid(group["Description"])

        # Listening Time
        if "Listening Time" in group.columns:
            row["Listening Time"] = first_valid(
                group["Listening Time"])

        # Ranks and Genre
        if "Ranks and Genre" in group.columns:
            row["Ranks and Genre"] = first_valid(
                group["Ranks and Genre"])

        # Merge keys
        row["Book_Name_Key"] = group["Book_Name_Key"].iloc[0]
        row["Author_Key"] = group["Author_Key"].iloc[0]

        grouped_rows.append(row)

    result = pd.DataFrame(grouped_rows)

    print(f"Rows before consolidation: {len(df)}")
    print(f"Rows after consolidation: {len(result)}")
    return result

# CLEAN DESCRIPTION
def clean_description(value):

    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text in ["", "nan", "None", "Unknown"]:
        return ""

    # Normalize escaped newline characters
    text = text.replace("\\n", " ")
    text = text.replace("\\r", " ")
    text = text.replace("\\t", " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Invalid scraped/error messages
    invalid_patterns = [
        r"Sorry, we just need to make sure you're not a robot",
        r"make sure your browser is accepting cookies",
        r"Oops!.*traffic is piling up",
        r"please try again in a short while",
        r"it will not have been processed",
        r"Go to the Amazon\.in home page",
        r"By completing your purchase",
        r"Conditions of Use",
        r"Amazon's Privacy Notice",
        r"Audible's Recurring Payment Terms",
        r"privacy notice"]

    for pattern in invalid_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return ""
    return text

# CONVERT LISTENING TIME TO MINUTES
def parse_listening_time(value):

    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text in ["","-1","Unknown","unknown","nan","None"]:
        return np.nan

    hours = 0
    minutes = 0

    hour_match = re.search(r"(\d+)\s*hours?",text,flags=re.IGNORECASE)
    minute_match = re.search(r"(\d+)\s*minutes?",text,flags=re.IGNORECASE)

    if hour_match:
        hours = int(hour_match.group(1))

    if minute_match:
        minutes = int(minute_match.group(1))

    total_minutes = hours * 60 + minutes
    if total_minutes == 0:
        return np.nan
    return total_minutes

def convert_listening_time(df):
    df = df.copy()
    if "Listening Time" not in df.columns:
        df["Listening Time"] = "Unknown"

    df["Listening Time Minutes"] = (df["Listening Time"].apply(parse_listening_time))
    return df

# EXTRACT GENRES
def extract_genres(value):
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if text in ["","-1","Unknown","unknown","nan","None"]:
        return "Unknown"
    genres = []
    pattern = (r"#\s*[\d,]+\s+in\s+" r"(.*?)(?=,\s*#\s*[\d,]+\s+in\s+|$)")

    matches = re.findall(pattern,text,flags=re.IGNORECASE)
    for genre in matches:
        genre = str(genre).strip()
        genre = re.sub(r"\s*\(See Top.*?\)","",
            genre,flags=re.IGNORECASE)

        genre = re.sub(r"\s*\(Audible Audiobooks & Originals\)","",
            genre,flags=re.IGNORECASE)

        if genre.casefold() == "audible audiobooks & originals":
            continue
        genre = genre.strip(" ,")
        if genre != "":
            genres.append(genre)

    # Remove duplicates while preserving order
    unique_genres = []
    for genre in genres:
        if genre not in unique_genres:
            unique_genres.append(genre)
    if not unique_genres:
        return "Unknown"
    return " | ".join(unique_genres)

# PROCESS GENRE
def process_genre(df):
    df = df.copy()
    if "Ranks and Genre" not in df.columns:
        df["Genre"] = "Unknown"
        return df

    df["Ranks and Genre"] = (df["Ranks and Genre"].astype(object)
        .where(df["Ranks and Genre"].notna(), "Unknown").astype(str).str.strip())

    df["Genre"] = df["Ranks and Genre"].apply(extract_genres)
    df["Genre"] = (df["Genre"].astype(object).where(df["Genre"].notna(), "Unknown")
        .astype(str).str.strip())

    df.loc[df["Genre"] == "", "Genre"] = "Unknown"
    return df

# EXTRACT AUDIBLE RANK
def extract_audible_rank(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    if text in ["","-1","Unknown","unknown","nan","None"]:
        return np.nan
    pattern = (r"#\s*([\d,]+)" r"\s+in\s+Audible\s+Audiobooks")
    match = re.search(pattern,text,flags=re.IGNORECASE)

    if match:
        rank_text = match.group(1).replace(",", "")
        try:
            return int(rank_text)
        except ValueError:
            return np.nan
    return np.nan

# PROCESS AUDIBLE RANK
def process_audible_rank(df):
    df = df.copy()
    if "Ranks and Genre" not in df.columns:
        df["Audible Rank"] = np.nan
        return df
    
    df["Ranks and Genre"] = (df["Ranks and Genre"].astype(object))
    df["Audible Rank"] = (df["Ranks and Genre"].apply(extract_audible_rank))
    return df

# CLEAN NUMERICAL FEATURES
def clean_numeric_features(df):
    df = df.copy()
    numeric_columns = ["Rating","Number of Reviews","Price"]
    for column in numeric_columns:
        if column not in df.columns:
            continue

        df[column] = pd.to_numeric(df[column],errors="coerce")

    # Rating
    if "Rating" in df.columns:
        df.loc[~df["Rating"].between(0, 5),"Rating"] = np.nan

    # Number of Reviews
    if "Number of Reviews" in df.columns:
        df.loc[df["Number of Reviews"] < 0,"Number of Reviews"] = np.nan

    # Price
    if "Price" in df.columns:
        df.loc[df["Price"] < 0,"Price"] = np.nan
    return df

# HANDLE MISSING VALUES
def handle_missing_values(df):
    df = df.copy()

    # Number of Reviews
    if "Number of Reviews" in df.columns:
        df["Number of Reviews"] = pd.to_numeric(df["Number of Reviews"],errors="coerce")
        df["Number of Reviews"] = (df["Number of Reviews"].fillna(0))

    # Price
    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"],errors="coerce")
        median_price = df["Price"].median()
        if pd.notna(median_price):
            df["Price"] = (df["Price"].fillna(median_price))

    # Description
    if "Description" in df.columns:
        df["Description"] = (df["Description"]
            .astype(object).where(df["Description"].notna(), "").apply(clean_description))

    # Listening Time
    if "Listening Time" in df.columns:
        df["Listening Time"] = (df["Listening Time"].astype(object)
            .where(df["Listening Time"].notna(), "Unknown").astype(str).str.strip())

    # Ranks and Genre
    if "Ranks and Genre" in df.columns:
        df["Ranks and Genre"] = (
            df["Ranks and Genre"].astype(object).where(df["Ranks and Genre"].notna(), "Unknown")
            .astype(str).str.strip())

    # Genre
    if "Genre" in df.columns:
        df["Genre"] = (df["Genre"].astype(object).where(df["Genre"].notna(), "Unknown").astype(str).str.strip())

    return df

# MERGE DATASETs
def merge_datasets(df1, df2):
    print("\nPREPARING DATASETS FOR MERGING")
    df1 = standardize_column_names(df1)
    df2 = standardize_column_names(df2)

    required_columns = [
        "Book Name",
        "Author",
        "Rating",
        "Number of Reviews",
        "Price"]

    # Validate Dataset 1
    for column in required_columns:
        if column not in df1.columns:
            raise KeyError(f"'{column}' missing from Dataset 1")

    # Validate Dataset 2
    for column in required_columns:
        if column not in df2.columns:
            raise KeyError(f"'{column}' missing from Dataset 2")

    # Replace -1
    df1 = replace_invalid_values(df1)
    df2 = replace_invalid_values(df2)

    # Clean strings
    df1 = clean_string_columns(df1)
    df2 = clean_string_columns(df2)

    # Exact duplicates
    df1 = remove_exact_duplicates(df1, "Dataset 1")
    df2 = remove_exact_duplicates(df2, "Dataset 2")

    # Consolidate duplicate books BEFORE merge
    df1 = consolidate_dataset(df1, "Dataset 1")
    df2 = consolidate_dataset(df2, "Dataset 2")

    # Rename overlapping columns
    df1 = df1.rename(columns={
        "Rating": "Rating_catalog",
        "Number of Reviews": "Number of Reviews_catalog",
        "Price": "Price_catalog"})

    df2 = df2.rename(columns={
        "Rating": "Rating_advanced",
        "Number of Reviews": "Number of Reviews_advanced",
        "Price": "Price_advanced"})

    # Merge
    df = pd.merge(df1,df2,
        on=["Book_Name_Key", "Author_Key"],
        how="outer",
        suffixes=("_catalog", "_advanced"),
        indicator=True)

    print("\nMerge Statistics:")
    print(df["_merge"].value_counts())
    print(f"\nMerged Dataset Shape: {df.shape}")

    # Data Source
    df["Data_Source"] = df["_merge"].map({
        "both": "Both",
        "left_only": "Catalog Only",
        "right_only": "Advanced Only"})

    # Book Name
    df["Book Name"] = (df["Book Name_catalog"].combine_first(df["Book Name_advanced"]))

    # Author
    df["Author"] = (df["Author_catalog"].combine_first(df["Author_advanced"]))

    # Rating
    df["Rating"] = (df["Rating_advanced"].combine_first(df["Rating_catalog"]))

    # Reviews
    df["Number of Reviews"] = (df["Number of Reviews_advanced"].combine_first(df["Number of Reviews_catalog"]))

    # Price
    df["Price"] = (df["Price_advanced"].combine_first(df["Price_catalog"]))

    # Description
    if "Description" not in df.columns:
        df["Description"] = ""

    # Listening Time
    if "Listening Time" not in df.columns:
        df["Listening Time"] = "Unknown"

    # Ranks and Genre
    if "Ranks and Genre" not in df.columns:
        df["Ranks and Genre"] = "Unknown"

    # Keep required columns
    columns_to_keep = [
        "Book Name",
        "Author",
        "Rating",
        "Number of Reviews",
        "Price",
        "Description",
        "Listening Time",
        "Ranks and Genre",
        "Book_Name_Key",
        "Author_Key",
        "Data_Source",
        "_merge"]

    df = df[[column for column in columns_to_keep if column in df.columns]]
    return df

# FINAL CLEANING
def clean_data(df):
    print("\nFINAL DATA CLEANING")
    print(f"Initial merged shape: {df.shape}")

    # Numerical Cleaning
    df = clean_numeric_features(df)

    # Genre
    df = process_genre(df)

    # Audible Rank
    df = process_audible_rank(df)

    # Listening Time
    df = convert_listening_time(df)

    # Missing Values
    df = handle_missing_values(df)

    # Description Cleaning
    if "Description" in df.columns:
        df["Description"] = df["Description"].apply(clean_description)

    # Clean Text Columns
    text_columns = [
        "Book Name",
        "Author",
        "Description",
        "Listening Time",
        "Ranks and Genre",
        "Genre",
        "Data_Source",
        "Book_Name_Key",
        "Author_Key"]

    for column in text_columns:
        if column in df.columns:
            df[column] = (df[column].astype(object).where(df[column].notna(), "Unknown").astype(str).str.strip())

    # Data Quality Flags
    df["Has_Description"] = (df["Description"].str.strip().ne("").astype(int))
    df["Has_Genre"] = (df["Genre"].str.strip().ne("Unknown").astype(int))
    df["Has_Listening_Time"] = (df["Listening Time Minutes"].notna().astype(int))

    # Rating Filled
    if "Rating" in df.columns:
        rating_median = df["Rating"].median()
        if pd.notna(rating_median):
            df["Rating_Filled"] = (df["Rating"].fillna(rating_median))
        else:
            df["Rating_Filled"] = df["Rating"].fillna(0)

    # Log Reviews
    if "Number of Reviews" in df.columns:
        df["Log Reviews"] = np.log1p(df["Number of Reviews"])

    # Remove Merge Indicator
    if "_merge" in df.columns:
        df.drop(columns=["_merge"], inplace=True)

    # Ensure Unique Book + Author
    before = len(df)
    df = df.drop_duplicates(subset=["Book_Name_Key", "Author_Key"],keep="first")
    removed = before - len(df)
    print("Duplicate Book + Author records removed after merge: "f"{removed}")

    # Reset Index
    df.reset_index(drop=True, inplace=True)

    # Column Order
    preferred_order = [
        "Book Name",
        "Author",
        "Rating",
        "Rating_Filled",
        "Number of Reviews",
        "Log Reviews",
        "Price",
        "Description",
        "Genre",
        "Ranks and Genre",
        "Audible Rank",
        "Listening Time",
        "Listening Time Minutes",
        "Data_Source",
        "Has_Description",
        "Has_Genre",
        "Has_Listening_Time",
        "Book_Name_Key",
        "Author_Key"]

    existing_columns = [column for column in preferred_order if column in df.columns]

    remaining_columns = [column for column in df.columns if column not in existing_columns]
    df = df[existing_columns + remaining_columns]

    # Final Report
    print("\nFINAL DATA QUALITY REPORT")
    print(f"Final Dataset Shape: {df.shape}")
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)
    print("\nFinal Columns:")
    for column in df.columns:
        print(f"  - {column}")

    # Data Source
    print("\nData Source Distribution:")
    if "Data_Source" in df.columns:
        print(df["Data_Source"].value_counts())

    # Data Quality Flags
    print("\nData Quality Flags:")
    quality_columns = ["Has_Description","Has_Genre","Has_Listening_Time"]

    for column in quality_columns:
        if column in df.columns:
            count = int(df[column].sum())
            percentage = count / len(df) * 100
            print(f"{column}: {count} "f"({percentage:.2f}%)")

    # Genre Summary
    if "Genre" in df.columns:
        print("\nTop Extracted Genres:")
        genre_counts = (df.loc[df["Genre"] != "Unknown","Genre"].value_counts().head(15))
        if len(genre_counts) > 0:
            print(genre_counts)
        else:
            print("No valid genres found.")

    # Rating Summary
    if "Rating" in df.columns:
        print("\nRating Summary:")
        print(df["Rating"].describe().round(2))

    # Review Summary
    if "Number of Reviews" in df.columns:
        print("\nReview Summary:")
        print(df["Number of Reviews"].describe().round(2))

    print("\nSample Data:")
    print(df.head().to_string())

    # Numerical Statistics
    print("\nNumerical Statistics:")
    numeric_columns = df.select_dtypes(include=np.number).columns
    if len(numeric_columns) > 0:
        print(df[numeric_columns].describe().round(2))
    return df

# SAVE DATA
def save_data(df):
    output_path = os.path.join(PROCESSED_DIR,"cleaned_books.csv")
    df.to_csv(output_path,index=False,encoding="utf-8-sig")

    print("\nDATA SAVING")
    print("Cleaned dataset saved to:")
    print(output_path)
    return output_path

# MAIN
if __name__ == "__main__":
    print("\nAUDIBLE INSIGHTS - DATA PREPROCESSING PIPELINE")
    df1, df2 = load_data()
    merged_df = merge_datasets(df1, df2)
    cleaned_df = clean_data(merged_df)
    save_data(cleaned_df)
    print("\nDATA PREPROCESSING COMPLETED SUCCESSFULLY")
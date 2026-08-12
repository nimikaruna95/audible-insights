# data_preprocessing.py
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_data():
    file1 = os.path.join(DATA_DIR, "Audible_Catlog.csv")
    file2 = os.path.join(DATA_DIR, "Audible_Catlog_Advanced_Features.csv")

    print("=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    if not os.path.exists(file1):
        raise FileNotFoundError(f"Dataset 1 not found: {file1}")
    if not os.path.exists(file2):
        raise FileNotFoundError(f"Dataset 2 not found: {file2}")

    df1, df2 = pd.read_csv(file1), pd.read_csv(file2)
    print(f"Dataset 1 Shape: {df1.shape}")
    print(f"Dataset 2 Shape: {df2.shape}")
    return df1, df2


def standardize_column_names(df):
    df = df.copy()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return df


def check_duplicate_keys(df, dataset_name):
    required = ["Book Name", "Author"]

    for col in required:
        if col not in df.columns:
            raise KeyError(f"'{col}' column not found in {dataset_name}")

    count = df.duplicated(subset=required).sum()
    print(f"{dataset_name} duplicate Book Name + Author records: {count}")


def merge_datasets(df1, df2):
    print("\n" + "=" * 70)
    print("MERGING DATASETS")
    print("=" * 70)

    df1, df2 = standardize_column_names(df1), standardize_column_names(df2)

    for col in ["Book Name", "Author"]:
        if col not in df1.columns:
            raise KeyError(f"'{col}' missing from Dataset 1")
        if col not in df2.columns:
            raise KeyError(f"'{col}' missing from Dataset 2")

    check_duplicate_keys(df1, "Dataset 1")
    check_duplicate_keys(df2, "Dataset 2")

    df = pd.merge(
        df1, df2,
        on=["Book Name", "Author"],
        how="outer",
        suffixes=("_catalog", "_advanced"),
        indicator=True
    )

    print("\nMerge Statistics:")
    print(df["_merge"].value_counts())
    print(f"\nMerged Dataset Shape: {df.shape}")

    return df.drop(columns="_merge")


def clean_numeric_column(df, column):
    if column in df.columns:
        df[column] = (
            df[column].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("£", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.strip()
        )
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def combine_columns(df, columns, new_column, method="mean"):
    columns = [c for c in columns if c in df.columns]

    for col in columns:
        df = clean_numeric_column(df, col)

    if len(columns) == 2:
        df[new_column] = getattr(df[columns], method)(axis=1)
        df.drop(columns=columns, inplace=True)
    elif len(columns) == 1:
        df[new_column] = df[columns[0]]
        df.drop(columns=columns, inplace=True)

    return df


def clean_data(df):
    print("\n" + "=" * 70)
    print("DATA CLEANING")
    print("=" * 70)

    df = standardize_column_names(df.copy())
    print(f"Initial Shape: {df.shape}")
    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nMissing Values BEFORE Cleaning:")
    print(df.isnull().sum())

    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"\nExact duplicate rows removed: {before - len(df)}")

    before = len(df)
    df.dropna(subset=["Book Name", "Author"], inplace=True)
    print(f"Rows removed due to missing Book Name/Author: {before - len(df)}")

    for col in ["Book Name", "Author"]:
        df[col] = df[col].astype(str).str.strip()

    before = len(df)
    duplicates = df.duplicated(subset=["Book Name", "Author"]).sum()
    df.drop_duplicates(subset=["Book Name", "Author"], inplace=True)
    print(f"Duplicate Book + Author records removed: {duplicates}")

    df = combine_columns(
        df,
        ["Number of Reviews_catalog", "Number of Reviews_advanced"],
        "Number of Reviews",
        "max"
    )

    df = combine_columns(
        df,
        ["Rating_catalog", "Rating_advanced"],
        "Rating",
        "mean"
    )

    df = combine_columns(
        df,
        ["Price_catalog", "Price_advanced"],
        "Price",
        "mean"
    )

    if "Description" in df.columns:
        df["Description"] = df["Description"].fillna("").astype(str).str.strip()
    else:
        df["Description"] = ""

    for col in ["Listening Time", "Ranks and Genre"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    if "Ranks and Genre" not in df.columns:
        df["Ranks and Genre"] = "Unknown"

    numeric_columns = ["Rating", "Number of Reviews", "Price"]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(median) else median)

    if "Rating" in df.columns:
        df["Rating"] = df["Rating"].clip(0, 5)

    if "Number of Reviews" in df.columns:
        df["Number of Reviews"] = df["Number of Reviews"].clip(lower=0)

    if "Price" in df.columns:
        df["Price"] = df["Price"].clip(lower=0)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    print(f"\nRemaining exact duplicates: {df.duplicated().sum()}")

    df.reset_index(drop=True, inplace=True)

    print("\n" + "=" * 70)
    print("FINAL DATA QUALITY REPORT")
    print("=" * 70)
    print(f"Final Dataset Shape: {df.shape}")

    print("\nMissing Values AFTER Cleaning:")
    print(df.isnull().sum())

    print("\nData Types:")
    print(df.dtypes)

    print("\nFinal Columns:")
    print(df.columns.tolist())

    print("\nBasic Numerical Statistics:")
    print(df.describe().round(2))

    return df


def save_data(df):
    output_path = os.path.join(DATA_DIR, "cleaned_books.csv")
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("DATA SAVING")
    print("=" * 70)
    print("Cleaned dataset saved to:")
    print(output_path)

    return output_path


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("AUDIBLE INSIGHTS - DATA PREPROCESSING PIPELINE")
    print("=" * 70)

    df1, df2 = load_data()
    merged_df = merge_datasets(df1, df2)
    cleaned_df = clean_data(merged_df)
    save_data(cleaned_df)

    print("\n" + "=" * 70)
    print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)


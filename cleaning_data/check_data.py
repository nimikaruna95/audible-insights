# check_data.py
import pandas as pd
import numpy as np
import os

# FILE PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

FILE_1 = os.path.join(DATA_DIR, "Audible_Catlog.csv")
FILE_2 = os.path.join(DATA_DIR, "Audible_Catlog_Advanced_Features.csv")

# LOAD DATA
def load_data(file_path):

    if not os.path.exists(file_path):
        print(f"ERROR: File not found -> {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded: {file_path}")
        return df

    except Exception as e:
        print(f"ERROR while loading {file_path}")
        print(e)
        return None

# BASIC DATASET INFORMATION
def basic_information(df, dataset_name):

    print(f"\nBASIC INFORMATION - {dataset_name}")
    print(f"Number of rows    : {df.shape[0]}")
    print(f"Number of columns : {df.shape[1]}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 records:")
    print(df.head())

# MISSING VALUES
def check_missing_values(df, dataset_name):
    print(f"\nMISSING VALUES - {dataset_name}")

    missing_count = df.isnull().sum()
    missing_percentage = (df.isnull().mean() * 100).round(2)

    missing_df = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": missing_percentage })

    print(missing_df)
    print("\nColumns containing missing values:")

    found_missing = False

    for column in df.columns:
        count = missing_count[column]

        if count > 0:
            found_missing = True
            print(f"  {column}: {count} missing "f"({missing_percentage[column]}%)")

    if not found_missing:
        print("  No missing values found.")

# CHECK -1 VALUES
def check_negative_one(df, dataset_name):
    print(f"\n-1 VALUES - {dataset_name}")

    found = False

    for column in df.columns:
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        count = (numeric_values == -1).sum()

        if count > 0:
            found = True
            percentage = (count / len(df)) * 100
            print(f"  {column}: {count} values "f"({percentage:.2f}%)")

    if not found:
        print("  No -1 values found.")

# DUPLICATE CHECK
def check_duplicates(df, dataset_name):
    print(f"\nDUPLICATES - {dataset_name}")

    duplicate_count = df.duplicated().sum()
    print(f"Exact duplicate rows: {duplicate_count}")

    if duplicate_count > 0:
        print("\nSample duplicate rows:")
        duplicates = df[df.duplicated(keep=False)]
        print(duplicates.head(10))

# BOOK + AUTHOR DUPLICATES
def check_book_author_duplicates(df, dataset_name):
    print(f"\nBOOK + AUTHOR DUPLICATES - {dataset_name}")

    required_columns = ["Book Name", "Author"]

    if not all(column in df.columns for column in required_columns):
        print("Book Name or Author column is missing.")
        return

    duplicate_count = df.duplicated(subset=["Book Name", "Author"]).sum()

    print(f"Duplicate Book Name + Author combinations: " f"{duplicate_count}")

    if duplicate_count > 0:
        duplicates = df[df.duplicated(subset=["Book Name", "Author"],keep=False)]
        print("\nSample duplicates:")
        print(duplicates[["Book Name", "Author"]].head(20))

# UNIQUE VALUES
def check_unique_values(df, dataset_name):
    print(f"\nUNIQUE VALUES - {dataset_name}")
    unique_df = pd.DataFrame({
        "Unique Values": df.nunique(),
        "Total Rows": len(df)})
    print(unique_df)

# NUMERICAL SUMMARY
def numerical_summary(df, dataset_name):
    print(f"\nNUMERICAL SUMMARY - {dataset_name}")
    numerical_columns = df.select_dtypes(include=np.number).columns

    if len(numerical_columns) == 0:
        print("No numerical columns found.")
        return
    print(df[numerical_columns].describe())

# STRING / OBJECT SUMMARY
def string_summary(df, dataset_name):
    print(f"\nSTRING COLUMN SUMMARY - {dataset_name}")
    string_columns = df.select_dtypes(include="object").columns

    for column in string_columns:
        print(f"\nColumn: {column}")
        print(f"Unique values: {df[column].nunique()}")
        print("Sample values:")

        values = df[column].dropna().head(5)
        for value in values:
            print(f"  {value}")

# CHECK COMMON COLUMNS
def compare_columns(df1, df2):
    print("\nCOLUMN COMPARISON")

    common_columns = df1.columns.intersection(df2.columns)
    dataset1_only = df1.columns.difference(df2.columns)
    dataset2_only = df2.columns.difference(df1.columns)

    print("\nColumns present in BOTH datasets:")
    for column in common_columns:
        print(f"  - {column}")

    print("\nColumns only in Audible_Catlog.csv:")
    for column in dataset1_only:
        print(f"  - {column}")

    print("\nColumns only in Audible_Catlog_Advanced_Features.csv:")
    for column in dataset2_only:
        print(f"  - {column}")

# CHECK COMMON BOOKS
def compare_books(df1, df2):
    print("\nBOOK OVERLAP BETWEEN DATASETS")

    required_columns = ["Book Name", "Author"]
    if not all(column in df1.columns for column in required_columns):
        print("Dataset 1 is missing Book Name or Author.")
        return

    if not all(column in df2.columns for column in required_columns):
        print("Dataset 2 is missing Book Name or Author.")
        return

    books1 = set(zip(
            df1["Book Name"].astype(str).str.strip(),
            df1["Author"].astype(str).str.strip()))

    books2 = set(zip(
            df2["Book Name"].astype(str).str.strip(),
            df2["Author"].astype(str).str.strip()))

    common_books = books1.intersection(books2)

    print(f"Books in Dataset 1 : {len(books1)}")
    print(f"Books in Dataset 2 : {len(books2)}")
    print(f"Common books       : {len(common_books)}")

# MAIN FUNCTION
def main():
    print("\nAUDIBLE INSIGHTS - DATA CHECK")

    # Load datasets
    df1 = load_data(FILE_1)
    df2 = load_data(FILE_2)

    if df1 is None or df2 is None:
        print("\nUnable to continue because a dataset could not be loaded.")
        return

    # Dataset 1
    basic_information(df1, "Audible_Catlog.csv")
    check_missing_values(df1, "Audible_Catlog.csv")
    check_negative_one(df1, "Audible_Catlog.csv")
    check_duplicates(df1, "Audible_Catlog.csv")
    check_book_author_duplicates(df1, "Audible_Catlog.csv")
    check_unique_values(df1, "Audible_Catlog.csv")
    numerical_summary(df1, "Audible_Catlog.csv")
    string_summary(df1, "Audible_Catlog.csv")

    # Dataset 2
    basic_information(df2, "Audible_Catlog_Advanced_Features.csv")
    check_missing_values(df2, "Audible_Catlog_Advanced_Features.csv")
    check_negative_one(df2, "Audible_Catlog_Advanced_Features.csv")
    check_duplicates(df2, "Audible_Catlog_Advanced_Features.csv")
    check_book_author_duplicates(df2, "Audible_Catlog_Advanced_Features.csv")
    check_unique_values(df2, "Audible_Catlog_Advanced_Features.csv")
    numerical_summary(df2, "Audible_Catlog_Advanced_Features.csv")
    string_summary(df2, "Audible_Catlog_Advanced_Features.csv")

    # Comparing the datasets
    compare_columns(df1, df2)
    compare_books(df1, df2)
    print("\nDATA CHECK COMPLETED")

# MAIN
if __name__ == "__main__":
    main()

#data_preprocessing.py
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def load_data():
    file1 = os.path.join(BASE_DIR, 'data', 'Audible_Catlog.csv')
    file2 = os.path.join(BASE_DIR, 'data', 'Audible_Catlog_Advanced_Features.csv')
    return pd.read_csv(file1), pd.read_csv(file2)

def merge_datasets(df1, df2):
    return pd.merge(df1, df2, on=['Book Name', 'Author'], how='inner')

def clean_data(df):
    df.columns = df.columns.str.strip()

    # Merge ratings
    df['Rating'] = (df['Rating_x'] + df['Rating_y']) / 2

    df.drop(columns=['Rating_x','Rating_y'], inplace=True)

    df = df.drop_duplicates()
    df = df.dropna(subset=['Book Name','Author'])

    df['Description'] = df['Description'].fillna("No description")

    return df

def save(df):
    path = os.path.join(BASE_DIR, 'data', 'cleaned_books.csv')
    df.to_csv(path, index=False)

if __name__ == "__main__":
    d1, d2 = load_data()
    merged = merge_datasets(d1, d2)
    cleaned = clean_data(merged)
    save(cleaned)
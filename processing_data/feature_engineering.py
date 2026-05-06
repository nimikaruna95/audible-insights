#feature_engineering.py
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def feature_engineering(df):

    df['Genre'] = df['Ranks and Genre'].apply(
        lambda x: x.split(',')[0] if isinstance(x, str) else ''
    )

    # COMBINED FEATURES (IMPORTANT)
    df['combined_features'] = (
        df['Genre'] + " " +
        df['Author'] + " " +
        df['Description']
    )

    # Popularity
    df['Popularity'] = df['Rating'] * df['Number of Reviews_x'].fillna(0)

    return df

if __name__ == "__main__":
    path = os.path.join(BASE_DIR, 'data', 'cleaned_books.csv')
    df = pd.read_csv(path)

    df = feature_engineering(df)

    out = os.path.join(BASE_DIR, 'data', 'engineered_books.csv')
    df.to_csv(out, index=False)
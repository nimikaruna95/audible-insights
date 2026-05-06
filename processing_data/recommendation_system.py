#recommendation_system.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

path = os.path.join(BASE_DIR, 'data', 'clustered_books.csv')
df = pd.read_csv(path)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# ---------------- CONTENT BASED ----------------
def content_based(book_name, top_n=10):
    matches = df[df['Book Name'].str.contains(book_name, case=False, na=False)]

    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]

    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    indices = [i[0] for i in scores]

    return df.iloc[indices]


# ---------------- CLUSTER BASED ----------------
def cluster_based(book_name):
    matches = df[df['Book Name'].str.contains(book_name, case=False, na=False)]

    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]
    cluster_id = df.loc[idx, 'Cluster']

    return df[df['Cluster'] == cluster_id].head(10)


# ---------------- HYBRID ----------------
def recommend_books(book_name, top_n=5):

    content = content_based(book_name, top_n)
    cluster = cluster_based(book_name)

    combined = pd.concat([content, cluster]).drop_duplicates()

    return combined[['Book Name','Author','Rating']].head(top_n)


# ---------------- HIDDEN GEMS ----------------
def get_hidden_gems():
    return df[
        (df['Rating'] > 4.5) &
        (df['Number of Reviews_x'] < 500)
    ][['Book Name','Author','Rating']].head(5)

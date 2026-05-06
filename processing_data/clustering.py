#clustering.py(with evaluation)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

path = os.path.join(BASE_DIR, 'data', 'engineered_books.csv')
df = pd.read_csv(path)

tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['combined_features'])

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

out = os.path.join(BASE_DIR, 'data', 'clustered_books.csv')
df.to_csv(out, index=False)


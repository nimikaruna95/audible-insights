#app.py

import streamlit as st
import pandas as pd
import os
import plotly.express as px
from recommendation_system import recommend_books, get_hidden_gems

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
path = os.path.join(BASE_DIR, 'data', 'clustered_books.csv')

df = pd.read_csv(path)

st.set_page_config(layout="wide")

st.title("📚 Audible Insights Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Filters")

min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 4.0)

# Extract Genre
df['Genre'] = df['Ranks and Genre'].apply(
    lambda x: x.split(',')[0] if isinstance(x, str) else ''
)

selected_genre = st.sidebar.selectbox("Select Genre", ["All"] + list(df['Genre'].unique()))

# Apply filter
if selected_genre != "All":
    df = df[df['Genre'] == selected_genre]

df = df[df['Rating'] >= min_rating]

# ---------------- DASHBOARD ----------------

col1, col2 = st.columns(2)

# Rating Distribution
with col1:
    fig1 = px.histogram(df, x='Rating', title="Rating Distribution")
    st.plotly_chart(fig1, use_container_width=True)

# Top Genres
with col2:
    genre_counts = df['Genre'].value_counts().head(10)
    fig2 = px.bar(x=genre_counts.index, y=genre_counts.values,
                  labels={'x': 'Genre', 'y': 'Count'},
                  title="Top Genres")
    st.plotly_chart(fig2, use_container_width=True)

# Scatter Plot
fig3 = px.scatter(
    df,
    x='Number of Reviews_x',
    y='Rating',
    title="Ratings vs Reviews",
    hover_data=['Book Name']
)
st.plotly_chart(fig3, use_container_width=True)

# ---------------- RECOMMENDATION ----------------

st.subheader("🔍 Get Book Recommendations")

book = st.selectbox("Choose a book", df['Book Name'].dropna().unique())

if st.button("Recommend"):
    recs = recommend_books(book)

    st.subheader("📖 Recommended Books")
    st.dataframe(recs)

# ---------------- HIDDEN GEMS ----------------

st.subheader("💎 Hidden Gems")
st.dataframe(get_hidden_gems())

# ---------------- TOP BOOKS ----------------

st.subheader("🔥 Top Rated Books")
top_books = df.sort_values(by='Rating', ascending=False).head(5)
st.dataframe(top_books[['Book Name','Author','Rating']])
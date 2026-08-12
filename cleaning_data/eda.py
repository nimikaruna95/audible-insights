# eda.py

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_books.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 70)
print("AUDIBLE INSIGHTS - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nCleaned dataset not found:\n{DATA_PATH}\n\n"
        "Please run data_preprocessing.py first."
    )

df = pd.read_csv(DATA_PATH)

if "Description" in df.columns:
    df["Description"] = (
        df["Description"].fillna("No description available")
        .astype(str).str.strip()
    )

text_columns = ["Book Name", "Author", "Listening Time", "Ranks and Genre"]
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

numeric_columns = ["Rating", "Number of Reviews", "Price"]
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()


def hist_plot(x, title, xlabel, filename, bins=30):
    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x=x, bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Books")
    save_plot(filename)


def box_plot(x, title, filename):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[x])
    plt.title(title)
    plt.xlabel(x)
    save_plot(filename)


def bar_plot(data, title, xlabel, ylabel, filename, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    sns.barplot(x=data.values, y=data.index)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_plot(filename)


print("\nDataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())
print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nSummary Statistics:")
print(df.describe(include="all"))

missing = df.isnull().sum().sort_values(ascending=False)
print("\nMissing Values:")
print(missing)

if missing.sum() > 0:
    plt.figure(figsize=(10, 5))
    missing[missing > 0].plot(kind="bar")
    plt.title("Missing Values by Column")
    plt.xlabel("Columns")
    plt.ylabel("Missing Count")
    plt.xticks(rotation=45)
    save_plot("missing_values.png")
else:
    print("\nNo missing values found.")

duplicate_rows = df.duplicated().sum()
duplicate_books = df.duplicated(
    subset=["Book Name", "Author"]
).sum()

print("\nDuplicate Rows:", duplicate_rows)
print("Duplicate Book + Author Records:", duplicate_books)

total_records = len(df)
unique_books = df["Book Name"].nunique()
unique_authors = df["Author"].nunique()

print("\nTotal Records:", total_records)
print("Unique Books:", unique_books)
print("Unique Authors:", unique_authors)

print("\nAverage Rating:", round(df["Rating"].mean(), 2))
print("\nRating Statistics:")
print(df["Rating"].describe())

hist_plot(
    "Rating",
    "Rating Distribution",
    "Rating",
    "rating_distribution.png",
    20
)

box_plot("Rating", "Rating Boxplot", "rating_boxplot.png")

print("\nPrice Statistics:")
print(df["Price"].describe())

hist_plot(
    "Price",
    "Book Price Distribution",
    "Price",
    "price_distribution.png"
)

box_plot("Price", "Price Boxplot", "price_boxplot.png")

print("\nNumber of Reviews Statistics:")
print(df["Number of Reviews"].describe())

hist_plot(
    "Number of Reviews",
    "Number of Reviews Distribution",
    "Number of Reviews",
    "reviews_distribution.png"
)

df["Log_Reviews"] = np.log1p(df["Number of Reviews"])

hist_plot(
    "Log_Reviews",
    "Log-Transformed Review Distribution",
    "log(1 + Number of Reviews)",
    "log_reviews_distribution.png"
)

author_counts = df["Author"].value_counts().head(10)
print("\nTop Authors by Number of Books:")
print(author_counts)

bar_plot(
    author_counts,
    "Top 10 Authors by Number of Books",
    "Number of Books",
    "Author",
    "top_authors_by_books.png"
)

author_ratings = df.groupby("Author").agg(
    Average_Rating=("Rating", "mean"),
    Books=("Book Name", "count"),
    Reviews=("Number of Reviews", "sum")
)

author_ratings_filtered = (
    author_ratings[author_ratings["Books"] >= 3]
    .sort_values("Average_Rating", ascending=False)
    .head(10)
)

print("\nTop Rated Authors:")
print(author_ratings_filtered)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=author_ratings_filtered.reset_index(),
    x="Average_Rating",
    y="Author"
)
plt.title("Top 10 Authors by Average Rating")
plt.xlabel("Average Rating")
plt.ylabel("Author")
plt.xlim(0, 5)
save_plot("top_rated_authors.png")

top_reviews = df.sort_values(
    "Number of Reviews", ascending=False
).head(10)

print("\nMost Reviewed Books:")
print(top_reviews[
    ["Book Name", "Author", "Rating", "Number of Reviews"]
])

plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_reviews,
    x="Number of Reviews",
    y="Book Name"
)
plt.title("Top 10 Most Reviewed Books")
plt.xlabel("Number of Reviews")
plt.ylabel("Book Name")
save_plot("most_reviewed_books.png")

top_rated = df.sort_values(
    ["Rating", "Number of Reviews"],
    ascending=[False, False]
).head(10)

print("\nTop Rated Books:")
print(top_rated[
    ["Book Name", "Author", "Rating", "Number of Reviews"]
])

top_rated.to_csv(
    os.path.join(REPORT_DIR, "top_rated_books.csv"),
    index=False
)

for x, y, title, xlabel, filename in [
    (
        "Number of Reviews", "Rating",
        "Rating vs Number of Reviews",
        "Number of Reviews", "rating_vs_reviews.png"
    ),
    (
        "Price", "Rating",
        "Rating vs Price",
        "Price", "rating_vs_price.png"
    ),
    (
        "Log_Reviews", "Rating",
        "Rating vs Log(Number of Reviews)",
        "log(1 + Number of Reviews)", "rating_vs_log_reviews.png"
    )
]:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Rating")
    save_plot(filename)

numeric_columns = [
    col for col in ["Rating", "Number of Reviews", "Price"]
    if col in df.columns
]

correlation = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Heatmap")
save_plot("correlation_heatmap.png")


def extract_genres(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value or value.lower() in ["unknown", "-1"]:
        return []

    genres = []

    for part in re.split(r",(?=#)", value):
        match = re.search(
            r"#\s*[\d,]+\s+in\s+(.+)",
            part.strip(),
            flags=re.IGNORECASE
        )

        if match:
            genre = re.sub(
                r"\s*\(See Top 100.*?\)",
                "",
                match.group(1).strip(),
                flags=re.IGNORECASE
            ).strip()

            if genre:
                genres.append(genre)

    return genres


df["Genre_List"] = df["Ranks and Genre"].apply(extract_genres)

genre_exploded = (
    df[
        [
            "Book Name",
            "Author",
            "Rating",
            "Number of Reviews",
            "Genre_List"
        ]
    ]
    .explode("Genre_List")
    .dropna(subset=["Genre_List"])
)

genre_exploded = genre_exploded[
    genre_exploded["Genre_List"].str.strip() != ""
]

genre_counts = genre_exploded["Genre_List"].value_counts()

print("\nTop Genres:")
print(genre_counts.head(15))

top_genres = genre_counts.head(15)

bar_plot(
    top_genres,
    "Top 15 Book Genres",
    "Number of Books",
    "Genre",
    "top_genres.png",
    (11, 7)
)

genre_rating = (
    genre_exploded.groupby("Genre_List")["Rating"]
    .agg(Average_Rating="mean", Book_Count="count")
)

genre_rating_filtered = (
    genre_rating[genre_rating["Book_Count"] >= 5]
    .sort_values("Average_Rating", ascending=False)
    .head(10)
)

print("\nTop Genres by Average Rating:")
print(genre_rating_filtered)

plt.figure(figsize=(11, 7))
sns.barplot(
    x=genre_rating_filtered["Average_Rating"].values,
    y=genre_rating_filtered.index
)
plt.title("Top Genres by Average Rating")
plt.xlabel("Average Rating")
plt.ylabel("Genre")
plt.xlim(0, 5)
save_plot("genre_average_rating.png")

genre_reviews = (
    genre_exploded.groupby("Genre_List")["Number of Reviews"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop Genres by Total Reviews:")
print(genre_reviews)

bar_plot(
    genre_reviews,
    "Top Genres by Total Number of Reviews",
    "Total Reviews",
    "Genre",
    "genre_total_reviews.png",
    (11, 7)
)

df["Description_Length"] = (
    df["Description"].fillna("").astype(str).str.len()
)

print("\nDescription Length Statistics:")
print(df["Description_Length"].describe())

hist_plot(
    "Description_Length",
    "Description Length Distribution",
    "Description Length (Characters)",
    "description_length.png"
)

description_available = (
    df["Description"] != "No description available"
)

description_counts = pd.Series({
    "Available": description_available.sum(),
    "Missing/Unavailable": (~description_available).sum()
})

print("\nDescription Availability:")
print(description_counts)

plt.figure(figsize=(7, 5))
sns.barplot(
    x=description_counts.index,
    y=description_counts.values
)
plt.title("Description Availability")
plt.xlabel("Description Status")
plt.ylabel("Number of Books")
save_plot("description_availability.png")

Q1_reviews = df["Number of Reviews"].quantile(0.25)
Q3_reviews = df["Number of Reviews"].quantile(0.75)
IQR_reviews = Q3_reviews - Q1_reviews
review_upper_limit = Q3_reviews + 1.5 * IQR_reviews
review_outliers = df[
    df["Number of Reviews"] > review_upper_limit
]

print("\nReview Outlier Analysis:")
print(f"Upper Bound: {review_upper_limit:.2f}")
print(f"Review Outliers: {len(review_outliers)}")

Q1_price = df["Price"].quantile(0.25)
Q3_price = df["Price"].quantile(0.75)
IQR_price = Q3_price - Q1_price
price_upper_limit = Q3_price + 1.5 * IQR_price
price_outliers = df[df["Price"] > price_upper_limit]

print("\nPrice Outlier Analysis:")
print(f"Upper Bound: {price_upper_limit:.2f}")
print(f"Price Outliers: {len(price_outliers)}")

genre_counts.reset_index().rename(
    columns={
        "Genre_List": "Genre",
        "count": "Number_of_Books"
    }
).to_csv(
    os.path.join(REPORT_DIR, "genre_analysis.csv"),
    index=False
)

genre_rating.reset_index().to_csv(
    os.path.join(REPORT_DIR, "genre_rating_analysis.csv"),
    index=False
)

author_ratings.reset_index().to_csv(
    os.path.join(REPORT_DIR, "author_analysis.csv"),
    index=False
)

correlation.to_csv(
    os.path.join(REPORT_DIR, "correlation_matrix.csv")
)

outlier_summary = pd.DataFrame({
    "Metric": ["Number of Reviews", "Price"],
    "Upper_Bound": [review_upper_limit, price_upper_limit],
    "Outlier_Count": [len(review_outliers), len(price_outliers)]
})

outlier_summary.to_csv(
    os.path.join(REPORT_DIR, "outlier_analysis.csv"),
    index=False
)

total_unique_genres = genre_exploded["Genre_List"].nunique()

summary = [
    "=" * 70,
    "AUDIBLE INSIGHTS - EDA SUMMARY",
    "=" * 70,
    f"Total Records: {len(df)}",
    f"Unique Books: {df['Book Name'].nunique()}",
    f"Unique Authors: {df['Author'].nunique()}",
    f"Total Unique Genres Identified: {total_unique_genres}",
    f"Average Rating: {df['Rating'].mean():.2f}",
    f"Highest Rating: {df['Rating'].max():.2f}",
    f"Average Reviews: {df['Number of Reviews'].mean():.0f}",
    f"Median Reviews: {df['Number of Reviews'].median():.0f}",
    f"Average Price: {df['Price'].mean():.2f}",
    f"Median Price: {df['Price'].median():.2f}"
]

if not genre_counts.empty:
    summary.append(f"Top Genre: {genre_counts.index[0]}")

if not top_reviews.empty:
    summary.append(
        f"Most Reviewed Book: {top_reviews.iloc[0]['Book Name']}"
    )

summary += [
    "",
    "Business Insights:",
    "- Ratings should be analyzed together with review counts.",
    "- Review counts are highly skewed; log transformation is useful.",
    "- Price distribution contains potential high-value outliers.",
    "- Genre-wise analysis helps identify popular book categories.",
    "- Highly reviewed books can indicate reader popularity.",
    "- Rating and review count can support popularity and quality features.",
    "- Title, author, genre and description can support NLP recommendations.",
    "- TF-IDF and cosine similarity can support content-based recommendation.",
    "- Clustering can group books with similar characteristics.",
    "- A hybrid system can combine content similarity with popularity."
]

report_path = os.path.join(REPORT_DIR, "eda_summary.txt")

with open(report_path, "w", encoding="utf-8") as file:
    file.write("\n".join(summary))

print("\n".join(summary))

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)
print("\nFigures saved to:", FIGURE_DIR)
print("Reports saved to:", REPORT_DIR)
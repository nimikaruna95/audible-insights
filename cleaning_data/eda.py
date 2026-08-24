# eda.py
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# PROJECT PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR,"processed_data","cleaned_books.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# DISPLAY SETTINGS
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 80)

# TITLE
print("\nAUDIBLE INSIGHTS - EXPLORATORY DATA ANALYSIS")

# LOAD DATA
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("\nCleaned dataset not found")

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"\nDataset loaded successfully")

# BASIC DATA STANDARDIZATION
df.columns = (df.columns.str.replace("\ufeff", "", regex=False).str.strip())

# TEXT COLUMNS
text_columns = ["Book Name","Author","Description","Genre",
    "Ranks and Genre","Listening Time","Data_Source","Book_Name_Key","Author_Key"]

for column in text_columns:
    if column in df.columns:
        df[column] = (df[column].astype("string").fillna("Unknown").str.strip())

# NUMERICAL COLUMNS
numeric_columns = ["Rating","Rating_Filled","Number of Reviews","Log Reviews","Price",
        "Audible Rank","Listening Time Minutes","Has_Description","Has_Genre","Has_Listening_Time"]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column],errors="coerce")

# BASIC INFORMATION
print("\n BASIC DATASET INFORMATION")
print(f"\nNumber of Records: {len(df):,}")
print(f"Number of Columns: {len(df.columns)}")
print(f"Unique Books: {df['Book Name'].nunique():,}")
print(f"Unique Authors: {df['Author'].nunique():,}")
print("\nColumns:")

for column in df.columns:
    print(f" - {column}")

print("\nFirst 5 Records:")
print(df.head().to_string())

# DATA TYPES
print("\nData Types:")
print(df.dtypes)

# MISSING VALUES
print("\n MISSING VALUE ANALYSIS")
missing_count = df.isnull().sum()

missing_percentage = (missing_count / len(df) * 100)
missing_report = pd.DataFrame({"Missing_Count": missing_count,"Missing_Percentage": missing_percentage.round(2)})
missing_report = (missing_report.sort_values("Missing_Count", ascending=False))

print(missing_report)
missing_report.to_csv(os.path.join(REPORT_DIR,"missing_value_analysis.csv"))

# MISSING VALUE PLOT
missing_plot = (missing_report[missing_report["Missing_Count"] > 0])

if not missing_plot.empty:
    
    # Graph
    plt.figure(figsize=(10, 6))
    sns.barplot(data=missing_plot.reset_index(),x="Missing_Count",y="index")
    plt.title("Missing Values by Column")
    plt.xlabel("Missing Count")
    plt.ylabel("Column")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR,"missing_values.png"),dpi=300)
    plt.close()

# DUPLICATE ANALYSIS
print("\n DUPLICATE ANALYSIS")

duplicate_rows = df.duplicated().sum()
duplicate_books = (df.duplicated(subset=["Book Name", "Author"]).sum())

print(f"Exact duplicate rows: {duplicate_rows}")
print(f"Duplicate Book + Author records: "f"{duplicate_books}")

# SUMMARY STATISTICS
print("\n SUMMARY STATISTICS")

analysis_numeric_columns = ["Rating","Number of Reviews","Price",
    "Listening Time Minutes","Audible Rank" ]

available_numeric = [column for column in analysis_numeric_columns if column in df.columns]

summary_statistics = (df[available_numeric].describe().round(2))
print(summary_statistics)
summary_statistics.to_csv(os.path.join(REPORT_DIR,"summary_statistics.csv"))

# RATING ANALYSIS
print("\n5. RATING ANALYSIS")

valid_ratings = df["Rating"].dropna()

print(f"\nBooks with valid ratings: "f"{len(valid_ratings):,}")

print(f"Average Rating: {valid_ratings.mean():.2f}")
print(f"Median Rating: {valid_ratings.median():.2f}")
print(f"Minimum Rating: {valid_ratings.min():.2f}")
print(f"Maximum Rating: {valid_ratings.max():.2f}")

print("\nRating Statistics:")
print(valid_ratings.describe().round(2))

# RATING DISTRIBUTION
plt.figure(figsize=(9, 5))
sns.histplot(valid_ratings,bins=20,kde=True)
plt.title("Distribution of Book Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Books")
plt.xlim(1, 5)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,"rating_distribution.png"),dpi=300)
plt.close()


# RATING COUNTS
rating_counts = (valid_ratings.round(1).value_counts().sort_index())

print("\nRating Frequency:")
print(rating_counts)

rating_counts.to_csv(os.path.join(REPORT_DIR,"rating_frequency.csv"),header=["Number_of_Books"])

# RATING BOXPLOT
plt.figure(figsize=(8, 4))
sns.boxplot(x=valid_ratings)
plt.title("Rating Boxplot")
plt.xlabel("Rating")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,"rating_boxplot.png"),dpi=300)
plt.close()

# PRICE ANALYSIS
print("\n PRICE ANALYSIS")
price_data = df["Price"].dropna()
print(price_data.describe().round(2))

plt.figure(figsize=(9, 5))
sns.histplot()
plt.title("Book Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Books")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,"price_distribution.png"),dpi=300)
plt.close()


plt.figure(figsize=(8, 4))
sns.boxplot(x=price_data)
plt.title("Book Price Boxplot")
plt.xlabel("Price")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,"price_boxplot.png"),dpi=300)
plt.close()

# REVIEW ANALYSIS
print("\n REVIEW COUNT ANALYSIS")
review_data = df["Number of Reviews"].fillna(0)
print(review_data.describe().round(2))

# LOG REVIEW DISTRIBUTION
if "Log Reviews" in df.columns:
    log_reviews = df["Log Reviews"].dropna()
else:
    log_reviews = np.log1p(review_data)

plt.figure(figsize=(9, 5))
sns.histplot(log_reviews,bins=30,kde=True)
plt.title("Log-Transformed Review Distribution")
plt.xlabel("log(1 + Number of Reviews)")
plt.ylabel("Number of Books")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,"log_reviews_distribution.png"),dpi=300)
plt.close()

# MOST REVIEWED BOOKS
most_reviewed = (df.sort_values("Number of Reviews",ascending=False).head(20))

print("\nTop 10 Most Reviewed Books:")
print(most_reviewed[["Book Name","Author","Rating",
            "Number of Reviews"]].head(10).to_string(index=False))

most_reviewed[["Book Name","Author","Rating",
        "Number of Reviews"]].to_csv(os.path.join(REPORT_DIR,"most_reviewed_books.csv"),index=False)

plt.figure(figsize=(12, 7))
plot_data = (most_reviewed.head(10).sort_values("Number of Reviews"))
sns.barplot(data=plot_data,x="Number of Reviews",y="Book Name")
plt.title("Top 10 Most Reviewed Books")
plt.xlabel("Number of Reviews")
plt.ylabel("Book")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,"most_reviewed_books.png"),dpi=300)
plt.close()

# TOP RATED BOOKS
print("\n TOP RATED BOOKS")

top_rated = (df[df["Rating"].notna()]
    .sort_values(["Rating", "Number of Reviews"],ascending=[False, False]).head(20))

print(top_rated[["Book Name","Author",
            "Rating","Number of Reviews"]].head(10).to_string(index=False))
top_rated.to_csv(os.path.join(REPORT_DIR,"top_rated_books.csv"),index=False)

# RATING VS REVIEW COUNT
print("\n RATING VS REVIEW COUNT")

# Create review groups
review_bins = [-1,50,200,500,1000,np.inf]
review_labels = ["0-50","51-200","201-500","501-1000","1001+"]
df["Review_Group"] = pd.cut(df["Number of Reviews"],bins=review_bins,labels=review_labels)

review_group_analysis = (df.groupby("Review_Group",observed=False).agg(
        Books=("Book Name", "count"),Average_Rating=("Rating", "mean"),
        Median_Rating=("Rating", "median"),Total_Reviews=("Number of Reviews", "sum")).reset_index())

print(review_group_analysis)

review_group_analysis.to_csv(os.path.join(REPORT_DIR,"review_group_analysis.csv"),index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=review_group_analysis,x="Review_Group",y="Average_Rating")
plt.title("Average Rating by Review Count Group")
plt.xlabel("Review Count Group")
plt.ylabel("Average Rating")
plt.ylim(0, 5)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,"rating_by_review_group.png"),dpi=300)
plt.close()

# RATING VS REVIEWS SCATTER
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df,x="Log Reviews",y="Rating",alpha=0.5)
plt.title("Rating vs Log-Transformed Review Count")
plt.xlabel("log(1 + Number of Reviews)")
plt.ylabel("Rating")
plt.ylim(0, 5.1)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,"rating_vs_reviews.png"),dpi=300)
plt.close()

# RATING VS PRICE
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df,x="Price",y="Rating",alpha=0.5)
plt.title("Rating vs Book Price")
plt.xlabel("Price")
plt.ylabel("Rating")
plt.ylim(0, 5.1)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,"rating_vs_price.png"),dpi=300)
plt.close()

# AUTHOR ANALYSIS
print("\n AUTHOR ANALYSIS")
author_analysis = (df.groupby("Author").agg(Books=("Book Name", "count"),
        Average_Rating=("Rating", "mean"),Total_Reviews=("Number of Reviews", "sum"),
        Average_Reviews=("Number of Reviews", "mean")))

# Most productive authors
top_authors_by_books = (author_analysis.sort_values("Books", ascending=False).head(10))
print("\nTop Authors by Number of Books:")
print(top_authors_by_books)

top_authors_by_books.to_csv(os.path.join(REPORT_DIR,"top_authors_by_books.csv"))

plt.figure(figsize=(10, 6))
sns.barplot(data=top_authors_by_books.reset_index(),x="Books",y="Author")
plt.title("Top 10 Authors by Number of Books")
plt.xlabel("Number of Books")
plt.ylabel("Author")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,"top_authors_by_books.png"),dpi=300)
plt.close()

# Highest-rated authors
# Require at least 3 books to avoid one-book bias.
top_rated_authors = (author_analysis[author_analysis["Books"] >= 3].sort_values(
        ["Average_Rating", "Total_Reviews"],ascending=[False, False]).head(10))

print("\nTop Rated Authors (minimum 3 books):")
print(top_rated_authors)

top_rated_authors.to_csv(os.path.join(REPORT_DIR,"top_rated_authors.csv"))

plt.figure(figsize=(10, 6))
sns.barplot(data=top_rated_authors.reset_index(),x="Average_Rating",y="Author")
plt.title("Top 10 Authors by Average Rating")
plt.xlabel("Average Rating")
plt.ylabel("Author")
plt.xlim(0, 5)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,"top_rated_authors.png"),dpi=300)
plt.close()

# AUTHOR POPULARITY VS RATING
author_valid = (author_analysis[author_analysis["Books"] >= 2].copy())

if len(author_valid) >= 2:
    author_correlation = (author_valid[["Books","Average_Rating",
                "Total_Reviews","Average_Reviews"]].corr())
    print("\nAuthor Popularity Correlation:")
    print(author_correlation.round(3))
    author_correlation.to_csv(os.path.join(REPORT_DIR,"author_popularity_correlation.csv" ))

# GENRE ANALYSIS
print("\n GENRE ANALYSIS")
# Use cleaned Genre column.
# Split combinations into separate genres.
genre_data = (df[["Book Name","Author",
            "Rating","Number of Reviews","Genre"]].copy())

genre_data["Genre_List"] = (genre_data["Genre"].astype(str).str.split("|"))
genre_exploded = genre_data.explode("Genre_List")
genre_exploded["Genre_List"] = (genre_exploded["Genre_List"].astype(str).str.strip())
genre_exploded = genre_exploded[~genre_exploded["Genre_List"].isin(["", "Unknown", "nan"])]

# Genre counts
genre_counts = (genre_exploded["Genre_List"].value_counts())
print("\nTop 15 Genres:")
print(genre_counts.head(15))
genre_counts.head(20).to_csv(os.path.join(REPORT_DIR,"genre_popularity.csv"),header=["Number_of_Books"])

plt.figure(figsize=(11, 8))
top_genres = (genre_counts.head(15).sort_values())
sns.barplot(x=top_genres.values,y=top_genres.index)
plt.title("Top 15 Most Popular Genres")
plt.xlabel("Number of Books")
plt.ylabel("Genre")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,"top_genres.png"),dpi=300)
plt.close()

# Genre rating
genre_rating = (genre_exploded
    .groupby("Genre_List").agg(Average_Rating=("Rating", "mean"),
        Book_Count=("Book Name", "count"),Total_Reviews=("Number of Reviews", "sum")))

genre_rating_filtered = (genre_rating[genre_rating["Book_Count"] >= 5]
    .sort_values("Average_Rating",ascending=False).head(10))

print("\nTop Genres by Average Rating:")
print(genre_rating_filtered)
genre_rating.to_csv(os.path.join(REPORT_DIR,"genre_rating_analysis.csv"))

plt.figure(figsize=(11, 7))
sns.barplot(data=genre_rating_filtered.reset_index(),x="Average_Rating",y="Genre_List")
plt.title("Top Genres by Average Rating")
plt.xlabel("Average Rating")
plt.ylabel("Genre")
plt.xlim(0, 5)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,"genre_average_rating.png"),dpi=300)
plt.close()

# Genre total reviews
genre_reviews = (genre_exploded.groupby("Genre_List")["Number of Reviews"].sum().sort_values(ascending=False).head(10))
print("\nTop Genres by Total Reviews:")
print(genre_reviews)
genre_reviews.to_csv(os.path.join(REPORT_DIR,"genre_total_reviews.csv"),header=["Total_Reviews"])

# HIDDEN GEMS
print("\n12. HIDDEN GEM ANALYSIS")

# A hidden gem:
# - Rating >= 4.5
# - At least 1 review
# - Review count in lower popularity range
# We use the 25th percentile as a data-driven
# definition of "low popularity".

review_low_threshold = (df["Number of Reviews"].quantile(0.25))

hidden_gems = (df[(df["Rating"] >= 4.5) & (df["Number of Reviews"] <= review_low_threshold)
        & (df["Number of Reviews"] > 0)].sort_values(["Rating", "Number of Reviews"],ascending=[False, False]))
print(f"Low popularity threshold: "f"{review_low_threshold:.0f} reviews")
print(f"Hidden gems identified: "f"{len(hidden_gems)}")
print("\nTop Hidden Gems:")
print(hidden_gems[["Book Name","Author","Rating","Number of Reviews","Genre"]]
    .head(20).to_string(index=False))
hidden_gems.to_csv(os.path.join(REPORT_DIR,"hidden_gems.csv"),index=False)

# LISTENING TIME ANALYSIS
print("\n LISTENING TIME ANALYSIS")
if "Listening Time Minutes" in df.columns:
    listening_data = (df[df["Listening Time Minutes"].notna()].copy())

    if not listening_data.empty:
        print("\nListening Time Statistics:")
        print(listening_data["Listening Time Minutes"].describe().round(2))

        # Distribution
        plt.figure(figsize=(9, 5))
        sns.histplot(listening_data["Listening Time Minutes"],bins=30,kde=True)
        plt.title("Listening Time Distribution")
        plt.xlabel("Listening Time (Minutes)")
        plt.ylabel("Number of Books")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR,"listening_time_distribution.png"),dpi=300)
        plt.close()

        # Rating vs listening time
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=listening_data,x="Listening Time Minutes",y="Rating",alpha=0.5)
        plt.title("Rating vs Listening Time")
        plt.xlabel("Listening Time (Minutes)")
        plt.ylabel("Rating")
        plt.ylim(0, 5.1)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR,"rating_vs_listening_time.png"),dpi=300)
        plt.close()

        # Longest books
        longest_books = (listening_data.sort_values("Listening Time Minutes",ascending=False).head(20))
        print("\nLongest Audiobooks:")
        print(longest_books[["Book Name","Author",
                    "Rating","Listening Time",
                    "Listening Time Minutes"]].head(10).to_string(index=False))
        longest_books.to_csv(os.path.join(REPORT_DIR,"longest_audiobooks.csv"),index=False)

# DATA SOURCE ANALYSIS
print("\n DATA SOURCE ANALYSIS")
if "Data_Source" in df.columns:
    source_analysis = (df.groupby("Data_Source")
        .agg(Books=("Book Name", "count"),Average_Rating=("Rating", "mean"),
            Average_Reviews=("Number of Reviews", "mean"),
            Total_Reviews=("Number of Reviews", "sum"),Average_Price=("Price", "mean")).round(2))

    print(source_analysis)
    source_analysis.to_csv(os.path.join(REPORT_DIR,"data_source_analysis.csv"))

    plt.figure(figsize=(9, 6))
    sns.barplot(data=source_analysis.reset_index(),x="Data_Source",y="Books")
    plt.title("Books by Dataset Source")
    plt.xlabel("Dataset Source")
    plt.ylabel("Number of Books")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR,"data_source_distribution.png"),dpi=300)
    plt.close()

# DESCRIPTION ANALYSIS
print("\n DESCRIPTION ANALYSIS")

if "Description" in df.columns:
    description_available = (df["Description"].notna() & df["Description"].ne("") 
                             & df["Description"].ne("Unknown"))
    df["Description_Length"] = np.where(description_available,df["Description"].str.len(),0)

    available_count = description_available.sum()
    unavailable_count = (~description_available).sum()

    print(f"Valid descriptions: " f"{available_count:,}")

    print(f"Missing/unusable descriptions: "f"{unavailable_count:,}")
    description_summary = pd.DataFrame({"Status": ["Available","Missing / Unusable"],
                                        "Number_of_Books": [available_count,unavailable_count]})

    description_summary.to_csv(os.path.join(REPORT_DIR,"description_availability.csv"),index=False)


    plt.figure(figsize=(8, 5))
    sns.barplot(data=description_summary,x="Status",y="Number_of_Books")
    plt.title("Description Availability")
    plt.xlabel("Description Status")
    plt.ylabel("Number of Books")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR,"description_availability.png"),dpi=300)
    plt.close()

    valid_description_lengths = (df.loc[description_available,"Description_Length"])
    if not valid_description_lengths.empty:
        print("\nDescription Length Statistics:")
        print(valid_description_lengths.describe().round(2))

        plt.figure(figsize=(9, 5))
        sns.histplot(valid_description_lengths,bins=30,kde=True)
        plt.title("Book Description Length Distribution")
        plt.xlabel("Description Length (Characters)")
        plt.ylabel("Number of Books")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR,"description_length.png"),dpi=300)
        plt.close()

# CORRELATION ANALYSIS
print("\n CORRELATION ANALYSIS")

correlation_columns = ["Rating","Number of Reviews","Log Reviews","Price",
    "Audible Rank","Listening Time Minutes"]

correlation_columns = [column for column in correlation_columns if column in df.columns]
correlation = (df[correlation_columns].corr())
print(correlation.round(3))
correlation.to_csv(os.path.join(REPORT_DIR,"correlation_matrix.csv"))


plt.figure(figsize=(9, 7))
sns.heatmap(correlation,annot=True,fmt=".2f",cmap="coolwarm",center=0,linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR,"correlation_heatmap.png"),dpi=300)
plt.close()

# OUTLIER ANALYSIS
print("\n OUTLIER ANALYSIS")

outlier_records = []

# Reviews
review_q1 = df["Number of Reviews"].quantile(0.25)
review_q3 = df["Number of Reviews"].quantile(0.75)
review_iqr = review_q3 - review_q1
review_upper = review_q3 + 1.5 * review_iqr

review_outliers = (df[df["Number of Reviews"] > review_upper])

print(f"Review upper bound: {review_upper:.2f}")
print(f"Review outliers: {len(review_outliers)}")

review_outliers.to_csv(os.path.join(REPORT_DIR,"review_outliers.csv"),index=False)
outlier_records.append({
    "Metric": "Number of Reviews",
    "Q1": review_q1,
    "Q3": review_q3,
    "IQR": review_iqr,
    "Upper_Bound": review_upper,
    "Outlier_Count": len(review_outliers)})

# Price
price_q1 = df["Price"].quantile(0.25)
price_q3 = df["Price"].quantile(0.75)
price_iqr = price_q3 - price_q1
price_upper = price_q3 + 1.5 * price_iqr

price_outliers = (df[df["Price"] > price_upper])

print(f"Price upper bound: {price_upper:.2f}")
print(f"Price outliers: {len(price_outliers)}")

price_outliers.to_csv(os.path.join(REPORT_DIR,"price_outliers.csv"),index=False)

outlier_records.append({
    "Metric": "Price",
    "Q1": price_q1,
    "Q3": price_q3,
    "IQR": price_iqr,
    "Upper_Bound": price_upper,
    "Outlier_Count": len(price_outliers)})

outlier_summary = pd.DataFrame(outlier_records)
outlier_summary.to_csv(os.path.join(REPORT_DIR,"outlier_analysis.csv"),index=False)

# PUBLICATION YEAR LIMITATION
print("\n PUBLICATION YEAR ANALYSIS")

publication_columns = [column for column in df.columns if "year" in column.lower() 
                       or "publication" in column.lower()]

if publication_columns:
    print("Publication-related columns found:")
    print(publication_columns)

else:
    print("Publication year cannot be done because source dataset don't contains publication year column.")
publication_note = (" Publication year analysis could not be performed because its not available on datasets.")

with open(os.path.join(REPORT_DIR,"publication_year_note.txt"),"w",encoding="utf-8") as file:
    file.write(publication_note)

# SCENARIO: SCIENCE FICTION
print("\n19. SCENARIO ANALYSIS - SCIENCE FICTION")

science_fiction_keywords = ["science fiction","sci-fi","science-fiction"]

science_fiction_mask = (df["Genre"].str.lower().apply(lambda value:any(
            keyword in value for keyword in science_fiction_keywords)))

science_fiction_books = (df[science_fiction_mask].sort_values(["Rating", "Number of Reviews"],
                                                              ascending=[False, False]).head(5))

print("\nTop 5 Science Fiction Books:")

if science_fiction_books.empty:
    print("No Science Fiction books were identified from the available genre metadata.")

else:
    print(science_fiction_books[["Book Name","Author",
                "Rating","Number of Reviews","Genre"]].to_string(index=False))

science_fiction_books.to_csv(os.path.join(REPORT_DIR,"science_fiction_top5.csv"),index=False)

# SCENARIO: THRILLER / MYSTERY
print("\n SCENARIO ANALYSIS - THRILLER")

thriller_keywords = ["thriller","crime","mystery"]

thriller_mask = (df["Genre"].str.lower().apply(lambda value:any(keyword in value for keyword in thriller_keywords)))
thriller_books = (df[thriller_mask].sort_values(
        ["Rating", "Number of Reviews"],
        ascending=[False, False]).head(10))

print("\nTop Thriller/Mystery Books:")
if thriller_books.empty:
    print("No thriller/mystery books were identified.")

else:
    print(thriller_books[["Book Name","Author",
                "Rating","Number of Reviews","Genre"]].to_string(index=False))

thriller_books.to_csv(os.path.join(REPORT_DIR,"thriller_top_books.csv"),index=False)

# BUSINESS INSIGHTS
print("\n BUSINESS INSIGHTS")

insights = []

# Rating
insights.append( f"1. The average book rating is " f"{valid_ratings.mean():.2f}.")

# Most popular genre
if not genre_counts.empty:
    insights.append(f"2. The most common identified genre is "f"'{genre_counts.index[0]}' with "
        f"{genre_counts.iloc[0]} book records.")

# Most reviewed
if not most_reviewed.empty:
    insights.append(f" The most reviewed book is " f"'{most_reviewed.iloc[0]['Book Name']}' " f"with "
        f"{int(most_reviewed.iloc[0]['Number of Reviews']):,} " f"reviews.")

# Hidden gems
insights.append(
    f"4. {len(hidden_gems):,} books meet "
    f"the hidden-gem criteria of rating "
    f">= 4.5 and low review count.")

# Description
description_percentage = (available_count / len(df) * 100)
insights.append(f"5. {description_percentage:.2f}% of books "f"have usable descriptions for NLP processing.")

# Genre availability
genre_percentage = (df["Has_Genre"].mean() * 100 if "Has_Genre" in df.columns else np.nan)
insights.append(f"6. {genre_percentage:.2f}% of books " f"have genre metadata.")

# Listening time
if "Has_Listening_Time" in df.columns:
    listening_percentage = (df["Has_Listening_Time"].mean() * 100)
    insights.append(f"7. {listening_percentage:.2f}% of books " f"have listening-time data.")

# Publication year
insights.append(
    " Publication-year trends cannot be analyzed because publication year is not present in the datasets.")

# Recommendation system
insights.extend([
    " Review count is highly skewed, so Log Reviews is useful as a recommendation feature.",
    " Genre, author and description features can support content-based recommendations.",
    " TF-IDF with cosine similarity can be used to compare book content.",
    " Clustering can group books with similar content and genre features.",
    " A hybrid recommendation system can combine content similarity, genre similarity and popularity."])

for insight in insights:
    print(insight)

# SAVE BUSINESS INSIGHTS
with open(os.path.join(REPORT_DIR,"business_insights.txt"),"w",encoding="utf-8") as file:
    file.write("\n".join(insights))

# FINAL EDA SUMMARY
summary = ["AUDIBLE INSIGHTS - EDA FINAL SUMMARY",
    "",
    f"Total Records: {len(df):,}",
    f"Unique Books: {df['Book Name'].nunique():,}",
    f"Unique Authors: {df['Author'].nunique():,}",
    f"Average Rating: {valid_ratings.mean():.2f}",
    f"Median Rating: {valid_ratings.median():.2f}",
    f"Average Reviews: {review_data.mean():.2f}",
    f"Median Reviews: {review_data.median():.2f}",
    f"Average Price: {price_data.mean():.2f}",
    f"Median Price: {price_data.median():.2f}",
    f"Valid Descriptions: {available_count:,}",
    f"Valid Genres: {df['Has_Genre'].sum():,}",
    f"Valid Listening Times: {df['Has_Listening_Time'].sum():,}",
    f"Hidden Gems: {len(hidden_gems):,}",
    "", "Publication Year:","Not available in source datasets.","",
    "EDA completed successfully."]

summary_text = "\n".join(summary)
print("\n" + summary_text)
with open(os.path.join(REPORT_DIR,"eda_summary.txt"),"w",encoding="utf-8") as file: file.write(summary_text)

# FINAL MESSAGE
print("\nEDA COMPLETED SUCCESSFULLY")
print(f"\nFigures saved to:\n" f"{FIGURE_DIR}")
print(f"\nReports saved to:\n"f"{REPORT_DIR}")
print("Use the EDA outputs to build feature_engineering,nlp,clustering and the recommendation models.")


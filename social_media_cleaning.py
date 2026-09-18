"""
=============================================================
 Social Media Analytics Project - Data Cleaning & Preparation
=============================================================
Dataset : social_media_performance.csv
Purpose : Clean and prepare the data for analysis and for a
          Power BI dashboard.
Tools   : Pandas, NumPy, Matplotlib, Seaborn
=============================================================
"""

# ----------------------------------------------------------
# Importing the libraries we need
# ----------------------------------------------------------
import pandas as pd          # for tables (DataFrames)
import numpy as np           # for numeric operations
import matplotlib
matplotlib.use("Agg")        # lets charts be saved without a screen
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings so that printed output is easy to read
pd.set_option("display.max_columns", None)   # show every column
pd.set_option("display.width", 200)          # wider console output

# A small helper so the output is neatly divided into sections
def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ==========================================================
# STEP 1 : Load the CSV file using Pandas
# ==========================================================
INPUT_FILE = "social_media_performance.csv"
OUTPUT_FILE = "cleaned_social_media_performance.csv"

df = pd.read_csv(INPUT_FILE)
header("STEP 1 : FILE LOADED SUCCESSFULLY")
print("File name :", INPUT_FILE)


# ==========================================================
# STEP 2 : Basic exploration of the dataset
# ==========================================================
header("STEP 2 : BASIC INFORMATION ABOUT THE DATASET")

# .shape returns (number_of_rows, number_of_columns)
print("Number of rows    :", df.shape[0])
print("Number of columns :", df.shape[1])

print("\nColumn names:")
print(list(df.columns))

print("\nData types of each column:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# describe() gives count, mean, std, min, quartiles and max
print("\nBasic statistical summary (numeric columns):")
print(df.describe())

# include='object' describes the text columns instead
print("\nBasic summary of text columns:")
print(df.describe(include="object"))


# ==========================================================
# STEP 3 : Check for missing / null values
# ==========================================================
header("STEP 3 : MISSING (NULL) VALUES IN EVERY COLUMN")

missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df) * 100).round(2)

missing_report = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_percent": missing_percent
})
print(missing_report)
print("\nTotal missing values in the whole dataset :", int(missing_counts.sum()))


# ==========================================================
# STEP 4 : Check for duplicate rows
# ==========================================================
header("STEP 4 : DUPLICATE ROWS")

duplicate_rows = df.duplicated().sum()
print("Number of fully duplicated rows :", duplicate_rows)

if duplicate_rows > 0:
    print("\nShowing the duplicated rows:")
    print(df[df.duplicated(keep=False)].sort_values("post_id").head(10))
    # Only remove them if they actually exist
    df = df.drop_duplicates()
    print("Duplicates removed. New shape :", df.shape)
else:
    print("No duplicate rows found, so nothing is removed.")


# ==========================================================
# STEP 5 : Check whether post_id has duplicate values
# ==========================================================
header("STEP 5 : DUPLICATE post_id VALUES")

post_id_duplicates = df["post_id"].duplicated().sum()
print("Number of duplicated post_id values :", post_id_duplicates)

if post_id_duplicates > 0:
    print("\nThe repeated post_id values are:")
    print(df[df["post_id"].duplicated(keep=False)]
          .sort_values("post_id")[["post_id", "platform", "post_datetime"]].head(10))
else:
    print("post_id is unique for every row -> it can be used as the primary key.")


# ==========================================================
# STEP 6 : Clean the text columns
# ==========================================================
header("STEP 6 : CLEANING TEXT COLUMNS")

text_columns = ["platform", "content_type", "topic", "language", "region", "hashtags"]

# Before cleaning, count how many values had extra spaces
spaces_before = {}
for col in text_columns:
    original = df[col].astype(str)
    stripped = original.str.strip()
    spaces_before[col] = int((original != stripped).sum())
print("Values having leading/trailing spaces BEFORE cleaning:")
print(spaces_before)

for col in text_columns:
    df[col] = (
        df[col]
        .astype(str)                           # make sure it is text
        .str.strip()                           # remove spaces at start and end
        .str.replace(r"\s+", " ", regex=True)  # turn multiple spaces into one
    )

# Standardising the values so that "linkedin", "LinkedIn " etc. become one value.
# .str.title() -> first letter capital  (Instagram, Linkedin)
# .str.upper() -> all capitals          (EN, IN)
# .str.lower() -> all small             (reel, article)
df["platform"] = df["platform"].str.title()
df["content_type"] = df["content_type"].str.lower()
df["topic"] = df["topic"].str.title()
df["language"] = df["language"].str.upper()
df["region"] = df["region"].str.upper()

# LinkedIn and YouTube have a capital letter in the middle, so .title() breaks
# them ("Linkedin", "Youtube"). We fix those names back to their correct form.
platform_fix = {"Linkedin": "LinkedIn", "Youtube": "YouTube", "Tiktok": "TikTok"}
df["platform"] = df["platform"].replace(platform_fix)

print("\nUnique values AFTER cleaning:")
for col in ["platform", "content_type", "topic", "language", "region"]:
    print(f"  {col:13s} ({df[col].nunique()}) -> {sorted(df[col].unique())}")

# hashtags is free text, so we only count them instead of listing all
df["hashtag_count"] = df["hashtags"].apply(
    lambda x: 0 if x.strip() == "" else len(x.split())
)
print("\nExample hashtags after cleaning:")
print(df[["hashtags", "hashtag_count"]].head(3))


# ==========================================================
# STEP 7 : Convert post_datetime into proper datetime format
# ==========================================================
header("STEP 7 : CONVERTING post_datetime TO DATETIME")

print("Data type BEFORE conversion :", df["post_datetime"].dtype)

# errors='coerce' -> any value that cannot be converted becomes NaT (missing)
df["post_datetime"] = pd.to_datetime(df["post_datetime"], errors="coerce")

print("Data type AFTER  conversion :", df["post_datetime"].dtype)
print("Rows that failed to convert (NaT) :", int(df["post_datetime"].isna().sum()))
print("Earliest post :", df["post_datetime"].min())
print("Latest   post :", df["post_datetime"].max())


# ==========================================================
# STEP 8 : Make sure numeric columns really are numeric
# ==========================================================
header("STEP 8 : CONVERTING NUMERIC COLUMNS")

numeric_columns = ["sentiment_score", "views", "likes", "comments",
                   "shares", "engagement_rate", "is_viral"]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Whole-number columns are stored as integers, decimals stay as float
integer_columns = ["views", "likes", "comments", "shares", "is_viral"]
for col in integer_columns:
    if df[col].isna().sum() == 0:            # astype(int) fails if NaN exists
        df[col] = df[col].astype(int)

print("Data types after conversion:")
print(df[numeric_columns].dtypes)
print("\nValues that could not be converted to numbers:")
print(df[numeric_columns].isna().sum())


# ==========================================================
# STEP 9 : Check for invalid (negative) numeric values
# ==========================================================
header("STEP 9 : INVALID / NEGATIVE NUMERIC VALUES")

invalid_checks = {
    "Negative views": (df["views"] < 0).sum(),
    "Negative likes": (df["likes"] < 0).sum(),
    "Negative comments": (df["comments"] < 0).sum(),
    "Negative shares": (df["shares"] < 0).sum(),
    "Zero views (division problem)": (df["views"] == 0).sum(),
    "Invalid sentiment_score (outside -1 to 1)":
        ((df["sentiment_score"] < -1) | (df["sentiment_score"] > 1)).sum(),
    "Invalid engagement_rate (below 0 or above 1)":
        ((df["engagement_rate"] < 0) | (df["engagement_rate"] > 1)).sum(),
}

for check_name, count in invalid_checks.items():
    status = "OK" if count == 0 else "PROBLEM FOUND"
    print(f"{check_name:45s} : {count:5d}  -> {status}")

# A logical check: likes + comments + shares should not be more than views
impossible_rows = df[(df["likes"] + df["comments"] + df["shares"]) > df["views"]]
print(f"\nRows where likes+comments+shares > views : {len(impossible_rows)}")


# ==========================================================
# STEP 10 : Range of sentiment_score
# ==========================================================
header("STEP 10 : SENTIMENT SCORE RANGE CHECK (expected -1 to 1)")

print("Minimum sentiment_score :", df["sentiment_score"].min())
print("Maximum sentiment_score :", df["sentiment_score"].max())
print("Average sentiment_score :", round(df["sentiment_score"].mean(), 4))

out_of_range_sentiment = df[(df["sentiment_score"] < -1) | (df["sentiment_score"] > 1)]
print("Rows outside the -1 to 1 range :", len(out_of_range_sentiment))
if len(out_of_range_sentiment) > 0:
    print(out_of_range_sentiment[["post_id", "sentiment_score"]].head(10))

# Extra: grouping sentiment into readable labels for the dashboard
df["sentiment_label"] = pd.cut(
    df["sentiment_score"],
    bins=[-1.01, -0.05, 0.05, 1.01],
    labels=["Negative", "Neutral", "Positive"]
)
print("\nSentiment distribution:")
print(df["sentiment_label"].value_counts())


# ==========================================================
# STEP 11 : Engagement rate check
# ==========================================================
header("STEP 11 : ENGAGEMENT RATE CHECK")

print("Minimum engagement_rate :", df["engagement_rate"].min())
print("Maximum engagement_rate :", df["engagement_rate"].max())
print("Average engagement_rate :", round(df["engagement_rate"].mean(), 4))

abnormal_er = df[(df["engagement_rate"] < 0) | (df["engagement_rate"] > 1)]
print("Abnormal engagement_rate rows (negative or above 100%) :", len(abnormal_er))
if len(abnormal_er) > 0:
    print(abnormal_er[["post_id", "views", "likes", "engagement_rate"]].head(10))


# ==========================================================
# STEP 12 : is_viral must contain only 0 and 1
# ==========================================================
header("STEP 12 : is_viral VALUE CHECK")

print("Unique values in is_viral :", sorted(df["is_viral"].dropna().unique()))
print("\nCount of each value:")
print(df["is_viral"].value_counts())

invalid_viral = df[~df["is_viral"].isin([0, 1])]
print("\nRows with a value other than 0 or 1 :", len(invalid_viral))


# ==========================================================
# STEP 13 : Create the viral_status column
# ==========================================================
header("STEP 13 : CREATING viral_status COLUMN")

# .map() replaces 1 with "Viral" and 0 with "Not Viral"
df["viral_status"] = df["is_viral"].map({1: "Viral", 0: "Not Viral"})

print(df["viral_status"].value_counts())
print("\nSample:")
print(df[["post_id", "is_viral", "viral_status"]].head())


# ==========================================================
# STEP 14 : Create date and time columns
# ==========================================================
header("STEP 14 : CREATING DATE / TIME COLUMNS")

df["post_date"] = df["post_datetime"].dt.date           # only the date part
df["year"] = df["post_datetime"].dt.year                # 2025
df["month"] = df["post_datetime"].dt.month              # 1 to 12
df["month_name"] = df["post_datetime"].dt.month_name()  # January, February...
df["day_name"] = df["post_datetime"].dt.day_name()      # Monday, Tuesday...
df["post_hour"] = df["post_datetime"].dt.hour           # 0 to 23

print(df[["post_datetime", "post_date", "year", "month",
          "month_name", "day_name", "post_hour"]].head())

print("\nNumber of posts per day of the week:")
print(df["day_name"].value_counts())


# ==========================================================
# STEP 15 : Outlier detection using the IQR method
# ==========================================================
header("STEP 15 : OUTLIER CHECK USING IQR (values are NOT deleted)")

outlier_columns = ["views", "likes", "comments", "shares"]
outlier_summary = []

for col in outlier_columns:
    Q1 = df[col].quantile(0.25)                 # 25th percentile
    Q3 = df[col].quantile(0.75)                 # 75th percentile
    IQR = Q3 - Q1                               # inter-quartile range
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]

    outlier_summary.append({
        "column": col,
        "Q1": round(Q1, 2),
        "Q3": round(Q3, 2),
        "IQR": round(IQR, 2),
        "lower_limit": round(lower_limit, 2),
        "upper_limit": round(upper_limit, 2),
        "outlier_count": len(outliers),
        "outlier_percent": round(len(outliers) / len(df) * 100, 2)
    })

    print(f"\n--- {col.upper()} ---")
    print(f"Q1 = {Q1:,.2f} | Q3 = {Q3:,.2f} | IQR = {IQR:,.2f}")
    print(f"Normal range : {lower_limit:,.2f} to {upper_limit:,.2f}")
    print(f"Outliers found : {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    if len(outliers) > 0:
        print("Top 5 highest outliers:")
        print(outliers.nlargest(5, col)[["post_id", "platform", "viral_status", col]])

print("\nOutlier summary table:")
print(pd.DataFrame(outlier_summary))

print("""
DECISION ON OUTLIERS:
These outliers are KEPT in the dataset. Reason:
1. Social media data is naturally right-skewed - a few posts go viral and
   collect millions of views while most posts get very few.
2. These high values are genuine, meaningful records, not typing mistakes.
   They are exactly the viral posts the project wants to study.
3. Deleting them would remove the most important rows and make the
   Power BI dashboard show a wrong picture of performance.
Outliers would only be removed if they were impossible values
(for example negative views), and no such values exist here.
""")


# ==========================================================
# STEP 16 : Relationship between the columns (correlation)
# ==========================================================
header("STEP 16 : RELATIONSHIPS BETWEEN COLUMNS")

# .corr() gives a value between -1 and 1.
# Close to  1 -> strong positive relationship
# Close to  0 -> almost no relationship
# Close to -1 -> strong negative relationship
pairs = [("views", "likes"), ("views", "comments"),
         ("views", "shares"), ("engagement_rate", "is_viral")]

for a, b in pairs:
    corr_value = df[a].corr(df[b])
    if abs(corr_value) >= 0.7:
        strength = "Strong"
    elif abs(corr_value) >= 0.4:
        strength = "Moderate"
    elif abs(corr_value) >= 0.2:
        strength = "Weak"
    else:
        strength = "Very weak / none"
    print(f"{a:16s} vs {b:16s} : {corr_value: .4f}  ({strength})")

print("\nFull correlation matrix:")
corr_matrix = df[["views", "likes", "comments", "shares",
                  "engagement_rate", "sentiment_score", "is_viral"]].corr()
print(corr_matrix.round(3))

print("\nAverage engagement_rate for viral and non-viral posts:")
print(df.groupby("viral_status")["engagement_rate"].agg(["count", "mean", "median"]).round(4))

# ---- Charts (saved as PNG files so they can be used in the report) ----
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
sns.scatterplot(data=df, x="views", y="likes", hue="viral_status",
                alpha=0.4, s=12, ax=axes[0, 0])
axes[0, 0].set_title("Views vs Likes")

sns.scatterplot(data=df, x="views", y="comments", hue="viral_status",
                alpha=0.4, s=12, ax=axes[0, 1])
axes[0, 1].set_title("Views vs Comments")

sns.scatterplot(data=df, x="views", y="shares", hue="viral_status",
                alpha=0.4, s=12, ax=axes[1, 0])
axes[1, 0].set_title("Views vs Shares")

sns.boxplot(data=df, x="viral_status", y="engagement_rate", ax=axes[1, 1])
axes[1, 1].set_title("Engagement Rate by Viral Status")

plt.tight_layout()
plt.savefig("relationship_plots.png", dpi=120)
plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt=".2f", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=120)
plt.close()

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, outlier_columns):
    sns.boxplot(y=df[col], ax=ax, color="#7fb3d5")
    ax.set_title(f"Outliers in {col}")
plt.tight_layout()
plt.savefig("outlier_boxplots.png", dpi=120)
plt.close()

print("\nCharts saved: relationship_plots.png, correlation_heatmap.png, outlier_boxplots.png")


# ==========================================================
# STEP 17 : Verify engagement_rate using our own calculation
# ==========================================================
header("STEP 17 : VERIFYING engagement_rate")

# Formula : (likes + comments + shares) / views
# np.where avoids dividing by zero if any row has views = 0
df["calculated_engagement_rate"] = np.where(
    df["views"] > 0,
    (df["likes"] + df["comments"] + df["shares"]) / df["views"],
    np.nan
).round(4)

# Difference between the given column and our calculation
df["engagement_rate_difference"] = (
    df["engagement_rate"] - df["calculated_engagement_rate"]
).round(4)

print("Comparison for the first 10 rows:")
print(df[["post_id", "views", "likes", "comments", "shares",
          "engagement_rate", "calculated_engagement_rate",
          "engagement_rate_difference"]].head(10))

abs_diff = df["engagement_rate_difference"].abs()
print("\nAverage absolute difference :", round(abs_diff.mean(), 5))
print("Largest absolute difference :", round(abs_diff.max(), 5))
print("Rows matching within 0.001 :", int((abs_diff <= 0.001).sum()),
      f"({(abs_diff <= 0.001).mean()*100:.2f}%)")
print("Rows matching within 0.01  :", int((abs_diff <= 0.01).sum()),
      f"({(abs_diff <= 0.01).mean()*100:.2f}%)")

# A flag column so mismatches can be filtered easily in Power BI
df["engagement_rate_matches"] = np.where(abs_diff <= 0.01, "Match", "Mismatch")
print("\nMatch / mismatch count:")
print(df["engagement_rate_matches"].value_counts())

mismatches = df[df["engagement_rate_matches"] == "Mismatch"]
if len(mismatches) > 0:
    print("\nExamples of mismatched rows:")
    print(mismatches[["post_id", "views", "likes", "comments", "shares",
                      "engagement_rate", "calculated_engagement_rate"]].head(10))


# ==========================================================
# STEP 18 & 19 : What we did NOT do
# ==========================================================
header("STEP 18 & 19 : COLUMNS AND ROWS THAT WERE NOT TOUCHED")
print("""
- No columns like clicks, followers or new_followers were invented,
  because they do not exist in the original dataset.
- Only helper columns built from existing data were added:
  hashtag_count, sentiment_label, viral_status, post_date, year, month,
  month_name, day_name, post_hour, calculated_engagement_rate,
  engagement_rate_difference, engagement_rate_matches.
- No valid rows were deleted. Outliers were kept (see Step 15).
""")


# ==========================================================
# STEP 20 : Final check after cleaning
# ==========================================================
header("STEP 20 : FINAL DATASET CHECK")

print("Final shape (rows, columns) :", df.shape)

print("\nFinal column names:")
print(list(df.columns))

print("\nMissing values in the final dataset:")
print(df.isnull().sum())

print("\nDuplicate rows in the final dataset :", df.duplicated().sum())
print("Duplicate post_id in the final dataset :", df["post_id"].duplicated().sum())

print("\nFinal data types:")
print(df.dtypes)

print("\nFirst 5 rows of the cleaned dataset:")
print(df.head())


# ==========================================================
# STEP 21 : Save the cleaned dataset
# ==========================================================
header("STEP 21 : SAVING THE CLEANED FILE")

# index=False stops Pandas from writing the row numbers as an extra column
df.to_csv(OUTPUT_FILE, index=False)
print("Cleaned dataset saved as :", OUTPUT_FILE)
print("Rows saved    :", df.shape[0])
print("Columns saved :", df.shape[1])
print("\nThis file is ready to be imported into Power BI.")
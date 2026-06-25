import os
import re
import json
import numpy as np
import pandas as pd
import torch

from pathlib import Path

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer


# ============================================================
# 1. הגדרות כלליות
# ============================================================

# שימי כאן את שם הקובץ שלך
INPUT_FILE = "stress_media_merged_with_duplicates_final.xlsx"

TITLE_COLUMN = "title_en"
CLEAN_TITLE_COLUMN = "clean_title_en"

OUTPUT_ARTICLES_FILE = "articles_with_topics.xlsx"
OUTPUT_TOPICS_FILE = "topics_summary.xlsx"

EMBEDDINGS_FILE = "title_embeddings_unique.npy"
CLEAN_TEXTS_FILE = "clean_titles_unique_for_model.json"

MODEL_OUTPUT_DIR = "bertopic_model"


# ============================================================
# 2. פונקציית ניקוי טקסט
# ============================================================

MEDIA_SUFFIX_PAT = re.compile(r'\s*-\s*[a-zA-Z0-9.]+$')
URL_PAT = re.compile(r'http\S+|www\S+|https\S+')
DIGIT_PAT = re.compile(r'\d+')
PUNCT_PAT = re.compile(r'[^\w\s]')
UNDERSCORE_PAT = re.compile(r'_')
WHITESPACE_PAT = re.compile(r'\s+')


def clean_text_fast(text):
    """
    ניקוי בסיסי ומהיר לכותרת אחת.
    הניקוי יחסית עדין כי כותרות הן קצרות וכל מילה יכולה להיות חשובה.
    """
    if not isinstance(text, str):
        return ""

    text = MEDIA_SUFFIX_PAT.sub('', text)
    text = text.lower()
    text = URL_PAT.sub('', text)
    text = DIGIT_PAT.sub('', text)
    text = PUNCT_PAT.sub(' ', text)
    text = UNDERSCORE_PAT.sub(' ', text)
    text = WHITESPACE_PAT.sub(' ', text)

    return text.strip()


# ============================================================
# 3. קריאת קובץ Excel או CSV
# ============================================================

def read_input_file(input_file):
    """
    קורא קובץ לפי הסיומת שלו:
    xlsx / xls / csv
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")

    suffix = Path(input_file).suffix.lower()

    print(f"Reading file: {input_file}")

    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(input_file)
    elif suffix == ".csv":
        df = pd.read_csv(input_file, encoding="utf-8-sig")
    else:
        raise ValueError("Input file must be .xlsx, .xls, or .csv")

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns found: {list(df.columns)}")

    if TITLE_COLUMN not in df.columns:
        raise ValueError(f"Column '{TITLE_COLUMN}' was not found in the file.")

    return df


# ============================================================
# 4. ניקוי הכותרות
# ============================================================
def standardize_date_columns(df):
    """
    מאחדת את פורמט עמודות התאריך לפורמט YYYY-MM-DD.
    אם קיימת עמודת published_at עם שעה, נוצרת גם עמודת published_at_datetime.
    """

    print("\nStandardizing date columns...")

    # עמודות תאריך אפשריות בקובץ
    date_columns = [
        "published_at",
        "issue_date",
        "query_date",
        "date",
        "date_str"
    ]

    for col in date_columns:
        if col in df.columns:
            parsed_dates = pd.to_datetime(
                df[col],
                errors="coerce",
                utc=True
            )

            # שמירת published_at גם כתאריך-שעה מלא
            if col == "published_at":
                df["published_at_datetime"] = parsed_dates.dt.strftime("%Y-%m-%d %H:%M:%S")

            # המרה לפורמט אחיד של תאריך בלבד
            df[col] = parsed_dates.dt.strftime("%Y-%m-%d")

            print(f"Standardized column: {col}")

    # year_month בפורמט אחיד YYYY-MM
    if "year_month" in df.columns:
        parsed_year_month = pd.to_datetime(
            df["year_month"],
            errors="coerce"
        )

        df["year_month"] = parsed_year_month.dt.strftime("%Y-%m")

        # אם year_month היה ריק אבל יש date, נבנה אותו מתוך date
        if "date" in df.columns:
            missing_year_month = df["year_month"].isna()
            df.loc[missing_year_month, "year_month"] = (
                pd.to_datetime(df.loc[missing_year_month, "date"], errors="coerce")
                .dt.strftime("%Y-%m")
            )

        print("Standardized column: year_month")

    # year כעמודה מספרית תקינה
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

        # אם year חסר אבל יש date, נבנה אותו מתוך date
        if "date" in df.columns:
            missing_year = df["year"].isna()
            df.loc[missing_year, "year"] = (
                pd.to_datetime(df.loc[missing_year, "date"], errors="coerce")
                .dt.year
                .astype("Int64")
            )

        print("Standardized column: year")

    print("Finished standardizing date columns.")

    return df

def clean_titles(df):
    """
    מוסיף עמודה clean_title_en.
    מסיר שורות שבהן אין כותרת תקינה אחרי ניקוי.
    """
    print("\nCleaning translated titles...")

    df[CLEAN_TITLE_COLUMN] = df[TITLE_COLUMN].apply(clean_text_fast)

    before = len(df)

    df = df[df[CLEAN_TITLE_COLUMN].str.len() > 0].copy()

    after = len(df)

    print(f"Rows before cleaning filter: {before:,}")
    print(f"Rows after cleaning filter: {after:,}")
    print(f"Removed empty titles: {before - after:,}")

    return df


# ============================================================
# 5. יצירת טבלה ייחודית למודל
# ============================================================

def create_unique_titles_df(df):
    """
    BERTopic ירוץ רק על כותרות ייחודיות.
    זה מקטין את זמן הריצה ומונע מכותרות שחוזרות הרבה להשתלט על המודל.

    חשוב:
    אנחנו לא מוחקים את הכפילויות מהפלט הסופי.
    בסוף נחזיר את topic_id לכל השורות המקוריות לפי clean_title_en.
    """
    print("\nCreating unique titles dataset for modeling...")

    df_unique = (
        df[[CLEAN_TITLE_COLUMN]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(f"Original cleaned rows: {len(df):,}")
    print(f"Unique cleaned titles for modeling: {len(df_unique):,}")

    return df_unique


# ============================================================
# 6. יצירת embeddings או טעינת embeddings קיימים
# ============================================================

def create_or_load_embeddings(texts):
    """
    יוצר embeddings לכותרות הייחודיות.
    אם כבר קיים קובץ embeddings שמתאים בדיוק לאותם texts,
    הוא ייטען במקום לחשב הכול מחדש.
    """
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(CLEAN_TEXTS_FILE):
        print("\nFound existing embeddings file. Checking compatibility...")

        with open(CLEAN_TEXTS_FILE, "r", encoding="utf-8") as f:
            saved_texts = json.load(f)

        if saved_texts == texts:
            embeddings = np.load(EMBEDDINGS_FILE)
            print(f"Embeddings loaded successfully: {embeddings.shape}")
            return embeddings
        else:
            print("Existing embeddings do not match current texts. Recomputing embeddings...")

    print("\nCreating sentence embeddings...")
    print("This may take time, especially for a large number of unique titles.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device=device
    )

    embeddings = embedding_model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    np.save(EMBEDDINGS_FILE, embeddings)

    with open(CLEAN_TEXTS_FILE, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)

    print(f"Embeddings created and saved: {embeddings.shape}")

    return embeddings


# ============================================================
# 7. הרצת BERTopic
# ============================================================

def run_bertopic(texts, embeddings):
    """
    מריץ BERTopic משופר:
    - UMAP להורדת ממדים
    - HDBSCAN לקלסטרינג
    - CountVectorizer מוגבל ל-20,000 features
    - KeyBERTInspired + MMR לשיפור ייצוג הנושאים
    - reduce_outliers לצמצום topic_id = -1
    """
    print("\nRunning BERTopic topic modeling...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using embedding model inside BERTopic on device: {device}")

    # חשוב:
    # גם אם אנחנו מעבירים embeddings מוכנים מראש,
    # KeyBERTInspired צריך embedding_model פנימי בשביל לשפר keywords.
    bertopic_embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device=device
    )

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=80,
        min_samples=5,
        metric="euclidean",
        prediction_data=True
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=5,
        max_features=20000,
        ngram_range=(1, 2)
    )

    representation_model = {
        "KeyBERT": KeyBERTInspired(),
        "MMR": MaximalMarginalRelevance(diversity=0.3)
    }

    topic_model = BERTopic(
        embedding_model=bertopic_embedding_model,  # זה התיקון החשוב
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        language="english",
        calculate_probabilities=False,
        verbose=True
    )

    topics, _ = topic_model.fit_transform(texts, embeddings)

    print("\nInitial BERTopic finished.")
    print(f"Initial number of topics excluding outliers: {len(set(topics)) - (1 if -1 in topics else 0)}")
    print(f"Initial number of outliers: {sum(1 for t in topics if t == -1):,}")

    print("\nReducing outliers with embeddings strategy...")

    topics = topic_model.reduce_outliers(
        texts,
        topics,
        strategy="embeddings",
        embeddings=embeddings
    )

    print("Updating topics after outlier reduction...")

    topic_model.update_topics(
        texts,
        topics=topics,
        vectorizer_model=vectorizer_model
    )

    print("\nAfter reducing outliers:")
    print(f"Number of topics excluding outliers: {len(set(topics)) - (1 if -1 in topics else 0)}")
    print(f"Number of outliers: {sum(1 for t in topics if t == -1):,}")

    topic_model.save(MODEL_OUTPUT_DIR, serialization="pickle")
    print(f"Model saved to folder: {MODEL_OUTPUT_DIR}")

    return topic_model, topics


# ============================================================
# 8. החזרת topic_id לכל הטבלה המקורית
# ============================================================

def assign_topics_to_all_rows(df_all, df_unique, topics):
    """
    המודל רץ רק על כותרות ייחודיות.
    כאן מחזירים את topic_id לכל 130,000 הרשומות המקוריות.
    """
    print("\nAssigning topic_id back to all original rows...")

    df_unique = df_unique.copy()
    df_unique["topic_id"] = topics

    topic_mapping = dict(
        zip(df_unique[CLEAN_TITLE_COLUMN], df_unique["topic_id"])
    )

    df_all = df_all.copy()
    df_all["topic_id"] = df_all[CLEAN_TITLE_COLUMN].map(topic_mapping)

    missing_topics = df_all["topic_id"].isna().sum()

    print(f"Rows in final article-level dataset: {len(df_all):,}")
    print(f"Rows without topic_id: {missing_topics:,}")

    df_all.to_excel(OUTPUT_ARTICLES_FILE, index=False)

    print(f"Saved article-level file: {OUTPUT_ARTICLES_FILE}")

    return df_all, df_unique


# ============================================================
# 9. יצירת טבלת סיכום topics
# ============================================================

def create_topics_summary(topic_model):
    """
    יוצר טבלת סיכום topics:
    - topic_id
    - מספר כתבות
    - שם אוטומטי
    - keywords
    - כותרות מייצגות
    """
    print("\nCreating topics summary...")

    topic_info = topic_model.get_topic_info()

    summary_rows = []

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]

        topic_words = topic_model.get_topic(topic_id)

        if topic_words:
            keywords = ", ".join([word for word, score in topic_words[:10]])
        else:
            keywords = ""

        try:
            representative_docs = topic_model.get_representative_docs(topic_id)
        except Exception:
            representative_docs = []

        if representative_docs:
            representative_titles = " | ".join(representative_docs[:5])
        else:
            representative_titles = ""

        summary_rows.append({
            "topic_id": topic_id,
            "num_unique_titles": row["Count"],
            "topic_name_auto": row["Name"],
            "keywords": keywords,
            "representative_titles": representative_titles
        })

    topics_summary = pd.DataFrame(summary_rows)

    topics_summary = topics_summary.sort_values(
        by="num_unique_titles",
        ascending=False
    )

    topics_summary.to_excel(OUTPUT_TOPICS_FILE, index=False)

    print(f"Saved topic-level summary file: {OUTPUT_TOPICS_FILE}")

    return topics_summary


# ============================================================
# 10. הוספת כמות כתבות מקוריות לכל topic
# ============================================================

def add_original_article_counts_to_summary(topics_summary, df_all):
    """
    מכיוון שהמודל רץ על כותרות ייחודיות,
    num_unique_titles מייצג מספר כותרות ייחודיות בכל topic.

    אבל חשוב לדעת גם כמה רשומות מקוריות יש בכל topic
    כולל כפילויות.
    """
    print("\nAdding original article counts to topics summary...")

    original_counts = (
        df_all
        .groupby("topic_id")
        .size()
        .reset_index(name="num_original_articles")
    )

    topics_summary = topics_summary.merge(
        original_counts,
        left_on="topic_id",
        right_on="topic_id",
        how="left"
    )

    topics_summary["num_original_articles"] = (
        topics_summary["num_original_articles"]
        .fillna(0)
        .astype(int)
    )

    topics_summary = topics_summary[
        [
            "topic_id",
            "num_unique_titles",
            "num_original_articles",
            "topic_name_auto",
            "keywords",
            "representative_titles"
        ]
    ]

    topics_summary = topics_summary.sort_values(
        by="num_original_articles",
        ascending=False
    )

    topics_summary.to_excel(OUTPUT_TOPICS_FILE, index=False)

    print(f"Updated topic-level summary file: {OUTPUT_TOPICS_FILE}")

    return topics_summary


# ============================================================
# 11. main
# ============================================================

def main():
    print("Starting topic modeling pipeline...")

    # קריאת הקובץ
    df = read_input_file(INPUT_FILE)

    # איחוד פורמט עמודות תאריך
    df = standardize_date_columns(df)

    # ניקוי כותרות
    df = clean_titles(df)

    # שמירת כל הדאטה המקורי אחרי ניקוי
    df_all = df.copy()

    # יצירת טבלה ייחודית למודל
    df_unique = create_unique_titles_df(df_all)

    # רשימת הטקסטים שעליהם המודל ילמד
    texts = df_unique[CLEAN_TITLE_COLUMN].tolist()

    # יצירת embeddings
    embeddings = create_or_load_embeddings(texts)

    # הרצת BERTopic
    topic_model, topics = run_bertopic(texts, embeddings)

    # החזרת topic_id לכל השורות המקוריות
    df_all, df_unique = assign_topics_to_all_rows(
        df_all=df_all,
        df_unique=df_unique,
        topics=topics
    )

    # יצירת טבלת סיכום topics
    topics_summary = create_topics_summary(topic_model)

    # הוספת כמות כתבות מקוריות לכל topic
    topics_summary = add_original_article_counts_to_summary(
        topics_summary=topics_summary,
        df_all=df_all
    )

    final_df = df_all.merge(
        topics_summary[
            [
                "topic_id",
                "topic_name_auto",
                "keywords",
                "representative_titles",
                "num_unique_titles",
                "num_original_articles"
            ]
        ],
        on="topic_id",
        how="left"
    )

    final_df.to_excel("articles_with_topic_details.xlsx", index=False)

    print("Saved final merged file: articles_with_topic_details.xlsx")

    print("\nProcess completed successfully!")

    print("\nCreated files:")
    print(f"1. Article-level file: {OUTPUT_ARTICLES_FILE}")
    print(f"2. Topic-level summary: {OUTPUT_TOPICS_FILE}")
    print(f"3. Embeddings file: {EMBEDDINGS_FILE}")
    print(f"4. BERTopic model folder: {MODEL_OUTPUT_DIR}")

    print("\nTop 10 topics by number of original articles:")
    print(topics_summary.head(10))


if __name__ == "__main__":
    main()
"""
04_analysis.py - wartime-stress analysis using TOPICS (not raw keywords).

Pipeline position (run order):
  01_translate_titles.py  -> adds English titles (title_en)
  02_topic_modeling.py    -> BERTopic, writes articles_with_topic_details.xlsx
  03_clean_alarms.py      -> writes alarms_clean.csv
  04_analysis.py          -> THIS FILE

Research question:
  What better predicts public anxiety (Google searches) during the war -
  physical threat (alarms) or media war-coverage?

What changed vs the keyword version:
  Media "war coverage" is now defined by TOPICS. Each article already carries a
  topic_id and that topic's keywords (from BERTopic). A topic is flagged as
  war-related if its keywords contain any war term; an article counts as war
  coverage if its topic is war-related. This is more robust than matching words
  in each individual title, because it groups titles that mean the same thing.

Inputs (place in ../data or next to this file):
  articles_with_topic_details.xlsx   (from 02_topic_modeling.py)
  alarms_clean.csv                   (from 03_clean_alarms.py)
  trends_anxiety.csv                 (Google Trends export)
"""
from pathlib import Path
from math import erf, sqrt
import numpy as np
import pandas as pd

# ---- paths: look in ../data first, then next to this script -----------------
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
def find(name):
    for p in (ROOT / "data" / name, SRC / name, ROOT / "output" / name):
        if p.exists():
            return p
    return ROOT / "data" / name      # default location for the error message
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

ARTICLES_FILE = find("articles_with_topic_details.xlsx")
ALARMS_FILE = find("alarms_clean.csv")
TRENDS_FILE = find("trends_anxiety.csv")

# column names inside the topic file (override here if yours differ)
DATE_COL = "date"
TOPIC_ID_COL = "topic_id"
KEYWORDS_COL = "keywords"          # BERTopic keywords string per topic

# A topic is war-related if its keywords contain any of these (English, because
# topics are built on the translated titles). Tune freely.
WAR_TERMS = ["war", "rocket", "missile", "hostage", "ceasefire", "gaza",
             "hezbollah", "iran", "hamas", "terror", "idf", "soldier", "killed",
             "truce", "lebanon", "attack", "strike", "siren", "alarm"]
# If you prefer to set the war topics by hand after seeing them, list their ids:
WAR_TOPIC_OVERRIDE = []           # e.g. [0, 3, 7]  (empty = auto-detect)

TERMS = {"חרדה": "anxiety", "התקף חרדה": "panic_attack", "נדודי שינה": "insomnia",
         "פסיכולוג": "psychologist", "תרופת הרגעה": "sedative"}
PRIMARY = "anxiety"


def corrp(x, y):
    s = pd.concat([x, y], axis=1).dropna(); n = len(s)
    r = s.iloc[:, 0].corr(s.iloc[:, 1])
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return r, p, n


def ols(y, X):
    s = pd.concat([y, X], axis=1).dropna(); yv = s.iloc[:, 0].values
    Xv = np.column_stack([np.ones(len(s)), s.iloc[:, 1:].values])
    b, *_ = np.linalg.lstsq(Xv, yv, rcond=None); res = yv - Xv @ b
    dof = len(s) - Xv.shape[1]
    se = np.sqrt(np.diag((res @ res) / dof * np.linalg.inv(Xv.T @ Xv)))
    r2 = 1 - (res @ res) / ((yv - yv.mean()) ** 2).sum()
    return pd.DataFrame({"coef": b, "t": b / se}, index=["const"] + list(X.columns)), r2, len(s)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def war_topic_ids(df):
    """Decide which topic_ids are war-related, from their keywords."""
    if WAR_TOPIC_OVERRIDE:
        return set(WAR_TOPIC_OVERRIDE)
    pat = "|".join(WAR_TERMS)
    topics = (df[[TOPIC_ID_COL, KEYWORDS_COL]]
              .dropna(subset=[TOPIC_ID_COL]).drop_duplicates(TOPIC_ID_COL))
    flagged = topics[topics[KEYWORDS_COL].fillna("").str.lower().str.contains(pat, regex=True)]
    print("War-related topics detected (id : keywords):")
    for _, r in flagged.iterrows():
        print(f"  {int(r[TOPIC_ID_COL]):>4} : {str(r[KEYWORDS_COL])[:70]}")
    return set(flagged[TOPIC_ID_COL].astype(int))


def monthly_war_coverage():
    df = pd.read_excel(ARTICLES_FILE)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TOPIC_ID_COL])
    war_ids = war_topic_ids(df)
    df["is_war"] = df[TOPIC_ID_COL].astype(int).isin(war_ids)
    df["month"] = df[DATE_COL].dt.to_period("M").astype(str)
    g = df.groupby("month").agg(articles=("is_war", "size"), war=("is_war", "sum"))
    return (g["war"] / g["articles"] * 100).rename("war_coverage_pct")


def build():
    tr = pd.read_csv(TRENDS_FILE)
    tr["month"] = pd.to_datetime(tr["Time"]).dt.to_period("M").astype(str)
    tr = tr.rename(columns=TERMS).set_index("month")[list(TERMS.values())]

    a = pd.read_csv(ALARMS_FILE, parse_dates=["time"])
    a["month"] = a["time"].dt.to_period("M").astype(str)
    alarms = a.groupby("month").size().rename("alarms")

    war_cov = monthly_war_coverage()

    d = pd.concat([alarms, war_cov, tr], axis=1)
    d = d[d["war_coverage_pct"].notna() & d[list(TERMS.values())].notna().all(axis=1)].copy()
    d["alarms"] = d["alarms"].fillna(0).astype(int)
    d["war_coverage_pct"] = d["war_coverage_pct"].round(1)
    return d.sort_index()


def main():
    d = build()
    d.to_csv(OUT / "topic_based_monthly.csv", encoding="utf-8-sig")
    A, W, X = "alarms", "war_coverage_pct", PRIMARY
    print(f"\nmonthly rows: {len(d)}  ({d.index.min()} .. {d.index.max()})\n")

    print("=== core correlations ===")
    for a, b in [(A, W), (A, X), (W, X)]:
        r, p, n = corrp(d[a], d[b])
        print(f"  {a:17s} vs {b:17s}  r={r:5.2f}  p={p:.4f} {stars(p)}  (n={n})")

    print("\n=== multiple regression: anxiety ~ alarms + war_coverage_pct ===")
    t, r2, n = ols(d[X], d[[A, W]]); print(t.round(3).to_string()); print(f"  R2={r2:.3f} n={n}")

    print("\nWrote output/topic_based_monthly.csv")


if __name__ == "__main__":
    main()

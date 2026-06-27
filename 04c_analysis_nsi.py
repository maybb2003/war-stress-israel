"""
04c_analysis_nsi.py - main analysis using the NSI as the media measure.

Research question:
  What better predicts public anxiety (Google searches) during the war -
  physical threat (alarms) or media war-coverage?

Here media war-coverage is measured by the Negative Sentiment Index (NSI), a
richer measure than a simple keyword count: for each month it combines the mean
negativity of the (translated) titles (AFINN) with the share of articles in
war/trauma topics:   NSI = mean_negativity * (1 + trauma_density).

Inputs (same folder):
  articles_with_topic_details1.xlsx   (from 02_topic_modeling.py)
  alarms_clean.csv  (or rocket_alarms_timeline.csv)  (from 03_clean_alarms.py)
  trends_anxiety.csv                  (Google Trends export)

Finding: even with this richer measure, media coverage predicts anxiety weakly,
while alarms predict it more strongly - which is the answer to the question.
"""
from pathlib import Path
from math import erf, sqrt
import numpy as np
import pandas as pd
from afinn import Afinn

BASE = Path(__file__).resolve().parent
ARTICLES = BASE / "articles_with_topic_details1.xlsx"
TRENDS = BASE / "trends_anxiety.csv"

# genuine war/trauma topics (by topic-name prefix from 02_topic_modeling.py)
TRAUMA_PREFIXES = ["3_", "6_", "8_", "21_", "22_", "24_", "36_"]


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
    se = np.sqrt(np.diag((res @ res) / (len(s) - Xv.shape[1]) * np.linalg.inv(Xv.T @ Xv)))
    r2 = 1 - (res @ res) / ((yv - yv.mean()) ** 2).sum()
    return pd.DataFrame({"coef": b, "t": b / se}, index=["const"] + list(X.columns)), r2, len(s)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def monthly_nsi():
    af = Afinn()
    df = pd.read_excel(ARTICLES, usecols=["date", "title", "clean_title_en", "topic_name_auto"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(["date", "title"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["neg"] = df["clean_title_en"].apply(lambda t: 0.0 if pd.isna(t) else -af.score(str(t)))
    topic = df["topic_name_auto"].astype(str).str.strip().str.lower()
    df["is_war"] = topic.apply(lambda s: any(s.startswith(p) for p in TRAUMA_PREFIXES))
    g = df.groupby("month").agg(neg=("neg", "mean"), war=("is_war", "mean"))
    return (g["neg"] * (1 + g["war"])).rename("NSI")


def main():
    nsi = monthly_nsi()

    af = BASE / "alarms_clean.csv"
    if not af.exists():
        af = BASE / "rocket_alarms_timeline.csv"
    a = pd.read_csv(af, parse_dates=["time"])
    a["month"] = a["time"].dt.to_period("M").astype(str)
    alarms = a.groupby("month").size().rename("alarms")

    tr = pd.read_csv(TRENDS)
    tr["month"] = pd.to_datetime(tr["Time"]).dt.to_period("M").astype(str)
    anx = tr.set_index("month")["חרדה"].rename("anxiety")

    d = pd.concat([alarms, anx, nsi], axis=1).dropna()
    d.to_csv(BASE / "nsi_analysis_monthly.csv", encoding="utf-8-sig")
    A, M, X = "alarms", "NSI", "anxiety"
    print(f"monthly rows: {len(d)}  ({d.index.min()} .. {d.index.max()})\n")

    print("=== core correlations ===")
    for a_, b_ in [(A, M), (A, X), (M, X)]:
        r, p, n = corrp(d[a_], d[b_])
        print(f"  {a_:7s} vs {b_:7s}  r={r:5.2f}  p={p:.4f} {stars(p)}  (n={n})")

    print("\n=== multiple regression: anxiety ~ alarms + NSI ===")
    t, r2, n = ols(d[X], d[[A, M]]); print(t.round(3).to_string()); print(f"  R2={r2:.3f} n={n}")
    print("\nWrote nsi_analysis_monthly.csv")


if __name__ == "__main__":
    main()

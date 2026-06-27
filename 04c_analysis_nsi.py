"""
04c_analysis_nsi.py - main analysis (WEEKLY), Israel 2023-2025.

Research question:
  What better predicts public anxiety (Google searches) during the war -
  physical threat (alarms) or media war-coverage?

Everything is aggregated to WEEKS (~84 points), which gives real statistical
power and lets us see lead-lag (does anxiety follow alarms with a delay?).

Media coverage is measured by the Negative Sentiment Index (NSI):
  NSI = mean_negativity(titles, AFINN) * (1 + war/trauma-topic density)

Inputs (same folder):
  articles_with_topic_details1.xlsx        (from 02_topic_modeling.py)
  alarms_clean.csv | rocket_alarms_timeline.csv   (from 03_clean_alarms.py)
  trends_anxiety_weekly.csv                 (stitched weekly Google Trends)

Finding: at weekly resolution alarms predict anxiety significantly, the effect
is strongest at a ~1-week lag (anxiety follows alarms), and alarms remain the
stronger predictor than coverage in a joint regression.
"""
from pathlib import Path
from math import erf, sqrt
import numpy as np
import pandas as pd
from afinn import Afinn

BASE = Path(__file__).resolve().parent
ARTICLES = BASE / "articles_with_topic_details1.xlsx"
TRENDS_WEEKLY = BASE / "trends_anxiety_weekly.csv"
TRAUMA_PREFIXES = ["3_", "6_", "8_", "21_", "22_", "24_", "36_"]
MIN_ARTICLES_PER_WEEK = 30      # drop sparse partial weeks


def to_week(s):
    return pd.to_datetime(s).dt.to_period("W-SAT").apply(lambda r: r.start_time)


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


def weekly_nsi():
    af = Afinn()
    df = pd.read_excel(ARTICLES, usecols=["date", "title", "clean_title_en", "topic_name_auto"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(["date", "title"])
    df["week"] = to_week(df["date"])
    df["neg"] = df["clean_title_en"].apply(lambda t: 0.0 if pd.isna(t) else -af.score(str(t)))
    topic = df["topic_name_auto"].astype(str).str.strip().str.lower()
    df["war"] = topic.apply(lambda s: any(s.startswith(p) for p in TRAUMA_PREFIXES))
    g = df.groupby("week").agg(neg=("neg", "mean"), war=("war", "mean"), n=("neg", "size"))
    g = g[g["n"] >= MIN_ARTICLES_PER_WEEK]
    return (g["neg"] * (1 + g["war"])).rename("NSI")


def main():
    nsi = weekly_nsi()

    af = BASE / "alarms_clean.csv"
    if not af.exists():
        af = BASE / "rocket_alarms_timeline.csv"
    a = pd.read_csv(af, parse_dates=["time"])
    a["week"] = to_week(a["time"])
    alarms = a.groupby("week").size().rename("alarms")

    tr = pd.read_csv(TRENDS_WEEKLY)
    tr["week"] = to_week(tr["Time"])
    anx = tr.set_index("week")["חרדה"].rename("anxiety")

    d = pd.concat([alarms, anx, nsi], axis=1).dropna().sort_index()
    d.to_csv(BASE / "weekly_nsi_analysis.csv", encoding="utf-8-sig")
    A, M, X = "alarms", "NSI", "anxiety"
    print(f"weekly rows: {len(d)}  ({d.index.min().date()} .. {d.index.max().date()})\n")

    print("=== correlations ===")
    for a_, b_ in [(A, M), (A, X), (M, X)]:
        r, p, n = corrp(d[a_], d[b_])
        print(f"  {a_:7s} vs {b_:8s}  r={r:5.2f}  p={p:.4f} {stars(p)}  (n={n})")

    print("\n=== lead-lag: anxiety_t vs alarms_(t-k weeks) ===")
    for k in range(0, 4):
        r, p, n = corrp(d[X], d[A].shift(k))
        print(f"  alarms lead by {k} wk: r={r:5.2f} p={p:.4f} {stars(p)}")

    print("\n=== regression: anxiety ~ alarms + NSI ===")
    t, r2, n = ols(d[X], d[[A, M]]); print(t.round(3).to_string()); print(f"  R2={r2:.3f} n={n}")
    print("\nWrote weekly_nsi_analysis.csv")


if __name__ == "__main__":
    main()

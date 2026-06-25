"""
Wartime-stress analysis - three high-resolution sources only (no survey).

Research question:
  What better predicts public anxiety (as reflected in Google searches) during
  the war - physical threat (alarms) or media war-coverage?

Sources (all monthly, Apr 2023 - Mar 2025, ~24 points):
  alarms            - physical threat exposure        (alarms_clean.csv)
  war_coverage_pct  - media information exposure (%)   (stress_media_..._with_duplicates.xlsx)
  anxiety           - public psychological response    (trends_anxiety.csv, "חרדה")

Analyses: correlations (+p), lead-lag, multiple regression (levels and first
differences). Google Trends is validated against alarms (it should rise with the
threat) rather than against any survey.
"""
from pathlib import Path
from math import erf, sqrt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE
TERMS = {"חרדה": "anxiety", "התקף חרדה": "panic_attack", "נדודי שינה": "insomnia",
         "פסיכולוג": "psychologist", "תרופת הרגעה": "sedative"}
PRIMARY = "anxiety"
WAR = ["מלחמה", "רקטה", "רקטות", "טיל", "חטוף", "חטופים", "הפסקת אש", "עזה",
       "חיזבאללה", "חזבאללה", "איראן", "חמאס", "פיגוע", "צהל", "נהרג", "נפל",
       "לבנון", "צבע אדום", "אזעקה", "מחבל", "חרבות ברזל", "חזית"]


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


def build():
    tr = pd.read_csv(BASE / "trends_anxiety.csv")
    tr["month"] = pd.to_datetime(tr["Time"]).dt.to_period("M").astype(str)
    tr = tr.rename(columns=TERMS).set_index("month")[list(TERMS.values())]

    alarms_file = BASE / "alarms_clean.csv"
    if not alarms_file.exists():
        alarms_file = BASE / "rocket_alarms_timeline.csv"   # raw fallback (only "time" is needed here)
    a = pd.read_csv(alarms_file, parse_dates=["time"])
    a["month"] = a["time"].dt.to_period("M").astype(str)
    alarms = a.groupby("month").size().rename("alarms")

    m = pd.read_excel(BASE / "stress_media_merged_with_duplicates.xlsx", usecols=["date", "title"])
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.dropna(subset=["date"]).drop_duplicates(["date", "title"])
    m["war"] = m["title"].fillna("").astype(str).str.contains("|".join(WAR), regex=True)
    m["month"] = m["date"].dt.to_period("M").astype(str)
    g = m.groupby("month").agg(articles=("war", "size"), war=("war", "sum"))
    war_cov = (g["war"] / g["articles"] * 100).rename("war_coverage_pct")

    d = pd.concat([alarms, war_cov, tr], axis=1)
    d = d[d["war_coverage_pct"].notna() & d[list(TERMS.values())].notna().all(axis=1)].copy()
    d["alarms"] = d["alarms"].fillna(0).astype(int)
    d["war_coverage_pct"] = d["war_coverage_pct"].round(1)
    return d.sort_index()


def main():
    d = build()
    d.to_csv(OUT / "three_sources_monthly.csv", encoding="utf-8-sig")
    A, W, X = "alarms", "war_coverage_pct", PRIMARY
    print(f"monthly rows: {len(d)}  ({d.index.min()} .. {d.index.max()})\n")

    print("=== validation: which search term rises with alarms? ===")
    for label in TERMS.values():
        r, p, _ = corrp(d[label], d[A])
        print(f"  {label:13s} vs alarms  r={r:5.2f} {stars(p)}")

    print("\n=== core correlations ===")
    for a, b in [(A, W), (A, X), (W, X)]:
        r, p, n = corrp(d[a], d[b])
        print(f"  {a:17s} vs {b:17s}  r={r:5.2f}  p={p:.4f} {stars(p)}  (n={n})")

    print("\n=== multiple regression: anxiety ~ alarms + war_coverage_pct (levels) ===")
    t, r2, n = ols(d[X], d[[A, W]]); print(t.round(3).to_string()); print(f"  R2={r2:.3f} n={n}")

    print("\n=== regression on first differences (controls for the shared Oct-23 jump) ===")
    dd = d[[X, A, W]].diff().dropna()
    t2, r2b, n2 = ols(dd[X], dd[[A, W]]); print(t2.round(3).to_string()); print(f"  R2={r2b:.3f} n={n2}")

    print("\nWrote three_sources_monthly.csv")


if __name__ == "__main__":
    main()

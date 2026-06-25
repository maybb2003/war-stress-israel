"""
05_plot.py - three-layer figure from the monthly analysis table.

Reads the monthly table written by 04_analysis.py (topic_based_monthly.csv) or
04b_analysis_keywords.py (three_sources_monthly.csv), z-scores the three series
so they share one scale, and plots them over time.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

# use whichever monthly table exists
for name in ("topic_based_monthly.csv", "three_sources_monthly.csv"):
    if (OUT / name).exists():
        d = pd.read_csv(OUT / name)
        break
else:
    raise SystemExit("Run 04_analysis.py or 04b_analysis_keywords.py first.")

d = d.sort_values("month").reset_index(drop=True)
z = lambda c: (d[c] - d[c].mean()) / d[c].std()
x = range(len(d))

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(x, z("alarms"), "o-", lw=2.4, color="#c0392b", label="Alarms (physical threat)")
ax.plot(x, z("war_coverage_pct"), "s--", lw=1.9, color="#2c3e50", label="War media coverage")
ax.plot(x, z("anxiety"), "^--", lw=1.9, color="#16a085", label='Anxiety searches ("חרדה")')
ax.axhline(0, color="#ccc", lw=.8)
if "2023-10" in list(d["month"]):
    i = list(d["month"]).index("2023-10")
    ax.axvline(i, color="#888", ls=":", lw=1)
    ax.text(i + 0.1, ax.get_ylim()[1] * 0.9, "Oct 2023", fontsize=8, color="#555")
ax.set_xticks(list(x))
ax.set_xticklabels(d["month"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("z-score (standardized)")
ax.set_title("Three layers of wartime stress, monthly (Israel, 2023-2025)",
             fontsize=12, weight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=9, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "three_layer_monthly.png", dpi=150)
print("Wrote output/three_layer_monthly.png")

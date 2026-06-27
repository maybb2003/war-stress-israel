"""Sum alarms per calendar month, with nationwide-coverage info.

Output: monthly_alarms.csv
  month         - YYYY-MM
  total_alarms  - total alarms across the whole country that month
  regions_hit   - how many of the 7 geographic regions had >=1 alarm (0-7)
  nationwide    - True when at least 4 of the 7 regions were hit that month

Filter however you like later, e.g. keep rows where nationwide is True (4+
regions = "broad" coverage), tighten to regions_hit == 7, or relax further.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
c = pd.read_csv(BASE / "alarms_clean.csv")
c["time"] = pd.to_datetime(c["time"], errors="coerce")
c = c.dropna(subset=["time"])

REGIONS = {"North", "Center", "South", "Yehuda & Shomron",
           "Jerusalem Area", "Haifa district", "Tel Aviv and Central area"}

c["month"] = c["time"].dt.to_period("M").astype(str)

total = c.groupby("month").size().rename("total_alarms")
hit = (c[c["region"].isin(REGIONS)]
       .groupby("month")["region"].nunique().rename("regions_hit"))

monthly = (pd.concat([total, hit], axis=1)
           .fillna({"regions_hit": 0})
           .astype({"total_alarms": int, "regions_hit": int})
           .reset_index())
monthly["nationwide"] = monthly["regions_hit"] >= 4

monthly.to_csv(BASE / "monthly_alarms.csv", index=False, encoding="utf-8-sig")
print(f"{len(monthly)} months, of which {int(monthly['nationwide'].sum())} hit 4+ regions")
print(monthly.tail(12).to_string(index=False))

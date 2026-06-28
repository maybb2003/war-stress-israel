# war-stress-israel

Analysis of wartime stress in Israel (2023–2025): do rocket alarms or media
war-coverage better predict public anxiety (measured by Google searches)?

**Main finding:** alarms predict anxiety searches more strongly than media
coverage. At weekly resolution both are individually significant (alarms↔anxiety
r≈0.26, p≈0.015; NSI↔anxiety r≈0.22, p≈0.04), but in a regression with both
predictors alarms are the stronger one. The alarm→anxiety link is strongest at a
~1-week lag (anxiety follows alarms). Data are weekly, Apr 2023 – Feb 2025
(~84 weeks after merging all sources).

---

## Tech stack
- **Python 3** — data processing and the main analysis (steps `01`–`04c`).
  Key libraries: pandas, numpy, openpyxl, deep-translator, BERTopic, afinn.
  Full list and install instructions in **`requirements.txt`**
  (`pip install -r requirements.txt`).
- **R** — plotting and regression (steps `05`–`06`).
  Packages: ggplot2, dplyr (`install.packages(c("ggplot2","dplyr"))`).

---

## Files (run in this order)

| File | What it does |
|------|--------------|
| `01_translate_titles.py` | Translates news titles Hebrew → English (`title_en`). |
| `02_topic_modeling.py` | Groups article titles into topics (BERTopic) → `articles_with_topic_details.xlsx`. |
| `03_clean_alarms.py` | Cleans raw alarms and maps each town to a region (needs `cities_reference.json`) → `alarms_clean.csv`. |
| `04c_analysis_nsi.py` | **Main analysis (weekly, ~84 wks).** Media coverage = NSI (sentiment × trauma-topic density). Compares alarms vs coverage as predictors of anxiety, including lead-lag. Alarms predict anxiety more strongly, strongest at a ~1-week lag. |
| `05_plot.R` | (R) Three-layer figure from `weekly_nsi_analysis.csv` → `three_layer_dark_style.png`. |
| `06_regression_analysis.R` | (R) Regression `anxiety ~ alarms + NSI`: writes coefficient / CI / model-fit CSVs and two diagnostic plots (residuals-vs-fitted, Q-Q). |
| `monthly_alarms.py` | Helper: monthly nationwide alarm counts (+ "4+ regions" flag). |

`04c` is the main analysis; it needs the topic model output from `02` first.

---

## Data files (NOT in git — keep them locally, next to the scripts)
Put these in the same folder as the code when you run it:
- `rocket_alarms_timeline.csv` (raw alarms) + `cities_reference.json` → needed by `03`
- `stress_media_merged_with_duplicates.xlsx` (news articles) → needed by `02`
- `trends_anxiety_weekly.csv` (weekly Google Trends, 5 terms; stitched from short-range exports) → needed by `04c`
- `icamh_addiction_prevalence.pdf` (ICAMH survey) → external reference only, not used in the analysis

The `.gitignore` keeps data and generated files (`.csv`, `.xlsx`, …) out of git,
so only code is committed. Every script reads and writes in its own folder.

## How to run
```bash
pip install -r requirements.txt          # Python steps (01-04c)
# R steps (05, 06) need R with: install.packages(c("ggplot2","dplyr"))

python 01_translate_titles.py      # Hebrew titles -> English (title_en)
python 02_topic_modeling.py        # -> articles_with_topic_details1.xlsx  (heavy / slow)
python 03_clean_alarms.py          # rocket_alarms_timeline.csv -> alarms_clean.csv
python 04c_analysis_nsi.py         # -> weekly_nsi_analysis.csv
Rscript 05_plot.R                  # -> three_layer_dark_style.png
Rscript 06_regression_analysis.R   # -> regression_*_appendix.csv + 2 diagnostic plots
```
Steps 01–02 are the heavy one-time setup that produces the topic file; once you
have it, day-to-day work is just 03 → 04c → 05 → 06.
(`04c` uses `alarms_clean.csv` if present, otherwise reads `rocket_alarms_timeline.csv` directly.)

## Notes / decisions
- Resolution is weekly (~84 weeks). The lead-lag check shows anxiety follows
  alarms by about one week.
- The ICAMH survey (5 time points) was dropped from the analysis — too few points
  for our weekly resolution. Worth one sentence in the writeup, not as a variable.
- Google Trends is validated against alarms (it rises with the threat), not a survey.
- "anxiety" (חרדה) is the primary search term: the only one that rises with alarms
  in the expected direction. We did not switch to a term that merely produced a
  significant fit (that would be an n=4 false positive).
- `monthly_alarms.py` is a separate descriptive helper and is not part of the
  weekly analysis chain.

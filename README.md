# war-stress-israel

Analysis of wartime stress in Israel (2023–2025): do rocket alarms or media
war-coverage better predict public anxiety (measured by Google searches)?

**Main finding:** alarms predict anxiety searches more strongly than media
coverage. Alarms ↔ anxiety move together (r≈0.47, p≈0.01); media coverage alone
does not. In a regression with both predictors, alarms are the stronger one.
Data are monthly, Apr 2023 – Mar 2025 (~24 points).

---

## Files (run in this order)

| File | What it does |
|------|--------------|
| `01_translate_titles.py` | Translates news titles Hebrew → English (`title_en`). |
| `02_topic_modeling.py` | Groups article titles into topics (BERTopic) → `articles_with_topic_details.xlsx`. |
| `03_clean_alarms.py` | Cleans raw alarms, maps to regions → `alarms_clean.csv`. |
| `04c_analysis_nsi.py` | **Main analysis.** Media coverage is measured by the NSI (sentiment × trauma-topic density); compares alarms vs coverage as predictors of anxiety. Alarms predict anxiety more strongly. |
| `05_plot.py` | Builds the three-layer figure from the monthly table. |
| `monthly_alarms.py` | Helper: monthly nationwide alarm counts (+ "4+ regions" flag). |

`04c` is the main analysis; it needs the topic model output from `02` first.

---

## Data files (NOT in git — keep them locally, next to the scripts)
Put these in the same folder as the code when you run it:
- `rocket_alarms_timeline.csv` (raw alarms) + `cities_reference.json` → needed by `03`
- `stress_media_merged_with_duplicates.xlsx` (news articles) → needed by `02`
- `trends_anxiety.csv` (Google Trends, 5 terms) → needed by `04c`
- `icamh_addiction_prevalence.pdf` (ICAMH survey) → external reference only, not used in the analysis

The `.gitignore` keeps data and generated files (`.csv`, `.xlsx`, …) out of git,
so only code is committed. Every script reads and writes in its own folder.

## How to run
```bash
pip install -r requirements.txt
python 03_clean_alarms.py          # rocket_alarms_timeline.csv -> alarms_clean.csv
python 04c_analysis_nsi.py         # -> nsi_analysis_monthly.csv
python 05_plot.py                  # -> three_layer_monthly.png
```
Full path: `01` → `02` → `03` → `04c` → `05`.
(`04c` uses `alarms_clean.csv` if present, otherwise reads `rocket_alarms_timeline.csv` directly.)

## Notes / decisions
- The ICAMH survey (5 time points) was dropped from the analysis — too few points
  for monthly resolution. Worth one sentence in the writeup, not as a variable.
- Google Trends is validated against alarms (it rises with the threat), not a survey.
- "anxiety" (חרדה) is the primary search term: the only one that rises with alarms
  in the expected direction. We did not switch to a term that merely produced a
  significant fit (that would be an n=4 false positive).

## Possible next step
Re-download Google Trends over a shorter range so it returns **weekly** data
(~100 points instead of 24) for more statistical power.

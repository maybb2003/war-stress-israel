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
| `04_analysis.py` | Main analysis: war-coverage from **topics** + alarms + Google Trends → correlations & regression. |
| `04b_analysis_keywords.py` | Same analysis, simpler: war-coverage from Hebrew **keywords** (no topic model needed). |
| `05_plot.py` | Builds the three-layer figure from the monthly table. |
| `monthly_alarms.py` | Helper: monthly nationwide alarm counts (+ "4+ regions" flag). |

`04` is the stronger version but needs the topic model (`02`) first.
`04b` is a self-contained baseline that runs immediately — start here.

---

## Data files (NOT in git — keep them locally, next to the scripts)
Put these in the same folder as the code when you run it:
- `alarms.xlsx` + `cities_reference.json`  → needed by `03`
- `stress_media_merged_with_duplicates.xlsx` → needed by `02` / `04b`
- `trends_anxiety.csv` → needed by `04` / `04b`

The `.gitignore` keeps data and generated files (`.csv`, `.xlsx`, …) out of git,
so only code is committed. Every script reads and writes in its own folder.

## How to run
```bash
pip install -r requirements.txt
python 03_clean_alarms.py          # -> alarms_clean.csv
python 04b_analysis_keywords.py    # -> three_sources_monthly.csv  (quick path)
python 05_plot.py                  # -> three_layer_monthly.png
```
Full topic path: `01` → `02` → move its output next to the scripts → `04` → `05`.

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

"""
Clean Israeli Home Front Command ("Pikud HaOref") alarm data.

What this script does:
  1. Loads the raw alarms table (alarms.xlsx).
  2. Translates the categorical columns to English (threat type, source).
  3. Maps every Hebrew city name to one of 7 geographic regions, using the
     official Oref "city -> alert zone" reference and a zone -> region rollup.
  4. Adds an English city name and a calendar date column.
  5. Produces two outputs:
        - alarms_clean.csv               (the cleaned, English, row-level table)
        - alarms_per_region_per_day.csv  (count of alarms per region per day)
     and prints a short report of any cities that could not be mapped.

The city reference (cities_reference.json) is the public Oref city list from
the MIT-licensed eladnava/pikud-haoref-api project. Every city in it carries a
Hebrew name, an English name, coordinates and an Oref "zone". Those ~30 zones
are rolled up here into the 7 administrative regions requested.
"""

import json
import re
import difflib
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
DATA = ROOT / "data"
OUT = ROOT / "output"; OUT.mkdir(exist_ok=True)
ALARMS_FILE = DATA / "alarms.xlsx"
REFERENCE_FILE = DATA / "cities_reference.json"
OUT_CLEAN = OUT / "alarms_clean.csv"
OUT_COUNTS = OUT / "alarms_per_region_per_day.csv"

# ---------------------------------------------------------------------------
# Static translation tables
# ---------------------------------------------------------------------------
# Hebrew threat description -> English label.
THREAT_TYPE_EN = {
    "ירי רקטות וטילים": "Rocket and missile fire",
    "חדירת כלי טיס עוין": "Hostile aircraft intrusion",
    "רעידת אדמה": "Earthquake",
    "חדירת מחבלים": "Terrorist infiltration",
    "אזהרה": "Warning",
}

# Oref alert zone (Hebrew) -> one of the 7 requested regions.
# "ים המלח" (Dead Sea) straddles two administrative regions and is resolved
# separately by latitude (see resolve_region).
ZONE_TO_REGION = {
    "אילת": "South",
    "בקעה": "Yehuda & Shomron",            # Jordan Valley settlements
    "בקעת בית שאן": "North",
    "גולן דרום": "North",
    "גולן צפון": "North",
    "גליל עליון": "North",
    "גליל תחתון": "North",
    "דן": "Tel Aviv and Central area",      # Gush Dan / Tel Aviv district
    "דרום הנגב": "South",
    "הכרמל": "Haifa district",
    "המפרץ": "Haifa district",              # Haifa Bay / Krayot
    "העמקים": "North",                      # Jezreel & Harod valleys
    "השפלה": "Center",
    "ואדי ערה": "Haifa district",
    "יהודה": "Yehuda & Shomron",
    "ירושלים": "Jerusalem Area",
    "ירקון": "Center",
    "לכיש": "South",
    "מנשה": "Haifa district",
    "מערב הנגב": "South",
    "מערב לכיש": "South",
    "מרכז הגליל": "North",
    "מרכז הנגב": "South",
    "עוטף עזה": "South",
    "ערבה": "South",
    "קו העימות": "North",                   # northern confrontation line
    "שומרון": "Yehuda & Shomron",
    "שפלת יהודה": "Jerusalem Area",         # Judea Foothills (Beit Shemesh area)
    "שרון": "Center",
}

# Latitude threshold that splits the Dead Sea zone between the northern
# settlements (Judea & Samaria) and the southern ones (South district).
DEAD_SEA_LAT_SPLIT = 31.55

# A few frequent places are missing from the city reference (campuses, farms,
# junctions, small settlements). These are mapped here by hand, only where the
# location is unambiguous. Keys are the *normalized* Hebrew names; values are
# (region, English name). Genuinely ambiguous names are left out on purpose so
# they surface as "Unknown" rather than being mis-assigned.
MANUAL_OVERRIDES = {
    'דוב"ב': ("North", "Dovev"),
    "אזור תעשייה אכזיב מילואות": ("North", "Achziv-Milouot Industrial Zone"),
    "חוות אירוח גורן": ("North", "Goren Guest Farm"),
    "שלוחות": ("North", "Shluhot"),
    "מכללת ספיר": ("South", "Sapir College"),
    "עמיעוז": ("South", "Amioz"),
    "אוהד": ("South", "Ohad"),
    "גבעולים": ("South", "Givolim"),
    "מלילות": ("South", "Melilot"),
    "כרמי גת": ("South", "Karmei Gat"),
    "תעשיון חצב": ("South", "Hatzav Industrial Zone"),
    'מתחם צומת שוקת': ("South", "Shoket Junction"),
    "ממשית": ("South", "Mamshit"),
    "אמציה": ("South", "Amatzia"),
    "כושי רמון": ("South", "Kushi Ramon"),
    "רומאנה": ("Haifa district", "Rumana"),
    "ימין אורד": ("Haifa district", "Yemin Orde"),
    "כפר הנוער ימין אורד": ("Haifa district", "Yemin Orde Youth Village"),
    "יערות הכרמל": ("Haifa district", "Ya'arot HaCarmel"),
    "כלא דמון": ("Haifa district", "Damon Prison"),
    "גבעת חביבה": ("Haifa district", "Givat Haviva"),
    'מתחם "חנה וסע" שפיים': ("Center", "Shefayim Park & Ride"),
    "כפר יעבץ": ("Center", "Kfar Yavetz"),
    "סינמה סיטי גלילות": ("Tel Aviv and Central area", "Cinema City Glilot"),
    "שער הגיא": ("Jerusalem Area", "Sha'ar HaGai"),
    "צומת האלה": ("Jerusalem Area", "HaEla Junction"),
    'ייט"ב': ("Yehuda & Shomron", "Yitav"),
    "חלמיש": ("Yehuda & Shomron", "Halamish"),
    "מעלה שומרון": ("Yehuda & Shomron", "Ma'ale Shomron"),
    "חברון": ("Yehuda & Shomron", "Hebron"),
    # corrections for near-identical names that fuzzy matching gets wrong
    "איבים": ("South", "Ibim"),                 # not אביבים (North)
    "שובה": ("South", "Shuva"),                 # not שואבה (Jerusalem hills)
    "גבעות גורל": ("South", "Givot Goral"),
    "שעלבים": ("Center", "Sha'alvim"),
    "ישע": ("South", "Yesha"),
    "נחשולים": ("Haifa district", "Nahsholim"),
    "תל יוסף": ("North", "Tel Yosef"),
    "כפר נחום": ("North", "Capernaum"),
    "בית סוהר השרון": ("Center", "HaSharon Prison"),
    "שהם": ("Center", "Shoham"),
    "קידר": ("Yehuda & Shomron", "Kedar"),
    "נבי סמואל": ("Yehuda & Shomron", "Nabi Samwil"),
    "רמת הנדיב": ("Haifa district", "Ramat HaNadiv"),
    "צומת בנימינה": ("Haifa district", "Binyamina Junction"),
    "בית ספר אורט בנימינה": ("Haifa district", "ORT Binyamina School"),
    "מרכז ימי קיסריה": ("Haifa district", "Caesarea Marine Center"),
    "אזור תעשייה חבל מודיעין": ("Center", "Modiin Region Industrial Zone"),
    "חוות שיקמים": ("South", "Havat Shikmim"),
    "כרמי קטיף": ("South", "Karmei Katif"),
    "צומת דבירה": ("South", "Devira Junction"),
    "עיר אובות": ("South", "Ir Ovot"),
    "חוף ניצנים": ("South", "Nitzanim Beach"),
    "בית סוהר מגידו": ("North", "Megiddo Prison"),
    "בית סוהר שיטה וגלבוע": ("North", "Shita-Gilboa Prison"),
    "עין כמונים": ("North", "Ein Kamonim"),
    "עינבר": ("North", "Inbar"),
    "שבלי": ("North", "Shibli"),
    "טבחה": ("North", "Tabgha"),
    "מצוק עורבים": ("North", "Arbel Cliff"),
    "כעביה": ("North", "Ka'abiyye"),
    "חוף אמנון": ("North", "Amnon Beach"),
    "חוף כינר": ("North", "Kinar Beach"),
    "חוף גולן": ("North", "Golan Beach"),
    "חוף סוסיתא": ("North", "Susita Beach"),
    "חוף כורסי": ("North", "Kursi Beach"),
    "חוף גופרה": ("North", "Gofra Beach"),
    # nationwide drills / countrywide alerts
    "ברחבי הארץ": ("Nationwide", "Nationwide"),
}


# ---------------------------------------------------------------------------
# City-name normalisation and matching
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Normalise a Hebrew place name for comparison."""
    text = str(text).strip().replace("\u05be", "-")          # maqaf -> hyphen
    text = re.sub(r"[\u2018\u2019\u05f3]", "'", text)         # apostrophes -> '
    text = re.sub(r"[\u201c\u201d\u05f4]", '"', text)         # quotes -> "
    text = re.sub(r"''", '"', text)                          # double ' -> gershayim
    text = re.sub(r"\s+", " ", text)
    return text


def core_name(text: str) -> str:
    """Reduce a name to the 'core' settlement name.

    Drops the "industrial zone" prefix, the "and its surroundings" suffix, and
    anything after a comma / parenthesis / " - " separator, so that sub-areas
    of a city (which all share the same zone) collapse to the same key.
    """
    text = normalize(text)
    text = re.sub(r"^אזור תעשייה\s+", "", text)
    text = re.sub(r"^איזור תעשייה\s+", "", text)
    text = re.sub(r"^אזור תעשיה\s+", "", text)
    text = re.sub(r"\s*והפזורה$", "", text)
    text = re.sub(r"\s*ופזורה$", "", text)
    text = re.split(r"\s*[,(]\s*| - ", text)[0].strip()
    text = text.replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def build_reference(reference_path: Path):
    """Build the lookup structures from the Oref city reference file."""
    cities = json.loads(reference_path.read_text(encoding="utf-8"))

    exact = {}        # normalized hebrew name      -> city record
    by_core = {}      # core settlement name        -> city record
    by_geresh = {}    # core without geresh/quotes  -> city record
    for city in cities:
        zone = city.get("zone")
        if not zone:                       # skip the "Select All" pseudo entry
            continue
        for key in {city.get("name"), city.get("value")}:
            if key:
                exact.setdefault(normalize(key), city)
        ck = core_name(city.get("value") or city.get("name"))
        if ck:
            by_core.setdefault(ck, city)
            by_geresh.setdefault(re.sub(r"['\"]", "", ck), city)
    return exact, by_core, by_geresh


def match_city(raw_name, exact, by_core, by_geresh, core_keys):
    """Return the best matching reference city record, or None.

    Matching goes from strict to loose and deliberately avoids broad substring
    matching, which is unsafe for short Hebrew names where a one-letter change
    means a different place (e.g. "איבים" vs "אביבים").
    """
    norm = normalize(raw_name)
    if norm in exact:
        return exact[norm]

    for variant in (re.sub(r"\s*והפזורה$", "", norm),
                    re.sub(r"^אזור תעשייה\s+", "", norm)):
        if variant in exact:
            return exact[variant]

    ck = core_name(raw_name)
    if ck in by_core:                       # exact core (collapses sub-areas)
        return by_core[ck]

    deg = re.sub(r"['\"]", "", ck)          # ignore geresh/gershayim
    if deg in by_geresh:
        return by_geresh[deg]

    first = ck.split(" ")[0]                # multi-place names -> first place
    if len(first) >= 4 and first in by_core:
        return by_core[first]

    close = difflib.get_close_matches(ck, core_keys, n=1, cutoff=0.9)
    if close:
        return by_core[close[0]]
    return None


def resolve_region(city_record):
    """Map a matched city record to one of the 7 regions."""
    zone = city_record.get("zone")
    if zone == "ים המלח":                  # Dead Sea: split by latitude
        lat = city_record.get("lat") or 0
        return "Yehuda & Shomron" if lat >= DEAD_SEA_LAT_SPLIT else "South"
    return ZONE_TO_REGION.get(zone)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    df = pd.read_excel(ALARMS_FILE)
    print(f"Loaded {len(df):,} rows, {df['cities'].nunique():,} unique cities.")

    # --- translate categorical columns -------------------------------------
    df["threat_type"] = df["description"].map(THREAT_TYPE_EN).fillna(df["description"])
    df["origin"] = df["origin"].fillna("Unknown")

    # --- build the city -> region map for the unique cities only -----------
    exact, by_core, by_geresh = build_reference(REFERENCE_FILE)
    core_keys = list(by_core)

    region_map, city_en_map = {}, {}
    for raw in df["cities"].astype(str).unique():
        override = MANUAL_OVERRIDES.get(normalize(raw))
        if override is not None:
            region_map[raw], city_en_map[raw] = override
            continue
        rec = match_city(raw, exact, by_core, by_geresh, core_keys)
        if rec is not None:
            region_map[raw] = resolve_region(rec) or "Unknown"
            city_en_map[raw] = rec.get("name_en") or raw
        else:
            region_map[raw] = "Unknown"
            city_en_map[raw] = raw

    df["city_en"] = df["cities"].astype(str).map(city_en_map)
    df["region"] = df["cities"].astype(str).map(region_map)

    # --- date column --------------------------------------------------------
    df["date"] = pd.to_datetime(df["time"]).dt.date

    # --- assemble the cleaned table ----------------------------------------
    clean = df[[
        "time", "date", "cities", "city_en", "region",
        "threat", "threat_type", "origin", "id", "rid",
    ]].rename(columns={"cities": "city_he"})
    clean.to_csv(OUT_CLEAN, index=False, encoding="utf-8-sig")

    # --- count alarms per region per day -----------------------------------
    counts = (
        clean.groupby(["date", "region"])
        .size()
        .reset_index(name="alarm_count")
        .sort_values(["date", "region"])
    )
    counts.to_csv(OUT_COUNTS, index=False, encoding="utf-8-sig")

    # --- report -------------------------------------------------------------
    unmatched = sorted({c for c, r in region_map.items() if r == "Unknown"})
    unmatched_rows = int(df["region"].eq("Unknown").sum())
    print(f"Mapped cities : {df['cities'].nunique() - len(unmatched):,} "
          f"/ {df['cities'].nunique():,}")
    print(f"Unmatched rows: {unmatched_rows:,} "
          f"({unmatched_rows / len(df) * 100:.2f}% of all rows)")
    print("\nAlarms per region (whole period):")
    print(clean["region"].value_counts().to_string())
    print(f"\nWrote: {OUT_CLEAN.name}  and  {OUT_COUNTS.name}")
    if unmatched:
        print(f"\n{len(unmatched)} unmatched city names (region='Unknown'); "
              "extend the reference or add manual overrides to map these:")
        print(", ".join(unmatched[:40]) + (" ..." if len(unmatched) > 40 else ""))


if __name__ == "__main__":
    main()

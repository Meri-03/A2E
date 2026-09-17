# aselam alikum
# Step 2 - Clean & Normalize

import logging
import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


# ==================== Normalization ====================

GEEZ_NORMALIZATION_MAP = {
    # ha family (ሐ series) — historical homophone of ሀ
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    # ha family (ኀ series) — also homophone of ሀ
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    # se family
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
    # glottal-a family
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    # tse family
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
    # ha family — labialized (wa-glide) forms; the only base family with this variant
    "ኈ": "ሗ", "ኊ": "ሗ", "ኋ": "ሗ", "ኌ": "ሗ", "ኍ": "ሗ",
    # se family — labialized form (mirrors the base ሠ→ሰ fold)
    "ሧ": "ሷ",
}

# Latin punctuation typed in Amharic text -> proper Ge'ez punctuation.
# Applied ONLY to the amharic column, never to english.
LATIN_TO_GEEZ_PUNCT_MAP = {
    ".": "።",
    ",": "፣",
    ";": "፤",
    ":": "፥",
    "?": "፧",
}
# "!" deliberately excluded — Amharic borrows "!" as-is, no Ge'ez equivalent.

_DASH_RE = re.compile(r"[-–—]+")
_ELLIPSIS_RE = re.compile(r"\.{3,}|…")
_CURLY_QUOTE_MAP = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}
_WHITESPACE_RE = re.compile(r"[\s\u00A0\u200B\uFEFF]+")


def normalize_punct_consistency(text):
    
    text = _DASH_RE.sub("—", text)
    text = _ELLIPSIS_RE.sub("…", text)
    text = "".join(_CURLY_QUOTE_MAP.get(ch, ch) for ch in text)
    return text

# added based on  mr ashu comment  16/11/18  

def strip_invisible_chars(text):
    return _INVISIBLE_CHAR_RE.sub("", text)

# END OF ADDED CODE
def normalize_amharic_text(text):
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFC", text)
    text = "".join(GEEZ_NORMALIZATION_MAP.get(ch, ch) for ch in text)
    text = "".join(LATIN_TO_GEEZ_PUNCT_MAP.get(ch, ch) for ch in text)
    text = normalize_punct_consistency(text)
    # added based on  mr ashu comment  16/11/18
    text = strip_invisible_chars(text)

    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_english_text(text):
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFC", text)
    text = normalize_punct_consistency(text)
    # added based on  mr ashu comment  16/11/18
    text = strip_invisible_chars(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def apply_normalization(df):
    out = df.copy()
    out["amharic"] = out["amharic"].apply(normalize_amharic_text)
    out["english"] = out["english"].apply(normalize_english_text)
    return out


# ==================== Cleaning ====================

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
_GEEZ_LETTER_RE = re.compile(r"[\u1200-\u137F]")
_VERSE_REF_RE = re.compile(r"\d+[:፥]\d+")

#  ADDED BASED ON MR ASHU COMMENT  17/11/18
_ALLOWED_CHAR_RE = re.compile(
    r"[\u1200-\u137F"              # Ethiopic
    r"A-Za-z"                      # English letters
    r"0-9"                         # digits
    r"\s"                          # whitespace
    r".,;:!?'\"()\[\]{}\-–—…/\\«»"  # punctuation
    r"።፣፤፥፦፧፨፠"                 # Ge'ez punctuation
    r"]"
)
_INVISIBLE_CHAR_RE = re.compile(r"[\u200b\u200c\u200d\u2060\u00ad\ufeff]")
# end of code
def has_target_script(text, target_script_re):
    
    return bool(target_script_re.search(text))

#  ADDED BASED ON MR ASHU COMMENT  17/11/18

def has_disallowed_chars(text):

    remainder = _ALLOWED_CHAR_RE.sub("", text)

    return len(remainder) > 0

# END OF ADDED CODE


def _is_noisy(am, en):
    if _URL_RE.search(am) or _URL_RE.search(en):
        return True
    if _HTML_TAG_RE.search(am) or _HTML_TAG_RE.search(en):
        return True
    # new  addtional based on mr ashu comment  16/11/18
    if has_disallowed_chars(am) or has_disallowed_chars(en):
        return True
    # end of code

    for text in (am, en):
        alpha = sum(1 for c in text if c.isalpha())
        if _VERSE_REF_RE.search(text):
            continue
        if len(text) > 0 and alpha / len(text) < 0.2:
            return True

    if am.strip().lower() == en.strip().lower():
        return True

    return False


    


@dataclass
class CleaningStats:
    starting_rows: int
    steps: list = field(default_factory=list)

    def log_step(self, name, before, after):
        self.steps.append((name, before - after, after))


def run_cleaning(df):
    stats = CleaningStats(starting_rows=len(df))
    cur = df

    # Remove null/empty rows
    before = len(cur)
    cur = cur[
        cur["amharic"].notna()
        & cur["english"].notna()
        & (cur["amharic"].astype(str).str.strip() != "")
        & (cur["english"].astype(str).str.strip() != "")
    ]
    stats.log_step("Remove null/empty rows", before, len(cur))

    # Remove duplicate pairs
    before = len(cur)
    cur = cur.drop_duplicates(subset=["amharic", "english"])
    stats.log_step("Remove duplicate pairs", before, len(cur))

    # Language ID filtering — target-script presence check, no threshold
    before = len(cur)
    am_ok = cur["amharic"].astype(str).apply(lambda t: has_target_script(t, _GEEZ_LETTER_RE))
    en_ok = cur["english"].astype(str).apply(lambda t: has_target_script(t, _LATIN_LETTER_RE))
    cur = cur[am_ok & en_ok]
    stats.log_step("Language ID filtering (target script presence check)", before, len(cur))

    # Noise removal
    before = len(cur)
    mask = ~cur.apply(lambda r: _is_noisy(str(r["amharic"]), str(r["english"])), axis=1)
    cur = cur[mask]
    stats.log_step("Remove noisy rows", before, len(cur))

    return cur, stats


def step2_report(stats):
    lines = ["# Step 2 Report - Clean & Normalize\n"]
    lines.append(f"Starting rows: {stats.starting_rows}\n")
    lines.append("| Step | Removed | Remaining |")
    lines.append("|---|---|---|")
    for name, removed, remaining in stats.steps:
        lines.append(f"|{name}|{removed}|{remaining}|")
    return "\n".join(lines)


if __name__ == "__main__":
    INPUT_FILE = "data/processed/merged_raw.csv"
    OUTPUT_FILE = "data/processed/cleaned_dataset.csv"
    REPORT_FILE = "data/processed/step2_clean_report.md"

    print("Loading merged dataset...")
    df = pd.read_csv(INPUT_FILE)
    print("Original rows:", len(df))

    print("Normalizing text...")
    normalized = apply_normalization(df)

    print("Cleaning dataset...")
    cleaned, stats = run_cleaning(normalized)
    print("Final rows:", len(cleaned))

    cleaned.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    report = step2_report(stats)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print("Saved cleaned dataset!")
    print("Saved cleaning report!")
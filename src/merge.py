# 10/11/18 E.C
#ASELAM ALIKUM THIS IS 576
#Step 1 - Collect & Merge



import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

SOURCE_COL_ALIASES = {"amharic", "am", "amh", "source", "src"}
TARGET_COL_ALIASES = {"english", "en", "eng", "target", "tgt"}


def _match_column(columns: list[str], aliases: set[str]) -> str:
    for col in columns:
        if str(col).strip().lower() in aliases:
            return col
    raise ValueError(f"No column in {columns} matches aliases {aliases}")


def collect_and_merge(raw_dir:  Path) -> pd.DataFrame:
    RAW_DIR = Path("data/raw")
    xlsx_files = sorted(raw_dir.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in {raw_dir}")

    frames = []
    for file_path in xlsx_files:
        xl = pd.ExcelFile(file_path)
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            df.columns = [str(c).strip() for c in df.columns]
            if df.shape[1] < 2:
                continue
            try:
                src_col = _match_column(list(df.columns), SOURCE_COL_ALIASES)
                tgt_col = _match_column(list(df.columns), TARGET_COL_ALIASES)
            except ValueError:
                logger.warning("Skipping %s/%s: unrecognized columns %s",
                                file_path.name, sheet_name, list(df.columns))
                continue
            standardized = pd.DataFrame({
                "amharic": df[src_col],
                "english": df[tgt_col],
                "source_file": file_path.name,
                "sheet_name": sheet_name,
            })
            frames.append(standardized)
            logger.info("Loaded %s/%s: %d rows", file_path.name, sheet_name, len(standardized))

    merged = pd.concat(frames, ignore_index=True)
    merged.insert(0, "row_id", range(len(merged)))
    logger.info("Merged %d tables -> %d total rows", len(frames), len(merged))
    return merged


def profile_dataset(df: pd.DataFrame) -> dict:
    return {
        "n_pairs": len(df),
        "n_sources": df["source_file"].nunique(),
        "sources_breakdown": df["source_file"].value_counts().to_dict(),
        "missing_amharic": int(df["amharic"].isna().sum()),
        "missing_english": int(df["english"].isna().sum()),
        "duplicate_pairs_exact": int(df.duplicated(subset=["amharic", "english"]).sum()),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }


def step1_report(profile: dict) -> str:
    lines = ["# Step 1 Report — Collect & Merge\n"]
    lines.append(f"**Total sentence pairs (raw, pre-clean):** {profile['n_pairs']:,}")
    lines.append(f"**Distinct source files:** {profile['n_sources']}\n")
    lines.append("| Source File | Rows | % of Corpus |")
    lines.append("|---|---|---|")
    for src, n in sorted(profile["sources_breakdown"].items(), key=lambda x: -x[1]):
        lines.append(f"| {src} | {n:,} | {100*n/profile['n_pairs']:.1f}% |")
    lines.append(f"\n**Missing Amharic cells:** {profile['missing_amharic']:,}")
    lines.append(f"**Missing English cells:** {profile['missing_english']:,}")
    lines.append(f"**Exact duplicate pairs (pre-clean):** {profile['duplicate_pairs_exact']:,}")
    lines.append(f"\n**Columns:** `{', '.join(profile['columns'])}`")
    lines.append(
        
    )
    return "\n".join(lines)

if __name__ == "__main__":
    RAW_DIR = Path("data/raw")

    merged_data = collect_and_merge(RAW_DIR)

    print("Merge completed!")
    print("Total rows:", len(merged_data))
    print(merged_data.head())

    profile = profile_dataset(merged_data)
    print(profile)
    merged_data.to_csv(
    "data/processed/merged_raw.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Saved merged dataset!")
report = step1_report(profile)

with open("data/processed/step1_merge_report.md", "w", encoding="utf-8") as f:
    f.write(report)

print("Saved Step 1 report!")
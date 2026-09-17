# 17/11/18
# ASELAM ALIKUM THIS IS 576

# Step 5 - Split & Finalize

import logging
from pathlib import Path

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


logger = logging.getLogger(__name__)

def _normalized_key(text: str) -> str:
    return " ".join(str(text).strip().lower().split())
# ==================== Splitting ====================

def split_dataset(
    df: pd.DataFrame, train_frac: float = 0.80, val_frac: float = 0.10,
    test_frac: float = 0.10, random_state: int = 42,# 
) -> dict[str, pd.DataFrame]:

    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6# asser expect this condtion must true other wise exclude....floting point precision,0.000001

    train_parts, val_parts, test_parts = [], [], []# empty lists

    for domain, group_df in df.groupby("domain"):#take one domain at a time
        group_df = group_df.copy()
        group_df["_group_key"] = group_df["amharic"].apply(_normalized_key)

        n_unique_groups = group_df["_group_key"].nunique()
        if n_unique_groups < 3:
            logger.warning(
                "Domain '%s' has only %d unique sentences — keeping all in train",
                domain, n_unique_groups
            )
            train_parts.append(group_df.drop(columns=["_group_key"]))
            continue

        gss1 = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=random_state)#split 1 make only one split
        train_idx, temp_idx = next(gss1.split(group_df, groups=group_df["_group_key"]))
        train_df, temp_df = group_df.iloc[train_idx], group_df.iloc[temp_idx]


        relative_val = val_frac / (val_frac + test_frac)
        gss2 = GroupShuffleSplit(n_splits=1, train_size=relative_val, random_state=random_state)
        val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["_group_key"]))
        val_df, test_df = temp_df.iloc[val_idx], temp_df.iloc[test_idx]# iloc is used to select rows by index positions

        train_parts.append(train_df.drop(columns=["_group_key"]))
        val_parts.append(val_df.drop(columns=["_group_key"]))
        test_parts.append(test_df.drop(columns=["_group_key"]))

    return {
        "train": pd.concat(train_parts, ignore_index=True),
        "validation": pd.concat(val_parts, ignore_index=True) if val_parts else df.iloc[0:0].copy(),
        "test": pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0].copy(),
    }


def verify_no_leakage(splits: dict[str, pd.DataFrame]) -> dict:
    
    keys = {}

    for name, d in splits.items():# one domain at a time

        pair_keys = (
            d["amharic"].astype(str)#convert to str
            + " || "
            + d["english"].astype(str)
        )

        keys[name] = set(pair_keys.apply(_normalized_key))

    overlaps = {}# empty dict 

    names = list(keys.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            a, b = names[i], names[j]

            overlaps[f"{a}_vs_{b}"] = len(
                keys[a] & keys[b]
            )

    return overlaps


# ==================== Domain distribution ====================

def domain_distribution_table(df: pd.DataFrame, overall_dist: pd.DataFrame | None = None) -> pd.DataFrame:

    if len(df) == 0:
        return pd.DataFrame(
            columns=["domain", "count", "percentage"]
        )

    counts = df["domain"].value_counts()
    percentage = (counts / len(df) * 100).round(2)
    table = pd.DataFrame({
        "domain": counts.index,
        "count": counts.values,
        "percentage": percentage.values,
    }).sort_values("percentage", ascending=False).reset_index(drop=True)

    if overall_dist is not None:
        overall_map = dict(zip(overall_dist["domain"], overall_dist["percentage"]))
        table["overall_percentage"] = (
    table["domain"]
    .map(overall_map)
    .fillna(0)
)
        table["drift"] = (table["percentage"] - table["overall_percentage"]).round(2)

    return table


# visulize emu


def plot_split_ratio(splits, output_path):

    split_sizes = {
        "Train": len(splits["train"]),
        "Validation": len(splits["validation"]),
        "Test": len(splits["test"])
    }

    # Remove empty splits from visualization
    split_sizes = {
        k: v for k, v in split_sizes.items()
        if v > 0
    }

    plt.figure(figsize=(6, 6))

    plt.pie(
        split_sizes.values(),
        labels=split_sizes.keys(),
        autopct="%1.1f%%"
    )

    plt.title("Dataset Split Ratio")

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_domain_distribution(splits, overall_dist, output_path):

    if overall_dist.empty:
        return

    comparison = overall_dist[["domain", "percentage"]].rename(
        columns={"percentage": "Overall"}
    )

    for name, data in splits.items():

        dist = domain_distribution_table(data)

        comparison[name.capitalize()] = (
            dist.set_index("domain")["percentage"]
            .reindex(comparison["domain"])
            .fillna(0)
            .values
        )

    comparison = comparison.set_index("domain")

    comparison.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.ylabel("Percentage (%)")
    plt.title("Domain Distribution Across Splits")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
# ==================== Report ====================

def step5_report(
    splits: dict[str, pd.DataFrame],
    overlaps: dict,
    overall_dist: pd.DataFrame
) -> str:

    lines = ["# Step 5 Report — Split & Finalize\n"]

    total = sum(len(d) for d in splits.values())

    # Split size report
    lines.append("| Split | Rows | Percentage |")
    lines.append("|---|---|---|")

    for name, d in splits.items():
        percentage = (100 * len(d) / total) if total > 0 else 0
        lines.append(
            f"| {name.capitalize()} | {len(d):,} | {percentage:.1f}% |"
        )

    # Leakage report
    lines.append("\n## Leakage verification\n")
    lines.append("| Split Pair | Overlapping sentences |")
    lines.append("|---|---|")

    for pair, n in overlaps.items():
        lines.append(f"| {pair} | {n} |")

    all_clean = all(n == 0 for n in overlaps.values())

    lines.append(
        f"\n**Result: {'Zero leakage confirmed' if all_clean else 'LEAKAGE DETECTED'}**\n"
    )

    # Domain distribution report
    lines.append("\n## Domain distribution per split\n")

    for name, d in splits.items():

        if "domain" not in d.columns:
            continue

        table = domain_distribution_table(d, overall_dist)

        lines.append(
            f"\n**{name.capitalize()}** ({len(d):,} rows)\n"
        )

        lines.append(
            "| Domain | Count | % of split | % of full corpus | Drift |"
        )
        lines.append(
            "|---|---|---|---|---|"
        )

        for _, row in table.iterrows():

            overall_percentage = row.get("overall_percentage", 0)
            if pd.isna(overall_percentage):
                overall_percentage = 0

            drift = row.get("drift", 0)
            if pd.isna(drift):
                drift = 0

            lines.append(
                f"| {row['domain']} | "
                f"{int(row['count']):,} | "
                f"{row['percentage']:.2f}% | "
                f"{overall_percentage:.2f}% | "
                f"{drift:+.2f} |"
            )

    lines.append(
        "\n**Why this matters:** stratified grouped splitting prevents "
        "leakage by construction (identical or near-identical sentence pairs "
        "cannot span multiple splits) while preserving each domain's "
        "distribution. The drift column quantifies how closely each split "
        "matches the original corpus distribution."
    )

    return "\n".join(lines)

if __name__ == "__main__":
    INPUT = Path("data/processed/tokenized_dataset.csv")
    SPLITS_DIR = Path("data/splits")
    REPORT_DIR = Path("outputs/reports")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tokenized dataset...")
    df = pd.read_csv(INPUT)
    print("Total sentence pairs:", len(df))

    print("Computing overall domain distribution...")
    overall_dist = domain_distribution_table(df)
    print(overall_dist)

    print("Splitting dataset (domain-stratified, sentence-grouped)...")
    splits = split_dataset(df)

    for name, d in splits.items():
        print(f"{name}: {len(d):,} rows")

    print("Verifying no leakage...")
    overlaps = verify_no_leakage(splits)
    print(overlaps)

    print("Creating split visualization...")

    plot_split_ratio(
        splits,
        REPORT_DIR / "split_ratio.png"
    )

    plot_domain_distribution(
        splits,
        overall_dist,
        REPORT_DIR / "domain_distribution.png"
    )

    print("Saving splits...")
    for name, split_df in splits.items():
        split_df.to_csv(
            SPLITS_DIR / f"{name}.csv",
            index=False,
            encoding="utf-8-sig"
        )

    print("Building report...")
    report = step5_report(splits, overlaps, overall_dist)

    with open(REPORT_DIR / "step5_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("Saved splits and report!")
    print("Step 5 completed successfully!")
    
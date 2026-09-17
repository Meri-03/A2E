import json
import pandas as pd


def compute_max_seq_len(csv_paths: list[str]) -> int:
    
    max_len = 0
    for path in csv_paths:
        df = pd.read_csv(path)
        for col in ("amharic_ids", "english_ids"):
            lengths = df[col].apply(lambda s: len(json.loads(s)))
            max_len = max(max_len, lengths.max())
    return max_len


if __name__ == "__main__":
    real_max_len = compute_max_seq_len([
        "data/splits/train.csv",
        "data/splits/validation.csv",
        "data/splits/test.csv",
    ])
    print("True max sequence length across all splits:", real_max_len)

    SAFE_MAX_LEN = real_max_len + 20

    
    print("Using max_len =", SAFE_MAX_LEN, "for positional encoding")
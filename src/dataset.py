"""
dataset.py — Bridges the split CSV files (train/validation/test) into
batches of padded PyTorch tensors the Transformer can actually train on.
"""

import json
import logging

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader # import Dataset and DataLoader from torch.utils.data

logger = logging.getLogger(__name__)

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class TranslationDataset(Dataset):
   

    def __init__(self, csv_path: str):# initialize the dataset with the path to the CSV file
        logger.info("Loading dataset from %s", csv_path)
        self.df = pd.read_csv(csv_path)

        required_cols = {"amharic_ids", "english_ids"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {missing}. "
                f"Available columns: {list(self.df.columns)}"
            )

        logger.info("Loaded %d sentence pairs", len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_ids = json.loads(row["amharic_ids"])#import json to load the amharic_ids and english_ids from the CSV file
        tgt_ids = json.loads(row["english_ids"])
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_batch(batch):
    
    src_batch, tgt_batch = zip(*batch)# unzip the batch of ,later we will pad the src_batch and tgt_batch to the same length

    max_src_len = max(len(s) for s in src_batch)
    max_tgt_len = max(len(t) for t in tgt_batch)

    batch_size = len(batch)
    src_padded = torch.full((batch_size, max_src_len), PAD_ID, dtype=torch.long)
    tgt_padded = torch.full((batch_size, max_tgt_len), PAD_ID, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):#enumerate gives index plus value, we will use the index to fill the padded tensors
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt

    src_mask = (src_padded != PAD_ID)
    tgt_mask = (tgt_padded != PAD_ID)

    return {
        "src_ids": src_padded,#dont look the pad
        "tgt_ids": tgt_padded,#lok a heaad mask ,on the decoder dont look at the futuere words
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
    }


def get_dataloader(csv_path: str, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    dataset = TranslationDataset(csv_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
    )


def dataset_report(splits: dict, sample_batch: dict, batch_size: int, padding_stats: dict) -> str:
    """Markdown report matching the Step 1-5 reporting pattern."""
    lines = ["# Dataset & DataLoader Report\n"]
    lines.append(f"**Batch size:** {batch_size}\n")

    lines.append("## Split sizes\n")
    lines.append("| Split | Sentence Pairs |")
    lines.append("|---|---|")
    for name, ds in splits.items():
        lines.append(f"| {name.capitalize()} | {len(ds):,} |")

    lines.append("\n## Special tokens used for batching\n")
    lines.append("| Token | ID |")
    lines.append("|---|---|")
    lines.append(f"| `<pad>` | {PAD_ID} |")
    lines.append(f"| `<bos>` | {BOS_ID} |")
    lines.append(f"| `<eos>` | {EOS_ID} |")
    lines.append(f"| `<unk>` | {UNK_ID} |")

    lines.append("\n## Example batch shapes (single random batch, illustrative only)\n")
    lines.append(f"- `src_ids`: {tuple(sample_batch['src_ids'].shape)}")
    lines.append(f"- `tgt_ids`: {tuple(sample_batch['tgt_ids'].shape)}")

    lines.append("\n## Padding Efficiency (averaged over "
                  f"{padding_stats['Batches sampled']} batches — stable, not single-batch noise)\n")
    lines.append(f"- Padding Ratio (%): {padding_stats['Padding Ratio (%)']}")
    lines.append(f"- Efficiency (%): {padding_stats['Efficiency (%)']}")

    lines.append(
        "\n**Why this matters:** confirms the Dataset/DataLoader correctly "
        "parses the tokenized CSVs and produces properly padded, "
        "equal-shaped tensors — the exact input format the Transformer's "
        "embedding layer expects."
    )
    return "\n".join(lines)


def compute_padding_stats(dataloader, num_batches: int = 200):
    """Average padding ratio across many batches — a single batch's ratio
    is essentially random noise, so this is the number actually worth
    reporting."""
    total_tokens = 0
    total_pad_tokens = 0
    batches_seen = 0

    for batch in dataloader:
        src_mask = batch["src_mask"]
        tgt_mask = batch["tgt_mask"]

        total_tokens += src_mask.numel() + tgt_mask.numel()
        total_pad_tokens += (~src_mask).sum().item() + (~tgt_mask).sum().item()

        batches_seen += 1
        if batches_seen >= num_batches:
            break

    padding_ratio = 100 * total_pad_tokens / total_tokens
    efficiency = 100 - padding_ratio
    return {
        "Padding Ratio (%)": round(padding_ratio, 2),
        "Efficiency (%)": round(efficiency, 2),
        "Batches sampled": batches_seen,
    }
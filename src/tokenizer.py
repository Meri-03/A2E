# 15/11/18 04;48
# ASELAM ALIKUM THIS IS 576 

# Step 4 - Tokenize & Vocabulary


import json
import logging
from pathlib import Path

import pandas as pd
import sentencepiece as spm

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}


def train_tokenizer(
    df: pd.DataFrame, work_dir: str | Path, model_prefix: str | Path, vocab_size: int = 16000# see the ref,dir is tempo
) -> spm.SentencePieceProcessor:
    
    work_dir = Path(work_dir)# tempo to fixed
    corpus_path = work_dir / "_spm_training_corpus.txt"# / meansjoin 
    with corpus_path.open("w", encoding="utf-8") as f:
        for text in df["amharic"]:
            f.write(str(text).strip() + "\n")
        for text in df["english"]:
            f.write(str(text).strip() + "\n")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),# name for out put model file
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        pad_piece="<pad>", bos_piece="<bos>", eos_piece="<eos>", unk_piece="<unk>",
        # input_sentence_size
        shuffle_input_sentence=True,# shaky
    )
    corpus_path.unlink()  # not needed after training

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_prefix) + ".model")
    logger.info("Trained SentencePiece tokenizer: vocab_size=%d", sp.get_piece_size())
    return sp


def export_vocabulary(sp: spm.SentencePieceProcessor, out_path: str | Path) -> None:# json
    out_path = Path(out_path)
    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}# creat vocab dict
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2) # allow amharic char saved normally


def numericalize(sp: spm.SentencePieceProcessor, text: str) -> list[int]:# convert text to token ids
    ids = sp.encode(str(text), out_type=int)
    return [SPECIAL_TOKENS["<bos>"]] + ids + [SPECIAL_TOKENS["<eos>"]]


def numericalize_dataframe(df: pd.DataFrame, sp: spm.SentencePieceProcessor) -> pd.DataFrame:
    out = df.copy()
    out["amharic_ids"] = out["amharic"].apply(lambda t: json.dumps(numericalize(sp, t)))
    out["english_ids"] = out["english"].apply(lambda t: json.dumps(numericalize(sp, t)))
    out["amharic_len"] = out["amharic_ids"].apply(lambda s: len(json.loads(s)))
    out["english_len"] = out["english_ids"].apply(lambda s: len(json.loads(s)))
    return out


def step4_report(sp: spm.SentencePieceProcessor, numericalized_df: pd.DataFrame,
                  sample_df: pd.DataFrame, vocab_size: int) -> str:
    lines = ["# Step 4 Report — Tokenize & Vocabulary\n"]
    lines.append(f"**Algorithm:** SentencePiece Unigram, shared vocab across both languages")
    lines.append(f"**Vocabulary size:** {vocab_size:,}\n")

    lines.append("## Special tokens\n")
    lines.append("| Token | ID | Purpose |")
    lines.append("|---|---|---|")
    lines.append("| `<pad>` | 0 | Pads sequences to batch max length |")
    lines.append("| `<bos>` | 1 | Marks start of sequence |")
    lines.append("| `<eos>` | 2 | Marks end — model learns to stop generating |")
    lines.append("| `<unk>` | 3 | Fallback for out-of-vocab sequences |")

    lines.append("\n## Sample tokenization\n")
    lines.append("| Original | Tokens |")
    lines.append("|---|---|")
    for _, row in sample_df.head(3).iterrows():
        for text in (row["amharic"], row["english"]):
            pieces = sp.encode(str(text), out_type=str)
            preview = " ".join(pieces[:12]) + (" ..." if len(pieces) > 12 else "")
            short = (str(text)[:45] + "...") if len(str(text)) > 45 else str(text)
            lines.append(f"| {short} | {preview} |")

    lines.append("\n## Numericalized sequence length (with BOS/EOS)\n")
    lines.append("| Stat | Amharic IDs | English IDs |")
    lines.append("|---|---|---|")
    am_desc = numericalized_df["amharic_len"].describe()
    en_desc = numericalized_df["english_len"].describe()
    for key, label in [("mean", "Mean"), ("max", "Max")]:
        lines.append(f"| {label} | {am_desc[key]:.1f} | {en_desc[key]:.1f} |")

    lines.append(
        "\n**Why this matters:** the tokenizer determines the model's "
        "effective vocabulary and its ability to generalize to unseen "
        "morphological forms. Numericalized IDs (stored as JSON strings) "
        "are the final bridge to PyTorch tensors at training time."
    )
    return "\n".join(lines)



if __name__ == "__main__":

    # ---------------- Paths ----------------

    INPUT = Path(
        "data/processed/analyzed_dataset.csv"
    )

    PROCESSED = Path(
        "data/processed"
    )

    TOKENIZER_DIR = Path(
        "models/tokenizer"
    )

    REPORT_DIR = Path(
        "outputs/reports"
    )


    # Create folders

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True
    )

    TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print("Loading analyzed dataset...")


    df = pd.read_csv(INPUT)


    print("Dataset loaded!")
    print("Total sentence pairs:", len(df))


    # ---------------- Train Tokenizer ----------------

    print("Training SentencePiece tokenizer...")


    tokenizer_path = TOKENIZER_DIR / "tokenizer"


    sp = train_tokenizer(
        df,
        TOKENIZER_DIR,
        tokenizer_path,
        vocab_size=16000
    )


    print(
        "Tokenizer vocabulary size:",
        sp.get_piece_size()
    )


    # ---------------- Export Vocabulary ----------------

    print("Saving vocabulary...")


    export_vocabulary(
        sp,
        TOKENIZER_DIR / "vocabulary.json"
    )


    # ---------------- Numericalization ----------------

    print("Converting text into token IDs...")


    tokenized_df = numericalize_dataframe(
        df,
        sp
    )


    # Save tokenized dataset

    tokenized_df.to_csv(
        PROCESSED / "tokenized_dataset.csv",
        index=False,
        encoding="utf-8-sig"
    )


    print(
        "Saved tokenized dataset!"
    )


    # ---------------- Report ----------------

    report = step4_report(
        sp,
        tokenized_df,
        df,
        16000
    )


    with open(
        REPORT_DIR / "step4_report.md",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)


    print(
        "Saved Step 4 report!"
    )


    print(
        "Step 4 completed successfully!"
    )
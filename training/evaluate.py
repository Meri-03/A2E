# 22/12/18 —


from pathlib import Path
import json
import os
import sys
import tempfile

import torch
import sacrebleu
from bert_score import score as bertscore_score
import sentencepiece as spm


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from transformer import Transformer
from dataset import get_dataloader
from compute_max_len import compute_max_seq_len


# ---------------------------------------------------------------------
# Special token IDs
# ---------------------------------------------------------------------

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


# ---------------------------------------------------------------------
# Evaluation paths
# ---------------------------------------------------------------------

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

PROGRESS_PATH = REPORT_DIR / "eval_progress.json"
METRICS_PATH = REPORT_DIR / "eval_metrics_progress.json"
HYPOTHESES_PATH = REPORT_DIR / "eval_hypotheses.json"
REFERENCES_PATH = REPORT_DIR / "eval_references.json"
FINAL_REPORT_PATH = REPORT_DIR / "final_test_evaluation.md"
LOG_PATH = REPORT_DIR / "evaluation.log"


# ---------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------

VOCAB_SIZE = 16000

D_MODEL = 512
NUM_HEADS = 8
D_FF = 2048
NUM_LAYERS = 6

BATCH_SIZE = 32

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "training"
    / "checkpoints"
    / "best.pt"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "test.csv"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "models"
    / "tokenizer"
    / "tokenizer.model"
)


# ---------------------------------------------------------------------
# Utility: atomic JSON save
# ---------------------------------------------------------------------

def atomic_json_save(data, path):
    """
    Safely save JSON.

    The file is first written to a temporary file.
    Only after the write succeeds is it moved into place.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

            f.flush()
            os.fsync(f.fileno())

        os.replace(
            temp_path,
            path,
        )

    except Exception:

        try:
            os.remove(temp_path)
        except OSError:
            pass

        raise


# ---------------------------------------------------------------------
# Greedy decoding
# ---------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(
    model,
    src_ids,
    max_len,
    device,
):
    """
    Greedy decoding for the entire batch.

    The decoder generates one token at a time.

    IMPORTANT:
    The batch stops only when every sentence has generated EOS.

    Therefore individual sentences may contain tokens after their
    own EOS. Those extra tokens are removed later during decoding.
    """

    model.eval()

    batch_size = src_ids.size(0)

    # -------------------------------------------------------------
    # Source mask
    # -------------------------------------------------------------

    src_mask = model.make_src_mask(
        src_ids
    )

    # -------------------------------------------------------------
    # Encoder
    # -------------------------------------------------------------

    encoder_output, _ = model.encoder(
        src_ids,
        src_mask,
    )

    # -------------------------------------------------------------
    # Start decoder with BOS
    # -------------------------------------------------------------

    decoder_input = torch.full(
        (batch_size, 1),
        BOS_ID,
        dtype=torch.long,
        device=device,
    )

    # -------------------------------------------------------------
    # Generate tokens
    # -------------------------------------------------------------

    for _ in range(max_len - 1):

        tgt_mask = model.make_tgt_mask(
            decoder_input
        )

        logits, _, _ = model.decoder(
            decoder_input,
            encoder_output,
            src_mask,
            tgt_mask,
        )

        # Greedy choice:
        # select the token with the highest probability.

        next_token = logits[
            :, -1, :
        ].argmax(
            dim=-1,
            keepdim=True,
        )

        decoder_input = torch.cat(
            [
                decoder_input,
                next_token,
            ],
            dim=1,
        )

        # Stop when every sentence has generated EOS.

        if (
            next_token == EOS_ID
        ).all():

            break

    return decoder_input


# ---------------------------------------------------------------------
# Load previous progress
# ---------------------------------------------------------------------

def load_progress():
    """
    Load previous evaluation progress.

    Returns:
        progress dictionary or None
        hypotheses list
        references list
        metrics list
    """

    hypotheses = []
    references = []
    metrics = []

    # -------------------------------------------------------------
    # No progress file
    # -------------------------------------------------------------

    if not PROGRESS_PATH.exists():

        print(
            "No previous evaluation progress found."
        )

        print(
            "Starting from the beginning."
        )

        return (
            None,
            hypotheses,
            references,
            metrics,
        )

    # -------------------------------------------------------------
    # Load progress
    # -------------------------------------------------------------

    try:

        with open(
            PROGRESS_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            progress = json.load(f)

    except Exception as e:

        print(
            f"WARNING: Could not read "
            f"{PROGRESS_PATH}: {e}"
        )

        print(
            "Starting from the beginning."
        )

        return (
            None,
            hypotheses,
            references,
            metrics,
        )

    # -------------------------------------------------------------
    # Load hypotheses
    # -------------------------------------------------------------

    if HYPOTHESES_PATH.exists():

        try:

            with open(
                HYPOTHESES_PATH,
                "r",
                encoding="utf-8",
            ) as f:

                hypotheses = json.load(f)

        except Exception as e:

            print(
                f"WARNING: Could not load hypotheses: {e}"
            )

            hypotheses = []

    # -------------------------------------------------------------
    # Load references
    # -------------------------------------------------------------

    if REFERENCES_PATH.exists():

        try:

            with open(
                REFERENCES_PATH,
                "r",
                encoding="utf-8",
            ) as f:

                references = json.load(f)

        except Exception as e:

            print(
                f"WARNING: Could not load references: {e}"
            )

            references = []

    # -------------------------------------------------------------
    # Load intermediate metrics
    # -------------------------------------------------------------

    if METRICS_PATH.exists():

        try:

            with open(
                METRICS_PATH,
                "r",
                encoding="utf-8",
            ) as f:

                metrics = json.load(f)

        except Exception as e:

            print(
                f"WARNING: Could not load metrics: {e}"
            )

            metrics = []

    # -------------------------------------------------------------
    # Read progress values
    # -------------------------------------------------------------

    completed_batches = int(
        progress.get(
            "completed_batches",
            0,
        )
    )

    sentences_completed = int(
        progress.get(
            "sentences_completed",
            0,
        )
    )

    total_sentences = int(
        progress.get(
            "total_sentences",
            0,
        )
    )

    # -------------------------------------------------------------
    # Display progress
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("PREVIOUS EVALUATION PROGRESS FOUND")
    print("=" * 70)

    print(
        f"Completed batches:    "
        f"{completed_batches}"
    )

    print(
        f"Sentences completed:  "
        f"{sentences_completed} / "
        f"{total_sentences}"
    )

    print(
        f"Loaded hypotheses:    "
        f"{len(hypotheses)}"
    )

    print(
        f"Loaded references:    "
        f"{len(references)}"
    )

    print(
        f"Saved metric records: "
        f"{len(metrics)}"
    )

    print("=" * 70)
    print()

    return (
        progress,
        hypotheses,
        references,
        metrics,
    )


# ---------------------------------------------------------------------
# Save evaluation progress
# ---------------------------------------------------------------------

def save_progress(
    batch_idx,
    completed_batches,
    hypotheses,
    references,
    total_sentences,
):
    """
    Save complete evaluation state.

    Called after every processed batch.
    """

    progress = {
        "batch_idx": batch_idx,
        "completed_batches": completed_batches,
        "sentences_completed": len(hypotheses),
        "total_sentences": total_sentences,
        "completed": False,
    }

    atomic_json_save(
        progress,
        PROGRESS_PATH,
    )

    atomic_json_save(
        hypotheses,
        HYPOTHESES_PATH,
    )

    atomic_json_save(
        references,
        REFERENCES_PATH,
    )


# ---------------------------------------------------------------------
# Compute intermediate BLEU / chrF++
# ---------------------------------------------------------------------

def compute_intermediate_metrics(
    hypotheses,
    references,
):
    """
    Compute BLEU and chrF++ on all translations evaluated so far.

    These are only progress metrics.

    The final metrics are calculated again on the complete test set.
    """

    if (
        not hypotheses
        or not references
    ):

        return {
            "bleu": 0.0,
            "chrf_plus_plus": 0.0,
        }

    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
    )

    chrf_pp = sacrebleu.corpus_chrf(
        hypotheses,
        [references],
        word_order=2,
    )

    return {
        "bleu": float(
            bleu.score
        ),
        "chrf_plus_plus": float(
            chrf_pp.score
        ),
    }


# ---------------------------------------------------------------------
# Save intermediate metrics
# ---------------------------------------------------------------------

def save_metrics(
    batch_idx,
    hypotheses,
    references,
    metrics,
):
    """
    Calculate and save intermediate BLEU and chrF++.
    """

    current = compute_intermediate_metrics(
        hypotheses,
        references,
    )

    record = {
        "batch_idx": batch_idx,
        "sentences": len(hypotheses),
        "bleu": round(
            current["bleu"],
            4,
        ),
        "chrf_plus_plus": round(
            current["chrf_plus_plus"],
            4,
        ),
    }

    # -------------------------------------------------------------
    # Remove an old record for this batch.
    # This prevents duplicates after resume.
    # -------------------------------------------------------------

    metrics = [
        m
        for m in metrics
        if m.get("batch_idx") != batch_idx
    ]

    metrics.append(record)

    atomic_json_save(
        metrics,
        METRICS_PATH,
    )

    return (
        metrics,
        current,
    )


# ---------------------------------------------------------------------
# Decode one generated sentence correctly
# ---------------------------------------------------------------------

def decode_hypothesis(
    token_ids,
    sp,
):
    """
    Decode one generated sequence.

    IMPORTANT FIX:
    Truncate at the first EOS token before removing special tokens.

    Example:

        BOS hello world EOS extra extra

    becomes:

        hello world
    """

    token_ids = list(token_ids)

    # -------------------------------------------------------------
    # Find first EOS
    # -------------------------------------------------------------

    if EOS_ID in token_ids:

        eos_pos = token_ids.index(
            EOS_ID
        )

        token_ids = token_ids[
            :eos_pos
        ]

    # -------------------------------------------------------------
    # Remove BOS and PAD
    # -------------------------------------------------------------

    token_ids = [
        token
        for token in token_ids
        if token not in (
            PAD_ID,
            BOS_ID,
        )
    ]

    # -------------------------------------------------------------
    # SentencePiece decoding
    # -------------------------------------------------------------

    return sp.decode(
        token_ids
    )


# ---------------------------------------------------------------------
# Decode reference sentence
# ---------------------------------------------------------------------

def decode_reference(
    token_ids,
    sp,
):
    """
    Decode a reference sequence.

    Removes PAD, BOS, and EOS.
    """

    token_ids = [
        token
        for token in token_ids
        if token not in (
            PAD_ID,
            BOS_ID,
            EOS_ID,
        )
    ]

    return sp.decode(
        token_ids
    )


# ---------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------

def evaluate_on_test_set(
    checkpoint_path=str(
        CHECKPOINT_PATH
    ),
):

    # -------------------------------------------------------------
    # Create report directory
    # -------------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # -------------------------------------------------------------
    # Compute maximum sequence length
    # -------------------------------------------------------------

    real_max_len = compute_max_seq_len(
        [
            str(
                PROJECT_ROOT
                / "data"
                / "splits"
                / "train.csv"
            ),
            str(
                PROJECT_ROOT
                / "data"
                / "splits"
                / "validation.csv"
            ),
            str(
                PROJECT_ROOT
                / "data"
                / "splits"
                / "test.csv"
            ),
        ]
    )

    max_len = int(
        real_max_len * 1.05
    )

    print(
        f"Real maximum sequence length: "
        f"{real_max_len}"
    )

    print(
        f"Evaluation maximum sequence length: "
        f"{max_len}"
    )

    # -------------------------------------------------------------
    # Load SentencePiece tokenizer
    # -------------------------------------------------------------

    sp = spm.SentencePieceProcessor()

    sp.load(
        str(TOKENIZER_PATH)
    )

    print(
        f"Loaded tokenizer: "
        f"{TOKENIZER_PATH}"
    )

    # -------------------------------------------------------------
    # Build Transformer
    # -------------------------------------------------------------

    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_len=max_len,
        pad_id=PAD_ID,
    ).to(device)

    # -------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------

    print(
        f"Loading checkpoint: "
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        f"Loaded — trained to epoch "
        f"{checkpoint['epoch']}, "
        f"step {checkpoint['step']}"
    )

    # -------------------------------------------------------------
    # Test dataloader
    # -------------------------------------------------------------

    test_loader = get_dataloader(
        str(TEST_PATH),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    total_sentences = len(
        test_loader.dataset
    )

    total_batches = len(
        test_loader
    )

    print(
        f"Test set: "
        f"{total_sentences} sentence pairs"
    )

    print(
        f"Total batches: "
        f"{total_batches}"
    )

    # -------------------------------------------------------------
    # Load previous progress
    # -------------------------------------------------------------

    (
        progress,
        hypotheses,
        references,
        metrics,
    ) = load_progress()

    # -------------------------------------------------------------
    # Determine starting batch
    # -------------------------------------------------------------

    if progress is None:

        completed_batches = 0

    else:

        completed_batches = int(
            progress.get(
                "completed_batches",
                0,
            )
        )

        # ---------------------------------------------------------
        # Safety check:
        # hypotheses and references must match.
        # ---------------------------------------------------------

        if (
            len(hypotheses)
            != len(references)
        ):

            print(
                "WARNING: hypotheses and references "
                "have different lengths."
            )

            print(
                f"Hypotheses: "
                f"{len(hypotheses)}"
            )

            print(
                f"References: "
                f"{len(references)}"
            )

            raise RuntimeError(
                "Saved evaluation files are inconsistent. "
                "Do not continue automatically."
            )

        # ---------------------------------------------------------
        # Safety check:
        # progress cannot claim batches
        # when there is no saved translation data.
        # ---------------------------------------------------------

        if (
            completed_batches > 0
            and len(hypotheses) == 0
        ):

            raise RuntimeError(
                "Progress file exists but hypotheses "
                "are empty. Evaluation state is inconsistent."
            )

        # ---------------------------------------------------------
        # Safety check:
        # saved sentences should not exceed
        # the test-set size.
        # ---------------------------------------------------------

        if (
            len(hypotheses)
            > total_sentences
        ):

            raise RuntimeError(
                "Saved hypotheses exceed the test-set size."
            )

        print(
            f"Resuming from batch "
            f"{completed_batches + 1}"
        )

    # -------------------------------------------------------------
    # Check whether evaluation is already complete
    # -------------------------------------------------------------

    if (
        progress is not None
        and progress.get(
            "completed",
            False,
        )
    ):

        print()
        print(
            "Evaluation is already marked as completed."
        )

        print(
            f"Saved sentences: "
            f"{len(hypotheses)}"
        )

        print(
            f"Expected sentences: "
            f"{total_sentences}"
        )

        return

    # -------------------------------------------------------------
    # Evaluation loop
    # -------------------------------------------------------------

    print()
    print(
        "Starting test-set evaluation..."
    )
    print()

    for i, batch in enumerate(
        test_loader
    ):

        # ---------------------------------------------------------
        # Skip batches already processed.
        # ---------------------------------------------------------

        if i < completed_batches:
            continue

        # ---------------------------------------------------------
        # Move source and target tensors to GPU/CPU.
        # ---------------------------------------------------------

        src_ids = batch[
            "src_ids"
        ].to(device)

        tgt_ids = batch[
            "tgt_ids"
        ].to(device)

        # ---------------------------------------------------------
        # Generate translations.
        # ---------------------------------------------------------

        generated = greedy_decode(
            model=model,
            src_ids=src_ids,
            max_len=max_len,
            device=device,
        )

        # ---------------------------------------------------------
        # Decode every sentence.
        # ---------------------------------------------------------

        for j in range(
            src_ids.size(0)
        ):

            # -----------------------------------------------------
            # FIXED HYPOTHESIS DECODING
            # -----------------------------------------------------

            hypothesis = decode_hypothesis(
                generated[j].tolist(),
                sp,
            )

            # -----------------------------------------------------
            # REFERENCE DECODING
            # -----------------------------------------------------

            reference = decode_reference(
                tgt_ids[j].tolist(),
                sp,
            )

            hypotheses.append(
                hypothesis
            )

            references.append(
                reference
            )

        # ---------------------------------------------------------
        # Current batch number
        # ---------------------------------------------------------

        batch_idx = i + 1

        completed_batches = batch_idx

        # ---------------------------------------------------------
        # SAVE AFTER EVERY BATCH
        # ---------------------------------------------------------

        save_progress(
            batch_idx=batch_idx,
            completed_batches=completed_batches,
            hypotheses=hypotheses,
            references=references,
            total_sentences=total_sentences,
        )

        # ---------------------------------------------------------
        # Calculate and save intermediate metrics.
        # ---------------------------------------------------------

        (
            metrics,
            current_metrics,
        ) = save_metrics(
            batch_idx=batch_idx,
            hypotheses=hypotheses,
            references=references,
            metrics=metrics,
        )

        # ---------------------------------------------------------
        # Progress output
        # ---------------------------------------------------------

        print(
            f"Batch "
            f"{batch_idx:4d}/"
            f"{total_batches} | "
            f"Sentences "
            f"{len(hypotheses):6d}/"
            f"{total_sentences} | "
            f"BLEU "
            f"{current_metrics['bleu']:.4f} | "
            f"chrF++ "
            f"{current_metrics['chrf_plus_plus']:.4f}",
            flush=True,
        )

        # ---------------------------------------------------------
        # Free temporary GPU tensors.
        # ---------------------------------------------------------

        del src_ids
        del tgt_ids
        del generated

        if device.type == "cuda":

            torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Safety check
    # -------------------------------------------------------------

    if (
        len(hypotheses)
        != total_sentences
    ):

        print()

        print(
            "WARNING: Evaluation loop ended, "
            "but not all sentences were processed."
        )

        print(
            f"Processed: "
            f"{len(hypotheses)}"
        )

        print(
            f"Expected:  "
            f"{total_sentences}"
        )

        raise RuntimeError(
            "Evaluation incomplete. "
            "Final metrics were NOT written."
        )

    # -------------------------------------------------------------
    # Final metrics
    # -------------------------------------------------------------

    print()

    print(
        "Computing final metrics on full test set..."
    )

    print()

    # -------------------------------------------------------------
    # BLEU
    # -------------------------------------------------------------

    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
    )

    # -------------------------------------------------------------
    # chrF++
    # -------------------------------------------------------------

    chrf_pp = sacrebleu.corpus_chrf(
        hypotheses,
        [references],
        word_order=2,
    )

    # -------------------------------------------------------------
    # BERTScore
    # -------------------------------------------------------------

    print(
        "Computing BERTScore..."
    )

    P, R, F1 = bertscore_score(
        hypotheses,
        references,
        lang="en",
        verbose=False,
        device=device,
    )

    # -------------------------------------------------------------
    # Convert metrics to Python floats.
    # -------------------------------------------------------------

    bleu_score = float(
        bleu.score
    )

    chrf_score = float(
        chrf_pp.score
    )

    bert_p = float(
        P.mean().item()
    )

    bert_r = float(
        R.mean().item()
    )

    bert_f1 = float(
        F1.mean().item()
    )

    # -------------------------------------------------------------
    # Final console results
    # -------------------------------------------------------------

    print()

    print(
        "=" * 60
    )

    print(
        "FINAL TEST SET RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Test set size: "
        f"{len(hypotheses)} sentence pairs"
    )

    print(
        f"BLEU:              "
        f"{bleu_score:.2f}"
    )

    print(
        f"chrF++:            "
        f"{chrf_score:.2f}"
    )

    print(
        f"BERTScore P:       "
        f"{bert_p:.4f}"
    )

    print(
        f"BERTScore R:       "
        f"{bert_r:.4f}"
    )

    print(
        f"BERTScore F1:      "
        f"{bert_f1:.4f}"
    )

    print(
        "=" * 60
    )

    # -------------------------------------------------------------
    # Sample translations
    # -------------------------------------------------------------

    print()

    print(
        "Sample translations (first 5):"
    )

    for i in range(
        min(
            5,
            len(hypotheses),
        )
    ):

        print()

        print(
            f"  Reference:  "
            f"{references[i]}"
        )

        print(
            f"  Model:      "
            f"{hypotheses[i]}"
        )

    # -------------------------------------------------------------
    # Build final Markdown report.
    # -------------------------------------------------------------

    report_text = []

    report_text.append(
        "# Final Test Set Evaluation\n\n"
    )

    report_text.append(
        f"**Checkpoint:** "
        f"{checkpoint_path}\n\n"
    )

    report_text.append(
        f"**Trained to:** epoch "
        f"{checkpoint['epoch']}, "
        f"step {checkpoint['step']}\n\n"
    )

    report_text.append(
        f"**Test set size:** "
        f"{len(hypotheses)} sentence pairs\n\n"
    )

    report_text.append(
        "## Metrics\n\n"
    )

    report_text.append(
        "| Metric | Score |\n"
    )

    report_text.append(
        "|---|---|\n"
    )

    report_text.append(
        f"| BLEU | {bleu_score:.2f} |\n"
    )

    report_text.append(
        f"| chrF++ | {chrf_score:.2f} |\n"
    )

    report_text.append(
        f"| BERTScore Precision | {bert_p:.4f} |\n"
    )

    report_text.append(
        f"| BERTScore Recall | {bert_r:.4f} |\n"
    )

    report_text.append(
        f"| BERTScore F1 | {bert_f1:.4f} |\n"
    )

    report_text.append(
        "\n"
    )

    report_text.append(
        "## Evaluation Methodology\n\n"
    )

    report_text.append(
        "- Checkpoint: `best.pt`\n"
    )

    report_text.append(
        "- Dataset: TEST set only\n"
    )

    report_text.append(
        "- Decoding: Greedy decoding\n"
    )

    report_text.append(
        "- Tokenization: SentencePiece\n"
    )

    report_text.append(
        "- BLEU: SacreBLEU `corpus_bleu`\n"
    )

    report_text.append(
        "- chrF++: SacreBLEU "
        "`corpus_chrf(word_order=2)`\n"
    )

    report_text.append(
        '- BERTScore: `lang="en"`\n'
    )

    report_text.append(
        "- Hypotheses truncated at individual first EOS\n"
    )

    report_text.append(
        "\n"
    )

    report_text.append(
        "## Sample translations\n\n"
    )

    for i in range(
        min(
            10,
            len(hypotheses),
        )
    ):

        report_text.append(
            f"### Example {i + 1}\n\n"
        )

        report_text.append(
            f"**Reference:** "
            f"{references[i]}\n\n"
        )

        report_text.append(
            f"**Model:** "
            f"{hypotheses[i]}\n\n"
        )

        report_text.append(
            "---\n\n"
        )

    report_text = "".join(
        report_text
    )

    # -------------------------------------------------------------
    # Atomic final report write.
    # -------------------------------------------------------------

    fd, temp_path = tempfile.mkstemp(
        dir=str(
            FINAL_REPORT_PATH.parent
        ),
        prefix=".final_test_evaluation.",
        suffix=".tmp",
    )

    try:

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                report_text
            )

            f.flush()

            os.fsync(
                f.fileno()
            )

        os.replace(
            temp_path,
            FINAL_REPORT_PATH,
        )

    except Exception:

        try:
            os.remove(
                temp_path
            )
        except OSError:
            pass

        raise

    # -------------------------------------------------------------
    # Mark evaluation as complete.
    # -------------------------------------------------------------

    completed_progress = {
        "batch_idx": total_batches,
        "completed_batches": total_batches,
        "sentences_completed": len(
            hypotheses
        ),
        "total_sentences": total_sentences,
        "completed": True,
    }

    atomic_json_save(
        completed_progress,
        PROGRESS_PATH,
    )

    # -------------------------------------------------------------
    # Save final metric record.
    # -------------------------------------------------------------

    final_metric_record = {
        "batch_idx": total_batches,
        "sentences": len(
            hypotheses
        ),
        "bleu": round(
            bleu_score,
            4,
        ),
        "chrf_plus_plus": round(
            chrf_score,
            4,
        ),
        "bertscore_precision": round(
            bert_p,
            4,
        ),
        "bertscore_recall": round(
            bert_r,
            4,
        ),
        "bertscore_f1": round(
            bert_f1,
            4,
        ),
        "final": True,
    }

    # Remove any previous final record for this batch.

    metrics = [
        m
        for m in metrics
        if m.get("batch_idx")
        != total_batches
    ]

    metrics.append(
        final_metric_record
    )

    atomic_json_save(
        metrics,
        METRICS_PATH,
    )

    # -------------------------------------------------------------
    # Final messages.
    # -------------------------------------------------------------

    print()

    print(
        f"Saved report to "
        f"{FINAL_REPORT_PATH}"
    )

    print()

    print(
        "Evaluation completed successfully."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    evaluate_on_test_set(
        str(
            CHECKPOINT_PATH
        )
    )
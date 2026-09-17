#04/01/19


import logging
from pathlib import Path

import torch
import torch.nn as nn
import sacrebleu

from bert_score import score as bertscore_score

from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.insert(0, "models")
sys.path.insert(0, "src")

from transformer import Transformer
from loss_schedule import (
    build_loss_fn,
    build_optimizer,
    TransformerLRSchedule,
)
from dataset import get_dataloader
from compute_max_len import compute_max_seq_len

import sentencepiece as spm


logger = logging.getLogger(__name__)



# Special token IDs
# ============================================================

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


# ============================================================
# Teacher forcing
# ============================================================

def shift_for_teacher_forcing(tgt_ids: torch.Tensor):
    
    decoder_input = tgt_ids[:, :-1]
    decoder_target = tgt_ids[:, 1:]

    return decoder_input, decoder_target


# ============================================================
# Early stopping
# ============================================================

class EarlyStopping:
   

    def __init__(self, patience: int = 3, min_delta: float = 0.001):

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:

        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss
            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def state_dict(self):

        return {
            "best_loss": self.best_loss,
            "counter": self.counter,
        }

    def load_state_dict(self, state):

        self.best_loss = state["best_loss"]
        self.counter = state["counter"]


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    path: Path,
    model,
    optimizer,
    schedule,
    early_stopping,
    epoch: int,
    step: int,
    best_val_loss: float,
):

    torch.save(
        {
            "epoch": epoch,
            "step": step,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "schedule_step_num":
                schedule.step_num,

            "best_val_loss":
                best_val_loss,

            "early_stopping_state":
                early_stopping.state_dict(),
        },
        path,
    )

    logger.info(
        "Saved checkpoint: %s (epoch=%d, step=%d)",
        path,
        epoch,
        step,
    )


# ============================================================
# Load checkpoint
# ============================================================

def load_checkpoint(
    path: Path,
    model,
    optimizer,
    schedule,
    early_stopping,
):

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    schedule.step_num = checkpoint[
        "schedule_step_num"
    ]

    if "early_stopping_state" in checkpoint:

        early_stopping.load_state_dict(
            checkpoint["early_stopping_state"]
        )

    logger.info(
        "Resumed from checkpoint: %s "
        "(epoch=%d, step=%d)",
        path,
        checkpoint["epoch"],
        checkpoint["step"],
    )

    return (
        checkpoint["epoch"],
        checkpoint["step"],
        checkpoint["best_val_loss"],
    )


# ============================================================
# Validation
# ============================================================

def run_validation(
    model,
    val_loader,
    loss_fn,
    device,
    vocab_size,
    max_batches=None,
):

    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():

        for i, batch in enumerate(val_loader):

            if max_batches is not None and i >= max_batches:
                break

            src_ids = batch["src_ids"].to(device)

            tgt_ids = batch["tgt_ids"].to(device)

            decoder_input, decoder_target = (
                shift_for_teacher_forcing(tgt_ids)
            )

            logits = model(
                src_ids,
                decoder_input,
            )

            loss = loss_fn(
                logits.reshape(-1, vocab_size),
                decoder_target.reshape(-1),
            )

            total_loss += loss.item()

            num_batches += 1

    model.train()

    return total_loss / max(num_batches, 1)


# ============================================================
# Greedy decoding
# ============================================================

@torch.no_grad()
def greedy_decode(
    model,
    src_ids,
    max_len,
    device,
):

    model.eval()

    batch_size = src_ids.size(0)

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    src_mask = model.make_src_mask(src_ids)

    encoder_output, _ = model.encoder(
        src_ids,
        src_mask,
    )

    # --------------------------------------------------------
    # Start decoder with BOS
    # --------------------------------------------------------

    decoder_input = torch.full(
        (batch_size, 1),
        BOS_ID,
        dtype=torch.long,
        device=device,
    )

    # --------------------------------------------------------
    # Generate token by token
    # --------------------------------------------------------

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device,
    )

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

        next_token = logits[:, -1, :].argmax(
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

        # ----------------------------------------------------
        # Track sentences that reached EOS
        # ----------------------------------------------------

        finished |= (
            next_token.squeeze(1) == EOS_ID
        )

        if finished.all():
            break

    model.train()

    return decoder_input


# ============================================================
# BLEU + chrF++ + BERTScore
# ============================================================

def compute_metrics(
    model,
    val_loader,
    sp,
    device,
    max_len,
    num_samples=200,
):

    hypotheses = []
    references = []

    count = 0

    for batch in val_loader:

        src_ids = batch["src_ids"].to(device)

        tgt_ids = batch["tgt_ids"].to(device)

        generated = greedy_decode(
            model,
            src_ids,
            max_len,
            device,
        )

        for i in range(src_ids.size(0)):

            # ------------------------------------------------
            # Generated translation
            # ------------------------------------------------

            hyp_ids = [
                token
                for token in generated[i].tolist()
                if token not in (
                    PAD_ID,
                    BOS_ID,
                    EOS_ID,
                )
            ]

            # ------------------------------------------------
            # Reference translation
            # ------------------------------------------------

            ref_ids = [
                token
                for token in tgt_ids[i].tolist()
                if token not in (
                    PAD_ID,
                    BOS_ID,
                    EOS_ID,
                )
            ]

            hypothesis = sp.decode(
                hyp_ids
            )

            reference = sp.decode(
                ref_ids
            )

            hypotheses.append(
                hypothesis
            )

            references.append(
                reference
            )

            count += 1

            if count >= num_samples:
                break

        if count >= num_samples:
            break

    # ========================================================
    # BLEU
    # ========================================================

    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
    )

    # ========================================================
    # chrF++
    # ========================================================

    chrf = sacrebleu.corpus_chrf(
        hypotheses,
        references,
        word_order=2,
    )

    # ========================================================
    # BERTScore
    # ========================================================

    P, R, F1 = bertscore_score(
        hypotheses,
        references,
        lang="en",
        verbose=False,
    )

    bert_f1 = F1.mean().item()

    return (
        bleu.score,
        chrf.score,
        bert_f1,
    )


# ============================================================
# Training
# ============================================================

def train():

    logging.basicConfig(
        level=logging.WARNING
    )

    # ========================================================
    # Configuration
    # ========================================================

    VOCAB_SIZE = 16000

    D_MODEL = 512

    NUM_HEADS = 8

    D_FF = 2048

    NUM_LAYERS = 6

    WARMUP_STEPS = 4000

    BATCH_SIZE = 32

    NUM_EPOCHS = 30

    # IMPORTANT:
    # The screen log shows validation every 1,000 steps.
    VALIDATE_EVERY_N_STEPS = 1000

    # The screen log also shows checkpoints every 1,000 steps.
    CHECKPOINT_EVERY_N_STEPS = 1000

    EARLY_STOPPING_PATIENCE = 3

    BLEU_SAMPLE_SIZE = 200

    # IMPORTANT:
    # The logged run was a fresh run.
    #
    # Set this to None so the code does NOT automatically
    # resume from best.pt.
    RESUME_CHECKPOINT = None

    # ========================================================
    # Directories
    # ========================================================

    CHECKPOINT_DIR = Path(
        "training/checkpoints"
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TB_LOG_DIR = Path(
        "outputs/tensorboard_logs"
    )

    TB_LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # Maximum sequence length
    # ========================================================

    real_max_len = compute_max_seq_len(
        [
            "data/splits/train.csv",
            "data/splits/validation.csv",
            "data/splits/test.csv",
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
        f"Model maximum sequence length: "
        f"{max_len}"
    )

    # ========================================================
    # Data loaders
    # ========================================================

    train_loader = get_dataloader(
        "data/splits/train.csv",
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = get_dataloader(
        "data/splits/validation.csv",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(
        f"Training batches per epoch: "
        f"{len(train_loader)}"
    )

    # ========================================================
    # SentencePiece tokenizer
    # ========================================================

    sp = spm.SentencePieceProcessor()

    sp.load(
        "models/tokenizer/tokenizer.model"
    )

    # ========================================================
    # Transformer
    # ========================================================

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

    # ========================================================
    # Loss
    # ========================================================

    loss_fn = build_loss_fn(
        PAD_ID
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = build_optimizer(
        model
    )

    # ========================================================
    # Transformer learning-rate schedule
    # ========================================================

    schedule = TransformerLRSchedule(
        optimizer,
        D_MODEL,
        warmup_steps=WARMUP_STEPS,
    )

    # ========================================================
    # Early stopping
    # ========================================================

    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE
    )

    # ========================================================
    # Starting state
    # ========================================================

    start_epoch = 0

    global_step = 0

    best_val_loss = float("inf")

    # ========================================================
    # Optional resume
    # ========================================================

    if (
        RESUME_CHECKPOINT is not None
        and Path(RESUME_CHECKPOINT).exists()
    ):

        start_epoch, global_step, best_val_loss = (
            load_checkpoint(
                Path(RESUME_CHECKPOINT),
                model,
                optimizer,
                schedule,
                early_stopping,
            )
        )

        start_epoch += 1

    else:

        print(
            "Starting fresh training run."
        )

    # ========================================================
    # TensorBoard
    # ========================================================

    writer = SummaryWriter(
        log_dir=str(TB_LOG_DIR)
    )

    # ========================================================
    # Training information
    # ========================================================

    print(
        f"Starting training: "
        f"epochs {start_epoch}->{NUM_EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Train batches/epoch: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation every "
        f"{VALIDATE_EVERY_N_STEPS} steps"
    )

    print(
        f"Checkpoint every "
        f"{CHECKPOINT_EVERY_N_STEPS} steps"
    )

    print(
        f"Early stopping patience: "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(
        start_epoch,
        NUM_EPOCHS,
    ):

        model.train()

        epoch_loss = 0.0

        num_batches_this_epoch = 0

        # ====================================================
        # Batch loop
        # ====================================================

        for batch in train_loader:

            src_ids = batch[
                "src_ids"
            ].to(device)

            tgt_ids = batch[
                "tgt_ids"
            ].to(device)

            # ------------------------------------------------
            # Teacher forcing
            # ------------------------------------------------

            decoder_input, decoder_target = (
                shift_for_teacher_forcing(
                    tgt_ids
                )
            )

            # ------------------------------------------------
            # Clear gradients
            # ------------------------------------------------

            optimizer.zero_grad()

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            logits = model(
                src_ids,
                decoder_input,
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = loss_fn(
                logits.reshape(
                    -1,
                    VOCAB_SIZE,
                ),
                decoder_target.reshape(
                    -1
                ),
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            # ------------------------------------------------
            # Optimizer update
            # ------------------------------------------------

            optimizer.step()

            # ------------------------------------------------
            # Learning-rate schedule
            # ------------------------------------------------

            current_lr = schedule.step()

            # ------------------------------------------------
            # Step counters
            # ------------------------------------------------

            global_step += 1

            epoch_loss += loss.item()

            num_batches_this_epoch += 1

            # ------------------------------------------------
            # TensorBoard
            # ------------------------------------------------

            writer.add_scalar(
                "train/loss",
                loss.item(),
                global_step,
            )

            writer.add_scalar(
                "train/learning_rate",
                current_lr,
                global_step,
            )

            # =================================================
            # Step-level validation
            # =================================================

            if (
                global_step
                % VALIDATE_EVERY_N_STEPS
                == 0
            ):

                val_loss = run_validation(
                    model,
                    val_loader,
                    loss_fn,
                    device,
                    VOCAB_SIZE,
                    max_batches=50,
                )

                writer.add_scalar(
                    "validation/loss",
                    val_loss,
                    global_step,
                )

                # ------------------------------------------------
                # Best checkpoint
                # ------------------------------------------------

                if val_loss < best_val_loss:

                    best_val_loss = val_loss

                    save_checkpoint(
                        CHECKPOINT_DIR
                        / "best.pt",

                        model,

                        optimizer,

                        schedule,

                        early_stopping,

                        epoch,

                        global_step,

                        best_val_loss,
                    )

            # =================================================
            # Regular checkpoint
            # =================================================

            if (
                global_step
                % CHECKPOINT_EVERY_N_STEPS
                == 0
            ):

                save_checkpoint(
                    CHECKPOINT_DIR
                    / "latest.pt",

                    model,

                    optimizer,

                    schedule,

                    early_stopping,

                    epoch,

                    global_step,

                    best_val_loss,
                )

        # ====================================================
        # Epoch statistics
        # ====================================================

        avg_epoch_loss = (
            epoch_loss
            / max(
                num_batches_this_epoch,
                1,
            )
        )

        # ====================================================
        # Full validation
        # ====================================================

        full_val_loss = run_validation(
            model,
            val_loader,
            loss_fn,
            device,
            VOCAB_SIZE,
        )

        # ====================================================
        # BLEU + chrF++ + BERTScore
        # ====================================================

        (
            bleu_score,
            chrf_score,
            bert_f1,
        ) = compute_metrics(
            model,
            val_loader,
            sp,
            device,
            max_len,
            num_samples=BLEU_SAMPLE_SIZE,
        )

        # ====================================================
        # TensorBoard epoch metrics
        # ====================================================

        writer.add_scalar(
            "validation/epoch_loss",
            full_val_loss,
            epoch,
        )

        writer.add_scalar(
            "validation/bleu",
            bleu_score,
            epoch,
        )

        writer.add_scalar(
            "validation/chrf",
            chrf_score,
            epoch,
        )

        writer.add_scalar(
            "validation/bertscore_f1",
            bert_f1,
            epoch,
        )

        # ====================================================
        # Print epoch result
        # ====================================================

        print(
            f"Epoch {epoch} complete | "
            f"train_loss={avg_epoch_loss:.4f} | "
            f"val_loss={full_val_loss:.4f} | "
            f"BLEU={bleu_score:.2f} | "
            f"chrF++={chrf_score:.2f} | "
            f"BERTScore F1={bert_f1:.3f}"
        )

        # ====================================================
        # Epoch checkpoint
        # ====================================================

        save_checkpoint(
            CHECKPOINT_DIR
            / f"epoch_{epoch}.pt",

            model,

            optimizer,

            schedule,

            early_stopping,

            epoch,

            global_step,

            best_val_loss,
        )

        # ====================================================
        # Early stopping
        # ====================================================

        if early_stopping.step(
            full_val_loss
        ):

            print(
                f"Early stopping triggered "
                f"at epoch {epoch} "
                f"(no improvement for "
                f"{EARLY_STOPPING_PATIENCE} "
                f"epochs)."
            )

            break

    # ========================================================
    # Finish
    # ========================================================

    writer.close()

    print(
        "Training complete."
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train()


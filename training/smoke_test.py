
import logging
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, "models")
sys.path.insert(0, "src")
sys.path.insert(0, "training")
from transformer import Transformer
from loss_schedule import build_loss_fn, build_optimizer, TransformerLRSchedule
from dataset import get_dataloader
from compute_max_len import compute_max_seq_len
from train import shift_for_teacher_forcing, save_checkpoint, run_validation

logger = logging.getLogger(__name__)

PAD_ID = 0


def smoke_test():
    logging.basicConfig(level=logging.INFO)

    VOCAB_SIZE = 16000
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048
    NUM_LAYERS = 6
    WARMUP_STEPS = 4000
    BATCH_SIZE = 8
    NUM_TRAIN_BATCHES = 10
    NUM_VAL_BATCHES = 5

    CHECKPOINT_DIR = Path("training/checkpoints")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TB_LOG_DIR = Path("outputs/tensorboard_logs")
    TB_LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    real_max_len = compute_max_seq_len([
        "data/splits/train.csv",
        "data/splits/validation.csv",
        "data/splits/test.csv",
    ])
    max_len = int(real_max_len * 1.05)

    train_loader = get_dataloader("data/splits/train.csv", batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader("data/splits/validation.csv", batch_size=BATCH_SIZE, shuffle=False)

    model = Transformer(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
        num_layers=NUM_LAYERS, max_len=max_len, pad_id=PAD_ID,
    ).to(device)

    loss_fn = build_loss_fn(PAD_ID)
    optimizer = build_optimizer(model)
    schedule = TransformerLRSchedule(optimizer, D_MODEL, warmup_steps=WARMUP_STEPS)
    writer = SummaryWriter(log_dir=str(TB_LOG_DIR))

    print(f"\nRunning smoke test: {NUM_TRAIN_BATCHES} real training batches, "
          f"batch_size={BATCH_SIZE}\n")

    model.train()
    for step, batch in enumerate(train_loader):
        if step >= NUM_TRAIN_BATCHES:
            break

        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)

        decoder_input, decoder_target = shift_for_teacher_forcing(tgt_ids)

        optimizer.zero_grad()
        logits = model(src_ids, decoder_input)
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), decoder_target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        current_lr = schedule.step()

        writer.add_scalar("smoke_test/loss", loss.item(), step)
        writer.add_scalar("smoke_test/learning_rate", current_lr, step)

        print(f"Batch {step} | Loss: {loss.item():.4f} | LR: {current_lr:.6f} | "
              f"src shape: {tuple(src_ids.shape)} | tgt shape: {tuple(tgt_ids.shape)}")

    print(f"\nRunning validation on {NUM_VAL_BATCHES} real validation batches...")
    val_loss = run_validation(model, val_loader, loss_fn, device, VOCAB_SIZE,
                               max_batches=NUM_VAL_BATCHES)
    print(f"Validation loss (sample): {val_loss:.4f}")
    writer.add_scalar("smoke_test/val_loss", val_loss, 0)

    print("\nSaving a test checkpoint...")
    save_checkpoint(CHECKPOINT_DIR / "smoke_test.pt", model, optimizer, schedule,
                     epoch=0, step=NUM_TRAIN_BATCHES, best_val_loss=val_loss)

    checkpoint_exists = (CHECKPOINT_DIR / "smoke_test.pt").exists()
    tb_logs_exist = any(TB_LOG_DIR.iterdir())

    writer.close()

    print("\n=== Smoke Test Results ===")
    print(f"Ran {NUM_TRAIN_BATCHES} real training batches without crashing.")
    print(f"Validation loss computed successfully: ({val_loss:.4f})")
    print(f"Checkpoint file saved: {'YES' if checkpoint_exists else 'MISSING'}")
    print(f"TensorBoard logs created: {'YES' if tb_logs_exist else 'MISSING'}")
    print("\nIf all four show YES, the training loop is confirmed working correctly.")

if __name__ == "__main__":
    smoke_test()
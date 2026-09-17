

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "models")
sys.path.insert(0, "src")
from transformer import Transformer
from compute_max_len import compute_max_seq_len

logger = logging.getLogger(__name__)

PAD_ID = 0


def build_loss_fn(pad_id: int = PAD_ID) -> nn.Module:
    
    return nn.CrossEntropyLoss(ignore_index=pad_id)


class TransformerLRSchedule:
  

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _compute_lr(self) -> float:
        step = max(self.step_num, 1)  # avoid step=0 division issues
        return (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Adam with the specific betas/eps from the original Transformer
    paper — these aren't PyTorch's defaults, and the paper found them
    to matter for training stability."""
    return torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)


# ---------------- Reporting ----------------

def plot_lr_schedule(d_model: int, warmup_steps: int, total_steps: int, out_path: Path):
   
    schedule = TransformerLRSchedule(
        torch.optim.Adam([torch.zeros(1, requires_grad=True)]),
        d_model, warmup_steps,
    )
    lrs = []
    for _ in range(total_steps):
        lr = schedule.step()
        lrs.append(lr)

    plt.figure(figsize=(9, 4))
    plt.plot(range(1, total_steps + 1), lrs)
    plt.axvline(x=warmup_steps, color="red", linestyle="--", label=f"warmup_steps={warmup_steps}")
    plt.xlabel("Training step")
    plt.ylabel("Learning rate")
    plt.title("Transformer Learning Rate Schedule (warmup + decay)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved LR schedule plot to %s", out_path)


def loss_schedule_report(d_model, warmup_steps, peak_lr, peak_step,
                          overfit_losses: list) -> str:
    lines = ["# Loss Function & LR Schedule Report\n"]

    lines.append("## Loss function\n")
    lines.append(f"- Cross-entropy loss, `ignore_index={PAD_ID}` (`<pad>` excluded from loss/gradient)")

    lines.append("\n## Learning rate schedule\n")
    lines.append("| Setting | Value |")
    lines.append("|---|---|")
    lines.append(f"| d_model | {d_model} |")
    lines.append(f"| warmup_steps | {warmup_steps} |")
    lines.append(f"| Peak learning rate | {peak_lr:.6f} at step {peak_step} |")

    lines.append(
        "\n**Why this matters:** learning rate climbs for the first "
        f"{warmup_steps} steps, letting the randomly-initialized weights "
        "settle before larger updates are applied, then decays smoothly "
        "for the rest of training. Training a Transformer with a flat "
        "learning rate from step 1 is a known source of instability."
    )

    lines.append("\n## Overfit sanity test\n")
    lines.append(
        "Trained repeatedly on ONE tiny fake batch to confirm gradients "
        "flow correctly through the entire architecture — loss should "
        "drop sharply toward zero, since a model CAN memorize one batch "
        "if backprop is wired correctly end to end.\n"
    )
    lines.append("| Step | Loss |")
    lines.append("|---|---|")
    for i, loss_val in enumerate(overfit_losses):
        if i % 10 == 0 or i == len(overfit_losses) - 1:
            lines.append(f"| {i} | {loss_val:.4f} |")

    loss_dropped = overfit_losses[-1] < overfit_losses[0] * 0.1
    lines.append(f"\n**Loss dropped by >90% on the toy batch:** "
                  f"{' Yes — gradients flow correctly' if loss_dropped else ' NO — something is broken in the architecture or loss wiring'}")

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    REPORT_DIR = Path("outputs/reports")
    PLOTS_DIR = Path("outputs/plots")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    VOCAB_SIZE = 16000
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048
    NUM_LAYERS = 6
    PAD_ID = 0
    WARMUP_STEPS = 4000

    real_max_len = compute_max_seq_len([
        "data/splits/train.csv",
        "data/splits/validation.csv",
        "data/splits/test.csv",
    ])
    MAX_LEN = int(real_max_len * 1.05)

    # ---- Plot the LR schedule shape ----
    plot_lr_schedule(D_MODEL, WARMUP_STEPS, total_steps=20000,
                      out_path=PLOTS_DIR / "lr_schedule.png")

    # find peak lr/step for the report
    temp_sched = TransformerLRSchedule(
        torch.optim.Adam([torch.zeros(1, requires_grad=True)]), D_MODEL, WARMUP_STEPS
    )
    lrs = [temp_sched.step() for _ in range(20000)]
    peak_lr = max(lrs)
    peak_step = lrs.index(peak_lr) + 1

    # ---- Overfit sanity test: can this model memorize ONE tiny batch? ----
    print("Running overfit sanity test on one tiny fake batch...")
    torch.manual_seed(42)

    model = Transformer(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
        num_layers=NUM_LAYERS, max_len=MAX_LEN, pad_id=PAD_ID,
    )
    loss_fn = build_loss_fn(PAD_ID)
    optimizer = build_optimizer(model)
    schedule = TransformerLRSchedule(optimizer, D_MODEL, warmup_steps=WARMUP_STEPS)

    # one tiny fake batch, reused every step on purpose
    src_ids = torch.randint(4, VOCAB_SIZE, (2, 10))
    tgt_input = torch.randint(4, VOCAB_SIZE, (2, 9))   # decoder input (shifted)
    tgt_output = torch.randint(4, VOCAB_SIZE, (2, 9))  # what it should predict

    overfit_losses = []
    for step in range(100):
        optimizer.zero_grad()
        logits = model(src_ids, tgt_input)                  # (2, 9, 16000)
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        schedule.step()
        overfit_losses.append(loss.item())

    print(f"Loss at step 0: {overfit_losses[0]:.4f}")
    print(f"Loss at step 99: {overfit_losses[-1]:.4f}")
    print(f"Dropped by >90%: {overfit_losses[-1] < overfit_losses[0] * 0.1}")

    report = loss_schedule_report(
        d_model=D_MODEL, warmup_steps=WARMUP_STEPS,
        peak_lr=peak_lr, peak_step=peak_step,
        overfit_losses=overfit_losses,
    )
    with open(REPORT_DIR / "loss_schedule_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved loss_schedule_report.md and lr_schedule.png")
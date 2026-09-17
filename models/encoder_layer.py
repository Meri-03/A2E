

import logging
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, "models")
from attention import MultiHeadAttention

logger = logging.getLogger(__name__)


class PositionwiseFeedForward(nn.Module):
   

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
   

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # --- Sub-layer 1: self-attention, with residual + norm ---
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))   # residual connection, then normalize

        # --- Sub-layer 2: feed-forward, with residual + norm ---
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))      # residual connection, then normalize

        return x, attn_weights


# ---------------- Reporting ----------------

def encoder_layer_report(d_model, num_heads, d_ff, num_params, before_shape, after_shape) -> str:
    lines = ["# Encoder Layer Report\n"]

    lines.append("## Configuration\n")
    lines.append("| Setting | Value |")
    lines.append("|---|---|")
    lines.append(f"| d_model | {d_model} |")
    lines.append(f"| Number of attention heads | {num_heads} |")
    lines.append(f"| d_ff (feed-forward inner dimension) | {d_ff} |")

    lines.append("\n## Structure\n")
    lines.append("```")
    lines.append("x -> MultiHeadAttention -> Add & Norm -> FeedForward -> Add & Norm -> output")
    lines.append("```")

    lines.append("\n## Learnable parameters\n")
    lines.append(f"- Total: {num_params:,} parameters "
                  "(attention Q/K/V/output projections + feed-forward layers + 2 layer norms)")

    lines.append("\n## Shape check (sanity test on a fake batch)\n")
    lines.append("| Stage | Shape |")
    lines.append("|---|---|")
    lines.append(f"| Input | {before_shape} |")
    lines.append(f"| Output | {after_shape} |")
    shapes_match = before_shape == after_shape
    lines.append(f"\n**Input/output shape preserved:** {' Yes' if shapes_match else ' NO — mismatch, check code'}")
    lines.append(
        "\nShape preservation matters here specifically because this "
        "layer will be stacked N times — each layer's output must be a "
        "valid input to the next identical layer."
    )

    lines.append(
        "\n**Why this matters:** residual connections give gradients a "
        "direct path backward through the network, which is what makes "
        "stacking many layers trainable rather than suffering vanishing "
        "gradients. Layer normalization keeps activation scale stable "
        "from layer to layer. The feed-forward network gives each "
        "position additional capacity to transform what it gathered "
        "from attention."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    REPORT_DIR = Path("outputs/reports")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048

    encoder_layer = EncoderLayer(D_MODEL, NUM_HEADS, D_FF)

    fake_input = torch.randn(4, 20, D_MODEL)  # (batch=4, seq_len=20, d_model=512)

    output, attn_weights = encoder_layer(fake_input)
    print("Input shape:", fake_input.shape)
    print("Output shape:", output.shape)
    print("Shapes match:", fake_input.shape == output.shape)

    num_params = sum(p.numel() for p in encoder_layer.parameters())
    print(f"\nEncoder layer parameters: {num_params:,}")

    report = encoder_layer_report(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_params=num_params,
        before_shape=tuple(fake_input.shape),
        after_shape=tuple(output.shape),
    )
    with open(REPORT_DIR / "encoder_layer_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved encoder_layer_report.md")
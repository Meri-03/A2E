

import logging
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, "models")
sys.path.insert(0, "src")
from embedding import TokenEmbedding, PositionalEncoding
from encoder_layer import EncoderLayer
from compute_max_len import compute_max_seq_len

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
   

    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 d_ff: int, num_layers: int, max_len: int,
                 pad_id: int = 0, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model, pad_id)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, token_ids: torch.Tensor, mask=None):
        x = self.token_embedding(token_ids)
        x = self.positional_encoding(x)

        attn_weights_per_layer = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_per_layer.append(attn_weights)

        return x, attn_weights_per_layer


# ---------------- Reporting ----------------

def encoder_report(vocab_size, d_model, num_heads, d_ff, num_layers, max_len,
                    num_params, before_shape, after_shape) -> str:
    lines = ["# Full Encoder Report\n"]

    lines.append("## Configuration\n")
    lines.append("| Setting | Value |")
    lines.append("|---|---|")
    lines.append(f"| Vocabulary size | {vocab_size:,} |")
    lines.append(f"| d_model | {d_model} |")
    lines.append(f"| Number of attention heads | {num_heads} |")
    lines.append(f"| d_ff | {d_ff} |")
    lines.append(f"| Number of stacked encoder layers | {num_layers} |")
    lines.append(f"| Max sequence length | {max_len} |")

    lines.append("\n## Learnable parameters\n")
    lines.append(f"- Total: {num_params:,} parameters "
                  f"(embedding table + {num_layers} encoder layers)")

    lines.append("\n## Shape check (sanity test on a fake batch)\n")
    lines.append("| Stage | Shape |")
    lines.append("|---|---|")
    lines.append(f"| Input token IDs | {before_shape} |")
    lines.append(f"| Output (encoded representation) | {after_shape} |")

    lines.append(
        "\n**Why this matters:** stacking multiple encoder layers lets "
        "the model build progressively richer representations — early "
        "layers can capture local word relationships, later layers can "
        "combine those into higher-level sentence understanding. This "
        "is the complete encoder side of the Transformer: raw Amharic "
        "token IDs go in, a contextualized representation comes out, "
        "ready for the decoder to attend to."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    REPORT_DIR = Path("outputs/reports")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    VOCAB_SIZE = 16000
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048
    NUM_LAYERS = 6
    PAD_ID = 0

    real_max_len = compute_max_seq_len([
        "data/splits/train.csv",
        "data/splits/validation.csv",
        "data/splits/test.csv",
    ])
    MAX_LEN = int(real_max_len * 1.05)
    print("True max sequence length:", real_max_len)
    print("Using max_len:", MAX_LEN)

    encoder = Encoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN,
        pad_id=PAD_ID,
    )

    fake_token_ids = torch.randint(4, VOCAB_SIZE, (4, 20))  # (batch=4, seq_len=20)

    output, attn_weights_per_layer = encoder(fake_token_ids)
    print("\nInput shape:", fake_token_ids.shape)
    print("Output shape:", output.shape)
    print("Number of attention weight sets (one per layer):", len(attn_weights_per_layer))

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal encoder parameters: {num_params:,}")

    report = encoder_report(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS,
        d_ff=D_FF, num_layers=NUM_LAYERS, max_len=MAX_LEN,
        num_params=num_params,
        before_shape=tuple(fake_token_ids.shape),
        after_shape=tuple(output.shape),
    )
    with open(REPORT_DIR / "encoder_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved encoder_report.md")
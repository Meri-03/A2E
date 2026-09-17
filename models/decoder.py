

import logging
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, "models")
sys.path.insert(0, "src")
from embedding import TokenEmbedding, PositionalEncoding
from decoder_layer import DecoderLayer
from attention import create_causal_mask
from compute_max_len import compute_max_seq_len

logger = logging.getLogger(__name__)


class Decoder(nn.Module):
   

    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 d_ff: int, num_layers: int, max_len: int,
                 pad_id: int = 0, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model, pad_id)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

       
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, target_ids: torch.Tensor, encoder_output: torch.Tensor,
                src_mask=None, tgt_mask=None):
        x = self.token_embedding(target_ids)
        x = self.positional_encoding(x)

        self_attn_weights_per_layer = []
        cross_attn_weights_per_layer = []
        for layer in self.layers:
            x, self_attn_w, cross_attn_w = layer(x, encoder_output, src_mask, tgt_mask)
            self_attn_weights_per_layer.append(self_attn_w)
            cross_attn_weights_per_layer.append(cross_attn_w)

        logits = self.output_projection(x)  # (batch, seq_len, vocab_size)

        return logits, self_attn_weights_per_layer, cross_attn_weights_per_layer


# ---------------- Reporting ----------------

def decoder_report(vocab_size, d_model, num_heads, d_ff, num_layers, max_len,
                    num_params, decoder_input_shape, encoder_output_shape,
                    logits_shape) -> str:
    lines = ["# Full Decoder Report\n"]

    lines.append("## Configuration\n")
    lines.append("| Setting | Value |")
    lines.append("|---|---|")
    lines.append(f"| Vocabulary size | {vocab_size:,} |")
    lines.append(f"| d_model | {d_model} |")
    lines.append(f"| Number of attention heads | {num_heads} |")
    lines.append(f"| d_ff | {d_ff} |")
    lines.append(f"| Number of stacked decoder layers | {num_layers} |")
    lines.append(f"| Max sequence length | {max_len} |")

    lines.append("\n## Learnable parameters\n")
    lines.append(f"- Total: {num_params:,} parameters "
                  f"(embedding table + {num_layers} decoder layers + output projection)")

    lines.append("\n## Shape check\n")
    lines.append("| Stage | Shape |")
    lines.append("|---|---|")
    lines.append(f"| Decoder input (target token IDs) | {decoder_input_shape} |")
    lines.append(f"| Encoder output (source representation) | {encoder_output_shape} |")
    lines.append(f"| Output logits | {logits_shape} |")

    lines.append(
        "\nThe final dimension of the output logits equals the "
        f"vocabulary size ({vocab_size:,}) — for every position, the "
        "model produces a raw score for every possible next token. "
        "Softmax (applied during loss computation / generation, not "
        "inside this module) turns these into probabilities."
    )

    lines.append(
        "\n**Why this matters:** this is the complete decoder side of "
        "the Transformer. Combined with the encoder, the full "
        "architecture is now assembled: Amharic token IDs go into the "
        "encoder, English token IDs (shifted right, teacher-forcing "
        "style) go into the decoder alongside the encoder's output, and "
        "the decoder produces a probability distribution over the "
        "English vocabulary at every position."
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
    print("Using max_len:", MAX_LEN)

    decoder = Decoder(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS,
        d_ff=D_FF, num_layers=NUM_LAYERS, max_len=MAX_LEN, pad_id=PAD_ID,
    )

   
    fake_target_ids = torch.randint(4, VOCAB_SIZE, (4, 15))
    fake_encoder_output = torch.randn(4, 20, D_MODEL)

    causal_mask = create_causal_mask(seq_len=15)

    logits, self_attn_w, cross_attn_w = decoder(
        fake_target_ids, fake_encoder_output, tgt_mask=causal_mask
    )

    print("Decoder input shape:", fake_target_ids.shape)
    print("Encoder output shape:", fake_encoder_output.shape)
    print("Output logits shape:", logits.shape)
    print("Number of decoder layers:", len(self_attn_w))

    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nTotal decoder parameters: {num_params:,}")

    report = decoder_report(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_heads=NUM_HEADS,
        d_ff=D_FF, num_layers=NUM_LAYERS, max_len=MAX_LEN,
        num_params=num_params,
        decoder_input_shape=tuple(fake_target_ids.shape),
        encoder_output_shape=tuple(fake_encoder_output.shape),
        logits_shape=tuple(logits.shape),
    )
    with open(REPORT_DIR / "decoder_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved decoder_report.md")
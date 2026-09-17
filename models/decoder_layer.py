

import logging
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, "models")
from attention import MultiHeadAttention, create_causal_mask
from encoder_layer import PositionwiseFeedForward

logger = logging.getLogger(__name__)


class DecoderLayer(nn.Module):
    """One decoder layer: MaskedSelfAttention -> Add&Norm ->
    CrossAttention(attends to encoder output) -> Add&Norm ->
    FeedForward -> Add&Norm.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask=None, tgt_mask=None):
        # --- Sub-layer 1: masked self-attention (decoder attends to itself) ---
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # --- Sub-layer 2: cross-attention (decoder attends to encoder output) ---
        # Query from decoder (x), Key/Value from encoder_output — this is
        # the actual translation mechanism.
        cross_attn_output, cross_attn_weights = self.cross_attn(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # --- Sub-layer 3: feed-forward ---
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x, self_attn_weights, cross_attn_weights


# ---------------- Reporting ----------------

def decoder_layer_report(d_model, num_heads, d_ff, num_params,
                          decoder_input_shape, encoder_output_shape, output_shape,
                          cross_attn_shape) -> str:
    lines = ["# Decoder Layer Report\n"]

    lines.append("## Configuration\n")
    lines.append("| Setting | Value |")
    lines.append("|---|---|")
    lines.append(f"| d_model | {d_model} |")
    lines.append(f"| Number of attention heads | {num_heads} |")
    lines.append(f"| d_ff | {d_ff} |")

    lines.append("\n## Structure\n")
    lines.append("```")
    lines.append("x -> MaskedSelfAttention -> Add&Norm")
    lines.append("  -> CrossAttention(K,V from encoder) -> Add&Norm")
    lines.append("  -> FeedForward -> Add&Norm -> output")
    lines.append("```")

    lines.append("\n## Learnable parameters\n")
    lines.append(f"- Total: {num_params:,} parameters "
                  "(self-attention + cross-attention + feed-forward + 3 layer norms)")

    lines.append("\n## Shape check — deliberately mismatched sequence lengths\n")
    lines.append("| Stage | Shape |")
    lines.append("|---|---|")
    lines.append(f"| Decoder input (target/English side) | {decoder_input_shape} |")
    lines.append(f"| Encoder output (source/Amharic side) | {encoder_output_shape} |")
    lines.append(f"| Decoder layer output | {output_shape} |")
    lines.append(f"| Cross-attention weights | {cross_attn_shape} |")

    shapes_ok = decoder_input_shape == output_shape
    lines.append(f"\n**Output shape matches decoder input shape:** "
                  f"{'Yes' if shapes_ok else ' NO — mismatch, check code'}")
    lines.append(
        "\nSource and target sequence lengths are deliberately different "
        "in this test — real Amharic and English sentences rarely have "
        "matching word counts. Cross-attention handles this natively: "
        "the Query sequence length (decoder) and Key/Value sequence "
        "length (encoder) don't need to match for the attention math to "
        "work."
    )

    lines.append(
        "\n**Why this matters:** cross-attention is the actual "
        "translation mechanism — it's the point where the decoder, "
        "generating English word by word, looks back at the encoded "
        "Amharic representation and decides which source words are "
        "relevant to the word it's about to produce. Everything built "
        "before this (embedding, positional encoding, the encoder stack) "
        "exists to prepare good Key/Value vectors for this one step."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    REPORT_DIR = Path("outputs/reports")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048

    decoder_layer = DecoderLayer(D_MODEL, NUM_HEADS, D_FF)

    # Deliberately DIFFERENT lengths: decoder (English) = 15 tokens,
    # encoder output (Amharic) = 20 tokens — proves cross-attention
    # works correctly even when source and target lengths differ.
    fake_decoder_input = torch.randn(4, 15, D_MODEL)
    fake_encoder_output = torch.randn(4, 20, D_MODEL)

    causal_mask = create_causal_mask(seq_len=15)  # decoder can't see its own future

    output, self_attn_w, cross_attn_w = decoder_layer(
        fake_decoder_input, fake_encoder_output, tgt_mask=causal_mask
    )

    print("Decoder input shape:", fake_decoder_input.shape)
    print("Encoder output shape:", fake_encoder_output.shape)
    print("Decoder layer output shape:", output.shape)
    print("Self-attention weights shape:", self_attn_w.shape)
    print("Cross-attention weights shape:", cross_attn_w.shape)

    num_params = sum(p.numel() for p in decoder_layer.parameters())
    print(f"\nDecoder layer parameters: {num_params:,}")

    report = decoder_layer_report(
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF, num_params=num_params,
        decoder_input_shape=tuple(fake_decoder_input.shape),
        encoder_output_shape=tuple(fake_encoder_output.shape),
        output_shape=tuple(output.shape),
        cross_attn_shape=tuple(cross_attn_w.shape),
    )
    with open(REPORT_DIR / "decoder_layer_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved decoder_layer_report.md")
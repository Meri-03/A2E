

import logging
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, "models")
sys.path.insert(0, "src")
from encoder import Encoder
from decoder import Decoder
from attention import create_causal_mask
from compute_max_len import compute_max_seq_len

logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    """Full encoder-decoder Transformer for Amharic -> English translation."""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int,
                 num_heads: int, d_ff: int, num_layers: int, max_len: int,
                 pad_id: int = 0, dropout: float = 0.1):
        super().__init__()
        self.pad_id = pad_id

        self.encoder = Encoder(
            vocab_size=src_vocab_size, d_model=d_model, num_heads=num_heads,
            d_ff=d_ff, num_layers=num_layers, max_len=max_len,
            pad_id=pad_id, dropout=dropout,
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size, d_model=d_model, num_heads=num_heads,
            d_ff=d_ff, num_layers=num_layers, max_len=max_len,
            pad_id=pad_id, dropout=dropout,
        )

    def make_src_mask(self, src_ids: torch.Tensor) -> torch.Tensor:
       
        return (src_ids != self.pad_id).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt_ids: torch.Tensor) -> torch.Tensor:
        
        batch_size, tgt_len = tgt_ids.shape
        pad_mask = (tgt_ids != self.pad_id).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)
        causal_mask = create_causal_mask(tgt_len).to(tgt_ids.device)   # (tgt_len, tgt_len)
       
        return pad_mask & causal_mask

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor):
        src_mask = self.make_src_mask(src_ids)
        tgt_mask = self.make_tgt_mask(tgt_ids)

        encoder_output, enc_attn_weights = self.encoder(src_ids, src_mask)
        logits, dec_self_attn, dec_cross_attn = self.decoder(
            tgt_ids, encoder_output, src_mask, tgt_mask
        )

        return logits


# ---------------- Reporting ----------------

def transformer_report(src_vocab_size, tgt_vocab_size, d_model, num_heads,
                        d_ff, num_layers, max_len, encoder_params, decoder_params,
                        total_params, src_shape, tgt_shape, logits_shape) -> str:
    lines = ["# Full Transformer Report\n"]

    lines.append("## Configuration\n")
    lines.append("| Setting | Value |")
    lines.append("|---|---|")
    lines.append(f"| Source (Amharic) vocabulary size | {src_vocab_size:,} |")
    lines.append(f"| Target (English) vocabulary size | {tgt_vocab_size:,} |")
    lines.append(f"| d_model | {d_model} |")
    lines.append(f"| Number of attention heads | {num_heads} |")
    lines.append(f"| d_ff | {d_ff} |")
    lines.append(f"| Number of encoder/decoder layers | {num_layers} |")
    lines.append(f"| Max sequence length | {max_len} |")

    lines.append("\n## Learnable parameters\n")
    lines.append(f"- Encoder: {encoder_params:,}")
    lines.append(f"- Decoder: {decoder_params:,}")
    lines.append(f"- **Total: {total_params:,}**")

    lines.append("\n## Shape check (full forward pass, real masks)\n")
    lines.append("| Stage | Shape |")
    lines.append("|---|---|")
    lines.append(f"| Source token IDs (Amharic) | {src_shape} |")
    lines.append(f"| Target token IDs (English) | {tgt_shape} |")
    lines.append(f"| Output logits | {logits_shape} |")

    lines.append(
        "\n**Why this matters:** this is the complete, assembled "
        "Transformer — the from-scratch architecture required by this "
        "project. It takes real padding masks (built from `<pad>` IDs, "
        "not placeholders) and a real causal mask, combined for the "
        "decoder so it correctly ignores both padding and future "
        "positions. This model, given a batch from `dataset.py`, is "
        "now ready to be trained."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    REPORT_DIR = Path("outputs/reports")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    SRC_VOCAB_SIZE = 16000   # shared vocab, so src and tgt are the same size
    TGT_VOCAB_SIZE = 16000
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

    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE, tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
        num_layers=NUM_LAYERS, max_len=MAX_LEN, pad_id=PAD_ID,
    )

    # fake a REALISTIC batch: includes actual <pad> tokens (ID 0) at the
    # end of some sequences, like dataset.py's collate_batch would produce
    fake_src_ids = torch.randint(4, SRC_VOCAB_SIZE, (4, 20))
    fake_src_ids[0, 15:] = PAD_ID  # pretend sentence 0 is shorter, padded

    fake_tgt_ids = torch.randint(4, TGT_VOCAB_SIZE, (4, 15))
    fake_tgt_ids[1, 10:] = PAD_ID  # pretend sentence 1 is shorter, padded

    logits = model(fake_src_ids, fake_tgt_ids)

    print("Source shape:", fake_src_ids.shape)
    print("Target shape:", fake_tgt_ids.shape)
    print("Output logits shape:", logits.shape)

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nEncoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total model parameters: {total_params:,}")

    report = transformer_report(
        src_vocab_size=SRC_VOCAB_SIZE, tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF, num_layers=NUM_LAYERS,
        max_len=MAX_LEN, encoder_params=encoder_params, decoder_params=decoder_params,
        total_params=total_params,
        src_shape=tuple(fake_src_ids.shape), tgt_shape=tuple(fake_tgt_ids.shape),
        logits_shape=tuple(logits.shape),
    )
    with open(REPORT_DIR / "transformer_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved transformer_report.md")
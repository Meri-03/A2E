#30/11/18-05:50--(❁´◡`❁).....stay as loba

import math
import logging
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def scaled_dot_product_attention(q, k, v, mask=None):#calcuate attention weights and output using scaled dot-product attention  
   
    d_k = q.size(-1)#embedding dimension stored in the(-1 ) last dimension of q

    
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)#calculate attention scores,batch,head,seq_len,vector_dim
  

    if mask is not None:
       
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)

    return output, weights


class MultiHeadAttention(nn.Module):
   
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)  # combines heads back together

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x, batch_size):
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self.split_heads(self.w_q(query), batch_size)
        k = self.split_heads(self.w_k(key), batch_size)
        v = self.split_heads(self.w_v(value), batch_size)

        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        combined = self.combine_heads(attn_output, batch_size)
        output = self.w_out(combined)
        output = self.dropout(output)

        return output, attn_weights


def create_causal_mask(seq_len: int) -> torch.Tensor:
   
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    return mask  # shape: (seq_len, seq_len), True = allowed to attend


# ---------------- Reporting ----------------

def plot_attention_weights(attn_weights: torch.Tensor, out_path: Path, head_idx: int = 0):
  
    weights = attn_weights[0, head_idx].detach().numpy()  # first batch item, chosen head

    plt.figure(figsize=(6, 5))
    plt.imshow(weights, cmap="viridis")
    plt.xlabel("Key position (attended TO)")
    plt.ylabel("Query position (attending FROM)")
    plt.title(f"Attention Weights — Head {head_idx}")
    plt.colorbar(label="Attention weight")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved attention weights plot to %s", out_path)


def attention_report(d_model, num_heads, d_k, num_params, before_shape, after_shape,
                      causal_mask_shape) -> str:
    lines = ["# Multi-Head Attention Report\n"]

    lines.append("## Configuration\n")
    lines.append("| Setting | Value |")
    lines.append("|---|---|")
    lines.append(f"| d_model | {d_model} |")
    lines.append(f"| Number of heads | {num_heads} |")
    lines.append(f"| d_k (dimension per head) | {d_k} |")

    lines.append("\n## Learnable parameters\n")
    lines.append(f"- Total: {num_params:,} parameters (Q, K, V, and output projection layers)")

    lines.append("\n## Shape check (sanity test on a fake batch)\n")
    lines.append("| Stage | Shape |")
    lines.append("|---|---|")
    lines.append(f"| Input (query=key=value) | {before_shape} |")
    lines.append(f"| Output after multi-head attention | {after_shape} |")
    shapes_match = before_shape == after_shape
    lines.append(f"\n**Input/output shape preserved:** {' Yes' if shapes_match else ' NO — mismatch, check code'}")

    lines.append(f"\n## Causal mask\n")
    lines.append(f"- Shape: {causal_mask_shape}")
    lines.append("- Lower-triangular: each position can only attend to itself and earlier positions")

    lines.append(
        "\n**Why this matters:** multi-head attention is the mechanism "
        "that lets every word in a sentence weigh how relevant every "
        "other word is to it — the actual computation that makes a "
        "Transformer work. Running multiple heads in parallel lets the "
        "model learn several different types of relationships "
        "simultaneously (e.g. grammatical agreement, coreference, local "
        "word order) instead of being limited to one."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    REPORT_DIR = Path("outputs/reports")
    PLOTS_DIR = Path("outputs/plots")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    D_MODEL = 512
    NUM_HEADS = 8

    mha = MultiHeadAttention(D_MODEL, NUM_HEADS)

    # fake a batch like the one embedding.py produces: (batch=4, seq_len=20, d_model=512)
    fake_input = torch.randn(4, 20, D_MODEL)

    output, attn_weights = mha(fake_input, fake_input, fake_input)  # self-attention: q=k=v
    print("Input shape:", fake_input.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attn_weights.shape)

    causal_mask = create_causal_mask(seq_len=20)
    print("\nCausal mask shape:", causal_mask.shape)
    print("Causal mask (first 5x5 corner):")
    print(causal_mask[:5, :5].int())

    num_params = sum(p.numel() for p in mha.parameters())
    print(f"\nMulti-head attention parameters: {num_params:,}")

    plot_attention_weights(attn_weights, PLOTS_DIR / "attention_weights.png", head_idx=0)

    report = attention_report(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_k=D_MODEL // NUM_HEADS,
        num_params=num_params,
        before_shape=tuple(fake_input.shape),
        after_shape=tuple(output.shape),
        causal_mask_shape=tuple(causal_mask.shape),
    )
    with open(REPORT_DIR / "attention_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\nSaved attention_report.md and attention_weights.png")
# 28/11/2018-05:43
# stay as loba

import math
import logging
from pathlib import Path

import torch#
import torch.nn as nn#neural network


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "src")
from compute_max_len import compute_max_seq_len

logger = logging.getLogger(__name__)


class TokenEmbedding(nn.Module):#convert token ids to embedding vectors, and scale by sqrt(d_model) for stability
   

    def __init__(self, vocab_size: int, d_model: int, pad_id: int = 0):# CONSTRACTOR
        super().__init__()#fam INTILIZE
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)# CRETAE EMBBADIBG LAYER
       

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
       
        return self.embedding(token_ids) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    
  
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):# REGULAZATION
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        
        div_term = torch.exp(#create different frequencies for each dimension of the embedding
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term) #
        pe[:, 1::2] = torch.cos(position * div_term)  

        pe = pe.unsqueeze(0)  # batch dim for broadcasting
        self.register_buffer("pe", pe)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]  # add positional pattern, sliced to this sentence's length
        return self.dropout(x)


def plot_positional_encoding(pos_encode: PositionalEncoding, out_path: Path, plot_len: int = 100):
    
    pe_matrix = pos_encode.pe[0, :plot_len, :].numpy()

    plt.figure(figsize=(10, 5))
    plt.imshow(pe_matrix.T, aspect="auto", cmap="RdBu")
    plt.xlabel("Position in sentence")
    plt.ylabel("Embedding dimension")
    plt.title("Positional Encoding Pattern (sine/cosine)")
    plt.colorbar(label="Value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved positional encoding plot to %s", out_path)


def embedding_report(
    vocab_size: int,
    d_model: int,
    pad_id: int,
    max_len: int,
    before_shape: tuple,
    after_embed_shape: tuple,
    after_pos_shape: tuple,
    num_params: int,
) -> str:
    lines = ["# Embedding & Positional Encoding Report\n"]

    lines.append("## Configuration\n")
    lines.append("| Setting | Value |")
    lines.append("|---|---|")
    lines.append(f"| Vocabulary size | {vocab_size:,} |")
    lines.append(f"| d_model (embedding dimension) | {d_model} |")
    lines.append(f"| Padding token ID | {pad_id} |")
    lines.append(f"| Max sequence length (positional encoding) | {max_len} |")

    lines.append("\n## Learnable parameters\n")
    lines.append(f"- Token embedding table: {num_params:,} parameters "
                  f"({vocab_size:,} vocab × {d_model} dimensions)")
    lines.append("- Positional encoding: 0 parameters (fixed sine/cosine math, not learned)")

    lines.append("\n## Shape check (sanity test on a fake batch)\n")
    lines.append("| Stage | Shape |")
    lines.append("|---|---|")
    lines.append(f"| Input token IDs | {before_shape} |")
    lines.append(f"| After token embedding | {after_embed_shape} |")
    lines.append(f"| After positional encoding | {after_pos_shape} |")

    shapes_match = after_embed_shape == after_pos_shape
    lines.append(f"\n**Shapes preserved through positional encoding:** "
                  f"{' Yes' if shapes_match else 'NO — mismatch, check code'}")

    lines.append(
        "\n**Why this matters:** the embedding table converts token IDs "
        "into meaningful vectors the model can learn from, and since "
        "Transformer attention has no built-in sense of word order, "
        "positional encoding is what lets the model tell one position "
        "in a sentence apart from another. Both steps are required "
        "before the first attention layer can run."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    REPORT_DIR = Path("outputs/reports")
    PLOTS_DIR = Path("outputs/plots")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    VOCAB_SIZE = 16000
    D_MODEL = 512
    PAD_ID = 0
    #MAX_LEN = 512


    real_max_len = compute_max_seq_len([
    "data/splits/train.csv",
    "data/splits/validation.csv",
    "data/splits/test.csv",
])

    MAX_LEN = int(real_max_len * 1.05)

    print("True max sequence length:", real_max_len)
    print("Using max_len:", MAX_LEN)

    token_embed = TokenEmbedding(VOCAB_SIZE, D_MODEL, pad_id=PAD_ID)
    pos_encode = PositionalEncoding(D_MODEL, max_len=MAX_LEN)

    fake_token_ids = torch.randint(4, VOCAB_SIZE, (4, 20))

    embedded = token_embed(fake_token_ids)
    print("After token embedding:", embedded.shape)

    with_position = pos_encode(embedded)
    print("After positional encoding:", with_position.shape)

    print("\nShapes match:", embedded.shape == with_position.shape)

    num_params = sum(p.numel() for p in token_embed.parameters())
    print(f"\nToken embedding parameters: {num_params:,}")

    plot_positional_encoding(pos_encode, PLOTS_DIR / "positional_encoding.png")

    report = embedding_report(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        pad_id=PAD_ID,
        max_len=MAX_LEN,
        before_shape=tuple(fake_token_ids.shape),
        after_embed_shape=tuple(embedded.shape),
        after_pos_shape=tuple(with_position.shape),
        num_params=num_params,
    )
    with open(REPORT_DIR / "embedding_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved embedding_report.md and positional_encoding.png")


    # 08:54 wakanda forever
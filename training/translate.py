#21/12/18..moring 03:19

"""
translate.py — Interactive inference: type an Amharic sentence, get an
English translation back.
"""

from pathlib import Path
import sys

import torch
import sentencepiece as spm

sys.path.insert(0, "models")
sys.path.insert(0, "src")

from transformer import Transformer
from compute_max_len import compute_max_seq_len

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


@torch.no_grad()
def greedy_decode(model, src_ids, max_len, device):
    model.eval()
    batch_size = src_ids.size(0)
    src_mask = model.make_src_mask(src_ids)
    encoder_output, _ = model.encoder(src_ids, src_mask)

    decoder_input = torch.full((batch_size, 1), BOS_ID, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(decoder_input)
        logits, _, _ = model.decoder(decoder_input, encoder_output, src_mask, tgt_mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decoder_input = torch.cat([decoder_input, next_token], dim=1)
        if (next_token == EOS_ID).all():
            break

    return decoder_input


def load_model(checkpoint_path: str, device):
    VOCAB_SIZE = 16000
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048
    NUM_LAYERS = 6

    real_max_len = compute_max_seq_len([
        "data/splits/train.csv", "data/splits/validation.csv", "data/splits/test.csv",
    ])
    max_len = int(real_max_len * 1.05)

    model = Transformer(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
        num_layers=NUM_LAYERS, max_len=max_len, pad_id=PAD_ID,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {checkpoint_path} "
          f"(epoch {checkpoint['epoch']}, step {checkpoint['step']})")

    return model, max_len


def translate(text: str, model, sp, max_len, device) -> str:
    ids = sp.encode(text, out_type=int)
    ids = [BOS_ID] + ids + [EOS_ID]

    src_ids = torch.tensor([ids], dtype=torch.long, device=device)

    generated = greedy_decode(model, src_ids, max_len, device)

    output_ids = [t for t in generated[0].tolist() if t not in (PAD_ID, BOS_ID, EOS_ID)]
    return sp.decode(output_ids)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load("models/tokenizer/tokenizer.model")

    model, max_len = load_model("training/checkpoints/best.pt", device)

    print("\n" + "=" * 60)
    print("Amharic -> English Translator")
    print("Type an Amharic sentence and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        text = input("Amharic: ").strip()
        if text.lower() in ("quit", "exit", ""):
            print("Goodbye.")
            break

        translation = translate(text, model, sp, max_len, device)
        print(f"English: {translation}\n")


if __name__ == "__main__":
    main()
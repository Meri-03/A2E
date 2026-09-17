#04/01/19
import base64
import sys
from pathlib import Path

import sentencepiece as spm
import streamlit as st
import torch

sys.path.insert(0, "models")
sys.path.insert(0, "src")

from compute_max_len import compute_max_seq_len
from transformer import Transformer

# ============================================================
# Model Configuration
# ============================================================

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

VOCAB_SIZE = 16000
D_MODEL = 512
NUM_HEADS = 8
D_FF = 2048
NUM_LAYERS = 6

CHECKPOINT_PATH = "training/checkpoints/best.pt"
TOKENIZER_PATH = "models/tokenizer/tokenizer.model"
LOGO_PATH = "assets/logo.png"


# ============================================================
# Greedy Decoding 
# ============================================================

@torch.no_grad()
def greedy_decode(model, src_ids, max_len, device):
    model.eval()
    batch_size = src_ids.size(0)
    src_mask = model.make_src_mask(src_ids)

    encoder_output, _ = model.encoder(src_ids, src_mask)

    decoder_input = torch.full(
        (batch_size, 1),
        BOS_ID,
        dtype=torch.long,
        device=device
    )

    for _ in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(decoder_input)

        logits, _, _ = model.decoder(
            decoder_input,
            encoder_output,
            src_mask,
            tgt_mask
        )

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        decoder_input = torch.cat([decoder_input, next_token], dim=1)

        if (next_token == EOS_ID).all():
            break

    return decoder_input


# ============================================================
# Load Model & Tokenizer
# ============================================================

@st.cache_resource
def load_everything():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_max_len = compute_max_seq_len([
        "data/splits/train.csv",
        "data/splits/validation.csv",
        "data/splits/test.csv",
    ])

    max_len = int(real_max_len * 1.05)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)

    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_len=max_len,
        pad_id=PAD_ID,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, sp, max_len, device, checkpoint


# ============================================================
# Translation Logic
# ============================================================

def translate(text, model, sp, max_len, device):
    ids = sp.encode(text, out_type=int)
    ids = [BOS_ID] + ids + [EOS_ID]

    src_ids = torch.tensor([ids], dtype=torch.long, device=device)

    generated = greedy_decode(model, src_ids, max_len, device)

    output_ids = [
        t for t in generated[0].tolist()
        if t not in (PAD_ID, BOS_ID, EOS_ID)
    ]

    return sp.decode(output_ids)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="A2E — Amharic to English Machine Translation",
    layout="wide",
)


# ============================================================
# Interface Styling (CSS)
# ============================================================

STYLES = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Ethiopic:wght@400;500&display=swap');

:root {
  --navy:        #211d63;
  --navy-deep:   #171448;
  --panel-blue:  #e8f0fa;
  --blue-border: #cbd8e8;
  --page:        #eef1f7;
  --ink:         #202333;
  --muted:       #505466;
  --line:        #dfe3eb;
}

.stApp {
  background: var(--page);
}

#MainMenu, footer, header {
  visibility: hidden;
}

/* Main Container Card */
.block-container {
  max-width: 62rem;
  margin: 0.5rem auto 1rem auto;
  padding: 1.35rem 1.8rem 1.15rem 1.8rem;
  background: rgba(255,255,255,.97);
  border: 1px solid #dce1ea;
  border-radius: 13px;
  box-shadow: 0 5px 18px rgba(35, 45, 75, .12);
}

/* Header Section */
.hdr {
  text-align: center;
  position: relative;
  margin-bottom: 1.55rem;
  padding-top: .1rem;
}

.hdr h1.wordmark,
.wordmark {
  font-family: 'Inter', system-ui, sans-serif !important;
  font-size: 2.8rem !important;
  font-weight: 700 !important;
  color: #211d63 !important;
  -webkit-text-fill-color: #211d63 !important;
  letter-spacing: -.035em !important;
  line-height: 1 !important;
  margin: 0 !important;
}

.subtitle {
  font-family: 'Inter', sans-serif !important;
  font-size: 1.05rem !important;
  font-weight: 500 !important;
  color: #202020 !important;
  margin: .4rem 0 0 0 !important;
}

.tagline {
  font-family: 'Inter', sans-serif !important;
  font-size: .68rem !important;
  color: var(--muted) !important;
  margin: .32rem 0 0 0 !important;
}

/* Panel Headers */
.panel-label {
  font-family: 'Inter', sans-serif;
  font-size: .78rem;
  font-weight: 600;
  color: #242631;
  margin-bottom: .35rem;
}

.panel-label-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: .35rem;
}

/* Amharic Text Area Styling */
[data-testid="stTextArea"] textarea {
  background: #ffffff !important;
  color: var(--ink) !important;
  border: 1px solid #d9dee7 !important;
  border-radius: 9px !important;
  font-family: 'Noto Sans Ethiopic', 'Inter', sans-serif !important;
  font-size: .95rem !important;
  line-height: 1.65 !important;
  padding: .75rem .8rem !important;
  box-shadow: 0 2px 7px rgba(30,42,94,.035) !important;
  resize: none !important;
}

[data-testid="stTextArea"] textarea:focus {
  border-color: #8d95c8 !important;
  box-shadow: 0 0 0 2px rgba(74,95,193,.10) !important;
}

[data-testid="stTextArea"] label {
  display: none !important;
}

/* English Output Styling */
.out-panel {
  position: relative;
  background: var(--panel-blue);
  border: 1px solid var(--blue-border);
  border-radius: 9px;
  padding: .75rem .8rem;
  min-height: 108px;
  font-family: 'Inter', sans-serif;
  font-size: .95rem;
  line-height: 1.65;
  color: var(--ink);
  box-shadow: 0 2px 7px rgba(30,42,94,.035);
  overflow-wrap: anywhere;
}

.out-panel.placeholder {
  color: #718096;
}

/* Layout Elements */
.arrow {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 132px;
  color: var(--navy);
  font-size: 1.55rem;
  font-weight: 500;
}

/* Centered Translate Button */
div.stButton {
  display: flex !important;
  justify-content: center !important;
  width: 100% !important;
  margin-top: .5rem;
}

div.stButton > button {
  background: var(--navy) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 999px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: .85rem !important;
  min-width: 160px !important;
  padding: .5rem 2rem !important;
  box-shadow: 0 3px 8px rgba(30,42,94,.20) !important;
  transition: background .15s ease, transform .12s ease;
}

div.stButton > button:hover {
  background: var(--navy-deep) !important;
  transform: translateY(-1px);
}

/* Bold Academic Footer */
.foot {
  text-align: center;
  margin-top: 1.3rem;
  padding-top: .85rem;
  border-top: 1px solid var(--line);
  font-family: 'Inter', sans-serif;
  font-size: .75rem;
  font-weight: 600;
  color: #2b303c;
  letter-spacing: .02em;
  line-height: 1.6;
}

[data-testid="stVerticalBlock"] {
  gap: .35rem;
}

@media (max-width: 700px) {
  .block-container {
    margin: .25rem;
    padding: 1rem;
    border-radius: 11px;
  }
  .wordmark {
    font-size: 2.35rem !important;
  }
  .subtitle {
    font-size: .9rem !important;
  }
  .arrow {
    height: 28px;
    transform: rotate(90deg);
  }
}

</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


# ============================================================
# Load Resources & Session State
# ============================================================

model, sp, max_len, device, checkpoint = load_everything()

if "result" not in st.session_state:
    st.session_state.result = None


# ============================================================
# Header Section
# ============================================================

logo_html = ""
if Path(LOGO_PATH).exists():
    b64 = base64.b64encode(Path(LOGO_PATH).read_bytes()).decode()
    logo_html = (
        f'<img src="data:image/png;base64,{b64}" '
        f'style="position:absolute; left:0; top:0; height:58px; width:auto;" alt="EAII Logo">'
    )

st.markdown(
    f"""
    <div class="hdr">
      {logo_html}
      <h1 class="wordmark">A2E</h1>
      <p class="subtitle">Amharic &rarr; English Machine Translation</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Translation Interface Panels
# ============================================================

left, mid, right = st.columns([1, 0.12, 1], gap="small")

with left:
    st.markdown(
        '<div class="panel-label-row"><span class="panel-label">Amharic</span></div>',
        unsafe_allow_html=True
    )
    text = st.text_area(
        "Amharic input",
        height=110,
        placeholder="",
        label_visibility="collapsed"
    )

with mid:
    st.markdown('<div class="arrow">&rarr;</div>', unsafe_allow_html=True)

with right:
    st.markdown(
        '<div class="panel-label-row"><span class="panel-label">English</span></div>',
        unsafe_allow_html=True
    )
    out_slot = st.empty()


def render_output():
    if st.session_state.result:
        out_slot.markdown(
            f'<div class="out-panel">{st.session_state.result}</div>',
            unsafe_allow_html=True
        )
    else:
        out_slot.markdown(
            '<div class="out-panel placeholder"></div>',
            unsafe_allow_html=True
        )

render_output()


# ============================================================
# Action Button & Execution
# ============================================================

if st.button("Translate"):
    if text.strip():
        out_slot.markdown(
            '<div class="out-panel placeholder">Translating&hellip;</div>',
            unsafe_allow_html=True
        )
        st.session_state.result = translate(
            text.strip(),
            model,
            sp,
            max_len,
            device
        )
        render_output()
    else:
        st.session_state.result = None
        render_output()
        st.warning("Enter some Amharic text to translate.")


# ============================================================
# Footer Section
# ============================================================

st.markdown(
    """
    <div class="foot">
      EAII Internship  2018/19 E.C.
    </div>
    """,
    unsafe_allow_html=True
)
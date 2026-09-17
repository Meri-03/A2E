# A2E — Amharic to English Machine Translation

**NLP-Based Amharic to English Machine Translation Using a Transformer Model Built from Scratch**

A2E is a neural machine translation system that translates **Amharic sentences into English** using a **Transformer architecture implemented from scratch in PyTorch**.

The project was developed during the **2019 E.C. internship at the Ethiopian Artificial Intelligence Institute (EAII)**.

## Project Overview

The goal of A2E is to build an Amharic-to-English translation model without using pretrained translation models.

The system includes:

* Dataset cleaning and preprocessing
* Train/validation/test splitting
* SentencePiece tokenization
* Transformer encoder-decoder architecture
* Multi-head attention
* Teacher forcing
* Learning-rate scheduling
* Gradient clipping
* Validation-based checkpoint selection
* BLEU, chrF++, and BERTScore evaluation
* Streamlit-based translation interface

## Model Architecture

The Transformer was implemented from scratch with the following configuration:

| Parameter                   |                       Value |
| --------------------------- | --------------------------: |
| Architecture                | Transformer Encoder-Decoder |
| Number of Encoder Layers    |                           6 |
| Number of Decoder Layers    |                           6 |
| Model Dimension (`d_model`) |                         512 |
| Attention Heads             |                           8 |
| Feed-Forward Dimension      |                        2048 |
| Vocabulary Size             |                      16,000 |
| Maximum Sequence Length     |                         543 |
| Training Precision          |                        BF16 |

The model contains approximately **68.7 million parameters**.

## Dataset

After cleaning and deduplication, the dataset contained:

**357,887 Amharic-English sentence pairs**

The data was divided into:

| Split      | Sentence Pairs |
| ---------- | -------------: |
| Training   |        286,146 |
| Validation |         35,813 |
| Test       |         35,928 |

The split was performed so that normalized Amharic sentence groups did not overlap between the datasets.

## Tokenization

A shared **SentencePiece Unigram tokenizer** was trained with a vocabulary size of **16,000**.

Special tokens:

```text
<PAD> = 0
<BOS> = 1
<EOS> = 2
<UNK> = 3
```

## Training

The model was trained using:

* Cross-entropy loss
* Padding-token loss masking
* Teacher forcing
* Adam-based optimization
* Transformer learning-rate schedule
* 4,000 warm-up steps
* Gradient clipping
* Gradient accumulation
* BF16 mixed precision
* Validation monitoring
* Early stopping

### Validation and Checkpoint Selection

Validation data was **not used to update the model weights**.

It was used during training to monitor generalization and select the best checkpoint.

The workflow was:

```text
Training Data
      ↓
Train Transformer
      ↓
Validation Data
      ↓
Calculate Validation Loss
      ↓
Select Best Checkpoint
      ↓
best.pt
      ↓
Final Test Evaluation
```

The best checkpoint was:

```text
Epoch: 29
Step: 268,000
Validation Loss: 1.79937596
```

The test set was kept separate from checkpoint selection and used for the final evaluation.

## Training Progress

| Metric          | Start (Epoch 0) | End (Epoch 29) |
| --------------- | --------------: | -------------: |
| Train Loss      |          4.5184 |         1.4706 |
| Validation Loss |          3.8035 |         1.7330 |
| BLEU            |            0.15 |           1.01 |
| chrF++          |            6.50 |          13.23 |
| BERTScore F1    |           0.756 |         ~0.756 |

## Final Test Results

The selected `best.pt` checkpoint was evaluated on **35,928 previously unseen sentence pairs**.

| Evaluation Metric   |                     Score |
| ------------------- | ------------------------: |
| Test Set Size       | **35,928 sentence pairs** |
| BLEU                |                 **21.26** |
| chrF++              |                 **39.52** |
| BERTScore Precision |                **0.9054** |
| BERTScore Recall    |                **0.9016** |
| BERTScore F1        |                **0.9034** |

## Project Structure

```text
a2e/
│
├── assets/
│   └── logo.png
│
├── data/
│   └── ...
│
├── models/
│   └── tokenizer/
│       └── tokenizer.model
│
├── src/
│   └── ...
│
├── training/
│   ├── checkpoints/
│   │   └── best.pt
│   ├── train.py
│   └── evaluate.py
│
├── outputs/
│   └── ...
│
├── interface.py
├── requirements.txt
└── README.md
```

## Running the Project

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd a2e
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit interface

```bash
streamlit run interface.py
```

The application will open in your browser.

## Important Files

### `interface.py`

Provides the user interface for entering an Amharic sentence and generating its English translation.

### `training/train.py`

Contains the Transformer training pipeline, including optimization, teacher forcing, validation, checkpointing, and learning-rate scheduling.

### `training/evaluate.py`

Evaluates the selected model on the held-out test set using:

* BLEU
* chrF++
* BERTScore

### `models/tokenizer/tokenizer.model`

The SentencePiece tokenizer used by the model.

### `training/checkpoints/best.pt`

The selected Transformer checkpoint based on validation performance.

## Important Note About the Model

A2E was built **without using pretrained translation models** such as:

```text
mBART
MarianMT
T5
mT5
NLLB
```

The Transformer model and training pipeline were implemented specifically for this project using **PyTorch**.

## Technologies

* Python
* PyTorch
* SentencePiece
* SacreBLEU
* BERTScore
* Streamlit
* Matplotlib
* TensorBoard

## Future Improvements

Possible future improvements include:

* Increasing the size and diversity of the dataset
* Improving low-resource Amharic translation quality
* Experimenting with larger or more efficient Transformer architectures
* Beam-search decoding
* Domain-specific evaluation
* Human evaluation by native Amharic and English speakers
* Further optimization for deployment

## Author

**Meron Ashenafi**

Information Science Student
University of Gondar

Developed during the **2019 E.C. internship at the Ethiopian Artificial Intelligence Institute (EAII)**.

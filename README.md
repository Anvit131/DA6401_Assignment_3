# Building_Transliteration_System
# English-to-Hindi Transliteration with Seq2Seq with and without Attention (PyTorch)

This repository implements a Sequence-to-Sequence (Seq2Seq) model with and without an Attention both mechanisms for transliteration tasks using PyTorch. It supports Beam Search decoding and experiment tracking with Weights & Biases (wandb).

Dataset: Dakshina Indic Transliteration Dataset.


This project implements a Sequence-to-Sequence (Seq2Seq) model for transliterating English text to Hindi using PyTorch. The model leverages recurrent neural networks (RNNs) with configurable LSTM or GRU cells and employs beam search during inference for improved prediction quality. The project uses the Dakshina dataset and integrates Weights & Biases (wandb) for experiment tracking and visualization.

Table of Contents

1. Project Overview
2. Prerequisites
3. Installation
4. Dataset
5. Model Architecture
6. Training
7. Inference
8. Results
9. Troubleshooting
    
---

**Project Overview**

This project builds a deep learning model to transliterate English words (e.g., "namaste") into their Hindi equivalents (e.g., "नमस्ते"). Key features include:

1. A Seq2Seq model with an encoder-decoder architecture.
2. Support for LSTM or GRU cells, configurable via hyperparameters.
3. Teacher forcing during training (50% ratio) for faster convergence.
4. Beam search during inference for better sequence generation.
5. Integration with Weights & Biases for logging metrics, predictions, and artifacts.
6. Evaluation metrics: word-level and character-level accuracies.

Prerequisites
Ensure the following are installed:
- Python 3.8+
- pip (Python package manager)
- Weights & Biases (wandb) account (optional, for logging)
- Devanagari-compatible font (e.g., NotoSansDevanagari)

---
**Installation**

```text
git clone https://github.com/yourusername/english-to-hindi-transliteration.git
cd english-to-hindi-transliteration
!fc-list | grep Devanagari
!wget https://noto-website-2.storage.googleapis.com/pkgs/NotoSansDevanagari-hinted.zip
!unzip NotoSansDevanagari-hinted.zip
```
Set up Weights & Biases :
```text
pip install wandb
wandb login
```
---
**Dataset**
This project uses the Dakshina Indic Transliteration Dataset. To set it up:
- Download the dataset from kaggle
- Extract and place the Hindi lexicon files in the data/ folder:
```text
data/DakshinaDataset/hi/lexicon/
├── hi.translit.sampled.train.tsv
├── hi.translit.sampled.dev.tsv
└── hi.translit.sampled.test.tsv
```
**Model Architecture**

The Seq2Seq model consists of:

Encoder:
1. Embedding layer to convert English characters to dense vectors.
2. RNN (LSTM or GRU) to encode the input sequence into a context vector.

Decoder:
1. Embedding layer for Hindi characters.
2. RNN to generate the output sequence, conditioned on the encoder's hidden state.
3. Linear layer to predict the next character.

Beam Search:
Used during inference to explore multiple sequence hypotheses, improving prediction quality.

**Hyperparameters **
1. 'emb_dim': Embedding dimension
2. 'hidden_dim': Hidden dimension
3. 'num_layers': Number of layers
4. 'cell_type': RNN, LSTM (or GRU)
5. 'dropout': Dropout
6. 'beam_size': Beam size


**Training**
```text
python train_model.ipynb --cell_type LSTM --hidden_dim 256 --num_layers 3 --dropout 0.3 --epochs 10 --batch_size 64
```
The best model is saved as models/best_model.pt.

---

**Inference**

To evaluate the model on the test set:

1. Ensure best_model.pt is in models/.
2. Run the inference script:
   python inference

Inference Details:
1. Uses beam search (beam size = 3) for sequence generation.
2. Computes word-level and character-level accuracies.
3. Logs results to wandb and saves predictions to outputs/predictions_vanilla.csv.
4. Displays sample predictions in the console.
5. Output: Predictions are saved in outputs/predictions.csv and logged to wandb.

        example:
   
           Input: "hello"
   
           Output: "हैलो"

##  Project Structure(in Sort)

My model structure **without attention**

```text
├── requirements.txt
├── data/
   └── DakshinaDataset.py
   └── create_vocab.py
   └──   pad_collate.py
---
├── Vanila_model/
    ├── Encoder.py/
    |   └── forward.py
    ├── Decoder.py/
        └── forward.py
    ├── Seq2Seq.py/
        └── forward.py
        └── predict.py/
           └──beam_search.py
    ├──train_model.py
    ├──test_model.py

```
My model structure **with attention**

```text

├── model_with_attention/
    ├── Attn_Encoder.py/           (Remain same as Vanilla )
    |   └── forward.py
    ├── Attn_Decoder.py/
        └── forward.py
    ├── Attn_Seq2Seq.py/
        └── forward.py
        └── predict.py/
            └──beam_search.py
    ├──Attn_train_model.py
    ├──test model with attention.py
```
**Results**

![Screenshot 2025-05-19 163840](https://github.com/user-attachments/assets/21a642dc-0e25-490f-83d1-df8850363278)


**Troubleshooting**
- Missing model file: Ensure best_model.pt is in models/.
- Wandb login issues: Run wandb login and follow the prompts.
- Dataset not found: Verify the dataset is correctly placed in data/.
---
##  Features
- Encoder-Decoder Seq2Seq architecture
- Attention mechanism
- Beam Search decoding for better predictions
- Character-level and Word-level accuracy metrics
- Weights & Biases integration for logging
- Early stopping and model checkpointing

I am working with the [Dakshina Dataset](https://github.com/google-research-datasets/dakshina). 

---

## Hyperparameter Sweep (wandb)

1. Initialize the sweep:
   wandb sweep

2. Run sweep agent:
   wandb agent 

---

##  Requirements

Contents of `requirements.txt`:
```text
torch
torchvision
numpy
pandas 
scikit-learn 
wandb 
matplotlib
sentencepiece
```
---

#  Attention connectivity vizualization for English-to-Hindi Transliteration

This module visualizes attention weights in a trained Seq2Seq model with attention for English-to-Hindi transliteration using the [Dakshina Dataset](https://github.com/google-research-datasets/dakshina). It generates animated GIFs showing how the model attends to different input characters during each step of output generation.

---

##  Features

- Loads a trained attention-based Seq2Seq model
- Uses beam search decoding
- Visualizes attention as animated curved lines
- Saves results as GIFs and renders in a W&B HTML panel
- Supports Devanagari font rendering
- Logs animations and predictions to [Weights & Biases (W&B)](https://wandb.ai)

---

## Code Overview

### 1. Font Setup
- Loads and registers Devanagari font for Hindi script rendering in matplotlib.

### 2. Model & Data Initialization
- Loads trained model (`Attn_Seq2Seq`) and Dakshina dataset for testing.
- Converts input and output tokens to vocab indices.

### 3. Attention Visualization
- Generates curved-line animations for attention weights per output character.
- Uses `matplotlib.animation.FuncAnimation` to save GIFs.
- Shows how model focuses on source tokens during decoding.

**Example**
  ![attention_samples_8 (1)](https://github.com/user-attachments/assets/30adf1c6-b9cc-4eec-a867-98541a27a5ad)

### 4. Logging
- Saves predictions and attention animations
- Logs GIFs and interactive HTML grid to W&B dashboard
- Also saves `.tsv` prediction file and `.html` animation grid locally

---

##  Output

- `/kaggle/working/attention_samples_*.gif` – GIFs of attention animations
- `/kaggle/working/predictions_with_attn.tsv` – model predictions
- `/kaggle/working/attention_grids1.html` – embedded attention grid (HTML)
- Logged to W&B:
  - Attention GIFs (`wandb.Video`)
  - Attention HTML Grid (`wandb.Html`)
  - Model predictions as CSV (`wandb.Artifact` if needed)

---

##  How to Run

Ensure all dependencies and paths are set correctly. Then run the Python script in a Kaggle or Jupyter Notebook environment.

---


Self Declaration
I, Anvit Kumar, swear on my honour that I have written the code and the report by myself and have not copied it from the internet or other students.

Wandb Report Link : https://wandb.ai/ma24m004-iit-madras/DL_ASSIGNMENT_3_RNN/reports/DA6401-ASSIGNMENT-3--VmlldzoxMjY4NzMzNw?accessToken=6rlnv27d41cw6rwds83if3kaw2r885fcivg8nbwngb9enyjpf2s2k3d2fdk19gf5


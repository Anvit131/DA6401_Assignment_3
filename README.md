# DA6401_Assignment_3
# Seq2Seq Transliteration Model with and without Attention (PyTorch)

This repository implements a Sequence-to-Sequence (Seq2Seq) model with and without an Attention both mechanisms for transliteration tasks using PyTorch. It supports Beam Search decoding and experiment tracking with Weights & Biases (wandb).

Dataset: Dakshina Indic Transliteration Dataset.

---

## 📁 Project Structure

My model structure **without attention**

```text
├── requirements.txt

├── data/

   └── DakshinaDataset

   └── create_vocab

   └──   pad_collate

├── Vanila_model

    ├── Encoder

    |   └── forward

    ├── Decoder

        └── forward

    ├── Seq2Seq

        └── forward

        └── predict

           └──beam_search

    ├──train_model

    ├──test model

```
My model structure **with attention**

```text

├── model_with_attention
    ├── Attn_Encoder
    |   └── forward
    ├── Attn_Decoder
        └── forward
    ├── Attn_Seq2Seq
        └── forward
        └── predict
            └──beam_search
    ├──Attn_train_model
    ├──test model with attention
```

##  Features
- Encoder-Decoder Seq2Seq architecture
- Attention mechanism
- Beam Search decoding for better predictions
- Character-level and Word-level accuracy metrics
- Weights & Biases integration for logging
- Early stopping and model checkpointing

---

## Hyperparameter Sweep (wandb)

1. Initialize the sweep:
   wandb sweep

2. Run sweep agent:
   wandb agent 

---

##  Requirements

Contents of `requirements.txt`:

torch
torchvision
numpy
pandas 
scikit-learn 
wandb 
matplotlib
sentencepiece 

---

##  Credits

Developed by [Anvit Kumar]  
For [DA6410 / Deep Learning]  

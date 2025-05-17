# DA6401_Assignment_3
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
    


**Project Overview**

This project builds a deep learning model to transliterate English words (e.g., "namaste") into their Hindi equivalents (e.g., "नमस्ते"). Key features include:

1. A Seq2Seq model with an encoder-decoder architecture.
2. Support for LSTM or GRU cells, configurable via hyperparameters.
3. Teacher forcing during training (50% ratio) for faster convergence.
4. Beam search during inference for better sequence generation.
5. Integration with Weights & Biases for logging metrics, predictions, and artifacts.
6. Evaluation metrics: word-level and character-level accuracies.

Prerequisites
 Set up Weights & Biases:
   Install wandb: pip install wandb
   Log in: wandb login 

Model Architecture

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

Hyperparameters (configurable in train.py or inference.py):
1. Embedding dimension
2. Hidden dimension
3. Number of layers
4. Cell type:RNN, LSTM (or GRU)
5. Dropout
6. Beam size

Inference

To evaluate the model on the test set:

1. Ensure best_model.pt is in models/.
2. Run the inference script:
   python inference

Inference Details:
1. Uses beam search (beam size = 3) for sequence generation.
2. Computes word-level and character-level accuracies.
3. Logs results to wandb and saves predictions to outputs/predictions_vanilla.csv.
4. Displays sample predictions in the console.

##  Project Structure(in Sort)

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
    ├── Attn_Encoder           (Remain same as Vanilla )
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

##  Credits

Developed by [Anvit Kumar]  
For [DA6410 / Deep Learning]  

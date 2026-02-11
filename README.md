# CBOW Word2Vec Implementation

A from-scratch implementation of the Continuous Bag of Words (CBOW) variant of Word2Vec using Python and NumPy.

## Overview

This project implements a neural network for learning word embeddings using the CBOW architecture. The model predicts a target word based on its surrounding context words.

## Features

- **Custom Neural Network Components**: Built from scratch without deep learning frameworks
- **CBOW Architecture**: Embedding layer → Linear layer → Softmax activation
- **SGD Optimizer**: Stochastic Gradient Descent for parameter updates
- **Negative Log Likelihood Loss**: Cross-entropy loss for training
- **Numerical Stability**: Softmax implementation with overflow prevention

## Project Structure

```
.
├── data/                    # Data processing modules
│   ├── dataset.py          # Dataset creation and batching
│   ├── textloader.py       # Text file loading
│   ├── textpreprocessor.py # Text tokenization and cleaning
│   └── vocabulary.py       # Vocabulary management
├── model/                   # Neural network components
│   ├── activations.py      # Softmax activation layer
│   ├── layers.py           # Linear and Embedding layers
│   ├── loss.py             # Negative log likelihood loss
│   ├── model.py            # CBOW model implementation
│   └── parameter.py        # Parameter wrapper class
├── training/                # Training utilities
│   ├── optimizer.py        # SGD optimizer
│   └── trainer.py          # Training loop
├── raw_data/                # Training data
│   └── data.txt            # Text corpus
├── main.py                  # Main training script
└── requirements.txt         # Python dependencies
```

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/bryanvela98/cbowmodel.git
cd cbowmodel
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Usage

Run the training script:

```bash
python main.py
```

**Hyperparameters** (configurable in `main.py`):

- `embedding_dim`: 50 (dimension of word embeddings)
- `context_size`: 2 (number of words on each side of target)
- `learning_rate`: 0.5
- `num_epochs`: 50
- `batch_size`: 32

## How It Works

1. **Text Preprocessing**: Tokenizes and cleans input text
2. **Vocabulary Building**: Creates word-to-index mappings
3. **Context-Target Pairs**: Generates training examples
4. **Training**:
   - Forward pass: context words → embeddings → concatenate → linear → softmax
   - Loss computation: negative log likelihood
   - Backward pass: gradient computation via backpropagation
   - Parameter update: SGD optimizer
5. **Output**: Trained word embeddings + loss plot

## Implementation Details

- **No PyTorch/TensorFlow**: Built entirely with NumPy
- **Backpropagation**: Manual gradient computation for all layers
- **Xavier Initialization**: Weights initialized using Glorot uniform distribution
- **Batch Training**: Mini-batch gradient descent for efficiency

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- tqdm

## Example Output

```
Vocabulary size: 1523
Training...
Epoch 1/50: 100%|██████████| 20/20 [00:01<00:00, 12.34it/s, loss=6.234]
Epoch 2/50: 100%|██████████| 20/20 [00:01<00:00, 13.45it/s, loss=5.876]
...
Training complete!
```

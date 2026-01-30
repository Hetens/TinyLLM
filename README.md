A lightweight transformer language model implementation built from scratch using PyTorch. This project features a **3.3 million parameter** model trained on the TinyStories dataset.

## Model Architecture

The DemoTransformer is a compact yet capable language model with the following specifications:

- **Parameters**: ~3.3 Million
- **Model Dimension (d_model)**: 32
- **Attention Heads**: 16
- **Head Dimension (d_head)**: 2
- **MLP Dimension**: 128 (4x d_model)
- **Layers**: 4
- **Context Length**: 128 tokens
- **Vocabulary Size**: 50,257 (GPT-2 tokenizer)

## Architecture Components

### Core Modules

- **`transformer.py`**: Main transformer model and transformer blocks
- **`attention.py`**: Multi-head self-attention with causal masking
- **`mlp.py`**: Feed-forward network with GELU activation
- **`layers.py`**: Embedding, positional embedding, unembedding, and layer normalization
- **`config.py`**: Model and training configuration
- **`trainer.py`**: Training loop, evaluation, and checkpointing
- **`sampler.py`**: Text generation with various sampling strategies
- **`main.py`**: Entry point for training and inference

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TinyLLM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

Train the model from scratch:

```bash
python main.py
```

The training script will:
- Load the TinyStories dataset
- Train for 10 epochs with 500 steps per epoch
- Save checkpoints to `./saved_models/`
- Display sample generations after each epoch

### Inference

Run inference with a saved checkpoint:

```bash
python main.py --inference ./saved_models/model_checkpoint.pt
```

This will start an interactive prompt where you can generate text.

## Features

### Sampling Strategies

The model supports multiple text generation strategies:

- **Greedy Search**: Always pick the most likely token
- **Temperature Sampling**: Control randomness (higher = more random)
- **Top-k Sampling**: Sample from the k most likely tokens
- **Top-p (Nucleus) Sampling**: Sample from tokens with cumulative probability p
- **Frequency Penalty**: Reduce repetition by penalizing frequent tokens

Example usage:
```python
from sampler import TransformerSampler

sampler = TransformerSampler(model, tokenizer)
text = sampler.sample(
    prompt="Once upon a time",
    max_tokens_generated=100,
    temperature=0.8,
    top_p=0.9
)
```

### Training Configuration

Default training parameters (configurable in `config.py`):

- **Batch Size**: 16
- **Learning Rate**: 1e-3
- **Weight Decay**: 1e-2
- **Optimizer**: AdamW
- **Epochs**: 10
- **Max Steps per Epoch**: 500

## Technical Details

### Attention Mechanism

The model uses multi-head self-attention with:
- Causal masking to prevent attending to future tokens
- Scaled dot-product attention
- Separate Q, K, V projections for each head

### Training

- **Dataset**: TinyStories by Roneneldan
- **Loss**: Cross-entropy (negative log-likelihood)
- **Evaluation**: Token-level accuracy on held-out test set
- **Hardware**: CUDA-enabled GPU (falls back to CPU if unavailable)

## Performance

The model is evaluated on:
- **Training Loss**: Cross-entropy loss on training batches
- **Test Accuracy**: Token prediction accuracy on test set
- **Sample Quality**: Qualitative evaluation of generated text

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with inspiration from the TransformerLens library
- Trained on the TinyStories dataset
- Uses GPT-2 tokenizer from HuggingFace Transformers

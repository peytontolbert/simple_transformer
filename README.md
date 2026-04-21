# Simple Transformer Implementation in PyTorch

A clean and simple implementation of the Transformer architecture using PyTorch. This implementation includes the core components of the Transformer model along with training and testing utilities.

## Features

- Clean implementation of Transformer architecture
- Multi-head attention mechanism
- Positional encoding
- Training script with sample data generation
- Runtime benchmark comparing eager PyTorch vs model-stack
- Comprehensive unit tests
- Easy to understand and modify

## Installation

1. Clone the repository:
```bash
git clone https://github.com/peytontolbert/simple-transformer.git
cd simple-transformer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

`requirements.txt` installs `model-stack` from GitHub so the model can route attention and sampling through the model-stack runtime when it is available, while leaving dense module paths on eager CUDA when that is faster on Ampere-class GPUs.

## Project Structure

```
simple-transformer/
├── benchmark_runtime.py  # Runtime benchmark for PyTorch vs model-stack
├── transformer/
│   └── model.py         # Transformer model implementation
├── tests/
│   └── test_transformer.py  # Unit tests
├── train.py             # Training script
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Usage

### Training

To train the model on sample data:

```bash
python train.py
```

This will:
1. Create sample training data
2. Initialize the Transformer model
3. Train for the specified number of epochs
4. Save the trained model

### Testing

To run the tests:

```bash
pytest tests/
```

### Benchmarking

To compare the eager PyTorch fallback path against the model-stack-backed path:

```bash
python benchmark_runtime.py --device auto --dtype auto
```

On a shared machine, you can target a specific GPU directly, for example `python benchmark_runtime.py --device cuda:1 --dtype bfloat16`.

The benchmark runs the same `SimpleTransformer` weights and inputs in two modes:

- `pytorch`: forces the eager fallback path
- `model-stack`: enables the model-stack runtime helpers

Performance is workload-dependent, so benchmark your own target shapes instead of assuming model-stack is always faster. On the RTX 3090 path, the current wrapper keeps attention on the tuned model-stack runtime and leaves dense module ops on eager CUDA because that combination is faster end to end than forcing every projection through the extension boundary.

#### Sample Runtime Stats

Measured sequentially in the `ai` Conda environment on an NVIDIA GeForce RTX 3090.

Repo-scale example configuration:

- `batch_size=32`
- `seq_len=20`
- `vocab_size=1000`
- `d_model=256`
- `num_heads=8`
- `num_layers=3`
- `d_ff=2048`
- `dtype=bfloat16`
- `warmup=20`, `iters=100`

| Mode | Mean Latency (ms) | Median (ms) | P95 (ms) | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| PyTorch | 1.433 | 1.424 | 1.498 | 446,692 |
| model-stack | 1.075 | 1.074 | 1.107 | 595,323 |

This configuration is now a clear win for model-stack on this machine: `PyTorch / model-stack = 1.33x`, with `max_abs_diff = 0.01611328125`.

Longer-context configuration:

- `batch_size=8`
- `seq_len=512`
- `vocab_size=4096`
- `d_model=512`
- `num_heads=8`
- `num_layers=6`
- `d_ff=2048`
- `dtype=bfloat16`
- `warmup=10`, `iters=30`

| Mode | Mean Latency (ms) | Median (ms) | P95 (ms) | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| PyTorch | 7.114 | 7.040 | 7.633 | 575,470 |
| model-stack | 4.149 | 4.148 | 4.248 | 987,557 |

On this longer prefill workload, `PyTorch / model-stack = 1.71x`, with `max_abs_diff = 0.0234375`.

## Model Architecture

The implementation includes:
- MultiHeadAttention: Implementation of multi-head attention mechanism
- PositionalEncoding: Adds positional information to the input embeddings
- TransformerBlock: Complete transformer block with attention and feed-forward layers
- SimpleTransformer: Main model class combining all components

The current implementation preserves the same public API while routing the core execution path through `model-stack` helpers with eager fallbacks for environments where the runtime package is unavailable.

## License

MIT License

## Contributing

Feel free to open issues and pull requests for improvements! 

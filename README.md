# Simple Transformer Implementation in PyTorch

A clean and simple implementation of the Transformer architecture using PyTorch. This implementation includes the core components of the Transformer model along with training and testing utilities.

## Features

- Clean implementation of Transformer architecture
- Multi-head attention mechanism
- Positional encoding
- Training script with sample data generation
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

## Project Structure

```
simple-transformer/
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

## Model Architecture

The implementation includes:
- MultiHeadAttention: Implementation of multi-head attention mechanism
- PositionalEncoding: Adds positional information to the input embeddings
- TransformerBlock: Complete transformer block with attention and feed-forward layers
- SimpleTransformer: Main model class combining all components

## License

MIT License

## Contributing

Feel free to open issues and pull requests for improvements! 
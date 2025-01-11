import pytest
import torch
from transformer.model import SimpleTransformer, MultiHeadAttention, PositionalEncoding, TransformerBlock

def test_multi_head_attention():
    batch_size = 4
    seq_length = 10
    d_model = 512
    num_heads = 8
    
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_length, d_model)
    
    output = mha(x, x, x)
    assert output.shape == (batch_size, seq_length, d_model)

def test_positional_encoding():
    batch_size = 4
    seq_length = 10
    d_model = 512
    
    pe = PositionalEncoding(d_model)
    x = torch.randn(batch_size, seq_length, d_model)
    
    output = pe(x)
    assert output.shape == (batch_size, seq_length, d_model)
    assert not torch.allclose(x, output)  # Ensure encoding was added

def test_transformer_block():
    batch_size = 4
    seq_length = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    
    block = TransformerBlock(d_model, num_heads, d_ff)
    x = torch.randn(batch_size, seq_length, d_model)
    
    output = block(x)
    assert output.shape == (batch_size, seq_length, d_model)

def test_simple_transformer():
    batch_size = 4
    seq_length = 10
    vocab_size = 1000
    d_model = 512
    
    model = SimpleTransformer(vocab_size, d_model)
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    output = model(x)
    assert output.shape == (batch_size, seq_length, vocab_size)

def test_transformer_training_step():
    batch_size = 4
    seq_length = 10
    vocab_size = 1000
    d_model = 512
    
    model = SimpleTransformer(vocab_size, d_model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create sample batch
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    y = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    output = model(x)
    loss = criterion(output.view(-1, vocab_size), y.view(-1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    assert not torch.isnan(loss)
    assert loss.item() > 0 
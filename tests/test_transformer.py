import pytest
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformer.model import SimpleTransformer, MultiHeadAttention, PositionalEncoding, TransformerBlock
from transformer import _model_stack as model_stack_mod

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


def test_multi_head_attention_uses_model_stack_helpers(monkeypatch):
    calls = {"qkv": 0, "attention": 0, "linear": 0}
    batch_size = 2
    seq_length = 5
    d_model = 16
    num_heads = 4

    mha = MultiHeadAttention(d_model, num_heads)

    def fake_runtime_qkv(q_module, k_module, v_module, q, k, v):
        calls["qkv"] += 1
        return q_module(q), k_module(k), v_module(v)

    def fake_runtime_attention(q, k, v, *, mask=None, is_causal=False):
        calls["attention"] += 1
        assert q.shape == (batch_size, num_heads, seq_length, d_model // num_heads)
        assert is_causal is True
        return torch.zeros_like(q)

    def fake_runtime_linear(module, x):
        calls["linear"] += 1
        return module(x)

    monkeypatch.setattr(model_stack_mod, "runtime_qkv", fake_runtime_qkv)
    monkeypatch.setattr(model_stack_mod, "runtime_attention", fake_runtime_attention)
    monkeypatch.setattr(model_stack_mod, "runtime_linear", fake_runtime_linear)

    x = torch.randn(batch_size, seq_length, d_model)
    output = mha(x, x, x, is_causal=True)

    assert output.shape == (batch_size, seq_length, d_model)
    assert calls == {"qkv": 1, "attention": 1, "linear": 1}


def test_transformer_block_uses_runtime_mlp_and_add_layer_norm(monkeypatch):
    calls = {"add_norm": 0, "mlp": 0}
    d_model = 32
    num_heads = 4
    d_ff = 64
    block = TransformerBlock(d_model, num_heads, d_ff, dropout=0.0)

    def fake_runtime_add_layer_norm(x, update, norm):
        calls["add_norm"] += 1
        return norm(x + update)

    def fake_runtime_mlp(w_in_module, w_out_module, x, *, activation="relu"):
        calls["mlp"] += 1
        assert activation == "relu"
        return w_out_module(torch.relu(w_in_module(x)))

    monkeypatch.setattr(model_stack_mod, "runtime_add_layer_norm", fake_runtime_add_layer_norm)
    monkeypatch.setattr(model_stack_mod, "runtime_mlp", fake_runtime_mlp)

    x = torch.randn(2, 6, d_model)
    output = block(x)

    assert output.shape == (2, 6, d_model)
    assert calls == {"add_norm": 2, "mlp": 1}


def test_simple_transformer_uses_runtime_embedding_and_output_linear(monkeypatch):
    calls = {"embedding": 0, "linear": 0}
    vocab_size = 128
    d_model = 32
    model = SimpleTransformer(vocab_size=vocab_size, d_model=d_model, num_layers=0)

    def fake_runtime_embedding(module, indices):
        calls["embedding"] += 1
        return torch.nn.functional.embedding(indices, module.weight, padding_idx=module.padding_idx)

    def fake_runtime_linear(module, x):
        calls["linear"] += 1
        return module(x)

    monkeypatch.setattr(model_stack_mod, "runtime_embedding", fake_runtime_embedding)
    monkeypatch.setattr(model_stack_mod, "runtime_linear", fake_runtime_linear)

    x = torch.randint(0, vocab_size, (3, 4))
    output = model(x)

    assert output.shape == (3, 4, vocab_size)
    assert calls == {"embedding": 1, "linear": 1}


def test_runtime_mode_temporarily_disables_helpers(monkeypatch):
    helper_names = tuple(model_stack_mod._RUNTIME_HELPER_NAMES)
    sentinel = object()
    for helper_name in helper_names:
        monkeypatch.setattr(model_stack_mod, helper_name, sentinel)

    with model_stack_mod.runtime_mode(enabled=False):
        during = model_stack_mod.runtime_status()["helpers_available"]

    after = model_stack_mod.runtime_status()["helpers_available"]

    assert model_stack_mod.runtime_status()["helpers_available"] is True
    assert during is False
    assert after is True


def test_dense_runtime_helpers_can_fall_back_to_module_paths(monkeypatch):
    monkeypatch.setattr(model_stack_mod, "_prefer_eager_dense_module_path", lambda tensor: True)

    def _unexpected(*args, **kwargs):
        raise AssertionError("runtime helper should not be called when dense module fallback is preferred")

    monkeypatch.setattr(model_stack_mod, "_runtime_embedding", _unexpected)
    monkeypatch.setattr(model_stack_mod, "_runtime_linear_module", _unexpected)
    monkeypatch.setattr(model_stack_mod, "_runtime_qkv_projection", _unexpected)
    monkeypatch.setattr(model_stack_mod, "_runtime_add_layer_norm", _unexpected)
    monkeypatch.setattr(model_stack_mod, "_runtime_mlp_module", _unexpected)

    embedding = torch.nn.Embedding(32, 8)
    linear = torch.nn.Linear(8, 8)
    norm = torch.nn.LayerNorm(8)
    ff_in = torch.nn.Linear(8, 16)
    ff_out = torch.nn.Linear(16, 8)
    x = torch.randn(2, 4, 8)
    token_ids = torch.randint(0, 32, (2, 4))

    assert torch.allclose(model_stack_mod.runtime_embedding(embedding, token_ids), embedding(token_ids))
    assert torch.allclose(model_stack_mod.runtime_linear(linear, x), linear(x))

    q, k, v = model_stack_mod.runtime_qkv(linear, linear, linear, x, x, x)
    assert torch.allclose(q, linear(x))
    assert torch.allclose(k, linear(x))
    assert torch.allclose(v, linear(x))

    update = torch.randn_like(x)
    assert torch.allclose(model_stack_mod.runtime_add_layer_norm(x, update, norm), norm(x + update))

    ref_mlp = ff_out(torch.relu(ff_in(x)))
    assert torch.allclose(model_stack_mod.runtime_mlp(ff_in, ff_out, x, activation="relu"), ref_mlp)

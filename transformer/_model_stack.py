from __future__ import annotations

import math
import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from runtime.hardware import cuda_arch_family

try:
    from runtime.native import has_native_op as _native_has_op
    from runtime.native import native_available as _native_available
except Exception:
    _native_has_op = None
    _native_available = None

try:
    from runtime.ops import (
        add_layer_norm as _runtime_add_layer_norm,
        attention as _runtime_attention,
        embedding as _runtime_embedding,
        linear_module as _runtime_linear_module,
        mlp_module as _runtime_mlp_module,
        qkv_projection as _runtime_qkv_projection,
        resolve_linear_module_tensors as _resolve_linear_module_tensors,
        sample_with_policies as _runtime_sample_with_policies,
        temperature as _runtime_temperature,
        topk_mask as _runtime_topk_mask,
        topp_mask as _runtime_topp_mask,
    )
except Exception:
    _runtime_add_layer_norm = None
    _runtime_attention = None
    _runtime_embedding = None
    _runtime_linear_module = None
    _runtime_mlp_module = None
    _runtime_qkv_projection = None
    _resolve_linear_module_tensors = None
    _runtime_sample_with_policies = None
    _runtime_temperature = None
    _runtime_topk_mask = None
    _runtime_topp_mask = None

_RUNTIME_HELPER_NAMES = (
    "_runtime_add_layer_norm",
    "_runtime_attention",
    "_runtime_embedding",
    "_runtime_linear_module",
    "_runtime_mlp_module",
    "_runtime_qkv_projection",
    "_resolve_linear_module_tensors",
    "_runtime_sample_with_policies",
    "_runtime_temperature",
    "_runtime_topk_mask",
    "_runtime_topp_mask",
)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _prefer_eager_dense_module_path(reference: torch.Tensor) -> bool:
    if not isinstance(reference, torch.Tensor) or not reference.is_cuda:
        return False
    if reference.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if _env_flag("MODEL_STACK_SIMPLE_TRANSFORMER_FORCE_RUNTIME_DENSE", "0"):
        return False
    return cuda_arch_family(reference.device) in {"ampere", "ada"}


def runtime_status() -> dict[str, object]:
    native_available = bool(_native_available()) if callable(_native_available) else False
    native_ops = {}
    for op_name in ("linear", "attention_prefill", "attention_decode", "embedding", "mlp", "add_layer_norm", "sampling"):
        native_ops[op_name] = bool(_native_has_op(op_name)) if callable(_native_has_op) else False
    return {
        "helpers_available": all(globals()[name] is not None for name in _RUNTIME_HELPER_NAMES),
        "native_available": native_available,
        "native_ops": native_ops,
    }


@contextmanager
def runtime_mode(enabled: bool):
    if enabled:
        yield
        return
    saved = {name: globals()[name] for name in _RUNTIME_HELPER_NAMES}
    try:
        for name in _RUNTIME_HELPER_NAMES:
            globals()[name] = None
        yield
    finally:
        for name, value in saved.items():
            globals()[name] = value


def runtime_linear(module, x: torch.Tensor) -> torch.Tensor:
    if _prefer_eager_dense_module_path(x):
        return module(x)
    if _runtime_linear_module is not None:
        return _runtime_linear_module(x, module)
    return module(x)


def runtime_embedding(module, indices: torch.Tensor) -> torch.Tensor:
    if _prefer_eager_dense_module_path(module.weight):
        return module(indices)
    if _runtime_embedding is not None:
        return _runtime_embedding(module.weight, indices, padding_idx=module.padding_idx)
    return F.embedding(indices, module.weight, padding_idx=module.padding_idx)


def _shared_qkv_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return q is k and k is v


def runtime_qkv(q_module, k_module, v_module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    if _prefer_eager_dense_module_path(q):
        return q_module(q), k_module(k), v_module(v)
    if (
        _runtime_qkv_projection is not None
        and _resolve_linear_module_tensors is not None
        and _shared_qkv_inputs(q, k, v)
    ):
        q_weight, q_bias = _resolve_linear_module_tensors(q_module, reference=q)
        k_weight, k_bias = _resolve_linear_module_tensors(k_module, reference=q)
        v_weight, v_bias = _resolve_linear_module_tensors(v_module, reference=q)
        return _runtime_qkv_projection(
            q,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
        )
    return (
        runtime_linear(q_module, q),
        runtime_linear(k_module, k),
        runtime_linear(v_module, v),
    )


def _prepare_attention_mask(mask: torch.Tensor | None, *, device: torch.device) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        masked = ~mask.to(device=device, dtype=torch.bool)
        additive = torch.zeros(masked.shape, device=device, dtype=torch.float32)
        additive = additive.masked_fill(masked, float("-inf"))
    else:
        additive = mask.to(device=device, dtype=torch.float32)
    if additive.ndim == 2:
        return additive.unsqueeze(0).unsqueeze(0)
    if additive.ndim == 3:
        return additive.unsqueeze(1)
    if additive.ndim == 4:
        return additive
    raise ValueError(f"Unsupported attention mask rank: {additive.ndim}")


def runtime_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    prepared_mask = _prepare_attention_mask(mask, device=q.device)
    if _runtime_attention is not None:
        return _runtime_attention(
            q,
            k,
            v,
            attn_mask=prepared_mask,
            is_causal=bool(is_causal and prepared_mask is None),
        )

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if prepared_mask is not None:
        scores = scores + prepared_mask.to(device=scores.device, dtype=scores.dtype)
    if is_causal and prepared_mask is None and q.shape[-2] > 1:
        causal = torch.triu(
            torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal.view(1, 1, q.shape[-2], k.shape[-2]), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def runtime_add_layer_norm(x: torch.Tensor, update: torch.Tensor, norm) -> torch.Tensor:
    if _prefer_eager_dense_module_path(x):
        return norm(x + update)
    if _runtime_add_layer_norm is not None:
        _, normalized = _runtime_add_layer_norm(
            x,
            update,
            norm.weight,
            norm.bias,
            residual_scale=1.0,
            eps=float(norm.eps),
        )
        return normalized
    combined = x + update
    return norm(combined)


def runtime_mlp(w_in_module, w_out_module, x: torch.Tensor, *, activation: str = "relu") -> torch.Tensor:
    if _prefer_eager_dense_module_path(x):
        hidden = w_in_module(x)
        act = str(activation).lower()
        if act == "relu":
            hidden = F.relu(hidden)
        elif act == "silu":
            hidden = F.silu(hidden)
        else:
            hidden = F.gelu(hidden)
        return w_out_module(hidden)
    if _runtime_mlp_module is not None:
        return _runtime_mlp_module(
            x,
            w_in_module,
            w_out_module,
            activation=activation,
            gated=False,
        )
    hidden = runtime_linear(w_in_module, x)
    act = str(activation).lower()
    if act == "relu":
        hidden = F.relu(hidden)
    elif act == "silu":
        hidden = F.silu(hidden)
    else:
        hidden = F.gelu(hidden)
    return runtime_linear(w_out_module, hidden)


def runtime_temperature(logits: torch.Tensor, tau: float) -> torch.Tensor:
    if _runtime_temperature is not None:
        return _runtime_temperature(logits, float(tau))
    return logits / max(float(tau), 1e-8)


def runtime_filter_logits(
    logits: torch.Tensor,
    *,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("inf"),
) -> torch.Tensor:
    filtered = logits.clone()
    combined_mask = torch.zeros_like(filtered, dtype=torch.bool)
    if int(top_k) > 0:
        if _runtime_topk_mask is not None:
            combined_mask |= _runtime_topk_mask(filtered, int(top_k))
        else:
            kth = torch.topk(filtered, int(top_k), dim=-1).values[..., -1, None]
            combined_mask |= filtered < kth
    if float(top_p) > 0.0:
        if _runtime_topp_mask is not None:
            combined_mask |= _runtime_topp_mask(filtered, float(top_p))
        else:
            probs = torch.softmax(filtered, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_remove = cumulative_probs > float(top_p)
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            top_p_mask = torch.zeros_like(sorted_remove, dtype=torch.bool)
            top_p_mask.scatter_(-1, sorted_indices, sorted_remove)
            combined_mask |= top_p_mask
    if combined_mask.any():
        filtered = filtered.masked_fill(combined_mask, float(filter_value))
    return filtered


def sample_next_token(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    if _runtime_sample_with_policies is not None:
        next_token = _runtime_sample_with_policies(
            logits,
            token_ids,
            do_sample=True,
            temperature=float(temperature),
            top_k=None if top_k in (None, 0) else int(top_k),
            top_p=None if top_p is None or float(top_p) <= 0.0 else float(top_p),
        )
        return next_token.unsqueeze(-1) if next_token.ndim == 1 else next_token

    scaled = runtime_temperature(logits, float(temperature))
    filtered = runtime_filter_logits(
        scaled,
        top_k=0 if top_k is None else int(top_k),
        top_p=0.0 if top_p is None else float(top_p),
    )
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import torch

from transformer import _model_stack
from transformer.model import SimpleTransformer


@dataclass
class BenchmarkStats:
    mode: str
    mean_ms: float
    median_ms: float
    p95_ms: float
    tokens_per_second: float


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch fallback vs model-stack runtime.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Target device: auto, cpu, cuda, or an explicit device such as cuda:1.",
    )
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return ordered[idx]


def _run_mode(
    model: SimpleTransformer,
    x: torch.Tensor,
    *,
    runtime_enabled: bool,
    warmup: int,
    iters: int,
    device: torch.device,
) -> tuple[BenchmarkStats, torch.Tensor]:
    latencies_ms: list[float] = []
    output = None

    with _model_stack.runtime_mode(enabled=runtime_enabled):
        with torch.inference_mode():
            for _ in range(int(warmup)):
                output = model(x)
            _sync(device)

            for _ in range(int(iters)):
                _sync(device)
                start = time.perf_counter()
                output = model(x)
                _sync(device)
                latencies_ms.append((time.perf_counter() - start) * 1000.0)

    tokens = int(x.shape[0] * x.shape[1])
    mean_ms = statistics.mean(latencies_ms)
    return (
        BenchmarkStats(
            mode="model-stack" if runtime_enabled else "pytorch",
            mean_ms=mean_ms,
            median_ms=statistics.median(latencies_ms),
            p95_ms=_percentile(latencies_ms, 0.95),
            tokens_per_second=tokens / (mean_ms / 1000.0),
        ),
        output,
    )


def main():
    args = _parse_args()
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)

    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    model = SimpleTransformer(
        vocab_size=int(args.vocab_size),
        d_model=int(args.d_model),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        d_ff=int(args.d_ff),
        dropout=0.0,
    ).to(device=device, dtype=dtype)
    model.eval()

    x = torch.randint(0, int(args.vocab_size), (int(args.batch_size), int(args.seq_len)), device=device)

    pytorch_stats, pytorch_output = _run_mode(
        model,
        x,
        runtime_enabled=False,
        warmup=int(args.warmup),
        iters=int(args.iters),
        device=device,
    )
    model_stack_stats, model_stack_output = _run_mode(
        model,
        x,
        runtime_enabled=True,
        warmup=int(args.warmup),
        iters=int(args.iters),
        device=device,
    )

    max_abs_diff = float((pytorch_output.float() - model_stack_output.float()).abs().max().item())
    speedup = float(pytorch_stats.mean_ms / model_stack_stats.mean_ms) if model_stack_stats.mean_ms > 0 else 0.0

    payload = {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "config": {
            "batch_size": int(args.batch_size),
            "seq_len": int(args.seq_len),
            "vocab_size": int(args.vocab_size),
            "d_model": int(args.d_model),
            "num_heads": int(args.num_heads),
            "num_layers": int(args.num_layers),
            "d_ff": int(args.d_ff),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
        },
        "runtime_status": _model_stack.runtime_status(),
        "stats": [asdict(pytorch_stats), asdict(model_stack_stats)],
        "speedup": speedup,
        "max_abs_diff": max_abs_diff,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print("Runtime Benchmark")
    print(f"device: {payload['device']}")
    print(f"dtype: {payload['dtype']}")
    print(f"config: batch={args.batch_size} seq={args.seq_len} vocab={args.vocab_size} d_model={args.d_model} heads={args.num_heads} layers={args.num_layers} d_ff={args.d_ff}")
    print(f"runtime helpers available: {payload['runtime_status']['helpers_available']}")
    print(f"native available: {payload['runtime_status']['native_available']}")
    print()
    print(f"{'mode':<12} {'mean_ms':>10} {'median_ms':>10} {'p95_ms':>10} {'tokens/s':>12}")
    for row in payload["stats"]:
        print(
            f"{row['mode']:<12} "
            f"{row['mean_ms']:>10.3f} "
            f"{row['median_ms']:>10.3f} "
            f"{row['p95_ms']:>10.3f} "
            f"{row['tokens_per_second']:>12.1f}"
        )
    print()
    print(f"speedup: {speedup:.2f}x")
    print(f"max_abs_diff: {max_abs_diff:.6g}")


if __name__ == "__main__":
    main()

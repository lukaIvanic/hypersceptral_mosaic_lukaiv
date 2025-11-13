from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

import src.metrics as metrics


MetricCallable = Callable[[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]], float]


METRIC_PAIRS: Dict[str, Tuple[MetricCallable, MetricCallable]] = {
    "sam": (metrics.sam, metrics.sam_fast),
    "sid": (metrics.sid, metrics.sid_fast),
    "ergas": (metrics.ergas, metrics.ergas_fast),
    "psnr_srgb": (metrics.psnr_srgb, metrics.psnr_srgb_fast),
    "ssim_srgb": (metrics.ssim_srgb, metrics.ssim_srgb_fast),
    "deltae00": (metrics.deltae00, metrics.deltae00_fast),
}

DTYPE_CHOICES: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class MetricResult:
    name: str
    baseline_ms: List[float]
    optimized_ms: List[float]
    abs_diffs: List[float]
    rel_diffs: List[float]

    def summary(self) -> Dict[str, float]:
        baseline_mean = sum(self.baseline_ms) / max(len(self.baseline_ms), 1)
        optimized_mean = sum(self.optimized_ms) / max(len(self.optimized_ms), 1)
        best_speedup = baseline_mean / optimized_mean if optimized_mean > 0 else float("inf")
        return {
            "baseline_ms": baseline_mean,
            "optimized_ms": optimized_mean,
            "speedup": best_speedup,
            "max_abs_diff": max(self.abs_diffs) if self.abs_diffs else 0.0,
            "max_rel_diff": max(self.rel_diffs) if self.rel_diffs else 0.0,
        }


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline metrics with optimized implementations on dummy data."
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for dummy tensors.")
    parser.add_argument("--channels", type=int, default=61, help="Spectral channels per sample.")
    parser.add_argument("--height", type=int, default=1024, help="Image height.")
    parser.add_argument("--width", type=int, default=1024, help="Image width.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=sorted(DTYPE_CHOICES.keys()),
        help="Tensor dtype for the benchmark.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device(),
        help="Compute device (cpu, cuda, cuda:0, etc.).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=list(METRIC_PAIRS.keys()),
        help="Metrics to benchmark. Defaults to all optimized metrics.",
    )
    parser.add_argument("--iterations", type=int, default=5, help="Timed iterations per metric.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations to ignore in stats.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dummy tensors.")
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for parity check.",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-4,
        help="Relative tolerance for parity check.",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to emit JSON summary.",
    )
    return parser.parse_args()


def make_dummy_tensors(
    shape: Tuple[int, int, int, int],
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pred = torch.rand(shape, generator=generator, dtype=dtype)
    target = torch.rand(shape, generator=generator, dtype=dtype)
    return pred.to(device=device), target.to(device=device)


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_metric(
    fn: MetricCallable,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[float, float]:
    cache: Dict[str, torch.Tensor] = {}
    start = time.perf_counter()
    value = fn(pred, target, cache)
    elapsed = (time.perf_counter() - start) * 1000.0
    return float(value), elapsed


def run_metric_benchmark(
    name: str,
    baseline_fn: MetricCallable,
    optimized_fn: MetricCallable,
    pred: torch.Tensor,
    target: torch.Tensor,
    iterations: int,
    warmup: int,
) -> MetricResult:
    device = pred.device
    baseline_timings: List[float] = []
    optimized_timings: List[float] = []
    abs_diffs: List[float] = []
    rel_diffs: List[float] = []

    total_runs = warmup + iterations
    for run_idx in range(total_runs):
        maybe_sync(device)
        baseline_value, baseline_ms = measure_metric(baseline_fn, pred, target)
        maybe_sync(device)
        optimized_value, optimized_ms = measure_metric(optimized_fn, pred, target)
        maybe_sync(device)

        abs_diff = abs(baseline_value - optimized_value)
        denom = max(abs(baseline_value), 1e-12)
        rel_diff = abs_diff / denom

        if run_idx >= warmup:
            baseline_timings.append(baseline_ms)
            optimized_timings.append(optimized_ms)
            abs_diffs.append(abs_diff)
            rel_diffs.append(rel_diff)

    return MetricResult(
        name=name,
        baseline_ms=baseline_timings,
        optimized_ms=optimized_timings,
        abs_diffs=abs_diffs,
        rel_diffs=rel_diffs,
    )


def format_table(results: Iterable[MetricResult]) -> str:
    header = f"{'metric':<12} {'baseline_ms':>12} {'optimized_ms':>13} {'speedup':>9} {'max_abs':>10} {'max_rel':>10}"
    lines = [header, "-" * len(header)]
    for result in results:
        summary = result.summary()
        lines.append(
            f"{result.name:<12} "
            f"{summary['baseline_ms']:>12.3f} "
            f"{summary['optimized_ms']:>13.3f} "
            f"{summary['speedup']:>9.3f} "
            f"{summary['max_abs_diff']:>10.3e} "
            f"{summary['max_rel_diff']:>10.3e}"
        )
    return "\n".join(lines)


def validate_tolerances(
    results: Iterable[MetricResult],
    abs_tol: float,
    rel_tol: float,
) -> List[str]:
    failures: List[str] = []
    for result in results:
        for abs_diff, rel_diff in zip(result.abs_diffs, result.rel_diffs):
            if abs_diff > abs_tol and rel_diff > rel_tol:
                failures.append(result.name)
                break
    return failures


def main() -> None:
    args = parse_args()

    selected_metrics: List[str] = []
    for name in args.metrics:
        if name not in METRIC_PAIRS:
            raise ValueError(f"Unknown metric '{name}'. Available: {list(METRIC_PAIRS.keys())}")
        selected_metrics.append(name)

    dtype = DTYPE_CHOICES[args.dtype]
    device = torch.device(args.device)

    pred, target = make_dummy_tensors(
        (args.batch_size, args.channels, args.height, args.width),
        dtype=dtype,
        device=device,
        seed=args.seed,
    )

    results: List[MetricResult] = []
    for name in selected_metrics:
        baseline_fn, optimized_fn = METRIC_PAIRS[name]
        result = run_metric_benchmark(
            name=name,
            baseline_fn=baseline_fn,
            optimized_fn=optimized_fn,
            pred=pred,
            target=target,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        results.append(result)

    print(format_table(results))
    failures = validate_tolerances(results, abs_tol=args.abs_tol, rel_tol=args.rel_tol)
    if failures:
        print("\nWARNING: metrics outside tolerances:", ", ".join(failures))
    else:
        print("\nAll metrics within tolerances.")

    if args.json_output:
        payload = {result.name: result.summary() for result in results}
        with open(args.json_output, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)


if __name__ == "__main__":
    main()


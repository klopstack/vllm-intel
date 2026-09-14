#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness, stress, and latency harness for XPU TP=2 all-reduce."""

import argparse
import hashlib
import inspect
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist

BLOCK = 1024
MAX_NUMEL = 262144
DTYPES = (torch.float16, torch.bfloat16)
SIZES = (1024, 5120, 35840, 262144)
FP16_PERF_SIZES = (
    1024,
    2048,
    5120,
    10240,
    15360,
    20480,
    30720,
    35840,
    40960,
    65536,
    262144,
)
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / ("vllm/distributed/device_communicators/xpu_triton_all_reduce.py")
ENV_KEYS = (
    "PYTHONPATH",
    "ZE_AFFINITY_MASK",
    "FI_PROVIDER_PATH",
    "FI_PROVIDER",
    "LD_LIBRARY_PATH",
    "ZE_FLAT_DEVICE_HIERARCHY",
    "TRITON_INTEL_DEVICE_ARCH",
    "VLLM_XPU_TRITON_ALLREDUCE",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mismatch(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    equal = (actual == expected) | (torch.isnan(actual) & torch.isnan(expected))
    signed_zero_mismatch = (
        (actual == 0)
        & (expected == 0)
        & (torch.signbit(actual) != torch.signbit(expected))
    )
    return ((~equal) | signed_zero_mismatch).any()


def aggregate_failures(local_failures: int, device: torch.device) -> int:
    value = torch.tensor(local_failures, dtype=torch.int64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return int(value.cpu().item())


def cpu_inputs(kind: str, size: int, dtype: torch.dtype, seed: int):
    if kind == "mantissa":
        step = 2.0**-10 if dtype == torch.float16 else 2.0**-7
        a = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=dtype)
        b = torch.tensor([step, 2 * step, -step, -2 * step], dtype=dtype)
    elif kind == "extreme":
        limit = 60000.0 if dtype == torch.float16 else 3.0e38
        a = torch.tensor([limit, -limit, limit, -limit], dtype=dtype)
        b = torch.tensor([-limit, limit, 1.0, -1.0], dtype=dtype)
    elif kind == "nonfinite":
        a = torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0])
        b = torch.tensor([1.0, float("inf"), -float("inf"), float("nan")])
        a, b = a.to(dtype), b.to(dtype)
    elif kind == "subnormal":
        tiny = torch.finfo(dtype).smallest_normal / 2
        a = torch.tensor([tiny, -tiny, tiny, -tiny], dtype=dtype)
        b = torch.tensor([tiny, tiny, -tiny, -tiny], dtype=dtype)
    else:
        generator_a = torch.Generator().manual_seed(seed)
        generator_b = torch.Generator().manual_seed(seed ^ 0x5A5A5A5A)
        step = 2.0**-10 if dtype == torch.float16 else 2.0**-7
        integers_a = torch.randint(-1024, 1024, (size,), generator=generator_a)
        integers_b = torch.randint(-1024, 1024, (size,), generator=generator_b)
        a = (integers_a.float() * step).to(dtype)
        b = (integers_b.float() * step).to(dtype)
        return a, b
    repeats = (size + a.numel() - 1) // a.numel()
    return a.repeat(repeats)[:size], b.repeat(repeats)[:size]


def check_one(
    communicator,
    a: torch.Tensor,
    b: torch.Tensor,
    rank: int,
    device: torch.device,
    check_preallocated: bool = True,
) -> int:
    expected_cpu = (a + b).clone()
    expected = expected_cpu.to(device)
    source = (a if rank == 0 else b).to(device)
    unchanged = source.clone()
    actual = communicator.all_reduce(source)
    baseline = source.clone()
    dist.all_reduce(baseline)
    failures = int(mismatch(actual, expected).item())
    failures += int(mismatch(baseline, expected).item())
    failures += int(mismatch(actual, baseline).item())
    failures += int(mismatch(source, unchanged).item())
    if check_preallocated:
        output = torch.full_like(source, float("nan"))
        communicator.ca_comm.all_reduce(source, output)
        failures += int(mismatch(output, expected).item())
    return failures


def edge_correctness(communicator, rank: int, device: torch.device) -> dict:
    failures = 0
    cases = 0
    for dtype in DTYPES:
        for size in SIZES:
            for kind in ("mantissa", "extreme", "nonfinite", "subnormal", "random"):
                a, b = cpu_inputs(kind, size, dtype, 9000 + size)
                failures += check_one(communicator, a, b, rank, device)
                cases += 1
        a, b = cpu_inputs("mantissa", 5120, dtype, 777)
        expected = (a + b).to(device)
        in_place = (a if rank == 0 else b).to(device)
        communicator.ca_comm.all_reduce(in_place, in_place)
        failures += int(mismatch(in_place, expected).item())
        cases += 1
    total = aggregate_failures(failures, device)
    if total:
        raise AssertionError(f"edge correctness failures across ranks: {total}")
    return {"calls_per_rank": cases, "aggregated_failures": total}


def randomized_correctness(
    communicator,
    rank: int,
    device: torch.device,
    iterations: int,
) -> dict:
    generator = torch.Generator().manual_seed(20260914)
    failures = 0
    counts = {str(dtype): 0 for dtype in DTYPES}
    for iteration in range(iterations):
        dtype = DTYPES[iteration % len(DTYPES)]
        if iteration < MAX_NUMEL // BLOCK:
            size = iteration + 1
        else:
            size = int(
                torch.randint(1, MAX_NUMEL // BLOCK + 1, (), generator=generator)
            )
        size *= BLOCK
        a, b = cpu_inputs("random", size, dtype, 100000 + iteration)
        failures += check_one(
            communicator,
            a,
            b,
            rank,
            device,
            check_preallocated=iteration % 10 == 0,
        )
        counts[str(dtype)] += 1
    torch.xpu.synchronize()
    total = aggregate_failures(failures, device)
    if total:
        raise AssertionError(f"randomized correctness failures: {total}")
    return {
        "calls_per_rank": iterations,
        "dtype_counts": counts,
        "aggregated_failures": total,
        "seed": 20260914,
        "size_blocks": "deterministic 1..256, then uniform random 1..256",
    }


def ordering_stress(
    communicator,
    rank: int,
    device: torch.device,
    iterations: int,
) -> dict:
    inputs = {dtype: torch.empty(5120, dtype=dtype, device=device) for dtype in DTYPES}
    outputs = {dtype: torch.empty(5120, dtype=dtype, device=device) for dtype in DTYPES}
    bad = torch.zeros((), dtype=torch.int64, device=device)
    counts = {str(dtype): 0 for dtype in DTYPES}
    for iteration in range(iterations):
        dtype = DTYPES[iteration % len(DTYPES)]
        base = float(iteration % 31 - 15)
        inputs[dtype].fill_(base + rank)
        outputs[dtype].fill_(float("nan"))
        communicator.ca_comm.all_reduce(inputs[dtype], outputs[dtype])
        bad += (outputs[dtype] != (2 * base + 1)).any().to(torch.int64)
        counts[str(dtype)] += 1
    torch.xpu.synchronize()
    total = aggregate_failures(int(bad.cpu().item()), device)
    if total:
        raise AssertionError(f"ordering stress failures: {total}")
    return {
        "calls_per_rank": iterations,
        "dtype_counts": counts,
        "aggregated_failures": total,
    }


def graph_stress(
    communicator,
    rank: int,
    cpu_group,
    device: torch.device,
    replays: int,
) -> dict:
    fp_input = torch.zeros(5120, dtype=torch.float16, device=device)
    bf_input = torch.zeros(35840, dtype=torch.bfloat16, device=device)
    fp_output = torch.empty_like(fp_input)
    bf_output = torch.empty_like(bf_input)
    for _ in range(5):
        communicator.ca_comm.all_reduce(fp_input, fp_output)
        communicator.ca_comm.all_reduce(bf_input, bf_output)
    torch.xpu.synchronize()
    dist.barrier(group=cpu_group)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        communicator.ca_comm.all_reduce(fp_input, fp_output)
        communicator.ca_comm.all_reduce(bf_input, bf_output)
    torch.xpu.synchronize()
    bad = torch.zeros((), dtype=torch.int64, device=device)
    for replay in range(replays):
        fp_base = float(replay % 29 - 14)
        bf_base = float(replay % 23 - 11)
        fp_input.fill_(fp_base + rank)
        bf_input.fill_(bf_base + rank)
        fp_output.fill_(float("nan"))
        bf_output.fill_(float("nan"))
        graph.replay()
        bad += (fp_output != (2 * fp_base + 1)).any().to(torch.int64)
        bad += (bf_output != (2 * bf_base + 1)).any().to(torch.int64)
    torch.xpu.synchronize()
    total = aggregate_failures(int(bad.cpu().item()), device)
    if total:
        raise AssertionError(f"graph stress failures: {total}")
    return {
        "graph_replays_per_rank": replays,
        "custom_calls_per_replay": 2,
        "changed_inputs": True,
        "poisoned_outputs": True,
        "aggregated_failures": total,
    }


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    median = statistics.median(ordered)
    return {
        "samples": samples,
        "median_us": median,
        "min_us": min(ordered),
        "max_us": max(ordered),
        "p10_us": ordered[int(0.1 * (len(ordered) - 1))],
        "p90_us": ordered[int(0.9 * (len(ordered) - 1))],
        "mad_us": statistics.median(abs(value - median) for value in ordered),
    }


def rank_max_us(local_us: float, device: torch.device) -> float:
    value = torch.tensor(local_us, dtype=torch.float64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.cpu().item())


def device_barrier(device: torch.device) -> None:
    marker = torch.ones((), dtype=torch.int32, device=device)
    dist.all_reduce(marker)
    torch.xpu.synchronize()


def event_time(operation, repeats: int, device: torch.device) -> float:
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    device_barrier(device)
    start.record()
    for _ in range(repeats):
        operation()
    end.record()
    end.synchronize()
    return rank_max_us(start.elapsed_time(end) * 1000 / repeats, device)


def capture_operation(operation, cpu_group):
    for _ in range(20):
        operation()
    torch.xpu.synchronize()
    dist.barrier(group=cpu_group)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        output = operation()
    torch.xpu.synchronize()
    return graph, output


def benchmark_case(communicator, rank: int, cpu_group, device, dtype, size) -> dict:
    source = torch.full((size,), 0.25 + rank, dtype=dtype, device=device)
    custom_output = torch.empty_like(source)
    ccl_output = torch.empty_like(source)

    def custom_wrapper():
        return communicator.all_reduce(source)

    def oneccl_wrapper():
        output = source.clone()
        dist.all_reduce(output)
        return output

    def custom_preallocated():
        return communicator.ca_comm.all_reduce(source, custom_output)

    def oneccl_preallocated_copy():
        ccl_output.copy_(source)
        dist.all_reduce(ccl_output)
        return ccl_output

    operations = {
        "custom_wrapper": custom_wrapper,
        "oneccl_clone_wrapper": oneccl_wrapper,
        "custom_preallocated": custom_preallocated,
        "oneccl_preallocated_copy": oneccl_preallocated_copy,
    }
    expected = torch.full_like(source, 1.5)
    for operation in operations.values():
        actual = operation()
        torch.xpu.synchronize()
        if mismatch(actual, expected).item():
            raise AssertionError(f"benchmark warmup incorrect for {dtype=} {size=}")
    repeat = 100 if size <= 5120 else 20 if size <= 35840 else 8
    eager = {name: [] for name in operations}
    for round_index in range(21):
        order = list(operations)
        if round_index % 2:
            order.reverse()
        for name in order:
            dist.barrier(group=cpu_group)
            eager[name].append(event_time(operations[name], repeat, device))

    graphs = {}
    graph_outputs = {}
    for name, operation in operations.items():
        graph, output = capture_operation(operation, cpu_group)
        graphs[name] = graph
        graph_outputs[name] = output
        output.fill_(float("nan"))
        graph.replay()
        torch.xpu.synchronize()
        if mismatch(output, expected).item():
            raise AssertionError(f"graph output incorrect for {name} {dtype=} {size=}")
        for _ in range(100):
            graph.replay()
    torch.xpu.synchronize()
    graph_samples = {name: [] for name in operations}
    for round_index in range(21):
        order = list(operations)
        if round_index % 2:
            order.reverse()
        for name in order:
            dist.barrier(group=cpu_group)
            graph_samples[name].append(event_time(graphs[name].replay, repeat, device))
    return {
        "dtype": str(dtype),
        "numel": size,
        "bytes_per_rank": size * torch.empty((), dtype=dtype).element_size(),
        "repeat_per_sample": repeat,
        "eager": {name: summarize(values) for name, values in eager.items()},
        "graph": {name: summarize(values) for name, values in graph_samples.items()},
        "boundary_note": {
            "custom_wrapper": "actual XpuCommunicator allocation + custom collective",
            "oneccl_clone_wrapper": (
                "XpuCommunicator fallback-equivalent clone + oneCCL"
            ),
            "custom_preallocated": "kernel path with caller-provided output",
            "oneccl_preallocated_copy": "caller output copy + in-place oneCCL",
            "graph": (
                "one collective per graph replay; latency includes the host "
                "graph.replay submission"
            ),
        },
    }


def performance(communicator, rank: int, cpu_group, device) -> list[dict]:
    results = []
    for dtype in DTYPES:
        sizes = FP16_PERF_SIZES if dtype == torch.float16 else SIZES
        for size in sizes:
            results.append(
                benchmark_case(communicator, rank, cpu_group, device, dtype, size)
            )
    return results


def write_rank_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def worker(args) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"xpu:{local_rank}")
    torch.xpu.set_device(local_rank)
    dist.init_process_group("xccl")
    cpu_group = dist.new_group(backend="gloo")
    properties = torch.xpu.get_device_properties(device)
    expected_uuids = set(filter(None, args.expected_uuids.split(",")))
    if expected_uuids and str(properties.uuid) not in expected_uuids:
        raise RuntimeError(f"unexpected XPU: {properties}")
    from vllm.distributed.device_communicators.xpu_communicator import (
        XpuCommunicator,
    )
    from vllm.distributed.device_communicators.xpu_triton_all_reduce import (
        OneShotAllReduce,
    )

    imported_sources = {}
    for name, item in (
        ("XpuCommunicator", XpuCommunicator),
        ("OneShotAllReduce", OneShotAllReduce),
    ):
        path = Path(inspect.getfile(item)).resolve()
        imported_sources[name] = {"path": str(path), "sha256": file_sha256(path)}
    actual_source_hash = imported_sources["OneShotAllReduce"]["sha256"]
    if (
        args.expected_source_sha256
        and actual_source_hash != args.expected_source_sha256
    ):
        raise RuntimeError(
            f"imported product source hash mismatch: {actual_source_hash}"
        )

    communicator = XpuCommunicator(
        cpu_group,
        device=device,
        device_group=dist.group.WORLD,
        unique_name="tp:kernel_benchmark",
    )
    if communicator.ca_comm is None:
        raise RuntimeError("custom XPU all-reduce did not initialize")
    report = {
        "rank": rank,
        "local_rank": local_rank,
        "device": str(properties),
        "uuid": str(properties.uuid),
        "torch": torch.__version__,
        "imported_sources": imported_sources,
        "benchmark_path": str(Path(__file__).resolve()),
        "benchmark_sha256_at_start": file_sha256(Path(__file__).resolve()),
        "environment": {key: os.environ.get(key) for key in ENV_KEYS},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "phases": {},
    }
    rank_path = args.output_dir / f"rank{rank}.json"
    try:
        report["phases"]["edge"] = edge_correctness(communicator, rank, device)
        write_rank_report(rank_path, report)
        report["phases"]["randomized"] = randomized_correctness(
            communicator, rank, device, args.randomized_calls
        )
        write_rank_report(rank_path, report)
        report["phases"]["ordering"] = ordering_stress(
            communicator, rank, device, args.ordering_calls
        )
        write_rank_report(rank_path, report)
        report["phases"]["graph_stress"] = graph_stress(
            communicator, rank, cpu_group, device, args.graph_replays
        )
        write_rank_report(rank_path, report)
        report["performance"] = performance(communicator, rank, cpu_group, device)
        report["success"] = True
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_rank_report(rank_path, report)
        dist.destroy_process_group()


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def cleanup_process_group(pgid: int) -> bool:
    if not process_group_exists(pgid):
        return False
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.25)
    if process_group_exists(pgid):
        os.killpg(pgid, signal.SIGKILL)
        time.sleep(0.5)
    return process_group_exists(pgid)


def parent(args) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    source_hash = file_sha256(SOURCE)
    benchmark_hash = file_sha256(Path(__file__).resolve())
    if args.expected_source_sha256 and source_hash != args.expected_source_sha256:
        raise RuntimeError(f"product source hash mismatch: {source_hash}")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(Path(__file__).resolve()),
        "--worker",
        "--output-dir",
        str(args.output_dir),
        "--randomized-calls",
        str(args.randomized_calls),
        "--ordering-calls",
        str(args.ordering_calls),
        "--graph-replays",
        str(args.graph_replays),
        "--expected-source-sha256",
        args.expected_source_sha256,
        "--expected-uuids",
        args.expected_uuids,
    ]
    environment = os.environ.copy()
    environment["VLLM_XPU_TRITON_ALLREDUCE"] = "1"
    started = time.monotonic()
    log_path = args.output_dir / "torchrun.log"
    process = None
    timed_out = False
    alive_after_cleanup = False
    try:
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                cleanup_process_group(process.pid)
                returncode = process.wait(timeout=10)
    finally:
        if process is not None:
            alive_after_cleanup = cleanup_process_group(process.pid)
    report = {
        "command": command,
        "cwd": str(Path.cwd()),
        "returncode": returncode,
        "timed_out": timed_out,
        "process_group_alive_after_cleanup": alive_after_cleanup,
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": source_hash,
        "benchmark_sha256_at_start": benchmark_hash,
        "benchmark_sha256_at_end": file_sha256(Path(__file__)),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "git_status": subprocess.run(
            ["git", "status", "--short"], check=True, capture_output=True, text=True
        ).stdout.splitlines(),
        "environment": {key: environment.get(key) for key in ENV_KEYS},
        "rank_reports": [],
    }
    for rank in range(2):
        path = args.output_dir / f"rank{rank}.json"
        if path.exists():
            report["rank_reports"].append(json.loads(path.read_text()))
    actual_uuids = {item["uuid"] for item in report["rank_reports"]}
    expected_uuids = set(filter(None, args.expected_uuids.split(",")))
    report["success"] = (
        returncode == 0
        and not timed_out
        and not alive_after_cleanup
        and len(report["rank_reports"]) == 2
        and len(actual_uuids) == 2
        and (not expected_uuids or actual_uuids == expected_uuids)
        and all(item.get("success") for item in report["rank_reports"])
    )
    (args.output_dir / "run.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if not report["success"]:
        raise RuntimeError(f"dual-XPU harness failed; see {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--randomized-calls", type=int, default=20000)
    parser.add_argument("--ordering-calls", type=int, default=200000)
    parser.add_argument("--graph-replays", type=int, default=2000)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--expected-source-sha256", default="")
    parser.add_argument("--expected-uuids", default="")
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()

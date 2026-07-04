# ruff: noqa: I001
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .config_space import BASELINE_CONFIG, NAMED_CONFIGS, V4ServingConfig
except ImportError:  # direct script execution
    from config_space import BASELINE_CONFIG, NAMED_CONFIGS, V4ServingConfig


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class HarnessConstants:
    image: str = "batchgen:v4flash-blackwell-src"
    model: str = "deepseek-ai/DeepSeek-V4-Flash"
    repo_mount: str = "/workspace/batchgen"
    checkpoint_host_dir: str = "/home/leyang/v4flash_converted_mp4"
    checkpoint_container_dir: str = "/ckpt_mp4"
    hf_cache_host_dir: str = "/mnt/raid0nvme0/public/huggingface"
    hf_cache_container_dir: str = "/root/.cache/huggingface"
    hf_snapshot_dir: str = (
        "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/"
        "snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136"
    )
    gpus: tuple[int, ...] = (0, 1, 2, 3)
    listen_port: int = 12345
    dist_init_addr: str = "localhost:12457"
    temp_root: str = "/tmp/autoresearch_v4"
    shm_size: str = "400g"
    warmup_requests: tuple[tuple[int, int], ...] = ((32, 4), (256, 8))
    prefill_prompt_tokens: tuple[int, ...] = (2048, 4096, 8192)
    decode_prompt: str = "The capital of France is"
    decode_output_tokens: int = 128
    decode_timeout_s: int = 2400
    accuracy_max_prompts: int = 5
    accuracy_max_decoding_length: int = 256
    accuracy_floor: float = 0.20
    shm_clean_threshold_bytes: int = 4 * 1024 * 1024 * 1024


CONST = HarnessConstants()
LOG_ROOT = Path(CONST.temp_root)
RESULTS_FILE = THIS_DIR / "results.tsv"


def _run(
    args: list[str],
    *,
    timeout: int | None = None,
    check: bool = True,
    capture_output: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        args,
        cwd=REPO_ROOT,
        text=True,
        capture_output=capture_output,
        timeout=timeout,
        env=env,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(args)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _safe_note(text: str) -> str:
    return text.replace("\t", " ").replace("\n", " ").strip()


def _repeat_sentence(sentence: str, target_tokens: int) -> str:
    target_words = max(1, int(target_tokens * 0.75))
    words = sentence.split()
    reps = max(1, (target_words + len(words) - 1) // len(words))
    return " ".join(words * reps)


def _prompt_with_target_tokens(target_tokens: int) -> str:
    base = (
        "Discuss the historical, economic, and cultural factors that shaped major "
        "civilizations, including trade routes, geography, institutions, and technological change."
    )
    return _repeat_sentence(base, target_tokens)


def _read_new_log_text(log_path: Path, offset: int) -> str:
    if not log_path.exists():
        return ""
    with log_path.open("rb") as fh:
        fh.seek(offset)
        return fh.read().decode("utf-8", errors="replace")


def _parse_latest_metric(text: str, pattern: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    return float(raw)


def _query_gpu_memory() -> list[dict[str, int]]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ).stdout
    rows: list[dict[str, int]] = []
    for line in out.strip().splitlines():
        idx_s, used_s, total_s = [part.strip() for part in line.split(",")]
        idx = int(idx_s)
        if idx in CONST.gpus:
            rows.append(
                {
                    "index": idx,
                    "memory_used_mb": int(used_s),
                    "memory_total_mb": int(total_s),
                }
            )
    return rows


def _ensure_gpus_idle() -> None:
    # <=64MB with no compute app = driver residual from a killed context, not a tenant.
    rows = _query_gpu_memory()
    busy = [row for row in rows if row["memory_used_mb"] > 64]
    if busy:
        raise RuntimeError(f"GPUs not idle before launch/after cleanup: {busy}")


def _gpu_bus_map() -> dict[str, int]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    ).stdout
    mapping: dict[str, int] = {}
    for line in out.strip().splitlines():
        idx_s, bus = [part.strip() for part in line.split(",")]
        idx = int(idx_s)
        if idx in CONST.gpus:
            mapping[bus.lower()] = idx
    return mapping


def _leftover_compute_pids() -> list[int]:
    proc = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_bus_id,pid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    bus_map = _gpu_bus_map()
    pids: list[int] = []
    for line in proc.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        bus_id, pid_s = parts
        if bus_id.lower() in bus_map:
            pids.append(int(pid_s))
    return sorted(set(pids))


def _clear_shm_leaks() -> None:
    _run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            "/dev/shm:/hostshm",
            CONST.image,
            "bash",
            "-lc",
            "rm -f /hostshm/shm_* /hostshm/batchgen_host_kv_cache",
        ],
        timeout=120,
    )


def _verify_shm_clean() -> None:
    leaked_names = [
        str(path)
        for path in Path("/dev/shm").glob("shm_*")
        if path.exists()
    ]
    if Path("/dev/shm/batchgen_host_kv_cache").exists():
        leaked_names.append("/dev/shm/batchgen_host_kv_cache")
    usage = shutil.disk_usage("/dev/shm")
    if leaked_names:
        raise RuntimeError(f"/dev/shm leak remains after cleanup: {leaked_names[:8]}")
    if usage.used > CONST.shm_clean_threshold_bytes:
        raise RuntimeError(
            f"/dev/shm still too full after cleanup: used={usage.used} bytes"
        )


def _docker_container_exists(name: str) -> bool:
    proc = _run(
        ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.ID}}"],
        check=False,
    )
    return bool(proc.stdout.strip())


def cleanup_experiment(container_name: str) -> None:
    errors: list[str] = []
    if _docker_container_exists(container_name):
        proc = _run(
            ["docker", "exec", container_name, "pkill", "-9", "-f", "launch_http_server"],
            check=False,
            timeout=30,
        )
        if proc.returncode not in (0, 1):
            errors.append(f"pkill failed: {proc.stderr.strip()}")
        proc = _run(["docker", "rm", "-f", container_name], check=False, timeout=120)
        if proc.returncode != 0:
            errors.append(f"docker rm -f failed: {proc.stderr.strip()}")

    for pid in _leftover_compute_pids():
        proc = _run(["kill", "-9", str(pid)], check=False, timeout=10)
        if proc.returncode != 0:
            errors.append(f"failed to kill leftover pid {pid}: {proc.stderr.strip()}")

    try:
        _clear_shm_leaks()
    except Exception as exc:  # pragma: no cover - operational cleanup path
        errors.append(str(exc))

    try:
        _ensure_gpus_idle()
        _verify_shm_clean()
    except Exception as exc:
        errors.append(str(exc))

    if errors:
        raise RuntimeError("cleanup failed: " + " | ".join(_safe_note(err) for err in errors))


def _server_log_path(tag: str) -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    return LOG_ROOT / f"{tag}.server.log"


def _health_check(base_url: str, timeout_s: int = 10) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


def _wait_for_server(log_path: Path, base_url: str, startup_timeout_s: int) -> None:
    deadline = time.time() + startup_timeout_s
    last_text = ""
    fatal_markers = (
        "ProcessExitedException",
        "Traceback",
        "No such file or directory",
        "ModuleNotFoundError",
        "RuntimeError:",
    )
    while time.time() < deadline:
        if log_path.exists():
            last_text = log_path.read_text(encoding="utf-8", errors="replace")
            if "Uvicorn running" in last_text and _health_check(base_url):
                return
            if any(marker in last_text for marker in fatal_markers):
                raise RuntimeError(f"server failed during startup:\n{last_text[-4000:]}")
        time.sleep(5)
    raise RuntimeError(f"timed out waiting for server startup:\n{last_text[-4000:]}")


def _post_json(url: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail[:500]}") from exc


def _send_inference(
    base_url: str,
    *,
    prompts: list[str],
    max_output_len: int,
    timeout_s: int,
) -> dict[str, Any]:
    return _post_json(
        f"{base_url}/v1/inference",
        {
            "prompts": prompts,
            "max_output_len": max_output_len,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        timeout_s,
    )


def _measure_request(log_path: Path, request_fn: Any) -> tuple[dict[str, Any], str]:
    offset = log_path.stat().st_size if log_path.exists() else 0
    result = request_fn()
    segment = _read_new_log_text(log_path, offset)
    return result, segment


def _parse_generation_metrics(segment: str) -> dict[str, float | None]:
    return {
        "prefill_ttft_s": _parse_latest_metric(segment, r"Prefill total time:\s*([0-9.]+)s"),
        "decode_tok_s": _parse_latest_metric(segment, r"Decode throughput:\s*([0-9.,]+)\s*tokens/s"),
    }


def _run_accuracy_guard(base_url: str, tag: str) -> dict[str, Any]:
    out_path = LOG_ROOT / f"{tag}.accuracy.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tests/e2e/v4flash_mmlu_pro_test/v4flash_mmlu_pro_batch_test.py"),
        "--hugging_face_checkpoint",
        CONST.model,
        "--base_url",
        base_url,
        "--max_prompts",
        str(CONST.accuracy_max_prompts),
        "--max_decoding_length",
        str(CONST.accuracy_max_decoding_length),
        "--temperature",
        "0.0",
        "--output",
        str(out_path),
    ]
    _run(cmd, timeout=7200)
    report = json.loads(out_path.read_text(encoding="utf-8"))
    total = int(report.get("total", 0))
    extraction_failures = int(report.get("extraction_failures", 0))
    accuracy = float(report.get("accuracy", 0.0))
    extraction_failure_rate = extraction_failures / total if total else 1.0
    guard_ok = bool(
        total >= CONST.accuracy_max_prompts
        and accuracy >= CONST.accuracy_floor
        and extraction_failure_rate < 1.0
    )
    return {
        "total": total,
        "correct": int(report.get("correct", 0)),
        "accuracy": accuracy,
        "extraction_failures": extraction_failures,
        "extraction_failure_rate": extraction_failure_rate,
        "pass": guard_ok,
    }


def _append_results_row(results_file: Path, row: dict[str, Any]) -> None:
    header = "tag\tconfig\tdecode_tok_s\tprefill_ttft_s\taccuracy_guard\tvram_mb\tstatus\tnotes\n"
    if not results_file.exists():
        results_file.write_text(header, encoding="utf-8")
    values = [
        str(row["tag"]),
        row["config"],
        str(row["decode_tok_s"]),
        str(row["prefill_ttft_s"]),
        row["accuracy_guard"],
        str(row["vram_mb"]),
        str(row["status"]),
        _safe_note(str(row["notes"])),
    ]
    with results_file.open("a", encoding="utf-8") as fh:
        fh.write("\t".join(values) + "\n")


def _container_name(tag: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "-", tag.strip()) or "baseline"
    return f"autoresearch-v4-{clean}"


def _docker_run_container(container_name: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "--gpus",
            '"device=0,1,2,3"',
            "--ipc=host",
            "--shm-size",
            CONST.shm_size,
            "--network=host",
            "-v",
            f"{REPO_ROOT}:{CONST.repo_mount}",
            "-v",
            f"{CONST.hf_cache_host_dir}:{CONST.hf_cache_container_dir}",
            "-v",
            f"{CONST.checkpoint_host_dir}:{CONST.checkpoint_container_dir}",
            "-v",
            f"{LOG_ROOT}:{LOG_ROOT}",
            "-e",
            "HF_HUB_OFFLINE=1",
            "-e",
            "BATCHGEN_V4_RESIDENT_EXPERTS="
            + os.environ.get("BATCHGEN_V4_RESIDENT_EXPERTS", "1"),
            "-e",
            "CUDA_VISIBLE_DEVICES=0,1,2,3",
            "-e",
            "PYTORCH_ALLOC_CONF=expandable_segments:True",
            "-w",
            CONST.repo_mount,
            CONST.image,
            "bash",
            "-lc",
            "sleep infinity",
        ],
        timeout=120,
    )


def _server_command(config: V4ServingConfig) -> list[str]:
    cmd: list[str] = []
    if config.numactl_node0:
        cmd.extend(["numactl", "--cpunodebind=0", "--membind=0"])
    cmd.extend(
        [
            "python",
            "-m",
            "batchgen.launch_http_server",
            "--model",
            CONST.model,
            "--converted-ckpt-dir",
            CONST.checkpoint_container_dir,
            "--cache-dir",
            CONST.hf_snapshot_dir,
            "--kv-dtype",
            config.kv_dtype,
            "--host-kv-cache-size",
            str(config.host_kv_cache_size_gb),
            "--gpu-memory-frac",
            str(config.gpu_memory_frac),
            "--gpu-arch",
            "blackwell",
            "--dist-init-addr",
            CONST.dist_init_addr,
            "--world-size",
            str(config.world_size),
            "--listen-port",
            str(CONST.listen_port),
            "--watchdog-timeout",
            str(config.watchdog_timeout_s),
        ]
    )
    if config.decode_step_timeout_s is not None:
        cmd.extend(["--decode-step-timeout", str(config.decode_step_timeout_s)])
    if config.initial_gpu_page_buffer is not None:
        cmd.extend(["--initial-gpu-page-buffer", str(config.initial_gpu_page_buffer)])
    if config.extension_gpu_page_buffer is not None:
        cmd.extend(["--extension-gpu-page-buffer", str(config.extension_gpu_page_buffer)])
    cmd.extend(list(config.server_extra_args))
    return cmd


def _start_server(container_name: str, config: V4ServingConfig, log_path: Path) -> None:
    server_cmd = shlex.join(_server_command(config))
    shell_cmd = f"{server_cmd} > {log_path} 2>&1"
    docker_cmd = ["docker", "exec", "-d"]
    env_pairs = {
        "BATCHGEN_DECODE_TIMING": "1",
        "BATCHGEN_DECODE_TIMING_INTERVAL": "1",
        "BATCHGEN_DECODE_TIMING_RANKS": "0,1,2,3",
        "BATCHGEN_DECODE_TIMING_CSV": str(LOG_ROOT / f"{container_name}.decode.csv"),
    }
    # Diagnostic-only host-env passthrough (no-op unless explicitly set); not a sweepable knob.
    for _diag_key in ("BATCHGEN_V4_SPARSE_PREFILL", "CUDA_LAUNCH_BLOCKING"):
        if os.environ.get(_diag_key) is not None:
            env_pairs[_diag_key] = os.environ[_diag_key]
    if config.nccl_p2p_level is not None:
        env_pairs["NCCL_P2P_LEVEL"] = config.nccl_p2p_level
    if config.nccl_algo is not None:
        env_pairs["NCCL_ALGO"] = config.nccl_algo
    if config.nccl_min_nchannels is not None:
        env_pairs["NCCL_MIN_NCHANNELS"] = str(config.nccl_min_nchannels)
    if config.nccl_max_nchannels is not None:
        env_pairs["NCCL_MAX_NCHANNELS"] = str(config.nccl_max_nchannels)
    if config.nccl_buffsize_bytes is not None:
        env_pairs["NCCL_BUFFSIZE"] = str(config.nccl_buffsize_bytes)
    if config.nccl_shm_disable is not None:
        env_pairs["NCCL_SHM_DISABLE"] = str(config.nccl_shm_disable)
    if config.attn_prefill_mb is not None:
        env_pairs["BATCHGEN_ATTN_PREFILL_MB"] = str(config.attn_prefill_mb)
    if config.moe_prefill_mb is not None:
        env_pairs["BATCHGEN_MOE_PREFILL_MB"] = str(config.moe_prefill_mb)
    if config.expert_prefill_cap is not None:
        env_pairs["BATCHGEN_EXPERT_PREFILL_CAP"] = str(config.expert_prefill_cap)
    if config.prefill_token_cap is not None:
        env_pairs["BATCHGEN_PREFILL_TOKEN_CAP"] = str(config.prefill_token_cap)
    if config.attn_decode_mb is not None:
        env_pairs["BATCHGEN_ATTN_DECODE_MB"] = str(config.attn_decode_mb)
    if config.moe_decode_mb is not None:
        env_pairs["BATCHGEN_MOE_DECODE_MB"] = str(config.moe_decode_mb)
    if config.expert_decode_cap is not None:
        env_pairs["BATCHGEN_EXPERT_DECODE_CAP"] = str(config.expert_decode_cap)
    for key, value in env_pairs.items():
        docker_cmd.extend(["-e", f"{key}={value}"])
    docker_cmd.extend([container_name, "bash", "-lc", shell_cmd])
    _run(docker_cmd, timeout=120)


def benchmark_config(
    config: V4ServingConfig,
    *,
    tag: str,
    results_file: Path,
) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{CONST.listen_port}"
    container_name = _container_name(tag)
    log_path = _server_log_path(tag)
    if log_path.exists():
        log_path.unlink()

    row = {
        "tag": tag,
        "config": config.compact_json(),
        "decode_tok_s": 0.0,
        "prefill_ttft_s": 0.0,
        "accuracy_guard": json.dumps({"pass": False}, sort_keys=True, separators=(",", ":")),
        "vram_mb": 0,
        "status": "crash",
        "notes": "",
    }

    prefill_samples: list[float] = []
    note_parts: list[str] = []
    try:
        cleanup_experiment(container_name)
        _docker_run_container(container_name)
        _start_server(container_name, config, log_path)
        _wait_for_server(log_path, base_url, config.startup_timeout_s)

        for prompt_tokens, output_tokens in CONST.warmup_requests:
            warm_prompt = _prompt_with_target_tokens(prompt_tokens)
            _measure_request(
                log_path,
                lambda prompt=warm_prompt, out_len=output_tokens: _send_inference(
                    base_url,
                    prompts=[prompt],
                    max_output_len=out_len,
                    timeout_s=CONST.decode_timeout_s,
                ),
            )

        _pf_override = os.environ.get("BENCH_PREFILL_TOKENS")
        _prefill_lengths = (
            tuple(int(x) for x in _pf_override.split(",") if x.strip())
            if _pf_override
            else CONST.prefill_prompt_tokens
        )
        _pf_conc = int(os.environ.get("BENCH_PREFILL_CONCURRENCY", "1"))
        _req_timeout = int(
            os.environ.get("BENCH_REQUEST_TIMEOUT", str(CONST.decode_timeout_s))
        )
        for prompt_tokens in _prefill_lengths:
            prompt = _prompt_with_target_tokens(prompt_tokens)
            # Unique per-slot prefixes defeat any prefix-cache/dedup inflating
            # the aggregate; identical prompts measure cache, not prefill.
            _batch_prompts = [
                f"[req {i:05d}] {prompt}" for i in range(_pf_conc)
            ]
            _t0 = time.time()
            _result, segment = _measure_request(
                log_path,
                lambda prompts=_batch_prompts: _send_inference(
                    base_url,
                    prompts=prompts,
                    max_output_len=1,
                    timeout_s=_req_timeout,
                ),
            )
            _wall = time.time() - _t0
            if _pf_conc > 1:
                _outs = _result.get("results", []) or []
                _n_ok = sum(1 for o in _outs if str(o).strip())
                _agg = _pf_conc * prompt_tokens / max(_wall, 1e-6)
                note_parts.append(
                    f"prefill_agg[{_pf_conc}x{prompt_tokens}]="
                    f"{_agg:.1f}tok/s wall={_wall:.1f}s "
                    f"results={len(_outs)} nonempty={_n_ok}"
                )
                prefill_samples.append(round(_wall, 3))
                continue
            metrics = _parse_generation_metrics(segment)
            if metrics["prefill_ttft_s"] is None:
                raise RuntimeError(
                    f"prefill metric missing for prompt_tokens={prompt_tokens}"
                )
            prefill_samples.append(float(metrics["prefill_ttft_s"]))

        row["prefill_ttft_s"] = round(sum(prefill_samples) / len(prefill_samples), 4)
        if os.environ.get("BENCH_SKIP_DECODE") == "1":
            note_parts.append("decode_skipped")
        else:
            prompts = [CONST.decode_prompt] * config.request_concurrency
            decode_result, decode_segment = _measure_request(
                log_path,
                lambda: _send_inference(
                    base_url,
                    prompts=prompts,
                    max_output_len=CONST.decode_output_tokens,
                    timeout_s=CONST.decode_timeout_s,
                ),
            )
            decode_metrics = _parse_generation_metrics(decode_segment)
            decode_tok_s = decode_metrics["decode_tok_s"]
            if decode_tok_s is None:
                raise RuntimeError(
                    "decode throughput metric missing from worker log"
                )
            row["decode_tok_s"] = round(float(decode_tok_s), 4)
            outputs = decode_result.get("results", [])
            first_output = outputs[0] if outputs else ""
            if not first_output.strip():
                raise RuntimeError("decode benchmark returned empty output")
            note_parts.append(
                f"coherence_sample={_safe_note(first_output[:120])}"
            )
        note_parts.append(
            "prefill_samples_s=" + json.dumps(prefill_samples, separators=(",", ":"))
        )

        accuracy = _run_accuracy_guard(base_url, tag)
        row["accuracy_guard"] = json.dumps(
            accuracy, sort_keys=True, separators=(",", ":")
        )
        row["vram_mb"] = max(
            (gpu["memory_used_mb"] for gpu in _query_gpu_memory()),
            default=0,
        )
        row["status"] = "ok" if accuracy["pass"] else "guard_fail"
        note_parts.append(f"request_concurrency={config.request_concurrency}")
        row["notes"] = " | ".join(note_parts)
        return row
    except Exception as exc:
        note_parts.append(str(exc))
        row["notes"] = " | ".join(_safe_note(part) for part in note_parts)
        return row
    finally:
        cleanup_error = None
        try:
            cleanup_experiment(container_name)
        except Exception as exc:  # pragma: no cover - operational cleanup path
            cleanup_error = exc
        if cleanup_error is not None:
            row["status"] = "cleanup_fail"
            row["notes"] = _safe_note(f"{row['notes']} | {cleanup_error}")
        _append_results_row(results_file, row)


def _config_from_args(args: argparse.Namespace) -> V4ServingConfig:
    if args.config_json is not None:
        payload = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        return V4ServingConfig(**payload)
    if args.config_name not in NAMED_CONFIGS:
        raise KeyError(f"unknown config name: {args.config_name}")
    return NAMED_CONFIGS[args.config_name]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed DeepSeek-V4-Flash serving-config benchmark harness"
    )
    parser.add_argument("--config-name", default=BASELINE_CONFIG.name)
    parser.add_argument(
        "--config-json",
        default=None,
        help="Path to a JSON file matching V4ServingConfig; overrides --config-name",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--results-file", default=str(RESULTS_FILE))
    args = parser.parse_args()

    config = _config_from_args(args)
    row = benchmark_config(config, tag=args.tag, results_file=Path(args.results_file))
    print(json.dumps(row, indent=2, sort_keys=True))
    if row["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

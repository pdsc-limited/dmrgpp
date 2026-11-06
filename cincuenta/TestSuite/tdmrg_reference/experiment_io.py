"""Run and summarize one DMRG experiment using only the Python standard library."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def sha256_file(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(_FLOAT, text)]


def parse_output(text: str) -> dict:
    """Parse known TargetingCommon and in-situ output; retain warnings and malformed records."""
    result = {"gs_weight": None, "weights": None, "gs_norms": None, "norms": None,
              "current_times": [], "measurements": [], "discarded_weights": [],
              "wft_warnings": [], "parse_errors": [], "missing_metrics": []}
    for line_no, line in enumerate(text.splitlines(), 1):
        m = re.search(r"gsWeight=(\S+)\s+weights=\s*(.*)", line)
        if m:
            try:
                result["gs_weight"] = float(m.group(1)); result["weights"] = _numbers(m.group(2))
            except ValueError:
                result["parse_errors"].append({"line": line_no, "text": line, "field": "weights"})
        m = re.search(r"gsNorms=(.*?)\s+norms=\s*(.*)", line)
        if m:
            result["gs_norms"] = _numbers(m.group(1)); result["norms"] = _numbers(m.group(2))
            if not result["gs_norms"] and m.group(1).strip() or not result["norms"] and m.group(2).strip():
                result["parse_errors"].append({"line": line_no, "text": line, "field": "norms"})
        m = re.search(r"CurrentTime=(\S+)(?:\s+CurrentTimeStep=(\S+))?", line)
        if m:
            try: result["current_times"].append({"time": float(m.group(1)), "step": int(m.group(2)) if m.group(2) else None})
            except ValueError: result["parse_errors"].append({"line": line_no, "text": line, "field": "time"})
        m = re.search(r"discarded\s+weight(?:\s*\([^)]*\))?\s*[=:]\s*(" + _FLOAT + r")", line, re.I)
        if m: result["discarded_weights"].append(float(m.group(1)))
        if (re.search(r"WFT|WaveFunctionTransf", line, re.I) and re.search(r"warn|fail|error|cannot|unable", line, re.I)) or re.search(r"Transformation completed\s+WARNING:", line, re.I):
            result["wft_warnings"].append({"line": line_no, "text": line})
        if line.strip().startswith("-------------&*&*&* In-situ measurements start"):
            result.setdefault("_in_situ", True)
            continue
        if line.strip().startswith("-------------&*&*&* In-situ measurements end"):
            result.pop("_in_situ", None); continue
        if result.get("_in_situ") or re.match(r"\s*\d+\s+.*<[^>]+>", line):
            # Some cocoon paths omit start/end markers. Match measurements directly.
            # TargetingCommon prints: site value time braket overlap.
            m = re.match(r"\s*(\d+)\s+(\([^)]*\)|\S+)\s+(" + _FLOAT + r")\s+(\S+)\s+(\S+)\s*$", line)
            if m:
                try:
                    result["measurements"].append({"site": int(m.group(1)), "value_raw": m.group(2), "time": float(m.group(3)), "bra_ket": m.group(4), "norm_raw": m.group(5), "raw": line})
                except ValueError:
                    result["parse_errors"].append({"line": line_no, "text": line, "field": "measurement"})
            elif line.strip() and not line.lstrip().startswith("cocoon:"):
                result["parse_errors"].append({"line": line_no, "text": line, "field": "measurement"})
    for key in ("gs_weight", "weights", "gs_norms", "norms"):
        if result[key] is None: result["missing_metrics"].append(key)
    result.pop("_in_situ", None)
    return result


def checkpoint_sizes(directory: os.PathLike | str, names: Iterable[str] | None = None) -> dict:
    root = Path(directory)
    paths = sorted(root.glob("*.hd5")) if names is None else [root / n for n in names]
    return {p.name: p.stat().st_size for p in paths if p.is_file()}


def atomic_write_json(path: os.PathLike | str, value: object) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, target)


def run_job(argv: Sequence[str], input_file: os.PathLike | str, cwd: os.PathLike | str,
            timeout: float | None = None, env: Mapping[str, str] | None = None,
            checkpoint_names: Iterable[str] | None = None, cout_names: Iterable[str] | None = None) -> dict:
    """Execute argv; an isolated stdlib helper measures the actual child with wait4."""
    if not argv: raise ValueError("argv must not be empty")
    if os.name != "posix" or not hasattr(os, "wait4"):
        raise NotImplementedError("run_job requires POSIX os.wait4 for accurate per-child peak RSS")
    directory = Path(cwd).resolve(); directory.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy(); child_env.update(env or {})
    child_env.setdefault("OMP_NUM_THREADS", "1"); child_env.setdefault("OPENBLAS_NUM_THREADS", "1")
    executable = shutil.which(str(argv[0]), path=child_env.get("PATH")) or str(Path(argv[0]).resolve())
    command = list(argv)
    # Files avoid pipe backpressure/deadlocks while retaining complete captured streams.
    stdout_path = directory / ".experiment-stdout"
    stderr_path = directory / ".experiment-stderr"
    metrics_path = directory / ".experiment-child-metrics.json"
    helper = Path(__file__).with_name("measure_child.py")
    wrapper_command = [sys.executable, str(helper), str(metrics_path), *command]
    started = time.monotonic(); timed_out = False
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        proc = subprocess.Popen(wrapper_command, cwd=directory, env=child_env, stdout=stdout_file, stderr=stderr_file,
                                start_new_session=True)
        deadline = None if timeout is None else started + timeout
        while proc.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait()
                break
            time.sleep(0.005)
    wall = time.monotonic() - started
    metrics = {}
    if not timed_out and metrics_path.is_file():
        try: metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): metrics = {}
    metrics_path.unlink(missing_ok=True)
    # If the helper could not launch the requested executable (or itself failed
    # before writing metrics), preserve the wrapper's failure status instead of
    # reporting an ambiguous None. Successful measurements use the target status.
    rc = metrics.get("returncode", proc.returncode)
    peak_rss = metrics.get("peak_child_rss_bytes")
    out, err = stdout_path.read_bytes(), stderr_path.read_bytes()
    stdout_path.unlink(missing_ok=True); stderr_path.unlink(missing_ok=True)
    if timed_out: rc = None
    logs = {}
    candidates = list(cout_names) if cout_names is not None else [p.name for p in directory.glob("runFor*.cout")]
    for name in candidates:
        p = directory / name
        if p.is_file(): logs[name] = p.read_text(errors="replace")
    decoded_out, decoded_err = out.decode(errors="replace"), err.decode(errors="replace")
    output = "\n".join([decoded_out, decoded_err, *logs.values()])
    return {"argv": command, "cwd": str(directory.resolve()), "executable": executable,
            "input_file": str(Path(input_file).resolve()), "input_sha256": sha256_file(input_file),
            "returncode": rc, "timed_out": timed_out, "wall_seconds": wall,
            "peak_child_rss_bytes": peak_rss, "stdout": decoded_out, "stderr": decoded_err,
            "cout_logs": logs, "parsed": parse_output(output),
            "checkpoints": checkpoint_sizes(directory, checkpoint_names)}

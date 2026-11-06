"""Lightweight isolated wait4 helper for per-command resource measurement."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: measure_child.py METRICS.json COMMAND [ARG ...]")
    metrics_path = Path(sys.argv[1])
    command = sys.argv[2:]
    started = time.monotonic()
    child = subprocess.Popen(command)
    while True:
        try:
            pid, status, usage = os.wait4(child.pid, 0)
            break
        except InterruptedError:
            continue
    child._handle_exitstatus(status)
    # Linux reports ru_maxrss in KiB; macOS reports bytes.
    peak_rss = int(usage.ru_maxrss) * (1024 if sys.platform.startswith("linux") else 1)
    metrics = {
        "returncode": child.returncode,
        "peak_child_rss_bytes": peak_rss,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "wall_seconds": time.monotonic() - started,
    }
    temporary = metrics_path.with_name(metrics_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, metrics_path)
    return child.returncode if child.returncode >= 0 else 128 - child.returncode


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a batch of trtship jobs on the GPU machine, non-interactively.

Why this exists: development happens on a Mac with no CUDA, while engine builds,
calibration and benchmarks run on a separate GPU laptop or a shared library
machine. Those sessions are time-boxed, sometimes non-admin, and not somewhere
you want to be debugging interactively. So the loop is:

    1. write and unit-test on the Mac
    2. commit and push
    3. on the GPU box:  git pull && python scripts/gpu_batch.py jobs/week3.yaml
    4. pull the results directory back and analyse on the Mac

Every job's stdout, stderr, exit code and wall time is captured, so a failed run
is diagnosable after the fact without re-booking GPU time.

Jobs file format (YAML):

    output_dir: .gpu_sync/week3
    jobs:
      - name: resnet18-fp16
        command: [trtship, build, --model, models/resnet18.pt, --precision, fp16]
      - name: resnet18-int8
        command: [trtship, build, --model, models/resnet18.pt, --precision, int8]
        allow_failure: true      # expected to fail; record it and keep going
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class JobResult:
    name: str
    command: list[str]
    returncode: int
    duration_s: float
    stdout_path: str
    stderr_path: str
    allowed_failure: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 or self.allowed_failure


def _environment_block() -> dict[str, Any]:
    """Capture the machine, so results pulled back to the Mac are attributable."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from trtship.env import detect

        env = detect().to_dict()
    except Exception as exc:  # the point is to never lose the batch over this
        env = {"error": f"trtship.env.detect() failed: {exc}"}
    env["hostname"] = platform.node()
    env["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return env


def run_batch(jobs_file: Path, output_dir: Path | None = None) -> int:
    spec = yaml.safe_load(jobs_file.read_text())
    if not isinstance(spec, dict) or "jobs" not in spec:
        print(f"error: {jobs_file} must be a mapping with a 'jobs' key", file=sys.stderr)
        return 2

    out = output_dir or Path(spec.get("output_dir", ".gpu_sync/run"))
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    env_block = _environment_block()
    print(f"host      {env_block.get('hostname')}")
    print(f"gpus      {[g.get('name') for g in env_block.get('gpus', [])] or 'none'}")
    print(f"tensorrt  {env_block.get('tensorrt')}")
    print(f"output    {out}")
    print()

    results: list[JobResult] = []
    for job in spec["jobs"]:
        name = job["name"]
        command = [str(c) for c in job["command"]]
        allow_failure = bool(job.get("allow_failure", False))

        stdout_path = logs / f"{name}.out.txt"
        stderr_path = logs / f"{name}.err.txt"

        print(f"[run ] {name}: {' '.join(command)}", flush=True)
        start = time.perf_counter()
        try:
            proc = subprocess.run(command, capture_output=True, text=True, check=False)
            returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except (OSError, subprocess.SubprocessError) as exc:
            returncode, stdout, stderr = 127, "", f"failed to launch: {exc}"
        duration = time.perf_counter() - start

        stdout_path.write_text(stdout)
        stderr_path.write_text(stderr)

        result = JobResult(
            name=name,
            command=command,
            returncode=returncode,
            duration_s=round(duration, 3),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            allowed_failure=allow_failure,
        )
        results.append(result)

        status = "ok  " if returncode == 0 else ("fail" if not allow_failure else "fail*")
        print(f"[{status}] {name} ({duration:.1f}s)", flush=True)
        if returncode != 0:
            tail = stderr.strip().splitlines()[-3:]
            for line in tail:
                print(f"        {line}", flush=True)

    manifest = {
        "jobs_file": str(jobs_file),
        "environment": env_block,
        "results": [asdict(r) for r in results],
    }
    manifest_path = out / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    failed = [r for r in results if not r.ok]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} succeeded — manifest: {manifest_path}")
    if failed:
        print("failed: " + ", ".join(r.name for r in failed))
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jobs_file", type=Path, help="YAML file describing the batch")
    parser.add_argument("--output-dir", type=Path, default=None, help="override output_dir")
    args = parser.parse_args()

    if not args.jobs_file.is_file():
        print(f"error: no such jobs file: {args.jobs_file}", file=sys.stderr)
        return 2
    return run_batch(args.jobs_file, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())

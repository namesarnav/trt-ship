# Architecture

The module layout follows `project.md` §2. This document covers the one thing
that document does not: which code can run where, and why that shapes the design.

## CPU-safe core, GPU-required edge

Primary development is on macOS/arm64, which has no CUDA and no TensorRT build.
Roughly 60% of trtship can still be built and tested there — but only because the
split is enforced rather than hoped for.

**CPU-safe (runs and is unit-tested on macOS):**

| Module | Why it needs no GPU |
|---|---|
| `core/model_spec.py` | dataclasses over torch tensors |
| `core/config.py` | pydantic validation |
| `env.py` | probes hardware; never requires it |
| `export/` | `torch.onnx` and `onnxruntime` (CPU EP) both work on arm64 |
| `build/cache.py` | cache *keys* are hashes |
| `build/shape_profiles.py` | profile *construction* is data manipulation |
| `deploy/triton/config_generator.py` | `config.pbtxt` is text generation |
| `benchmark/report.py` | reads manifests produced elsewhere |

**GPU-required:**

`build/engine_builder.py`, all of `calibrate/`, all of `benchmark/backends/`
(including the PyTorch eager baseline — MPS numbers are not comparable to CUDA),
`deploy/triton/smoke_test.py`, and the published Docker image (`linux/amd64` on a
CUDA base, not buildable on arm64).

## The three rules

1. **`import trtship` must never import `tensorrt`.** TensorRT imports live
   inside the functions that use them. Enforced by
   `tests/unit/test_no_tensorrt_import.py`, which blocks the module in a
   subprocess — checking `sys.modules` in-process would pass trivially on the
   very machine that cannot detect the regression.

2. **Every GPU module gets a CPU-testable seam.** `engine_builder` receives a
   config object built and validated on CPU; the untestable part is the thin API
   call, not the logic around it. This is what keeps coverage reachable (NFR-5.2)
   when most CI runs have no GPU.

3. **GPU work is batched and scripted.** `scripts/gpu_batch.py` runs a YAML job
   list, captures per-job stdout/stderr/timing, and writes a manifest. Write on
   the Mac, push, run one batch on the GPU box, pull results back. A workflow
   that assumes an interactive GPU shell fails on a time-boxed library machine.

## Why ModelSpec

Every stage operates on a `ModelSpec`, never a raw `nn.Module`. Adding a second
source format — a bare ONNX file with no PyTorch model behind it — then changes
only ingestion. It is also what makes the run manifest possible: `to_dict()`
drops the live torch objects and keeps the metadata.

## Config has one code path

CLI flags and YAML both parse into the same pydantic models in `core/config.py`.
Validation lives there, so a bad config fails with a field-level message before
any slow work starts, and `--config pipeline.yaml` is guaranteed equivalent to
the flag form (FR-8.3) rather than approximately equivalent.

# trtship

Take a trained PyTorch model to a benchmarked, calibrated, Triton-servable
TensorRT deployment — in one reproducible, debuggable pipeline.

> **Status: pre-alpha (Phase 0).** The scaffolding, core types and environment
> preflight exist. The export, build, calibrate, benchmark and deploy stages do
> not yet. See [the build plan](#project-status) for what lands when.

## Why

Getting a PyTorch model into production on TensorRT today means exporting to
ONNX and debugging opset mismatches one at a time, hand-writing INT8 calibration
per model, fighting dynamic shape profiles, hand-writing Triton `config.pbtxt`,
and redoing all of it for the next architecture. `torch-tensorrt` and raw
TensorRT each solve pieces. trtship is the pipeline around them.

## Install

```bash
pip install trtship              # core: ingest, export, Triton config, reports
pip install "trtship[export]"    # + ONNX Runtime numerical verification
pip install "trtship[gpu]"       # + TensorRT engine build, calibration (Linux/Windows + NVIDIA)
```

Requires Python 3.10–3.12.

## Check your environment first

```bash
trtship doctor
```

It reports what this machine can actually do, and says why for anything it
cannot. On a Mac you will see the CPU stages available and the GPU stages
explained rather than silently missing:

```
Pipeline stages
stage          status
export         available
verify         unavailable
build          unavailable
triton-config  available
report         available

Notes
  • No CUDA: macOS has no CUDA support at all, and NVIDIA ships no TensorRT
    build for it. Engine build, calibration, benchmarking and the Triton smoke
    test must run on a Linux/Windows machine with an NVIDIA GPU.
```

## Quickstart

*Not yet functional — this is the target interface (FR-8.1).*

```python
import torch, torchvision
from trtship import ship

model = torchvision.models.resnet18(weights="DEFAULT").eval().cuda()

result = ship(
    model,
    example_inputs=torch.randn(1, 3, 224, 224).cuda(),
    precision="int8",
    calibration_data="./imagenet_calib_subset/",
    output_dir="./resnet18_deployment/",
)
print(result.benchmark_report_path)  # interactive HTML comparison
print(result.triton_repo_path)       # ready for tritonserver
```

## Development

```bash
# Python 3.11 is the primary target; 3.13 is not yet supported by the
# ONNX Runtime / TensorRT wheels this project depends on.
conda create -n trtship python=3.11 -y && conda activate trtship

pip install -e ".[export,dev]"      # on a GPU box, add: gpu,triton,report
pre-commit install
pytest tests/unit                   # runs anywhere, no GPU needed
pytest tests/integration -m gpu     # requires CUDA + TensorRT
```

trtship is developed on a machine that cannot run it end to end. The CPU-safe
core imports and tests without CUDA or TensorRT present; see
[docs/architecture.md](docs/architecture.md) for the split and the rules that
keep it true, and [docs/compatibility.md](docs/compatibility.md) for the tested
version matrix.

## Project status

| Phase | Scope | State |
|---|---|---|
| 0 | Scaffolding, `ModelSpec`, config, `doctor`, CI | done |
| 1 | Export chain, diagnostics, numerical verification | next |
| 2 | Engine build, dynamic shapes, cache | |
| 3 | Benchmark harness across three backends | |
| 4 | INT8 calibration, per-layer sensitivity | |
| 5 | HTML report, run manifests | |
| 6 | Triton repo generation, smoke test | |
| 7 | `ship` command, YAML config, error-message pass | |
| 8 | 8+ models, docs site, Docker image | |
| 9 | PyPI release, real users | |

## License

MIT — see [LICENSE](LICENSE).

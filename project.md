trtship — Model-Agnostic PyTorch → TensorRT Deployment Tool
Full Project Specification: Requirements, User Stories, Architecture, and Build Plan

1. Problem Statement
Getting a PyTorch model into a production TensorRT deployment today means manually:

Exporting to ONNX and debugging opset/operator mismatches one at a time
Hand-writing INT8 calibration scripts per model
Fighting dynamic shape profile configuration
Hand-writing Triton config.pbtxt files with correct I/O names/shapes/dtypes
Re-doing all of this from scratch for every new model architecture

torch-tensorrt and raw TensorRT solve pieces of this, but there is no tool that takes you from "I have a PyTorch model" to "I have a benchmarked, calibrated, Triton-servable TensorRT deployment" in one reproducible, debuggable pipeline. That gap is trtship.
2. Goal
Build a CLI + Python library, trtship, that:

Takes an arbitrary PyTorch nn.Module (or a small set of supported source formats)
Exports it through a resilient multi-strategy export pipeline
Builds an optimized TensorRT engine (FP16/INT8, static or dynamic shapes)
Calibrates INT8 automatically with a pluggable calibration dataset interface
Emits a ready-to-serve Triton Inference Server model repository
Produces a reproducible benchmark report (latency, throughput, accuracy delta) comparing baseline vs optimized

The tool ships as an installable, documented, open-source package with real external users — this is the "production" bar, not internal-only usage.
3. Target Users
ML engineers who need to deploy PyTorch models with low latency and don't want to hand-roll the export/calibration/serving pipeline every time
MLOps/platform teams standardizing how models get pushed to inference serving
Researchers who want a quick, honest FP16/INT8 accuracy-vs-speed comparison without writing throwaway scripts
4. Non-Goals (explicitly out of scope)
Training or fine-tuning — this tool only touches already-trained models
Support for frameworks other than PyTorch (no TensorFlow/JAX in v1)
A hosted SaaS version (v1 is a local CLI/library; hosting is a stretch goal, not core scope)
Multi-GPU distributed inference orchestration (single-GPU engine building only in v1)
Automatic model architecture search or NAS — this optimizes serving, not the model itself
5. Success Criteria ("done" for this project)
Package published on PyPI, installable via pip install trtship
Successfully converts and benchmarks at least 8 real, distinct model architectures (CNNs, transformers, at least one detection/segmentation model)
At least one PR merged upstream into torch-tensorrt, ONNX Runtime, or Triton (stretch but strongly recommended — see Path C discussion)
Public GitHub repo with docs site, >0 real external users (issues/stars/downloads from people who are not you)
Full test suite with CI passing on GPU runners
Blog/write-up series documenting real op-support failures encountered and fixed
Functional Requirements (FR)
Each FR has an ID, description, and acceptance criteria. Use these IDs directly as GitHub issue/milestone tags.
FR-1: Model Ingestion
FR-1.1 — Accept a PyTorch nn.Module instance plus example input tensor(s) (shape + dtype) as the primary input format. Acceptance: trtship.load(model, example_inputs=...) succeeds for a plain nn.Module and returns an internal ModelSpec object.

FR-1.2 — Accept a path to a saved model: .pt/.pth (via torch.load, with weights_only safety handling), TorchScript .pt, or a HuggingFace model identifier (via transformers). Acceptance: CLI command trtship convert --model path/to/model.pt --example-input ... loads successfully for each supported format.

FR-1.3 — Validate that example inputs are shape/dtype-consistent with the model's forward signature before attempting export; fail fast with a clear error if not. Acceptance: Passing a mismatched input shape produces a specific error message naming the mismatch, not a raw traceback from deep inside PyTorch.
FR-2: Export Pipeline
FR-2.1 — Attempt export via torch.onnx.export (dynamo-based exporter) first; on failure, fall back to the legacy TorchScript-based exporter; on failure, fall back to torch.export + torch-tensorrt's native path. Acceptance: For each of the 8+ benchmark models, the tool either succeeds via one of the three paths or produces a clear "unsupported" diagnostic naming the exact failing operator.

FR-2.2 — On export failure, parse the underlying error to identify the specific unsupported operator(s) and surface them in a structured diagnostic (op name, location in model — module path if available, suggested next step). Acceptance: A deliberately-broken test model (custom op with no ONNX mapping) produces a diagnostic that names the op and module path, not a raw stack trace.

FR-2.3 — Support a plugin/registration system where a user can register a custom op-handling strategy (e.g., a symbolic function for ONNX export, or a TensorRT plugin) for an operator the default pipeline can't handle. Acceptance: Registering a custom symbolic function for a synthetic unsupported op allows a previously-failing model to export successfully.

FR-2.4 — Verify exported ONNX graph numerically matches the original PyTorch model output within a configurable tolerance, on the example input and at least N additional random inputs. Acceptance: trtship verify reports max absolute/relative error between PyTorch and ONNX Runtime outputs; fails the pipeline if error exceeds threshold (default 1e-3 relative).
FR-3: Precision & Calibration
FR-3.1 — Support FP32 (passthrough/baseline), FP16, and INT8 engine builds. Acceptance: trtship build --precision fp16 and --precision int8 both produce valid .engine files for all benchmark models where the target GPU supports the precision.

FR-3.2 — Provide a pluggable calibration dataset interface (CalibrationDataset base class) that users implement to supply representative data for INT8 calibration; ship default implementations for common data types (image folders, tokenized text batches, raw tensor .npy/.pt directories). Acceptance: At least 2 built-in calibrator dataset types work out of the box; a user can subclass and register a custom one in under 20 lines of code.

FR-3.3 — Support multiple calibration algorithms: entropy (default TensorRT IInt8EntropyCalibrator2), MinMax, and percentile-based calibration; expose as a config option. Acceptance: All three calibration modes produce a working INT8 engine on at least 3 of the benchmark models; the benchmark report shows accuracy delta per calibration method.

FR-3.4 — Run automatic per-layer sensitivity analysis for INT8: identify which layers cause the largest accuracy drop when quantized, and support falling back to FP16 for those specific layers (mixed precision). Acceptance: trtship analyze-sensitivity outputs a per-layer ranked report; trtship build --precision int8 --auto-mixed-precision produces an engine that keeps the worst N layers in FP16 and shows improved accuracy vs full-INT8 in the report.
FR-4: Shape Handling
FR-4.1 — Support static-shape engine builds (fixed input dimensions, matching the example input). Acceptance: Default trtship build with no shape config produces a static engine matching example input shape.

FR-4.2 — Support dynamic-shape engine builds via TensorRT optimization profiles: user specifies min/opt/max shape per input dimension. Acceptance: trtship build --dynamic-shapes "input:1x3x224x224,4x3x224x224,16x3x224x224" (min,opt,max batch) produces an engine that correctly runs inference at batch sizes 1, 4, 8, and 16.

FR-4.3 — Auto-detect likely dynamic dimensions (e.g., batch dimension) from the model and example input where the user hasn't specified, with a sane default profile, and warn the user this is a guess. Acceptance: Running without explicit shape config on a model with an obviously dynamic batch dim produces a working engine and a logged warning explaining the assumed profile.
FR-5: Engine Building & Caching
FR-5.1 — Build a .engine file from the (optionally calibrated) ONNX graph using the TensorRT builder API, with configurable builder flags (workspace size, timing cache, tactic sources). Acceptance: Engine builds succeed and are loadable via tensorrt.Runtime for all benchmark models.

FR-5.2 — Cache built engines keyed by a hash of (model weights hash, input shape config, precision config, TensorRT version, GPU compute capability); skip rebuild if a valid cached engine exists. Acceptance: Running trtship build twice with identical config on the same machine completes the second run in under 2 seconds (cache hit), and running after a config change triggers a rebuild.

FR-5.3 — Invalidate cache automatically when TensorRT version or GPU changes are detected, with a clear log message explaining why a rebuild is happening. Acceptance: Simulating a TensorRT version bump (mock) triggers cache invalidation and rebuild.
FR-6: Triton Deployment Artifact Generation
FR-6.1 — Generate a complete Triton model repository directory: versioned engine file, config.pbtxt with correctly inferred input/output names, shapes, dtypes, and max batch size. Acceptance: trtship export-triton --output ./model_repo produces a directory that Triton Inference Server (tritonserver --model-repository=./model_repo) loads without manual edits, for all benchmark models.

FR-6.2 — Support generating Triton dynamic batching config (preferred batch sizes, max queue delay) with sane defaults and user override. Acceptance: Generated config includes a dynamic_batching block; overriding --max-queue-delay-us changes the generated value.

FR-6.3 — Provide a one-command local smoke test: spin up Triton in Docker with the generated repo, send a test inference request, verify output matches expected (within tolerance) from FR-2.4, tear down. Acceptance: trtship smoke-test exits 0 on success and non-zero with a clear message on failure, for all benchmark models.
FR-7: Benchmarking & Reporting
FR-7.1 — Measure baseline PyTorch (eager) latency (p50, p90, p99), throughput (samples/sec at various batch sizes), and peak GPU memory. Acceptance: trtship benchmark --backend pytorch outputs a structured report (JSON + human-readable table) with all above metrics.

FR-7.2 — Measure the same metrics for the built TensorRT engine, and for an ONNX Runtime baseline (non-TensorRT) as a middle comparison point. Acceptance: Same report format across all three backends (PyTorch eager, ONNX Runtime, TensorRT), directly comparable.

FR-7.3 — Measure accuracy delta (task-appropriate metric — configurable, default top-1 accuracy or output MSE) between baseline and each precision level, using a user-supplied validation dataset. Acceptance: Report includes an accuracy column per precision level when a validation dataset is supplied; gracefully omits it with a warning when not supplied.

FR-7.4 — Generate a visual report (matplotlib/plotly charts: latency vs precision, throughput vs batch size, accuracy vs precision) saved as HTML/PNG artifacts. Acceptance: trtship report --format html produces a self-contained HTML file with embedded charts, viewable without internet access.

FR-7.5 — Support exporting the full run (all configs, all metrics) as a reproducible JSON manifest that can be re-loaded to regenerate the report without re-running inference. Acceptance: trtship report --from-manifest run.json regenerates identical charts from a saved manifest.
FR-8: CLI & Developer Experience
FR-8.1 — Single-command end-to-end path: trtship ship --model model.pt --example-input ... --precision int8 --calibration-data ./cal_data/ runs export → calibrate → build → benchmark → generate Triton repo → smoke test, with progress output at each stage. Acceptance: This single command succeeds end-to-end on at least 5 of the benchmark models with no additional flags beyond the ones shown.

FR-8.2 — Every pipeline stage is independently invocable (trtship export, trtship calibrate, trtship build, trtship benchmark, trtship export-triton) for users who want to run/debug one stage at a time. Acceptance: Each subcommand works standalone given the correct prior-stage artifacts on disk.

FR-8.3 — Config file support (YAML) as an alternative to CLI flags, so full pipeline configs can be checked into version control. Acceptance: trtship ship --config pipeline.yaml produces identical results to the equivalent CLI-flag invocation.

FR-8.4 — Structured, leveled logging (debug/info/warning/error) with --verbose flag; every failure produces an actionable message, never a bare traceback as the only output. Acceptance: Manual review: induce at least 10 distinct failure modes across the pipeline and confirm each produces a human-readable diagnostic.
Non-Functional Requirements (NFR)
NFR-1: Performance
NFR-1.1 — Engine caching (FR-5.2) must reduce a no-op re-run to under 2 seconds for models up to 500MB.

NFR-1.2 — The tool's own overhead (excluding actual TensorRT build time, which is inherently slow) should add no more than ~10% to end-to-end pipeline time versus manually running each step.

NFR-1.3 — Benchmark measurements must use proper GPU warmup (minimum 20 iterations discarded) and CUDA synchronization before timing to avoid measuring async-dispatch artifacts rather than real latency.
NFR-2: Reliability
NFR-2.1 — No pipeline stage should leave the working directory or GPU memory in a corrupted/partial state on failure — use atomic writes (write to temp, rename on success) for all artifacts.

NFR-2.2 — GPU memory must be explicitly released between pipeline stages (PyTorch model, ONNX Runtime session, TensorRT engine/context) to avoid OOM on long multi-model benchmark runs.

NFR-2.3 — The tool must detect and clearly report environment mismatches (wrong CUDA version, missing TensorRT, incompatible GPU compute capability) at startup, before attempting any work.
NFR-3: Compatibility
NFR-3.1 — Support TensorRT 10.x (current major version at time of writing) with an explicit compatibility matrix documented for each supported TensorRT/CUDA/PyTorch version combination.

NFR-3.2 — Support Python 3.10–3.12.

NFR-3.3 — Support both consumer (e.g., RTX 30/40-series) and datacenter (A100/H100) GPUs; document known behavior differences (e.g., sparsity support only on Ampere+, FP8 support only on Hopper/Ada+).

NFR-3.4 — Triton config generation must target a documented, pinned Triton Inference Server version and note compatibility explicitly rather than assuming forward compatibility.
NFR-4: Usability
NFR-4.1 — A new user should be able to go from pip install trtship to a working benchmark report on a stock torchvision model in under 10 minutes, following only the README quickstart.

NFR-4.2 — Every CLI command must support --help with example invocations, not just flag descriptions.

NFR-4.3 — Error messages must never require the user to read trtship source code to understand what went wrong for the top 20 most common failure modes (enumerate these during development and test against them explicitly).
NFR-5: Maintainability
NFR-5.1 — Core pipeline stages (export, calibrate, build, benchmark, deploy) must be separated into independent, independently-testable modules with clear interfaces — no stage should require importing internals of another.

NFR-5.2 — Test coverage minimum 75% on core library code (excluding CLI glue and example scripts), enforced in CI.

NFR-5.3 — All public APIs must have type hints and docstrings sufficient to auto-generate API reference docs (e.g., via mkdocstrings or sphinx-autodoc).
NFR-6: Security
NFR-6.1 — Loading user-supplied .pt/.pth files must default to safe loading (weights_only=True where possible) and clearly warn when falling back to unsafe pickle-based loading, since arbitrary PyTorch checkpoints can execute arbitrary code on load.

NFR-6.2 — The tool must not execute or eval() any content from calibration data files beyond intended tensor deserialization.
NFR-7: Portability / Deployment
NFR-7.1 — Ship an official Docker image with all dependencies (correct CUDA/TensorRT/PyTorch versions) pinned and pre-installed, so users aren't required to solve the CUDA/TensorRT/PyTorch version-compatibility puzzle themselves.

NFR-7.2 — CI must run the full test suite on actual GPU runners (not GPU-mocked), since TensorRT behavior cannot be meaningfully tested on CPU-only CI.
NFR-8: Observability
NFR-8.1 — Every pipeline run must produce a run manifest (JSON) capturing full config, environment (GPU, driver, CUDA, TensorRT, PyTorch versions), timings per stage, and artifact paths/hashes — sufficient to fully reproduce or debug a run after the fact.
User Stories
Format: As a [role], I want to [action], so that [outcome]. Each maps back to FR IDs.
Epic 1: First-Run Experience
US-1.1 — As a new user, I want to run one command against a standard torchvision model and get a working TensorRT engine and benchmark report, so that I can evaluate whether this tool is worth adopting before investing time in my own model. (FR-8.1, NFR-4.1)

US-1.2 — As a new user, I want clear, non-cryptic error messages when my environment is misconfigured (wrong CUDA/TensorRT version), so that I don't waste an hour debugging a version mismatch. (NFR-2.3, NFR-4.3)

US-1.3 — As a new user, I want an official Docker image with everything pre-installed, so that I can try the tool without solving the CUDA/TensorRT dependency puzzle myself. (NFR-7.1)
Epic 2: Model Export
US-2.1 — As an ML engineer, I want the tool to try multiple export strategies automatically, so that a single unsupported op doesn't block my entire deployment. (FR-2.1)

US-2.2 — As an ML engineer, when export fails, I want to know exactly which operator and where in my model it failed, so that I can fix or work around it instead of guessing from a stack trace. (FR-2.2, NFR-4.3)

US-2.3 — As an advanced user with a custom op, I want to register my own export handling for that op, so that I'm not blocked waiting for upstream support. (FR-2.3)

US-2.4 — As an ML engineer, I want automatic numerical verification that the exported model matches my original PyTorch model's outputs, so that I can trust the optimized model won't silently produce wrong results. (FR-2.4)
Epic 3: Precision & Accuracy
US-3.1 — As an ML engineer optimizing for latency, I want to build an INT8 engine using my own representative data for calibration, so that quantization accuracy loss is minimized for my actual use case. (FR-3.2)

US-3.2 — As an ML engineer, I want to see which layers are most sensitive to quantization, so that I can make an informed decision about mixed precision instead of accepting a blanket accuracy hit. (FR-3.4)

US-3.3 — As an ML engineer, I want to compare accuracy across FP32/FP16/INT8 on my own validation set, so that I can make a data-driven speed-vs-accuracy tradeoff decision rather than guessing. (FR-7.3)
Epic 4: Dynamic Shapes
US-4.1 — As an ML engineer serving variable batch sizes in production, I want to build an engine that supports a range of input shapes, so that I don't need a separate engine per batch size. (FR-4.2)

US-4.2 — As a new user unfamiliar with TensorRT optimization profiles, I want the tool to guess a sane dynamic shape config from my model, so that I don't need to learn TensorRT internals just to get variable batch size support. (FR-4.3)
Epic 5: Deployment
US-5.1 — As an MLOps engineer, I want a ready-to-serve Triton model repository generated automatically, so that I don't have to hand-write config.pbtxt files. (FR-6.1)

US-5.2 — As an MLOps engineer, I want to verify the generated Triton deployment actually works before pushing to production, so that I catch config errors locally instead of in a production incident. (FR-6.3)

US-5.3 — As an MLOps engineer, I want dynamic batching configured sensibly by default with the option to tune it, so that I get reasonable production throughput without needing to be a Triton expert on day one. (FR-6.2)
Epic 6: Benchmarking & Decision-Making
US-6.1 — As a technical lead evaluating whether TensorRT optimization is worth the engineering effort for a given model, I want a clear before/after report (latency, throughput, memory, accuracy), so that I can make a go/no-go decision backed by data. (FR-7.1, FR-7.2, FR-7.4)

US-6.2 — As an engineer sharing results with a non-technical stakeholder, I want a visual HTML report I can send as a link or attachment, so that I don't have to explain raw JSON numbers in a meeting. (FR-7.4)

US-6.3 — As an engineer re-running experiments over time, I want to save and reload past benchmark runs, so that I can track regressions or improvements across TensorRT/driver version upgrades. (FR-7.5, NFR-8.1)
Epic 7: Iteration & Debugging
US-7.1 — As a power user debugging a specific pipeline stage, I want to run just that stage (e.g., only calibration) against existing artifacts, so that I don't have to re-run the entire slow pipeline to test one change. (FR-8.2)

US-7.2 — As a team standardizing deployment configs, I want to define the full pipeline in a YAML file checked into version control, so that deployments are reproducible and reviewable like code. (FR-8.3)

US-7.3 — As a user re-running the same build repeatedly during iteration, I want unchanged builds to be cached and skipped, so that my iteration loop isn't dominated by redundant multi-minute TensorRT builds. (FR-5.2, NFR-1.1)
Architecture
1. High-Level System Diagram (described)
                          ┌─────────────────────────────────┐

                          │            CLI Layer             │

                          │  (Click/Typer commands, YAML     │

                          │   config parsing, progress UI)   │

                          └────────────────┬──────────────────┘

                                           │

                          ┌────────────────▼──────────────────┐

                          │         Orchestrator / Pipeline    │

                          │   (stage sequencing, manifest,     │

                          │    caching decisions)              │

                          └────┬───────┬───────┬───────┬───────┘

                               │       │       │       │

                    ┌──────────▼─┐ ┌───▼────┐ ┌▼───────┐ ┌▼──────────┐

                    │  Export     │ │Calibrate│ │ Build  │ │ Benchmark │

                    │  Module     │ │ Module  │ │ Module │ │  Module   │

                    └──────────┬──┘ └───┬────┘ └┬───────┘ └┬──────────┘

                               │       │       │           │

                    ┌──────────▼───────▼───────▼───────────▼──────────┐

                    │              Artifact Store                      │

                    │  (ONNX files, engines, cache index, manifests)   │

                    └──────────────────────────┬────────────────────────┘

                                                │

                                    ┌───────────▼────────────┐

                                    │   Triton Export Module  │

                                    │  (config.pbtxt + repo)  │

                                    └──────────────────────────┘
2. Module Breakdown
2.1 trtship.core.model_spec
Holds the internal representation of "a model to be optimized": the nn.Module reference (or loaded checkpoint), example inputs, input/output names and shapes, and detected dynamic dimensions. Every other module operates on a ModelSpec, never on raw PyTorch objects directly — this is the seam that keeps modules decoupled (NFR-5.1).
2.2 trtship.export
exporters/dynamo.py — wraps torch.onnx.export(..., dynamo=True)
exporters/legacy.py — wraps the TorchScript-tracing-based ONNX exporter (fallback)
exporters/torch_export.py — wraps torch.export.export + torch_tensorrt.dynamo path (bypasses ONNX entirely for models where that's more reliable)
diagnostics.py — parses exporter exceptions into structured ExportFailure objects (op name, module path, exporter attempted, suggested fix)
registry.py — plugin registration system for custom op handlers (FR-2.3)
verify.py — runs ONNX Runtime on the exported graph and diffs against PyTorch eager output (FR-2.4)

Design decision: each exporter implements a common Exporter interface (.try_export(model_spec) -> ExportResult | ExportFailure). The orchestrator tries them in a configurable priority order and stops at first success. This is the extension point for adding new export strategies later without touching orchestration logic.
2.3 trtship.calibrate
datasets/base.py — abstract CalibrationDataset (FR-3.2)
datasets/image_folder.py, datasets/tensor_dir.py, datasets/text_batches.py — built-in implementations
calibrators.py — thin wrappers around TensorRT's IInt8EntropyCalibrator2, IInt8MinMaxCalibrator, and a custom percentile calibrator (FR-3.3)
sensitivity.py — per-layer sensitivity analysis: quantize one layer at a time (or use TensorRT's built-in per-layer precision constraints), measure accuracy delta, rank layers (FR-3.4)
2.4 trtship.build
engine_builder.py — wraps the TensorRT Builder/BuilderConfig/NetworkDefinition API; applies precision flags, workspace size, optimization profiles for dynamic shapes
cache.py — computes the cache key (model weight hash + config hash + environment fingerprint), checks/writes the local cache index (FR-5.2, FR-5.3)
shape_profiles.py — dynamic shape profile construction and auto-detection heuristics (FR-4.2, FR-4.3)
2.5 trtship.benchmark
backends/pytorch_backend.py, backends/onnxruntime_backend.py, backends/tensorrt_backend.py — each implements a common run_inference(inputs, n_iters) -> Timings interface
metrics.py — latency percentiles, throughput calculation, GPU memory tracking (via torch.cuda.max_memory_allocated / pynvml)
accuracy.py — task-configurable accuracy comparison (top-1, MSE, custom callable)
report.py — manifest serialization, HTML/PNG report generation (matplotlib or plotly)
2.6 trtship.deploy.triton
config_generator.py — builds config.pbtxt from ModelSpec I/O metadata
repo_builder.py — assembles the versioned model repository directory structure
smoke_test.py — spins up Triton in Docker (via docker Python SDK or subprocess), sends a test request via tritonclient, tears down (FR-6.3)
2.7 trtship.cli
Thin layer using Typer (preferred over argparse for this — auto-generated --help, easy subcommands). Each subcommand maps 1:1 to a pipeline stage plus the composite ship command. Parses YAML config (FR-8.3) into the same config objects the Python API uses, so CLI and library usage are equivalent, not two separate code paths.
2.8 trtship.orchestrator
Sequences stages, decides what needs to (re)run based on cache state, builds the run manifest incrementally, handles GPU memory cleanup between stages (NFR-2.2).
3. Tech Stack
Concern
Choice
Why
Language
Python 3.10+
Matches the PyTorch/TensorRT ecosystem; no reason to leave it
CLI framework
Typer
Better DX than argparse/click alone; auto --help, type-hint-driven
Core DL framework
PyTorch 2.x
Given, per project scope
Export
torch.onnx (dynamo exporter), torch.export, torch_tensorrt
Official, most-maintained paths
Inference engine
TensorRT 10.x Python API
Core to the whole project
ONNX runtime baseline
onnxruntime-gpu
Needed for the ORT comparison backend and export verification
Calibration data handling
Plain PyTorch Dataset/DataLoader
No reason to invent a new data-loading abstraction
Serving target
Triton Inference Server (Docker image)
Industry standard, NVIDIA's own tool — directly relevant
Triton client (smoke test)
tritonclient[grpc,http]
Official client library
Config files
YAML via pydantic models (not raw dict parsing)
Validates config shape, gives good error messages for free
Packaging
pyproject.toml, hatchling or setuptools build backend
Modern standard
Testing
pytest, pytest-cov
Standard
CI
GitHub Actions with a self-hosted or cloud GPU runner (see NFR-7.2)
GPU-dependent tests can't run on GitHub's free CPU runners
Docs
mkdocs-material + mkdocstrings for API reference
Clean, fast to set up, good search
Charts
plotly (interactive HTML) with matplotlib fallback for static PNG
Plotly gives you interactive hover tooltips in the HTML report for free
Containerization
Docker, official image pinned to a specific CUDA/TensorRT/PyTorch combo
NFR-7.1

4. Key Design Decisions & Rationale
Why a ModelSpec abstraction instead of passing nn.Module everywhere? Decouples every downstream stage from PyTorch internals. If you later want to support a second source format (e.g., a raw ONNX file with no PyTorch model behind it), only the ingestion layer changes.

Why try multiple exporters instead of picking the "best" one? Because in practice, no single PyTorch→ONNX export path handles every model reliably — this is the actual pain point the tool exists to solve. Hard-coding one exporter would make this a thin wrapper, not a real tool.

Why build calibration as a pluggable interface rather than assuming image classification? Because real usage spans vision, NLP, and other modalities. A tool that only works for ImageFolder-style data is a demo, not a devtool.

Why generate Triton configs rather than support multiple serving backends? Scope control. Triton is the NVIDIA-standard serving layer and the most relevant to the internship framing; supporting TorchServe, KServe, etc. is a reasonable v2 direction but not necessary to prove the concept.

Why cache engines keyed on a hash rather than by filename? Filenames drift and lie (stale engine with a misleading name is worse than no cache). Content-addressed caching keyed on the actual determinants of engine validity (weights, config, environment) is the only approach that's actually safe to trust (NFR-5.3 depends on this).
End-to-End Build Plan (Month by Month)
Assume ~15-20 focused hours/week alongside coursework. Adjust pacing to your actual availability, but keep the order — later months depend on earlier ones being solid, and calibration/export edge cases are genuinely unpredictable in how long they take, so don't compress those.
Month 1: Foundations + Export Pipeline
Week 1 — Project setup & scaffolding

Repo structure, pyproject.toml, pre-commit hooks (black, ruff, mypy), initial CI skeleton (lint + CPU-only unit tests first — GPU runner comes later)
Define ModelSpec and core Pydantic config models
Pick your first 3 benchmark models: one small CNN (ResNet-18), one from your own prior work (given your NER/transformer background — a small BERT variant is a natural fit and lets you speak to it from experience), one that's intentionally a bit awkward (something with a custom forward path or control flow) to stress-test export early

Week 2-3 — Export pipeline (FR-2.1, FR-2.2)

Implement the dynamo ONNX exporter path end-to-end for the easy model (ResNet-18) first — get one model working fully before generalizing
Implement the legacy TorchScript exporter as fallback
Build the failure-diagnosis layer (FR-2.2) — this is genuinely hard and worth the time; you'll be reading PyTorch/ONNX GitHub issues to understand real failure signatures
Get the transformer model exporting (attention masks, dynamic sequence length are the likely pain points here)

Week 4 — Export verification + plugin registry

Implement verify.py (FR-2.4) — numerical diffing between PyTorch and ONNX Runtime outputs
Implement the custom op registration system (FR-2.3)
Write your first blog post: document 2-3 real export failures you hit and how you diagnosed/fixed them — start this habit now, not retroactively at the end

Milestone check: 3 models export successfully with verified numerical correctness.
Month 2: TensorRT Build + Calibration
Week 5-6 — Engine building (FR-5.1, FR-4.1)

Static-shape FP32 and FP16 engine builds for all 3 models
Get comfortable with the raw TensorRT builder API — this is where you'll spend real time reading NVIDIA's own documentation and sample code
Implement engine caching (FR-5.2, FR-5.3) — content-addressed hashing

Week 7 — Dynamic shapes (FR-4.2, FR-4.3)

Optimization profile construction for variable batch size
Auto-detection heuristic for likely dynamic dims

Week 8 — INT8 calibration (FR-3.1, FR-3.2, FR-3.3)

Implement CalibrationDataset base + 2 built-in implementations (image folder, tensor directory)
Wire up entropy calibration first (TensorRT's default/most common), then MinMax and percentile
This week is likely to run long — INT8 calibration debugging is notoriously fiddly. Budget slack.

Milestone check: All 3 models build FP16 and INT8 engines; INT8 accuracy is measured (even informally) and roughly sane.
Month 3: Benchmarking + Sensitivity Analysis
Week 9-10 — Benchmark harness (FR-7.1, FR-7.2, NFR-1.3)

Implement the three backends (PyTorch eager, ONNX Runtime, TensorRT) behind a common interface
Correct GPU warmup/synchronization methodology — this is easy to get subtly wrong and produce misleading numbers; cross-check your latency numbers against trtexec's own benchmark output as a sanity check
Latency percentiles, throughput at multiple batch sizes, memory tracking

Week 11 — Accuracy comparison + sensitivity analysis (FR-7.3, FR-3.4)

Wire up validation-set accuracy comparison across precision levels
Implement per-layer sensitivity analysis — this is one of the more novel/impressive pieces, worth extra polish
Implement mixed-precision fallback (auto-keep worst layers in FP16)

Week 12 — Reporting (FR-7.4, FR-7.5)

Plotly-based HTML report generation
JSON manifest save/reload
Second blog post: your INT8 accuracy findings across models — this is genuinely interesting content if you have real numbers

Milestone check: Full benchmark report (3 backends × 3 precisions × multiple batch sizes) generated for all 3 models, with charts.
Month 4: Triton Deployment + Expand Model Coverage
Week 13-14 — Triton export (FR-6.1, FR-6.2)

config.pbtxt generation from ModelSpec
Model repository directory assembly
Dynamic batching config

Week 15 — Smoke testing (FR-6.3)

Docker-based Triton spin-up/teardown automation
tritonclient test request + response verification
This is a good week to also add a 4th and 5th benchmark model (a detection model like YOLO is a strong addition — different I/O shape complexity than classification)

Week 16 — CLI polish + composite ship command (FR-8.1, FR-8.2, FR-8.3)

Wire the single-command end-to-end path
YAML config support
--help text and example invocations for every subcommand

Milestone check: trtship ship works end-to-end on 5 models, output is a Triton-servable, smoke-tested deployment.
Month 5: Expand Coverage, Testing, Docs, Docker Image
Week 17-18 — Model coverage expansion

Push to 8+ models: add a segmentation model, another transformer variant, something with an unusual op if you haven't hit one naturally yet
Every new model is a chance to find another export edge case — treat this as core work, not padding

Week 19 — Test suite + CI on GPU runners (NFR-5.2, NFR-7.2)

Get coverage to 75%+ on core library code
Set up a GPU-enabled CI runner (GitHub Actions with a self-hosted GPU runner, or a cloud GPU CI service) — this is a real infra task in itself and worth documenting

Week 20 — Docs site + Docker image (NFR-4.1, NFR-7.1)

mkdocs-material site with quickstart, API reference, troubleshooting guide (your top-20 failure modes from NFR-4.3, written up properly)
Official Docker image, pinned versions, published to Docker Hub or GHCR

Milestone check: New user can go from zero to working benchmark report in under 10 minutes using only the docs.
Month 6: Polish, Real Users, Upstream Contribution
Week 21-22 — PyPI release + soliciting real usage

Publish v0.1.0 to PyPI
Share in relevant places: r/MachineLearning, HuggingFace forums, your research network, relevant Discord/Slack communities for ML infra
Actively ask 3-5 people to try it on their own models — this is how you find the failure modes you'd never find yourself, and it's what makes "production" claim real

Week 23 — Respond to real feedback

Fix whatever real users actually hit (this will not match what you expected)
This week's bug reports are your best interview material — "here's a real user's model that broke my assumptions and how I fixed it" is a great story

Week 24 — Upstream contribution attempt (Path C from earlier discussion)

With real experience now in hand, look for a concrete, scoped issue in torch-tensorrt, ONNX Runtime, or Triton's client libraries that your trtship work makes you well-positioned to fix
Submit a PR. This may not land in week 24 exactly (review cycles take time) — start this conversation with maintainers earlier if a good issue surfaces sooner; don't wait until the last week to open the PR

Final deliverable checklist:

PyPI package, real download numbers
GitHub repo with real stars/issues from non-you accounts
Docs site live
8+ models benchmarked, results in the repo
Blog series (aim for 3-4 posts minimum)
Docker image published
CI green on GPU runners
Upstream PR opened (merged is ideal, opened-and-under-review is still a strong story)
Time Budget Reality Check
This plan assumes things go roughly to plan. They won't, entirely — INT8 calibration and export edge cases are the two likeliest places to blow the schedule, because their difficulty is inherently unpredictable (you don't know which op will break until it does). Build in slack by treating Month 5's "expand to 8+ models" as flexible scope you can compress to 6 models if Months 1-2 ran long, rather than something that compresses the deployment/docs work at the end. The deployment, docs, and real-user weeks are what make this "production" rather than "a working repo" — don't let them be the first thing cut.
What to Expect From the Final Deliverable
1. Repository Structure
trtship/

├── pyproject.toml

├── README.md                    # quickstart, install, one worked example

├── LICENSE                      # Apache 2.0 or MIT — pick one, matches ecosystem norms

├── CONTRIBUTING.md

├── docker/

│   ├── Dockerfile               # pinned CUDA/TensorRT/PyTorch versions

│   └── docker-compose.yml       # tool + Triton for local smoke testing

├── src/trtship/

│   ├── __init__.py

│   ├── core/

│   │   ├── model_spec.py

│   │   └── config.py            # pydantic config models

│   ├── export/

│   │   ├── exporters/

│   │   ├── diagnostics.py

│   │   ├── registry.py

│   │   └── verify.py

│   ├── calibrate/

│   │   ├── datasets/

│   │   ├── calibrators.py

│   │   └── sensitivity.py

│   ├── build/

│   │   ├── engine_builder.py

│   │   ├── cache.py

│   │   └── shape_profiles.py

│   ├── benchmark/

│   │   ├── backends/

│   │   ├── metrics.py

│   │   ├── accuracy.py

│   │   └── report.py

│   ├── deploy/

│   │   └── triton/

│   ├── orchestrator.py

│   └── cli.py

├── tests/

│   ├── unit/                    # per-module, mockable, run on every push

│   ├── integration/             # multi-stage, GPU-required

│   └── models/                  # test fixture models incl. deliberately-broken ones

├── examples/

│   ├── resnet18/

│   ├── bert_classification/

│   ├── yolo_detection/

│   └── ...                      # one dir per benchmark model, each with a README

├── docs/                        # mkdocs source

│   ├── quickstart.md

│   ├── troubleshooting.md       # top-20 failure modes, written up properly

│   ├── api/                     # auto-generated from docstrings

│   └── architecture.md

└── .github/workflows/

    ├── lint.yml

    ├── test-cpu.yml

    └── test-gpu.yml             # self-hosted or cloud GPU runner
2. What "Done" Actually Looks Like, Concretely
Install experience:

pip install trtship

# or

docker pull ghcr.io/<you>/trtship:latest

Quickstart experience (this needs to actually work, first try, for a new user):

import torch

import torchvision

from trtship import ship

model = torchvision.models.resnet18(pretrained=True).eval().cuda()

example_input = torch.randn(1, 3, 224, 224).cuda()

result = ship(

    model,

    example_inputs=example_input,

    precision="int8",

    calibration_data="./imagenet_calib_subset/",

    output_dir="./resnet18_deployment/",

)

print(result.benchmark_report_path)   # HTML report

print(result.triton_repo_path)        # ready to serve

CLI equivalent:

trtship ship \

  --model resnet18.pt \

  --example-input "1,3,224,224" \

  --precision int8 \

  --calibration-data ./imagenet_calib_subset/ \

  --output-dir ./resnet18_deployment/

What comes out the other end:

resnet18_deployment/model.onnx — verified export
resnet18_deployment/model_int8.engine — built, cached TensorRT engine
resnet18_deployment/triton_repo/resnet18/1/model.plan + config.pbtxt — ready for tritonserver --model-repository=...
resnet18_deployment/report.html — interactive benchmark comparison
resnet18_deployment/manifest.json — full reproducibility record

Serving it for real:

docker run --gpus all -p 8000:8000 -p 8001:8001 \

  -v ./resnet18_deployment/triton_repo:/models \

  nvcr.io/nvidia/tritonserver:24.xx-py3 \

  tritonserver --model-repository=/models

This should just work, no manual config editing, because trtship already generated a correct config.pbtxt.
3. The Benchmark Report Content (what's actually in report.html)
Summary table: latency (p50/p90/p99), throughput at batch 1/4/16, peak memory, accuracy — one row per backend×precision combination
Latency-vs-precision bar chart
Throughput-vs-batch-size line chart, one line per backend
Accuracy-vs-precision chart (only if validation data was supplied)
Per-layer sensitivity chart (INT8 runs only)
Environment block: GPU model, driver version, CUDA version, TensorRT version, PyTorch version — for reproducibility
4. Evidence of "Production" You Should Be Able to Point To
By the end, you should be able to answer each of these with a real link or number, not a claim:

"How many people besides you have used it?" → GitHub issues/discussions from other accounts, PyPI download count
"Where does it run in the real world?" → your own deployed example (even a small demo service counts if it's externally reachable and actually serving), or a documented case of someone else using it on their model
"What broke that you didn't expect?" → your blog posts, your issue tracker history
"Did anything you built get used by a project bigger than yours?" → the upstream PR, merged or under review
5. How This Should Show Up on a Resume / in an Interview
Resume line (roughly):

Built and published trtship, an open-source PyTorch→TensorRT deployment tool with automatic export fallback, INT8 calibration, and Triton config generation; benchmarked across 8+ model architectures; N downloads/stars; contributed [upstream PR] to [torch-tensorrt/ORT/Triton].

Interview material this naturally produces:

A specific, technical story about an export failure you diagnosed and fixed (you'll have several — pick the most interesting)
A real, data-backed opinion on FP16 vs INT8 tradeoffs, because you measured it yourself across multiple architectures, not read it in a blog post
An answer to "tell me about a time you had to debug something with no clear error message" — the export diagnostics work is exactly this
A concrete answer to "have you contributed to open source" that isn't a typo-fix PR
6. Honest Risk Notes
INT8 calibration accuracy can legitimately be bad for some architectures (especially transformers, which are more quantization-sensitive than CNNs). This is not a failure of your tool — document it honestly in your report rather than hiding it. A tool that's honest about a precision level's limitations is more credible than one that oversells.
GPU access is a real constraint. You need a CUDA-capable NVIDIA GPU for essentially all of this. If you don't have one locally, budget for cloud GPU costs (a single RTX 4090 or A10 instance for the build/test-heavy weeks) — this is a real line-item cost of this project, not optional.
TensorRT version churn is real. NVIDIA ships new TensorRT versions fairly often, and APIs do shift between major versions. Pin your dependency versions explicitly and document exactly what you tested against, rather than claiming broad compatibility you haven't verified.


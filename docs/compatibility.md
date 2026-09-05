# Compatibility matrix

Per NFR-3.1 and NFR-3.3, trtship documents exactly what it has been tested
against rather than claiming broad compatibility. This file is the record.

**Status: unfilled.** Run `trtship doctor` on each machine and paste the results
below. Until the GPU row is filled in, every claim about INT8, FP8 or sparsity in
this project is untested.

## Development machines

| Role | OS / arch | Python | torch | CUDA | TensorRT | GPU | SM | Stages available |
|---|---|---|---|---|---|---|---|---|
| Primary dev | macOS 27 / arm64 (M4 Pro) | 3.11 | 2.10.0 | — | — | — | — | export, verify, triton-config, report |
| GPU laptop | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Library machine | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

macOS has no CUDA and NVIDIA publishes no TensorRT build for it. This is not a
packaging gap that can be worked around — the GPU stages must run elsewhere.

## Precision support by compute capability

Encoded in `src/trtship/env.py` and reported by `trtship doctor`, so an
unsupported precision is caught in week 1 rather than mid-calibration.

| Precision | Minimum SM | Architecture |
|---|---|---|
| FP32 | any | — |
| FP16 | 5.3 | Maxwell 5.3+ |
| INT8 | 6.1 | Pascal 6.1+ (dp4a); fast INT8 tensor cores from Turing 7.5 |
| FP8 | 8.9 | Ada 8.9 / Hopper 9.0 |
| Structured sparsity | 8.0 | Ampere+ |

## Open questions for the first GPU session

Answer these before Phase 2 starts; each one changes scope if the answer is no.

- [ ] Does the GPU laptop's card reach SM 7.5? Below that, INT8 works but without
      tensor-core acceleration, and the benchmark story gets much weaker.
- [ ] Does TensorRT 10.x install cleanly there, and against which CUDA?
- [ ] Is Docker available? Without it the Triton smoke test (FR-6.3) cannot run.
- [ ] Can the laptop stay on as a self-hosted GitHub Actions runner (Phase 2)?
- [ ] Library machines: admin rights? Docker? persistent storage? session limits?
- [ ] Enough VRAM for the largest planned model at the largest planned batch size?

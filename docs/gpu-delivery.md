# GPU image delivery

DefectDock keeps the lightweight API image as the default. GPU training is an
explicit delivery variant built from the same pinned base image, source tree,
and `uv.lock`.

## Validated baseline

The following local baseline passed on 2026-08-30:

- NVIDIA GeForce RTX 2060, compute capability 7.5, 6 GiB VRAM;
- host driver 595.97 and Docker NVIDIA runtime;
- Python 3.13.15;
- PyTorch 2.13.0+cu130 and TorchVision 0.28.0+cu130;
- one epoch of the synthetic object-detection smoke workload on `device=cuda`;
- generated `best.ckpt`, `last.ckpt`, metrics, events, and `run.manifest.json`;
- fixable High/Critical container vulnerability gate: zero findings.

This proves the packaged training path on the stated baseline. It does not
replace accuracy, latency, stability, or capacity acceptance on representative
customer data and the customer's exact driver/hardware combination.

## Build and run

```bash
docker compose -f compose.yaml -f compose.gpu.yaml build api
docker compose -f compose.yaml -f compose.gpu.yaml up api
```

For a traceable release build, set `DEFECTDOCK_BUILD_REVISION` to the full Git
commit before building. The value is validated before it is copied into a run
manifest and is also written to the OCI `org.opencontainers.image.revision`
label. GitHub GPU acceptance injects `GITHUB_SHA` automatically.

The GPU override installs the locked `train` extra and requests all available
GPUs through the Compose `gpus` field. The API remains available on port 8000
and uses the same `/data` persistence contract as the lightweight image.

Run the two local acceptance checks directly against the built image:

```bash
docker run --rm --gpus all \
  -v "$PWD/scripts:/smoke:ro" defectdock:gpu \
  python /smoke/check_gpu_runtime.py

docker run --rm --gpus all \
  -v "$PWD/scripts:/smoke:ro" defectdock:gpu \
  python /smoke/smoke_train.py /data/hardware-smoke --device cuda
```

On Windows PowerShell, `$PWD` can be used in the same bind mount expression.
The smoke workspace must be empty because the command rejects accidental
overwrites.

## Automated acceptance

`.github/workflows/gpu-smoke.yml` is intentionally manual and targets a
self-hosted Linux runner carrying the labels `gpu` and `nvidia`. It builds the
locked image, verifies CUDA visibility, runs the one-epoch workload, generates
an SBOM and both vulnerability reports, and archives checksummed evidence.

Before distributing a GPU image, record the exact image digest, GPU model,
compute capability, driver, CUDA build, test result, SBOM, vulnerability
decision, and CUDA/NVIDIA license review beside the release.

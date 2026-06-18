# ComfyUI-Human3R — RunComfy port bundle

Artifacts for deploying the **ComfyUI-Human3R** custom node pack (video → SMPL
human motion, wrapping [fanegg/Human3R](https://github.com/fanegg/Human3R)) onto
**RunComfy**.

This pack runs in **pure PyTorch** — no compiled CUDA extension required. The
upstream `curope` extension is optional; its pure-PyTorch RoPE fallback was patched
to compute rotations the same way the CUDA kernel does, so it works on any torch
(verified numerically equivalent to the CUDA path, max abs diff 6.6e-4).

## Contents
- **`ComfyUI-Human3R-runcomfy.tar.gz`** (4.9 MB) — the full node pack: all code,
  the SMPL aux files, and the fixed RoPE fallback. The 3.39 GB model checkpoint is
  **not** included (download separately, see below).
  sha256 `1c937213e89e05f2b59a9ba3570457d63e5efcf07cfcdb5b25e00e5b7cf5cf12`
- **`docs/RUNCOMFY_PORT_STEPS.md`** — step-by-step operator runbook for RunComfy.
- **`pos_embed_FIXED_reference.py`** — the patched RoPE file, for reference.

## Install on RunComfy (summary — full detail in the runbook)
1. Get the tarball onto the machine (browser upload, or RunComfy "download from URL"
   pointed at this file's raw GitHub link) and extract into
   `ComfyUI/custom_nodes/` → `ComfyUI/custom_nodes/ComfyUI-Human3R/`.
2. Let ComfyUI Manager `pip install -r requirements.txt` (installs `smplx`, `roma`,
   etc. — pure-Python). Do not let torch be changed.
3. Download the checkpoint to `ComfyUI-Human3R/human3r_src/human3r_672S.pth`:
   `huggingface-cli download faneggg/human3r human3r_672S.pth`
4. Restart ComfyUI, run the smoke workflow, then **Cloud Save**.

> Note: this repo intentionally ships the pack as a compressed tarball rather than
> as a raw, Git-URL-installable node folder, because two SMPL `.pkl` aux files are
> ~578 MB each (over GitHub's 100 MB per-file limit). The tarball compresses the
> whole pack to 4.9 MB.

# Porting ComfyUI-Human3R into RunComfy — operator runbook

> Audience: Robert, doing the manual GUI steps. Claude can't reach RunComfy
> (no terminal/API), so everything here is by hand in the web UI. Steps are
> ordered; **STOP points** mark where Claude needs a result back before you
> continue.

## ✅ Status (2026-06-18) — port is unblocked
We probed a live RunComfy machine and fixed the one real blocker:

- **RunComfy runs** Python 3.12.12, **torch 2.11.0+cu130**, CUDA 13.0, GPU RTX
  A6000 (sm_86), 48 GB. This is a total mismatch with the pack's original build
  env (py3.11 / torch 2.4.1+cu124 / sm_89), so the bundled compiled `curope.so`
  **cannot load there**.
- Human3R used to **crash** without `curope` (a bug in its pure-PyTorch RoPE
  fallback: a `-1` padding position became an out-of-range table lookup).
- **We fixed the fallback** so it computes rotations the same way the CUDA kernel
  does (`freq = pos × inv_freq`, on the fly). Verified on RunPod: identical
  outputs to the CUDA path (max abs diff 6.6e-4, i.e. float noise) at the same
  speed (~30 s). **So Human3R now needs NO compiled extension on any torch** — it
  runs in pure PyTorch on RunComfy's 2.11 stack.

**Net:** just install the pack, let Manager pull the pip deps, add the checkpoint,
and run. The bundled `.so` is now optional dead weight on RunComfy (ignored
because of the version mismatch) and only helps on an exactly-matching host.

**One residual unknown:** torch 2.11 is much newer than the 2.4.1 we validated
on. The RoPE crash is fixed, but other parts of Human3R could trip on 2.11 — that
can only be shaken out by the smoke test in Phase C, on RunComfy itself.

## 0. Before you start
- Machine: a **private on-demand** machine. RunComfy's 48 GB A6000 is plenty
  (Human3R needs ~8 GB). State must persist via **Cloud Save**, so don't use the
  shared/playground machine.
- File on your Mac, in `runcomfy/`:
  - `ComfyUI-Human3R-runcomfy.tar.gz` (4.9 MB) — the cleaned pack **with the fixed
    fallback**: all code + aux files. The 3.2 GB checkpoint is excluded
    (re-download in Phase B). (It also still carries the prebuilt `.so`, harmless
    and unused on RunComfy.)

---

## Live port progress (2026-06-18) — runtime fixes baked into the shipped pack
Running the smoke test on RunComfy (torch 2.8 / ComfyUI 0.7) surfaced a chain of
porting issues, each now fixed in the shipped `nodes.py` / pack:
1. ✅ `No module named 'smplx'` — manual uploads don't auto-pip. Fixed by the deps
   helper repo (`ComfyUI-Human3R-Deps`, install via Manager Git-URL).
2. ✅ `No module named 'models.blocks'` — croco's bundled top-level `models`
   (a namespace package) was shadowed by another node's regular `models`. Fixed in
   `nodes.py` via `_ensure_croco_imports()` (binds `sys.modules['models']` to
   croco's models dir before importing dust3r). Logs a marker line when active.
3. ✅ RoPE crash without curope — already fixed earlier (pure-PyTorch fallback
   patch in `pos_embed.py`); the expected log line `cannot find cuda-compiled
   version of RoPE2D, using a slow pytorch version` confirms it.
4. ✅ `UnpicklingError: omegaconf.dictconfig.DictConfig was not an allowed global`
   — torch ≥2.6 defaults `torch.load(weights_only=True)`; the checkpoint stores
   omegaconf configs. Fixed in `nodes.py` load_model: scoped `weights_only=False`
   override (trusted official HF checkpoint) around `from_pretrained`.

**Next unknown:** whether the run completes after fix #4, or surfaces another
torch-2.8 difference. The shipped `nodes.py` (≈14.6 KB, in the bundle repo +
`runcomfy/nodes.py`) contains fixes #2 and #4; the tarball/pack contains all four.

## PHASE A — environment probe ✅ DONE
Already run via the `ComfyUI-EnvProbe` node; result recorded above. Nothing to do
unless RunComfy later changes its machine image — if a run misbehaves, re-run the
probe (`https://github.com/SchwenderOne/ComfyUI-EnvProbe`, Manager → Install via
Git URL → add **Env Probe** → Queue Prompt) and send Claude the versions.

---

## PHASE B — install the pack + checkpoint
1. Get the pack into `ComfyUI/custom_nodes/` as
   `ComfyUI/custom_nodes/ComfyUI-Human3R/`. **Reliable way (confirmed approach):**
   extract the `.tar.gz` on your Mac (Finder double-click → `ComfyUI-Human3R`
   folder) and **drag-drop that folder** into the web file browser under
   `custom_nodes/`. Uploading the raw tarball and extracting it *on* RunComfy only
   works if its file browser can extract archives — **TODO: verify what RunComfy's
   file browser actually supports (folder upload? archive extraction?)**; until
   confirmed, prefer the Mac-extract-then-drag-drop route.
   **⚠️ File-placement gotcha (cost hours):** the node file is
   `ComfyUI-Human3R/nodes.py` at the **pack ROOT — beside `__init__.py`,
   `requirements.txt`, and the `human3r_src/` folder**. When replacing just
   `nodes.py` later, it must go there, NOT inside `human3r_src/`. The RunComfy file
   browser silently put earlier single-file uploads into `human3r_src/`, so ComfyUI
   kept importing the original pack-root file (no error changed, no marker line).
   To confirm a replacement took: check the file's byte size and look for the
   `[Human3R] _ensure_croco_imports …` marker line in the log.
2. **Install the pip deps** — ⚠️ a manually-uploaded folder does **NOT** get its
   `requirements.txt` auto-installed (only Manager's *Install via Git URL* runs
   pip; the startup "installing dependencies done." line is for other nodes). The
   pack will import fine at load (its heavy deps are lazy) but then fail at run time
   with `No module named 'smplx'`. Fix: install the **deps helper repo** via
   ComfyUI-Manager → **Install via Git URL** →
   `https://github.com/SchwenderOne/ComfyUI-Human3R-Deps`
   (a no-op pack carrying Human3R's requirements; Manager pip-installs smplx, roma,
   scipy, accelerate, …). Do **not** let anything change torch. Then **Restart**.
   (Confirmed needed 2026-06-18 on a torch-2.8/ComfyUI-0.7 RunComfy machine.)
3. Download the checkpoint `human3r_672S.pth` (3.39 GB) into the pack at
   **`ComfyUI-Human3R/human3r_src/human3r_672S.pth`** (the loader's default path).
   Use RunComfy's in-platform **download-from-URL** from HuggingFace; the in-pack
   command Human3R documents is:
   `huggingface-cli download faneggg/human3r human3r_672S.pth`
   (public repo; the aux SMPL files are already inside the tarball).
4. **Restart** ComfyUI. Watch the log panel for `IMPORT FAILED` on the pack.

   **➤ STOP if you see `IMPORT FAILED`** — copy the traceback to Claude.

   (You may see `cannot find cuda-compiled version of RoPE2D, using a slow pytorch
   version` — that is now **expected and fine**; the fixed fallback is what runs.)

---

## PHASE C — smoke test
1. Load a workflow equivalent to `test_assets/workflows/human3r_smoke.json` and
   upload the test clip `GoodMornin1.mp4` (Claude can hand you a RunComfy-friendly
   version if paths differ). DINOv2 is pulled via `torch.hub` on first run (needs
   outbound network once, then cached).
2. **Queue Prompt.** Expected: completes in tens of seconds, returns SMPL params,
   no crash.

   **➤ STOP — report to Claude:** completed?, runtime, and any error. This is the
   real test of whether torch 2.11 has any *other* surprises for Human3R.

---

## PHASE D — persist
Once green, **Cloud Save** to snapshot nodes + deps + checkpoint into a
reproducible container so you don't redo Phase B next session.

---

### Reminder
When done for the session, **stop the RunComfy machine AND the RunPod pod** to
stop billing — RunComfy persists via Cloud Save, RunPod persists on `/workspace`.

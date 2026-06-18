"""
ComfyUI nodes for Human3R — everyone everywhere all at once.

Loader node: loads ARCroco3DStereo from a .pth checkpoint.
Inference node: runs inference on a video and saves SMPL-X parameters per frame.

No SMPL-X body model files are required for inference (those are only needed
for mesh rendering via prepare_output, which is not called here).
"""
import os
import sys
import tempfile
import shutil
import time

import numpy as np
import torch

_PACK_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_PACK_DIR, "human3r_src")

# Ensure src is in path (also set by __init__.py but nodes.py may be imported directly).
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_DEFAULT_MODEL_PATH = os.path.join(_SRC_DIR, "human3r_672S.pth")


def _ensure_croco_imports():
    """Make croco's bundled top-level ``models`` package win over any OTHER ComfyUI
    custom node that also registers a top-level ``models``.

    croco uses absolute imports (``from models.blocks import Mlp``), and
    ``dust3r.heads`` imports them *before* dust3r's own ``path_to_croco`` adds
    croco to ``sys.path``. On a busy ComfyUI (many custom nodes) the name
    ``models`` then resolves to the wrong package, raising
    ``No module named 'models.blocks'``. Fix: put croco first on ``sys.path`` and
    drop any cached foreign ``models`` so the re-import resolves to ours.
    """
    croco = os.path.join(_SRC_DIR, "croco")
    if not os.path.isdir(croco):
        return
    if croco in sys.path:
        sys.path.remove(croco)
    sys.path.insert(0, croco)
    croco_abs = os.path.abspath(croco)

    def _under_croco(mod):
        f = getattr(mod, "__file__", None)
        if f:
            return os.path.abspath(f).startswith(croco_abs)
        return any(
            os.path.abspath(pp).startswith(croco_abs)
            for pp in (getattr(mod, "__path__", None) or [])
        )

    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            mod = sys.modules.get(name)
            if mod is not None and not _under_croco(mod):
                del sys.modules[name]


def _extract_frames(video_path, subsample=1, max_frames=None):
    """Extract frames from video or directory into a temp dir. Returns (img_paths, tmpdir)."""
    import cv2
    import glob

    if os.path.isdir(video_path):
        img_paths = sorted(glob.glob(os.path.join(video_path, "*")))
        if max_frames is not None:
            img_paths = img_paths[:max_frames]
        img_paths = img_paths[::subsample]
        return img_paths, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = list(range(0, total_frames, 1))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    frame_indices = frame_indices[::subsample]

    tmpdir = tempfile.mkdtemp()
    img_paths = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        fp = os.path.join(tmpdir, f"frame_{i:06d}.jpg")
        cv2.imwrite(fp, frame)
        img_paths.append(fp)
    cap.release()
    return img_paths, tmpdir


def _prepare_views(img_paths, size, img_res, reset_interval):
    """Build the view dicts expected by inference_recurrent_lighter."""
    from copy import deepcopy
    from dust3r.utils.image import load_images, pad_image
    from dust3r.utils.geometry import get_camera_parameters

    images = load_images(img_paths, size=size)
    K_mhmr = None
    if img_res is not None:
        K_mhmr = get_camera_parameters(img_res, device="cpu")

    views = []
    for i, img in enumerate(images):
        view = {
            "img": img["img"],
            "ray_map": torch.full(
                (img["img"].shape[0], 6, img["img"].shape[-2], img["img"].shape[-1]),
                float("nan"),
            ),
            "true_shape": torch.from_numpy(img["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0),
        }
        if img_res is not None:
            view["img_mhmr"] = pad_image(view["img"], img_res)
            view["K_mhmr"] = K_mhmr
        views.append(view)
        if (i + 1) % reset_interval == 0:
            overlap = deepcopy(view)
            overlap["reset"] = torch.tensor(False).unsqueeze(0)
            views.append(overlap)
    return views


def _save_outputs(outputs, out_dir):
    """
    Save raw SMPL-X parameters and camera data from inference outputs.

    No SMPL-X body model files are needed here — we save the tensors as-is.
    """
    import roma

    preds = outputs["pred"]
    views = outputs["views"]

    # Strip reset-overlap views (same logic as prepare_output in demo.py).
    reset_mask = torch.cat([v["reset"] for v in views], 0)
    shifted = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], 0)
    preds = [p for p, m in zip(preds, shifted) if not m]
    views = [v for v, m in zip(views, shifted) if not m]

    os.makedirs(os.path.join(out_dir, "smpl"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "camera"), exist_ok=True)

    from dust3r.utils.camera import pose_encoding_to_camera

    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu() for pred in preds
    ]

    for f_id, (pred, pr_pose) in enumerate(zip(preds, pr_poses)):
        smpl_shape = pred.get("smpl_shape", torch.empty(1, 0, 10))[0]
        smpl_rotvec = roma.rotmat_to_rotvec(
            pred.get("smpl_rotmat", torch.empty(1, 0, 53, 3, 3))[0]
        )
        smpl_transl = pred.get("smpl_transl", torch.empty(1, 0, 3))[0]
        smpl_expression = pred.get("smpl_expression", [None])[0]
        smpl_id = pred.get("smpl_id", torch.empty(1, 0))[0]

        np.savez(
            os.path.join(out_dir, "smpl", f"{f_id:06d}.npz"),
            shape=smpl_shape.numpy(),
            rotvec=smpl_rotvec.numpy(),
            transl=smpl_transl.numpy(),
            expression=(smpl_expression.numpy() if smpl_expression is not None else np.array([])),
            smpl_id=smpl_id.numpy(),
        )
        c2w = pr_pose[0].numpy()
        np.savez(os.path.join(out_dir, "camera", f"{f_id:06d}.npz"), cam2world=c2w)

    return len(preds)


# ---------------------------------------------------------------------------
# Node: Human3RModelLoader
# ---------------------------------------------------------------------------

class Human3RModelLoader:
    """
    Loads a Human3R checkpoint (ARCroco3DStereo) onto the specified device.
    The loaded model is cached in-process — re-running with the same path is fast.
    """

    _cache: dict = {}

    @classmethod
    def INPUT_TYPES(cls):
        default_path = _DEFAULT_MODEL_PATH if os.path.isfile(_DEFAULT_MODEL_PATH) else ""
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": default_path,
                    "multiline": False,
                    "tooltip": "Absolute path to a human3r_*.pth checkpoint file.",
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("HUMAN3R_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Human3R"
    DESCRIPTION = "Load a Human3R checkpoint (ARCroco3DStereo) ready for inference."

    def load_model(self, model_path, device):
        _ensure_croco_imports()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {model_path}\n"
                f"Download with: huggingface-cli download faneggg/human3r human3r_672S.pth "
                f"--local-dir {_SRC_DIR}"
            )

        cache_key = (model_path, device)
        if cache_key in Human3RModelLoader._cache:
            print(f"[Human3R] Using cached model for {model_path}")
            return (Human3RModelLoader._cache[cache_key],)

        print(f"[Human3R] Loading model from {model_path} ...")
        from dust3r.model import ARCroco3DStereo

        model = ARCroco3DStereo.from_pretrained(model_path).to(device)
        model.eval()
        Human3RModelLoader._cache[cache_key] = model
        print(f"[Human3R] Model loaded on {device}.")
        return (model,)


# ---------------------------------------------------------------------------
# Node: Human3RVideoInference
# ---------------------------------------------------------------------------

class Human3RVideoInference:
    """
    Runs Human3R inference on a video file and saves per-frame SMPL-X parameters
    (shape, rotvec, transl, expression) and camera data as .npz files.

    SMPL-X body model files are NOT required — we save the raw network outputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HUMAN3R_MODEL", {}),
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to an MP4 video or a directory of frame images.",
                }),
                "output_dir": ("STRING", {
                    "default": "/tmp/human3r_output",
                    "multiline": False,
                    "tooltip": "Directory where SMPL and camera .npz files are saved.",
                }),
                "size": ("INT", {
                    "default": 512, "min": 224, "max": 1024, "step": 32,
                    "tooltip": "Rescale input to this resolution (512 for 672/896 models).",
                }),
                "subsample": ("INT", {
                    "default": 1, "min": 1, "max": 10,
                    "tooltip": "Keep every N-th frame.",
                }),
                "max_frames": ("INT", {
                    "default": 0, "min": 0, "max": 10000,
                    "tooltip": "Max frames to process (0 = all frames).",
                }),
                "reset_interval": ("INT", {
                    "default": 100, "min": 1, "max": 10000,
                    "tooltip": "Recurrent state reset interval (100 is recommended).",
                }),
            },
            "optional": {
                "use_ttt3r": ("BOOLEAN", {"default": False,
                    "tooltip": "Enable test-time training (slower but more accurate)."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("output_dir", "frame_count", "status")
    FUNCTION = "run_inference"
    CATEGORY = "Human3R"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Run Human3R inference on a video. Outputs per-frame SMPL-X parameters "
        "(shape, rotvec, transl) and camera matrices saved as .npz in output_dir."
    )

    def run_inference(
        self,
        model,
        video_path,
        output_dir,
        size,
        subsample,
        max_frames,
        reset_interval,
        use_ttt3r=False,
    ):
        _ensure_croco_imports()
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input not found: {video_path}")

        device = next(model.parameters()).device
        max_frames_arg = max_frames if max_frames > 0 else None

        print(f"[Human3R] Extracting frames from {video_path} ...")
        img_paths, tmpdir = _extract_frames(video_path, subsample=subsample, max_frames=max_frames_arg)
        if not img_paths:
            raise ValueError(f"No frames found in: {video_path}")
        print(f"[Human3R] {len(img_paths)} frames to process.")

        img_res = getattr(model, "mhmr_img_res", None)
        views = _prepare_views(img_paths, size, img_res, reset_interval)

        if tmpdir is not None:
            shutil.rmtree(tmpdir)

        from dust3r.inference import inference_recurrent_lighter

        print(f"[Human3R] Running inference ({len(views)} views, use_ttt3r={use_ttt3r}) ...")
        t0 = time.time()
        with torch.no_grad():
            outputs, _ = inference_recurrent_lighter(
                views, model, device, use_ttt3r=use_ttt3r
            )
        elapsed = time.time() - t0
        print(f"[Human3R] Inference done in {elapsed:.1f}s.")

        os.makedirs(output_dir, exist_ok=True)
        frame_count = _save_outputs(outputs, output_dir)
        status = f"OK — {frame_count} frames in {elapsed:.1f}s ({elapsed/max(frame_count,1):.2f}s/frame)"
        print(f"[Human3R] {status}")
        print(f"[Human3R] Output saved to {output_dir}")
        return (output_dir, frame_count, status)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "Human3RModelLoader": Human3RModelLoader,
    "Human3RVideoInference": Human3RVideoInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Human3RModelLoader": "Human3R Model Loader",
    "Human3RVideoInference": "Human3R Video Inference",
}

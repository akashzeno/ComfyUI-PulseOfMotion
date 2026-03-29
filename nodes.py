"""ComfyUI nodes for Pulse of Motion (Visual Chronometer) PhyFPS prediction."""

import os
import time
import torch
import numpy as np

import comfy.model_management as mm
import comfy.utils
import folder_paths

LOG_PREFIX = "\033[96m[PulseOfMotion]\033[0m"


def get_available_devices():
    """Detect available devices dynamically."""
    devices = ["auto", "cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            devices.append(f"cuda:{i}")
    return devices

from .model.fps_predictor import FPSPredictor

# Register model folder
MODELS_DIR = os.path.join(folder_paths.models_dir, "pulse_of_motion")
os.makedirs(MODELS_DIR, exist_ok=True)

HF_REPO_ID = "xiangbog/Visual_Chronometer"
HF_CKPT_FILENAME = "vc_common_10_60fps.ckpt"

# Model config matching the original config_fps.yaml
MODEL_CONFIG = {
    "ddconfig": {
        "double_z": True,
        "z_channels": 4,
        "resolution": 216,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 2, 4, 4],
        "temporal_down_factor": 1,
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0,
    },
    "ppconfig": {
        "temporal_scale_factor": 4,
        "z_channels": 4,
        "out_ch": 4,
        "ch": 4,
        "attn_temporal_factor": [],
    },
    "embed_dim": 4,
    "use_quant_conv": True,
    "hidden_dim": 1024,
    "n_layers": 1,
    "freeze_encoder": True,
}


def get_checkpoint_path():
    """Return local checkpoint path, downloading from HuggingFace if needed."""
    ckpt_path = os.path.join(MODELS_DIR, HF_CKPT_FILENAME)
    if os.path.exists(ckpt_path):
        return ckpt_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "Checkpoint not found and `huggingface_hub` is not installed.\n"
            "Install with: pip install huggingface_hub\n"
            f"Or manually download from https://huggingface.co/{HF_REPO_ID} to: {ckpt_path}"
        )

    print(f"[PulseOfMotion] Downloading checkpoint from HuggingFace ({HF_REPO_ID})...")
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_CKPT_FILENAME,
        local_dir=MODELS_DIR,
    )
    print(f"[PulseOfMotion] Checkpoint saved to: {ckpt_path}")
    return ckpt_path


def load_fps_predictor(device, dtype, ckpt_path=None):
    """Instantiate and load the FPSPredictor model."""
    if ckpt_path is None:
        ckpt_path = get_checkpoint_path()
    config = MODEL_CONFIG.copy()

    # Build model without loading base VAE checkpoint (we load the full FPS predictor checkpoint)
    model = FPSPredictor(
        ddconfig=config["ddconfig"],
        ppconfig=config["ppconfig"],
        embed_dim=config["embed_dim"],
        use_quant_conv=config["use_quant_conv"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        freeze_encoder=False,  # We load weights manually
        ckpt_path=None,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


class LoadVisualChronometer:
    """Loads the Visual Chronometer (Pulse of Motion) model for PhyFPS prediction."""

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = []
        if os.path.isdir(MODELS_DIR):
            for f in os.listdir(MODELS_DIR):
                if f.endswith((".ckpt", ".pt", ".pth", ".safetensors")):
                    checkpoints.append(f)
        if not checkpoints:
            checkpoints = [HF_CKPT_FILENAME]
        devices = get_available_devices()
        return {
            "required": {
                "model_name": (sorted(checkpoints), {"default": checkpoints[0], "tooltip": "Visual Chronometer checkpoint file. Auto-downloads from HuggingFace if not found locally."}),
                "device": (devices, {"default": "auto", "tooltip": "Device to load the model on. 'auto' uses ComfyUI's default (usually GPU). Use 'cpu' to free up VRAM for other models."}),
            },
        }

    RETURN_TYPES = ("VC_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Pulse of Motion"
    DESCRIPTION = "Loads the Visual Chronometer model for predicting Physical FPS (PhyFPS) from video. Based on the paper 'The Pulse of Motion: Measuring Physical Frame Rate from Visual Dynamics'."

    def load(self, model_name, device="auto"):
        if device == "auto":
            device = mm.get_torch_device()
        else:
            device = torch.device(device)

        dtype = torch.float32 if device.type == "cpu" else mm.unet_dtype()
        if dtype == torch.float16:
            dtype = torch.float32  # model needs float32 for stable inference

        ckpt_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(ckpt_path):
            ckpt_path = get_checkpoint_path()

        device_label = device if isinstance(device, str) else str(device)
        if torch.cuda.is_available() and device.type == "cuda":
            device_label = f"{device} ({torch.cuda.get_device_name(device)})"
        print(f"{LOG_PREFIX} Loading model: {model_name} (device={device_label}, dtype={dtype})")
        t0 = time.time()
        model = load_fps_predictor(device, dtype, ckpt_path)
        print(f"{LOG_PREFIX} Model loaded in {time.time() - t0:.2f}s")
        return (model,)


class PredictPhyFPS:
    """Predicts Physical FPS (PhyFPS) from a batch of video frames using the Visual Chronometer.

    Accepts a ComfyUI IMAGE batch (sequential video frames). Splits the frames into
    overlapping 30-frame clips using a sliding window, predicts PhyFPS per clip,
    and returns the average.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VC_MODEL", {"tooltip": "Visual Chronometer model from the loader node."}),
                "images": ("IMAGE", {"tooltip": "Sequential video frames as an IMAGE batch. Frames are auto-resized to 216x216 internally."}),
                "clip_length": ("INT", {"default": 30, "min": 2, "max": 120, "step": 1, "tooltip": "Number of frames per analysis clip. The model was trained on 30-frame clips. Lower values are faster but less accurate."}),
                "stride": ("INT", {"default": 4, "min": 1, "max": 30, "step": 1, "tooltip": "Step size between clips. Lower = more overlapping clips = smoother average but slower. Higher = fewer clips = faster but coarser."}),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("phyfps", "report")
    FUNCTION = "predict"
    CATEGORY = "Pulse of Motion"
    DESCRIPTION = "Predicts Physical FPS (PhyFPS) from video frames using a sliding window of overlapping clips. Returns the average PhyFPS and a detailed per-segment report."

    def predict(self, model, images, clip_length=30, stride=4):
        """
        Args:
            model: FPSPredictor model
            images: ComfyUI IMAGE tensor [B, H, W, C] in [0, 1] range, RGB
            clip_length: Number of frames per clip
            stride: Sliding window stride
        """
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        t_total = time.time()

        num_frames = images.shape[0]
        h_in, w_in = images.shape[1], images.shape[2]
        if num_frames < clip_length:
            raise RuntimeError(
                f"Not enough frames ({num_frames}) for clip_length={clip_length}. "
                f"Need at least {clip_length} frames."
            )

        print(f"{LOG_PREFIX} Input: {num_frames} frames at {h_in}x{w_in}, clip_length={clip_length}, stride={stride}")

        # Resize frames to 216x216 and normalize to [-1, 1]
        # images: [B, H, W, C] -> [B, C, H, W]
        t0 = time.time()
        frames = images.permute(0, 3, 1, 2)
        frames = torch.nn.functional.interpolate(
            frames, size=(216, 216), mode="bilinear", align_corners=False
        )
        frames = frames * 2.0 - 1.0  # [0,1] -> [-1,1]
        print(f"{LOG_PREFIX} Preprocessing done in {time.time() - t0:.2f}s (resized to 216x216, normalized)")

        # Build sliding window clips
        segment_results = []
        starts = list(range(0, num_frames - clip_length + 1, stride))
        total_clips = len(starts)
        pbar = comfy.utils.ProgressBar(total_clips)
        print(f"{LOG_PREFIX} Running inference on {total_clips} clips...")
        t0 = time.time()

        with torch.no_grad():
            for i, start in enumerate(starts):
                clip = frames[start:start + clip_length]  # [T, C, H, W]
                clip = clip.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
                clip = clip.to(device=device, dtype=dtype)

                pred_log_fps = model(clip)
                fps = torch.exp(pred_log_fps).item()

                mid_frame = start + clip_length // 2
                segment_results.append({
                    "start": start,
                    "end": start + clip_length - 1,
                    "mid": mid_frame,
                    "fps": round(fps, 1),
                })
                pbar.update(1)
                print(f"{LOG_PREFIX}   Clip {i+1}/{total_clips} (frames {start}-{start+clip_length-1}) -> {fps:.1f} PhyFPS")

        if not segment_results:
            raise RuntimeError("No segments could be extracted from the input frames.")

        avg_fps = round(np.mean([r["fps"] for r in segment_results]), 1)
        inference_time = time.time() - t0
        total_time = time.time() - t_total
        print(f"{LOG_PREFIX} Inference done in {inference_time:.2f}s ({inference_time/total_clips:.2f}s per clip)")
        print(f"{LOG_PREFIX} Average PhyFPS: {avg_fps} | Total time: {total_time:.2f}s")

        # Build report string
        lines = [
            f"Pulse of Motion — PhyFPS Prediction",
            f"Total frames: {num_frames} | Clips: {len(segment_results)} | Clip length: {clip_length} | Stride: {stride}",
            f"",
            f"{'Segment':>8}  {'Frames':>12}  {'Mid':>6}  {'PhyFPS':>8}",
            f"{'─' * 8}  {'─' * 12}  {'─' * 6}  {'─' * 8}",
        ]
        for i, r in enumerate(segment_results):
            lines.append(f"{i:>8d}  {r['start']:>5d}-{r['end']:<5d}  {r['mid']:>6d}  {r['fps']:>8.1f}")
        lines.append(f"{'─' * 8}  {'─' * 12}  {'─' * 6}  {'─' * 8}")
        lines.append(f"{'AVG':>8}  {'':>12}  {'':>6}  {avg_fps:>8.1f}")

        report = "\n".join(lines)
        return (avg_fps, report)


class PredictPhyFPSBatch:
    """Predicts PhyFPS for multiple video segments, returning per-segment values as a list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("VC_MODEL", {"tooltip": "Visual Chronometer model from the loader node."}),
                "images": ("IMAGE", {"tooltip": "Sequential video frames as an IMAGE batch. Frames are auto-resized to 216x216 internally."}),
                "clip_length": ("INT", {"default": 30, "min": 2, "max": 120, "step": 1, "tooltip": "Number of frames per analysis clip. The model was trained on 30-frame clips. Lower values are faster but less accurate."}),
                "stride": ("INT", {"default": 4, "min": 1, "max": 30, "step": 1, "tooltip": "Step size between clips. Lower = more overlapping clips = smoother average but slower. Higher = fewer clips = faster but coarser."}),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("avg_phyfps", "segment_fps_list", "report")
    OUTPUT_IS_LIST = (False, True, False)
    FUNCTION = "predict_batch"
    CATEGORY = "Pulse of Motion"
    DESCRIPTION = "Predicts PhyFPS per segment and returns both the average and a list of per-segment FPS values. Useful for analyzing FPS variation across a video."

    def predict_batch(self, model, images, clip_length=30, stride=4):
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        t_total = time.time()

        num_frames = images.shape[0]
        h_in, w_in = images.shape[1], images.shape[2]
        if num_frames < clip_length:
            raise RuntimeError(
                f"Not enough frames ({num_frames}) for clip_length={clip_length}."
            )

        print(f"{LOG_PREFIX} [Batch] Input: {num_frames} frames at {h_in}x{w_in}, clip_length={clip_length}, stride={stride}")

        t0 = time.time()
        frames = images.permute(0, 3, 1, 2)
        frames = torch.nn.functional.interpolate(
            frames, size=(216, 216), mode="bilinear", align_corners=False
        )
        frames = frames * 2.0 - 1.0
        print(f"{LOG_PREFIX} [Batch] Preprocessing done in {time.time() - t0:.2f}s")

        fps_values = []
        starts = list(range(0, num_frames - clip_length + 1, stride))
        total_clips = len(starts)
        pbar = comfy.utils.ProgressBar(total_clips)
        print(f"{LOG_PREFIX} [Batch] Running inference on {total_clips} clips...")
        t0 = time.time()

        with torch.no_grad():
            for i, start in enumerate(starts):
                clip = frames[start:start + clip_length]
                clip = clip.permute(1, 0, 2, 3).unsqueeze(0)
                clip = clip.to(device=device, dtype=dtype)

                pred_log_fps = model(clip)
                fps = round(torch.exp(pred_log_fps).item(), 1)
                fps_values.append(fps)
                pbar.update(1)
                print(f"{LOG_PREFIX}   Clip {i+1}/{total_clips} (frames {start}-{start+clip_length-1}) -> {fps} PhyFPS")

        avg_fps = round(np.mean(fps_values), 1)
        inference_time = time.time() - t0
        total_time = time.time() - t_total
        print(f"{LOG_PREFIX} [Batch] Inference done in {inference_time:.2f}s ({inference_time/total_clips:.2f}s per clip)")
        print(f"{LOG_PREFIX} [Batch] Average PhyFPS: {avg_fps} | Total time: {total_time:.2f}s")

        lines = [
            f"PhyFPS per segment: {fps_values}",
            f"Average PhyFPS: {avg_fps}",
            f"Segments: {len(fps_values)} | Clip length: {clip_length} | Stride: {stride}",
        ]
        report = "\n".join(lines)

        return (avg_fps, fps_values, report)


NODE_CLASS_MAPPINGS = {
    "LoadVisualChronometer": LoadVisualChronometer,
    "PredictPhyFPS": PredictPhyFPS,
    "PredictPhyFPSBatch": PredictPhyFPSBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadVisualChronometer": "Load Visual Chronometer (Pulse of Motion)",
    "PredictPhyFPS": "Predict PhyFPS",
    "PredictPhyFPSBatch": "Predict PhyFPS (Batch Details)",
}

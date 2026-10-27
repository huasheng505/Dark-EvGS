"""Small shared utilities; no imports of the legacy CUDA-only Scene package."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def resize_camera(frame, max_width=0):
    if max_width and frame.width > max_width:
        height = max(1, round(frame.height * max_width / frame.width))
        return replace(frame, width=max_width, height=height,
                       cx=None if frame.cx is None else frame.cx * max_width / frame.width,
                       cy=None if frame.cy is None else frame.cy * height / frame.height)
    return frame


def image_tensor(path, camera=None, device="cpu"):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if camera is not None and image.size != (camera.width, camera.height):
            image = image.resize((camera.width, camera.height), Image.Resampling.LANCZOS)
        array = np.array(image, dtype=np.float32) / 255.
    return torch.from_numpy(array).permute(2, 0, 1).to(device)


def save_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(image).save(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_device(device, backend):
    if backend == "cuda" and not str(device).startswith("cuda"):
        raise ValueError("CUDA rasterizer requires --device cuda; use --backend reference for tiny CPU checks")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; CPU reference is for small functional tests only")

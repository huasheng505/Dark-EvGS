"""Camera/frame metadata for Dark-EvGS; bright evaluation images are never read.

COLMAP binary poses are preferred.  Exported ``output/cameras.json`` supplies a
fallback when present (e.g. Panda).  No guessed camera trajectory is fabricated.
The split is deterministic: every fifth chronologically ordered frame is held out.
Static checkpoint inference can opt out of timestamps entirely; it then orders
frames by filename and leaves their timestamps explicitly unknown (None).
"""
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
from typing import Optional
import warnings
import numpy as np
from .events import seconds_scale


@dataclass(frozen=True)
class Frame:
    name: str
    image_path: Path
    timestamp: Optional[float]
    R: np.ndarray
    T: np.ndarray
    fovx: float
    fovy: float
    width: int
    height: int
    split: str
    cx: float = None
    cy: float = None


@lru_cache(maxsize=1)
def _colmap_loader():
    # Import this independent parser without scene/__init__, which imports CUDA.
    path = Path(__file__).resolve().parents[1] / 'scene' / 'colmap_loader.py'
    spec = importlib.util.spec_from_file_location('_dark_evgs_colmap', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity(name):
    stem = Path(name).stem
    return int(stem) if stem.isdecimal() else None


def _match_name(name, paths):
    if name in paths:
        return paths[name]
    identity = _identity(name)
    matches = [path for other, path in paths.items()
               if identity is not None and _identity(other) == identity]
    if len(matches) != 1:
        raise ValueError(f'Camera image {name!r} has no unique corresponding frame')
    return matches[0]


def _read_timestamps(path, frame_paths, timestamp_scale):
    path = Path(path)
    # Avoid np.loadtxt on accidental multi-million-row event exports (Cat).
    rows = []
    with path.open() as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            row = line.replace(',', ' ').split()
            if len(row) not in (1, 2):
                raise ValueError(f'{path}: expected frame timestamps, not a multi-column event export')
            rows.append(row)
            if len(rows) > len(frame_paths):
                raise ValueError(f'{path}: more timestamps than the {len(frame_paths)} frames; '
                                 'provide an explicit filename,timestamp manifest; no subsampling is guessed')
    if not rows:
        raise ValueError(f'{path}: frame timestamp file is empty')
    if all(len(row) == 1 for row in rows):
        if len(rows) != len(frame_paths):
            raise ValueError(f'{path}: {len(rows)} timestamps for {len(frame_paths)} frames; '
                             'provide an explicit filename,timestamp manifest; no subsampling is guessed')
        names = [p.name for p in frame_paths]
        raw = np.asarray([float(row[0]) for row in rows], dtype=np.float64)
    elif all(len(row) == 2 for row in rows):
        names = [row[0] for row in rows]
        raw = np.asarray([float(row[1]) for row in rows], dtype=np.float64)
        if len(set(names)) != len(names):
            raise ValueError(f'{path}: duplicate frame timestamp names')
        if set(names) != {p.name for p in frame_paths}:
            raise ValueError(f'{path}: timestamp manifest must name every input frame exactly once')
    else:
        raise ValueError(f'{path}: expected one frame timestamp per line or filename,timestamp; '
                         'this may be an event export rather than frame timestamps')
    if not np.isfinite(raw).all():
        raise ValueError('Frame timestamps must be finite')
    scale = seconds_scale(raw, timestamp_scale)
    mapping = dict(zip(names, raw * scale))
    ordered = np.asarray([mapping[p.name] for p in frame_paths])
    if np.any(ordered[1:] <= ordered[:-1]):
        raise ValueError('Frame timestamps must strictly increase in filename order')
    return mapping, scale


class SceneDataset:
    def __init__(self, source, split='train', holdout=5, images='low_light_frames',
                 timestamps=None, timestamp_scale=None, require_timestamps=True):
        self.source = Path(source).expanduser().resolve()
        if split not in {'train', 'test', 'all'}:
            raise ValueError('split must be train, test or all')
        if holdout < 2 or int(holdout) != holdout:
            raise ValueError('holdout must be an integer >= 2')
        self.holdout = int(holdout)
        image_dir = self.source / images
        frame_paths = sorted(p for p in image_dir.iterdir()
                             if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}) if image_dir.is_dir() else []
        if len(frame_paths) < 2:
            raise ValueError(f'{image_dir}: need at least two input frames')
        self.require_timestamps = bool(require_timestamps)
        if self.require_timestamps:
            if timestamps is None:
                candidates = [image_dir/'timestamps.txt', image_dir/'timestamp.txt',
                              self.source/'timestamps.txt', self.source/'timestamp.txt']
                timestamps = next((p for p in candidates if p.is_file()), None)
                if timestamps is None:
                    raise FileNotFoundError('No frame timestamps found; supply timestamps=filename')
            self.timestamps_path = Path(timestamps).resolve()
            mapping, self.timestamp_scale = _read_timestamps(timestamps, frame_paths, timestamp_scale)
        else:
            # Static checkpoint rendering needs calibrated poses, not temporal
            # supervision. Do not invent timestamps or parse irrelevant exports.
            self.timestamps_path = None
            self.timestamp_scale = None
            mapping = {path.name: None for path in frame_paths}
        filename_order = {path.name: i for i, path in enumerate(frame_paths)}
        paths = {p.name: p for p in frame_paths}
        poses = self._poses()
        if len(poses) != len(frame_paths):
            raise ValueError(f'{len(poses)} camera poses for {len(frame_paths)} frames; '
                             'provide complete corresponding poses rather than guessing a trajectory')
        records = []
        used_paths = set()
        numeric_matches = 0
        for name, metadata in poses.items():
            path = _match_name(name, paths)
            numeric_matches += path.name != name
            if path in used_paths:
                raise ValueError('Multiple camera poses map to the same input frame')
            used_paths.add(path)
            from PIL import Image
            with Image.open(path) as image:
                if image.size != (metadata['width'], metadata['height']):
                    raise ValueError(f'{path}: image size {image.size} differs from calibrated camera '
                                     f"{metadata['width']}x{metadata['height']}")
            if not (np.isfinite(metadata['R']).all() and np.isfinite(metadata['T']).all()):
                raise ValueError(f'{name}: camera pose contains nonfinite values')
            sort_key = mapping[path.name] if self.require_timestamps else filename_order[path.name]
            records.append((sort_key, name, path, metadata))
        if numeric_matches:
            warnings.warn(f'{numeric_matches} camera filenames were matched by unique numeric frame ID '
                          '(different zero padding); canonical camera filenames are retained', stacklevel=2)
        records.sort(key=lambda row: row[0])
        self.all_frames = [Frame(name, path, float(mapping[path.name]) if self.require_timestamps else None, split='test' if i % self.holdout == 0 else 'train',
                                 **metadata)
                           for i, (time, name, path, metadata) in enumerate(records)]
        self.train = [f for f in self.all_frames if f.split == 'train']
        self.test = [f for f in self.all_frames if f.split == 'test']
        self.frames = self.all_frames if split == 'all' else self.train if split == 'train' else self.test

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]

    def _poses(self):
        sparse = self.source / 'sparse' / '0'
        if (sparse / 'cameras.bin').is_file() and (sparse / 'images.bin').is_file():
            loader = _colmap_loader()
            intrinsics = loader.read_intrinsics_binary(sparse / 'cameras.bin')
            extrinsics = loader.read_extrinsics_binary(sparse / 'images.bin')
            result = {}
            for camera in extrinsics.values():
                intr = intrinsics[camera.camera_id]
                if intr.model == 'SIMPLE_PINHOLE':
                    fx, cx, cy = intr.params
                    fy = fx
                elif intr.model == 'PINHOLE':
                    fx, fy, cx, cy = intr.params
                else:
                    raise ValueError(f'Unsupported distorted camera {intr.model}; undistort frames and events first')
                name = Path(camera.name).name
                if name in result:
                    raise ValueError(f'Duplicate camera basename {name}')
                result[name] = dict(R=camera.qvec2rotmat().T if hasattr(camera, 'qvec2rotmat')
                                    else loader.qvec2rotmat(camera.qvec).T,
                                    T=np.asarray(camera.tvec, dtype=np.float64),
                                    fovx=float(2*np.arctan(intr.width/(2*fx))),
                                    fovy=float(2*np.arctan(intr.height/(2*fy))),
                                    width=int(intr.width), height=int(intr.height),
                                    cx=float(cx), cy=float(cy))
            return result
        path = self.source / 'output' / 'cameras.json'
        if not path.is_file():
            raise FileNotFoundError(f'{self.source}: missing COLMAP binary camera poses and output/cameras.json; '
                                    'calibrated camera poses are required')
        result = {}
        for camera in json.loads(path.read_text()):
            name = Path(camera['img_name']).name
            if not Path(name).suffix:
                name += '.png'
            if name in result:
                raise ValueError(f'Duplicate camera basename {name}')
            rotation = np.asarray(camera['rotation'], dtype=np.float64)
            position = np.asarray(camera['position'], dtype=np.float64)
            if rotation.shape != (3, 3) or position.shape != (3,):
                raise ValueError(f'{path}: invalid camera rotation/position shape')
            if not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-4) or np.linalg.det(rotation) < 0:
                raise ValueError(f'{path}: camera rotation must be proper orthonormal')
            width, height = int(camera['width']), int(camera['height'])
            result[name] = dict(R=rotation, T=-rotation.T@position,
                                fovx=float(2*np.arctan(width/(2*camera['fx']))),
                                fovy=float(2*np.arctan(height/(2*camera['fy']))),
                                width=width, height=height,
                                cx=float(camera.get('cx', width/2)), cy=float(camera.get('cy', height/2)))
        return result


def load_image(path):
    """Load RGB float32 CHW [0,1]; torch is needed only for this operation."""
    from PIL import Image
    import torch
    with Image.open(path) as image:
        array = np.array(image.convert('RGB'), dtype=np.float32) / 255.
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()

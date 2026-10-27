"""Generate training-view pseudo images using a trained independent CTCM prior."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True, help='trained independent prior; random inference is forbidden')
    p.add_argument('--images', default='low_light_frames')
    p.add_argument('--holdout', type=int, default=5)
    p.add_argument('--timestamps', type=Path)
    p.add_argument('--timestamp-scale', type=float, help='frame timestamp conversion to seconds')
    p.add_argument('--event-timestamp-scale', type=float, help='independent event timestamp conversion to seconds')
    p.add_argument('--device', default='cuda')
    return p


def main():
    args = parser().parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from dark_evgs.data import SceneDataset, load_image
    from dark_evgs.events import EventStream
    from dark_evgs.prior import load_prior, normalize_voxel
    if args.output.resolve().is_relative_to(args.source.resolve()):
        raise ValueError('write pseudo images outside the source scene to preserve raw data')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('output must be empty to avoid mixing incompatible pseudo runs')
    model, checkpoint = load_prior(args.checkpoint, args.device)
    if checkpoint.get('epoch', 0) < 1:
        raise ValueError('prior checkpoint has no completed training epoch; random inference is forbidden')
    scene_name = args.source.resolve().name
    provenance = checkpoint.get('provenance', {})
    held = provenance.get('held_out_scenes', [])
    if scene_name not in held:
        raise ValueError(f'{scene_name!r} must be declared held_out_scenes in the prior checkpoint')
    if scene_name in provenance.get('train_scenes', []) + provenance.get('val_scenes', []):
        raise ValueError('prior was trained or selected on the target scene')
    model.eval()
    dataset = SceneDataset(args.source, split='train', holdout=args.holdout, images=args.images,
                           timestamps=args.timestamps, timestamp_scale=args.timestamp_scale)
    frames = dataset.frames
    if len(frames) < 2:
        raise ValueError('at least two training frames are required')
    events = EventStream(args.source / 'event.npz', timestamp_scale=args.event_timestamp_scale)
    # Consecutive train frames only. A held-out image never enters the prior.
    # Average the two independent neighbor predictions for interior train frames.
    sums, counts = {}, {}
    with torch.inference_mode():
        for f0, f1 in zip(frames[:-1], frames[1:]):
            image0 = load_image(f0.image_path).unsqueeze(0).to(args.device)
            image1 = load_image(f1.image_path).unsqueeze(0).to(args.device)
            if image0.shape != image1.shape:
                raise ValueError('paired training frames differ in resolution')
            height, width = image0.shape[-2:]
            voxel = events.voxel(f0.timestamp, f1.timestamp, model.config['event_bins'], height, width)
            voxel = normalize_voxel(voxel).unsqueeze(0).to(args.device)
            predictions = model(image0, image1, voxel)
            for frame, prediction in zip((f0, f1), predictions):
                array = prediction.squeeze(0).cpu().numpy()
                if frame.name not in sums:
                    sums[frame.name] = array
                    counts[frame.name] = 1
                else:
                    sums[frame.name] += array
                    counts[frame.name] += 1
    metadata = {'format': 'dark-evgs-pseudo-v1', 'source': str(args.source.resolve()),
                'scene': scene_name, 'checkpoint': str(args.checkpoint.resolve()),
                'checkpoint_digest': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                'prior_provenance': provenance, 'model_config': model.config, 'holdout': args.holdout,
                'images': args.images, 'timestamp_scale': dataset.timestamp_scale,
                'event_timestamp_scale': events.timestamp_scale,
                'timestamps': str(dataset.timestamps_path.resolve()),
                'timestamp_digest': hashlib.sha256(dataset.timestamps_path.read_bytes()).hexdigest(),
                'frame_names': [f.name for f in frames],
                'method': 'independent CTCM; adjacent training views, average neighbor predictions',
                'ground_truth_used': False}
    # Publish a complete directory atomically; failed inference leaves no partial run.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pseudo-', dir=args.output.parent) as temporary:
        staging = Path(temporary)
        for frame in frames:
            image = sums[frame.name] / counts[frame.name]
            pixels = np.rint(np.clip(image.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
            Image.fromarray(pixels).save(staging / frame.name)
        (staging / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError('output changed during inference; refusing to replace existing files')
        staging.replace(args.output)
    print(f'Saved {len(frames)} training-view pseudo images to {args.output}')


if __name__ == '__main__':
    main()

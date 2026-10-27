"""Render original-author Gaussian PLYs or independently trained checkpoints."""
import argparse
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-m', '--model', type=Path, required=True)
    p.add_argument('-s', '--source', type=Path)
    p.add_argument('--images', default=None)
    p.add_argument('--timestamps', type=Path)
    p.add_argument('--timestamp-scale', type=float)
    p.add_argument('--holdout', type=int)
    p.add_argument('--split', choices=['train','test','all'], default='test')
    p.add_argument('--iteration', type=int, default=-1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--backend', choices=['cuda','reference'], default='cuda')
    p.add_argument('--max-width', type=int, default=0)
    p.add_argument('--output', type=Path, help='default: model/split/ours_ITER/renders')
    return p


def main():
    args = parser().parse_args()
    import torch
    from dark_evgs.backend import GaussianScene, infer_sh_degree
    from dark_evgs.data import SceneDataset
    from dark_evgs.runtime import resize_camera, save_image, write_json, require_device
    require_device(args.device, args.backend)
    path = args.model/'run_config.json'
    config = json.loads(path.read_text()) if path.is_file() else {}
    source = args.source or (Path(config['source']) if 'source' in config else None)
    if source is None:
        raise ValueError('Author checkpoints contain stale paths; provide -s /path/to/scene explicitly')
    holdout = args.holdout or config.get('holdout', 5)
    images = args.images or config.get('images', 'low_light_frames')
    timestamps = args.timestamps or config.get('timestamps')
    timestamp_scale = args.timestamp_scale if args.timestamp_scale is not None else config.get('timestamp_scale')
    if config and (holdout != config['holdout'] or Path(source).resolve() != Path(config['source']).resolve()):
        raise ValueError('Inference source/split differs from saved training run')
    dataset = SceneDataset(source, split=args.split, holdout=holdout, images=images,
                           timestamps=timestamps, timestamp_scale=timestamp_scale,
                           require_timestamps=bool(config))
    if args.iteration < 0:
        candidates = [int(p.name.split('_')[-1]) for p in (args.model/'point_cloud').glob('iteration_*')
                      if p.name.split('_')[-1].isdigit() and (p/'point_cloud.ply').is_file()]
        if not candidates:
            raise FileNotFoundError('No Gaussian checkpoints found')
        args.iteration = max(candidates)
    ply = args.model/'point_cloud'/f'iteration_{args.iteration}'/'point_cloud.ply'
    model = GaussianScene(infer_sh_degree(ply), args.device)
    model.load_ply(ply)
    # PLY has no active-degree metadata. Trained runs ramp SH degree at 1000-step intervals.
    if config:
        model.active_sh_degree = min(args.iteration//1000, model.max_sh_degree)
    output = args.output or args.model/args.split/f'ours_{args.iteration}'/'renders'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Render output is nonempty; use a new --output directory: {output}')
    with torch.inference_mode():
        for frame in dataset.frames:
            camera = resize_camera(frame, args.max_width)
            result = model.render(camera, backend=args.backend)
            save_image(result['render'], output/frame.name)
    write_json(output.parent/'render_manifest.json', {
        'format':'dark-evgs-renders-v1', 'source':str(Path(source).resolve()),
        'model':str(args.model.resolve()), 'iteration':args.iteration,
        'backend':args.backend, 'split':args.split, 'holdout':holdout,
        'max_width':args.max_width, 'frame_names':[f.name for f in dataset.frames],
        'temporal_alignment_checked':bool(config),
        'note':'reference backend is for small smoke tests, not paper metrics'})
    print(f'Rendered {len(dataset)} {args.split} views to {output}')


if __name__ == '__main__':
    main()

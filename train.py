"""Train a Gaussian field with paper-guided triplet losses and fixed pseudo targets."""
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-s', '--source', type=Path, required=True)
    p.add_argument('-m', '--model', type=Path, required=True)
    p.add_argument('--pseudo-dir', type=Path, required=True)
    p.add_argument('--images', default='low_light_frames')
    p.add_argument('--timestamps', type=Path)
    p.add_argument('--timestamp-scale', type=float)
    p.add_argument('--event-timestamp-scale', type=float)
    p.add_argument('--holdout', type=int, default=5)
    p.add_argument('--device', default='cuda')
    p.add_argument('--backend', choices=['cuda', 'reference'], default='cuda')
    p.add_argument('--max-width', type=int, default=0, help='0=native; reduced size is for debugging, not paper metrics')
    p.add_argument('--iterations', type=int, default=30000)
    p.add_argument('--num-points', type=int, default=1000)
    p.add_argument('--init-bounds', nargs=6, type=float, metavar=('X0','Y0','Z0','X1','Y1','Z1'),
                   help='explicit random-init world bounds for poorly constrained camera trajectories')
    p.add_argument('--sh-degree', type=int, default=3)
    p.add_argument('--event-contrast', type=float, default=0.2, help='assumption: sensor contrast threshold, not reported by paper')
    p.add_argument('--event-filter', choices=['none', 'density'], default='none')
    p.add_argument('--filter-window-us', type=float, default=2000)
    p.add_argument('--filter-radius', type=int, default=1)
    p.add_argument('--filter-min-neighbors', type=int, default=2)
    p.add_argument('--lambda-event', type=float, default=0.25)
    p.add_argument('--lambda-mix', type=float, default=0.25)
    p.add_argument('--gamma', type=float, default=2.2)
    p.add_argument('--epsilon', type=float, default=1e-5)
    p.add_argument('--position-lr-init', type=float, default=1.6e-4)
    p.add_argument('--position-lr-final', type=float, default=1.6e-6)
    p.add_argument('--position-lr-max-steps', type=int, default=30000)
    p.add_argument('--feature-lr', type=float, default=2.5e-3)
    p.add_argument('--opacity-lr', type=float, default=5e-2)
    p.add_argument('--scaling-lr', type=float, default=5e-3)
    p.add_argument('--rotation-lr', type=float, default=1e-3)
    p.add_argument('--percent-dense', type=float, default=0.01)
    p.add_argument('--densify-from', type=int, default=500)
    p.add_argument('--densify-until', type=int, default=15000)
    p.add_argument('--densify-interval', type=int, default=100)
    p.add_argument('--densify-threshold', type=float, default=2e-4)
    p.add_argument('--max-points', type=int, default=1000000)
    p.add_argument('--opacity-reset-interval', type=int, default=3000)
    p.add_argument('--save-every', type=int, default=7000)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', type=Path)
    return p


def main():
    args = parser().parse_args()
    import numpy as np
    import torch
    from torch.nn import functional as F
    from dark_evgs.backend import GaussianScene
    from dark_evgs.data import SceneDataset
    from dark_evgs.events import EventStream
    from dark_evgs.losses import triplet_loss
    from dark_evgs.runtime import image_tensor, resize_camera, write_json, require_device

    if min(args.iterations, args.num_points, args.max_points, args.save_every, args.log_every,
           args.densify_interval, args.opacity_reset_interval) < 1 or args.max_width < 0:
        raise ValueError('Iteration counts, point limits, intervals must be positive; max-width nonnegative')
    require_device(args.device, args.backend)
    args.source, args.model, args.pseudo_dir = (p.resolve() for p in (args.source, args.model, args.pseudo_dir))
    if args.model.is_relative_to(args.source):
        raise ValueError('Use a new model directory outside the source scene; preserve supplied checkpoints')
    metadata_path = args.pseudo_dir / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('format') != 'dark-evgs-pseudo-v1':
        raise ValueError('Unsupported pseudo manifest format')
    if metadata.get('ground_truth_used') is not False:
        raise ValueError('Pseudo metadata must explicitly declare ground_truth_used=false')
    dataset = SceneDataset(args.source, split='train', holdout=args.holdout, images=args.images,
                           timestamps=args.timestamps, timestamp_scale=args.timestamp_scale)
    if len(dataset) < 2:
        raise ValueError('At least two training views are required')
    names = [f.name for f in dataset.frames]
    if set(metadata.get('frame_names', [])) != set(names) or metadata.get('holdout') != args.holdout:
        raise ValueError('Pseudo frames/split do not match training dataset; regenerate pseudo frames')
    if Path(metadata.get('source', '')).resolve() != args.source:
        raise ValueError('Pseudo source scene does not match --source')
    provenance = metadata.get('prior_provenance', {})
    if args.source.name not in provenance.get('held_out_scenes', []):
        raise ValueError('Prior must explicitly hold out the target reconstruction scene')
    if args.source.name in provenance.get('train_scenes', []) + provenance.get('val_scenes', []):
        raise ValueError('Prior training or validation contains target scene')
    if metadata.get('images') != args.images or metadata.get('timestamp_scale') != dataset.timestamp_scale:
        raise ValueError('Pseudo input images/time units do not match training')
    if metadata.get('timestamp_digest') != hashlib.sha256(dataset.timestamps_path.read_bytes()).hexdigest():
        raise ValueError('Frame timestamps changed since pseudo preparation')
    for frame in dataset.frames:
        if not (args.pseudo_dir / frame.name).is_file():
            raise FileNotFoundError(args.pseudo_dir / frame.name)
    events = EventStream(args.source / 'event.npz', timestamp_scale=args.event_timestamp_scale)
    if metadata.get('event_timestamp_scale') != events.timestamp_scale:
        raise ValueError('Pseudo and field training event time units differ')
    if dataset.frames[0].timestamp < events.time_range[0] or dataset.frames[-1].timestamp > events.time_range[1]:
        raise ValueError('Training frame span exceeds event coverage; supply complete synchronized data')
    # Store exact identities and input content digests for safe continuation.
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'}
    config['implementation'] = 'independent-paper-guided-v1'
    config['train_frames'] = names
    config['test_frames'] = [f.name for f in dataset.test]
    config['pseudo_digest'] = hashlib.sha256(metadata_path.read_bytes() + b''.join(
        hashlib.sha256((args.pseudo_dir/f.name).read_bytes()).digest() for f in dataset.frames)).hexdigest()
    config['timestamp_digest'] = hashlib.sha256(dataset.timestamps_path.read_bytes()).hexdigest()
    event_hash = hashlib.sha256()
    with (args.source/'event.npz').open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            event_hash.update(chunk)
    config['event_digest'] = event_hash.hexdigest()
    pose_records = [{'name':f.name, 'R':f.R.tolist(), 'T':f.T.tolist(),
                     'fov':[f.fovx,f.fovy], 'size':[f.width,f.height], 'principal':[f.cx,f.cy]}
                    for f in dataset.all_frames]
    config['camera_digest'] = hashlib.sha256(json.dumps(pose_records,sort_keys=True).encode()).hexdigest()
    config['event_range'] = list(events.time_range)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = GaussianScene(args.sh_degree, args.device)
    first = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=True)
        if checkpoint.get('format') != 'dark-evgs-training-v1':
            raise ValueError('Not a training checkpoint from this implementation')
        ignored = {'iterations', 'log_every', 'save_every', 'device', 'backend'}
        for key in set(config) | set(checkpoint['config']):
            if key not in ignored and config.get(key) != checkpoint['config'].get(key):
                raise ValueError(f'Resume configuration mismatch: {key}')
        first = checkpoint['iteration']
        model.restore(checkpoint['model'], args)
        torch.set_rng_state(checkpoint['rng'].cpu())
        if torch.cuda.is_available() and 'cuda_rng' in checkpoint:
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['cuda_rng']])
    else:
        if args.model.exists() and any(args.model.iterdir()):
            raise FileExistsError('Model directory must be empty, or explicitly resume a checkpoint')
        bounds = None if args.init_bounds is None else (args.init_bounds[:3], args.init_bounds[3:])
        model.initialize_random(dataset.frames, args.num_points, args.seed, bounds=bounds)
        model.training_setup(args)
    if first >= args.iterations:
        raise ValueError('--iterations must exceed the saved iteration')
    args.model.mkdir(parents=True, exist_ok=True)
    write_json(args.model / 'run_config.json', config)
    write_json(args.model / 'split.json', {'train': names, 'test': config['test_frames'], 'holdout': args.holdout})
    frames = [resize_camera(f, args.max_width) for f in dataset.frames]
    # CPU cache avoids holding the complete training set on the GPU.
    pseudo = [image_tensor(args.pseudo_dir/f.name, camera=f) for f in frames]
    background = torch.zeros(3, device=args.device)

    @lru_cache(maxsize=64)
    def event_target(index):
        native0, native1 = dataset.frames[index:index+2]
        target = events.accumulate(native0.timestamp, native1.timestamp, native0.height, native0.width,
                                  contrast=args.event_contrast, filter_mode=args.event_filter,
                                  density_radius=args.filter_radius, min_neighbors=args.filter_min_neighbors,
                                  density_window_us=args.filter_window_us)
        target = torch.from_numpy(target)[None]
        camera = frames[index]
        if target.shape[-2:] != (camera.height, camera.width):
            target = F.interpolate(target[None], size=(camera.height, camera.width), mode='area')[0]
        return target

    def save(iteration):
        model.save_ply(args.model / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply')
        state = {'format': 'dark-evgs-training-v1', 'iteration': iteration,
                 'model': model.capture(), 'config': config, 'rng': torch.get_rng_state()}
        if torch.cuda.is_available():
            state['cuda_rng'] = torch.cuda.get_rng_state_all()
        folder = args.model/'checkpoints'
        folder.mkdir(exist_ok=True)
        path = folder/'latest.pt'
        torch.save(state, folder/'latest.tmp')
        (folder/'latest.tmp').replace(path)

    with (args.model/'train_log.jsonl').open('a') as log:
        for iteration in range(first + 1, args.iterations + 1):
            index = int(torch.randint(len(frames)-1, ()).item())
            camera0, camera1 = frames[index:index+2]
            if (camera0.height, camera0.width) != (camera1.height, camera1.width):
                raise ValueError('Paired camera resolutions must match')
            target = event_target(index).to(args.device)
            lr = model.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                model.oneupSHdegree()
            model.optimizer.zero_grad(set_to_none=True)
            outputs = [model.render(c, backend=args.backend, background=background) for c in (camera0,camera1)]
            losses = triplet_loss(outputs[0]['render'], outputs[1]['render'],
                                  pseudo[index].to(args.device), pseudo[index+1].to(args.device), target,
                                  args.lambda_event, args.lambda_mix, args.gamma, args.epsilon)
            if not torch.isfinite(losses['total']):
                raise RuntimeError(f'Nonfinite loss at iteration {iteration}')
            losses['total'].backward()
            with torch.no_grad():
                for output in outputs:
                    visible = output['visibility_filter']
                    model.max_radii2D[visible] = torch.maximum(model.max_radii2D[visible], output['radii'][visible].float())
                    if iteration < args.densify_until:
                        model.add_densification_stats(output['viewspace_points'], visible)
                model.optimizer.step()
                if args.densify_from <= iteration < args.densify_until and iteration % args.densify_interval == 0:
                    model.densify_and_prune(args.densify_threshold, 0.005, model.spatial_lr_scale,
                                           max_screen_size=20 if iteration > args.opacity_reset_interval else None,
                                           max_points=args.max_points)
                if iteration < args.densify_until and iteration % args.opacity_reset_interval == 0:
                    model.reset_opacity()
            if iteration == first+1 or iteration % args.log_every == 0 or iteration == args.iterations:
                entry = {'iteration': iteration, 'pair': [camera0.name,camera1.name],
                         'points': len(model.get_xyz), 'position_lr': lr,
                         **{key: float(value.detach()) for key,value in losses.items()}}
                line = json.dumps(entry, allow_nan=False)
                print(line, flush=True)
                log.write(line+'\n')
                log.flush()
            if iteration % args.save_every == 0 or iteration == args.iterations:
                save(iteration)
    print(f'Completed {args.iterations} iterations; model: {args.model}')


if __name__ == '__main__':
    main()

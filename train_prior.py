"""Train the independent CTCM on auxiliary scene-disjoint dark/bright pairs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True, help='JSON with train, val, held_out_scenes; see REPRODUCTION.md')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--event-bins', type=int, default=5)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--crop-size', type=int, default=128, help='training crop, 0 = full image')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', type=Path)
    return p


def validate_manifest(manifest):
    held = set(manifest.get('held_out_scenes', []))
    if not held:
        raise ValueError('manifest must list held_out_scenes (target reconstruction/evaluation scenes)')
    scene_sets = []
    for split in ('train', 'val'):
        records = manifest.get(split, [])
        if not records:
            raise ValueError(f'manifest requires nonempty {split} records')
        for record in records:
            if not all(key in record for key in ('scene', 'dark0', 'dark1', 'bright0', 'bright1', 'event_voxel')):
                raise ValueError(f'{split} record lacks required scene/images/event_voxel fields')
        scenes = {r['scene'] for r in records}
        if scenes & held:
            raise ValueError(f'target scene leakage in {split}: {sorted(scenes & held)}')
        scene_sets.append(scenes)
    if scene_sets[0] & scene_sets[1]:
        raise ValueError('auxiliary train/val scenes must be disjoint')
    return scene_sets


def main():
    args = parser().parse_args()
    import numpy as np
    import torch
    from torch.nn import functional as F
    from PIL import Image
    from dark_evgs.prior import CTCM, FORMAT, load_prior, normalize_voxel
    if args.epochs < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.crop_size < 0:
        raise ValueError('positive epochs/learning rate and nonnegative crop-size required')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for a small prior smoke experiment')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError('Prior output must be empty, or explicitly resume a checkpoint; refusing to overwrite weights')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    training_config = {'learning_rate': args.learning_rate, 'event_bins': args.event_bins,
                       'width': args.width, 'crop_size': args.crop_size, 'seed': args.seed,
                       'manifest_sha256': manifest_digest}
    train_scenes, val_scenes = validate_manifest(manifest)
    root = args.manifest.resolve().parent
    model = CTCM(args.event_bins, args.width).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    first_epoch = 0
    if args.resume:
        model, checkpoint = load_prior(args.resume, args.device)
        if checkpoint.get('training_config') != training_config:
            raise ValueError('resume training configuration or manifest SHA256 differs from checkpoint')
        if model.config != {'event_bins': args.event_bins, 'width': args.width}:
            raise ValueError('resume architecture differs from CLI configuration')
        if checkpoint['provenance']['held_out_scenes'] != sorted(manifest['held_out_scenes']):
            raise ValueError('resume held-out scene declaration differs')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer'])
        first_epoch = checkpoint['epoch']
        if first_epoch >= args.epochs:
            raise ValueError('--epochs is the total target epoch count and must exceed the resumed epoch')
        saved_train = set(checkpoint['provenance'].get('train_scenes', []))
        saved_val = set(checkpoint['provenance'].get('val_scenes', []))
        if saved_train != train_scenes or saved_val != val_scenes:
            raise ValueError('resume auxiliary scene splits differ from the checkpoint')
        rng = checkpoint.get('rng')
        if not rng or 'python' not in rng or 'torch' not in rng:
            raise ValueError('checkpoint lacks RNG state required for reproducible prior resume')
        random.setstate(rng['python'])
        torch.set_rng_state(rng['torch'].cpu())
        if torch.cuda.is_available() and 'cuda' in rng:
            torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])
    args.output.mkdir(parents=True, exist_ok=True)

    def read_record(record, training):
        images = []
        for key in ('dark0', 'dark1', 'bright0', 'bright1'):
            with Image.open(root / record[key]) as image:
                array = np.asarray(image.convert('RGB'), dtype=np.float32).copy() / 255.0
            images.append(torch.from_numpy(array).permute(2, 0, 1))
        path = root / record['event_voxel']
        if path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as data:
                voxel = data['voxel'].copy()
        else:
            voxel = np.load(path, allow_pickle=False)
        voxel = torch.as_tensor(voxel, dtype=torch.float32)
        shape = images[0].shape[-2:]
        if not all(x.shape == images[0].shape for x in images) or voxel.shape != (args.event_bins, *shape):
            raise ValueError('auxiliary image and voxel dimensions do not match')
        if not torch.isfinite(voxel).all():
            raise ValueError('auxiliary event voxel contains nonfinite values')
        # Normalize before cropping, matching full-frame pseudo preparation.
        voxel = normalize_voxel(voxel)
        if training and args.crop_size:
            h, w = shape
            crop = min(args.crop_size, h, w)
            y, x = random.randrange(h - crop + 1), random.randrange(w - crop + 1)
            images = [v[:, y:y + crop, x:x + crop] for v in images]
            voxel = voxel[:, y:y + crop, x:x + crop]
        return [x.unsqueeze(0).to(args.device) for x in images], voxel.unsqueeze(0).to(args.device)

    for epoch in range(first_epoch, args.epochs):
        losses = {}
        for split in ('train', 'val'):
            training = split == 'train'
            model.train(training)
            records = list(manifest[split])
            if training:
                random.shuffle(records)
            total = 0.0
            with torch.set_grad_enabled(training):
                for record in records:
                    (d0, d1, b0, b1), voxel = read_record(record, training)
                    p0, p1 = model(d0, d1, voxel)
                    loss = (F.mse_loss(p0, b0) + F.mse_loss(p1, b1)) * 0.5
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    total += loss.item()
            losses[split] = total / len(records)
        checkpoint = {'format': FORMAT, 'model_config': model.config, 'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'losses': losses,
                      'training_config': training_config,
                      'rng': {'python': random.getstate(), 'torch': torch.get_rng_state()},
                      'provenance': {'implementation': 'independent compact CTCM approximation; not author weights',
                                     'manifest': str(args.manifest.resolve()),
                                     'manifest_sha256': manifest_digest,
                                     'train_scenes': sorted(train_scenes), 'val_scenes': sorted(val_scenes),
                                     'held_out_scenes': sorted(manifest['held_out_scenes']),
                                     'seed': args.seed}}
        if torch.cuda.is_available():
            checkpoint['rng']['cuda'] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, args.output / 'last.tmp')
        (args.output / 'last.tmp').replace(args.output / 'last.pt')
        print(json.dumps({'epoch': epoch + 1, **losses}), flush=True)
        with (args.output / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch': epoch + 1, **losses}) + '\n')
    print(f'Independent prior checkpoint: {args.output / "last.pt"}')


if __name__ == '__main__':
    main()

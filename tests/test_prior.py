"""Behavior tests for the independent prior, not an author-model equivalence test."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from dark_evgs.prior import CTMB, CTCM, FORMAT, load_prior, normalize_voxel
from train_prior import validate_manifest


def record(scene):
    return {key: scene if key == 'scene' else 'unused' for key in
            ('scene', 'dark0', 'dark1', 'bright0', 'bright1', 'event_voxel')}


def test_manifest_rejects_leakage():
    valid = {'held_out_scenes': ['lion'], 'train': [record('aux_a')], 'val': [record('aux_b')]}
    validate_manifest(valid)
    with pytest.raises(ValueError, match='leakage'):
        validate_manifest({**valid, 'train': [record('lion')]})
    with pytest.raises(ValueError, match='disjoint'):
        validate_manifest({**valid, 'val': [record('aux_a')]})
    with pytest.raises(ValueError, match='held_out_scenes'):
        validate_manifest({**valid, 'held_out_scenes': []})


def test_prior_gradient_odd_shapes_and_checkpoint(tmp_path):
    torch.manual_seed(2)
    torch.set_num_threads(1)
    model = CTCM(event_bins=3, width=8)
    f0 = torch.rand(1, 3, 17, 19) * 0.2
    f1 = torch.rand_like(f0) * 0.2
    events = torch.randn(1, 3, 17, 19)
    p0, p1 = model(f0, f1, events)
    assert p0.shape == f0.shape and p1.shape == f1.shape
    assert ((p0 > 0) & (p0 < 1)).all()
    loss = ((p0 - .5).square().mean() + (p1 - .5).square().mean())
    loss.backward()
    assert model.event_encoder[0].weight.grad.abs().sum() > 0
    assert model.fusion[2].qkv.weight.grad.abs().sum() > 0
    assert model.exposure[-1].weight.grad.abs().sum() > 0
    path = tmp_path / 'prior.pt'
    torch.save({'format': FORMAT, 'model_config': model.config, 'state_dict': model.state_dict()}, path)
    restored, _ = load_prior(path)
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(restored(f0, f1, events)[0], p0)
    torch.save({'state_dict': model.state_dict()}, path)
    with pytest.raises(ValueError, match='adapter'):
        load_prior(path)


def test_voxel_normalization_keeps_zero_and_polarity():
    source = torch.tensor([[[0., 4., -2.]]])
    actual = normalize_voxel(source)
    assert actual[0, 0, 0] == 0
    assert actual[0, 0, 1] > 0 and actual[0, 0, 2] < 0
    assert torch.allclose(normalize_voxel(torch.zeros(2, 2, 2)), torch.zeros(2, 2, 2))


def test_training_checkpoint_one_epoch(tmp_path):
    from PIL import Image
    root = Path(__file__).resolve().parents[1]
    for name, value in [('d0', 15), ('d1', 18), ('b0', 100), ('b1', 105)]:
        Image.fromarray(np.full((12, 14, 3), value, np.uint8)).save(tmp_path / f'{name}.png')
    np.save(tmp_path / 'voxel.npy', np.ones((3, 12, 14), dtype=np.float32))
    item = dict(dark0='d0.png', dark1='d1.png', bright0='b0.png', bright1='b1.png', event_voxel='voxel.npy')
    manifest = {'held_out_scenes': ['lion'], 'train': [{'scene': 'aux_train', **item}],
                'val': [{'scene': 'aux_val', **item}]}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    subprocess.run([sys.executable, str(root/'train_prior.py'), '--manifest', str(tmp_path/'manifest.json'),
                    '--output', str(tmp_path/'output'), '--epochs', '1', '--device', 'cpu',
                    '--width', '4', '--event-bins', '3', '--crop-size', '8'], check=True,
                   env={**__import__('os').environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}, timeout=90)
    _, checkpoint = load_prior(tmp_path/'output/last.pt')
    assert checkpoint['epoch'] == 1
    assert checkpoint['provenance']['held_out_scenes'] == ['lion']
    assert np.isfinite(checkpoint['losses']['val'])

    # Exercise actual pseudo generation; every fifth input image is held out.
    source = tmp_path / 'lion'
    (source/'low_light_frames').mkdir(parents=True)
    (source/'output').mkdir()
    cameras = []
    for index in range(5):
        name = f'{index:06d}.png'
        Image.fromarray(np.full((12, 14, 3), 15 + index, np.uint8)).save(source/'low_light_frames'/name)
        cameras.append({'img_name': name, 'width': 14, 'height': 12, 'fx': 20., 'fy': 20.,
                        'rotation': np.eye(3).tolist(), 'position': [index * .1, 0., 0.]})
    (source/'output/cameras.json').write_text(json.dumps(cameras))
    (source/'timestamps.txt').write_text('0\n1\n2\n3\n4\n')
    times = np.linspace(0, 4, 30)
    np.savez(source/'event.npz', x=np.zeros(30), y=np.zeros(30), t=times * 1e6, p=np.ones(30))
    subprocess.run([sys.executable, str(root/'prepare_pseudo.py'), '--source', str(source),
                    '--output', str(tmp_path/'pseudo'), '--checkpoint', str(tmp_path/'output/last.pt'),
                    '--device', 'cpu', '--timestamp-scale', '1', '--event-timestamp-scale', '1e-6'], check=True,
                   env={**__import__('os').environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}, timeout=90)
    metadata = json.loads((tmp_path/'pseudo/metadata.json').read_text())
    assert metadata['ground_truth_used'] is False
    assert len(metadata['frame_names']) == 4
    assert '000000.png' not in metadata['frame_names']
    assert len(list((tmp_path/'pseudo').glob('*.png'))) == 4
    assert metadata['timestamp_scale'] == 1
    assert metadata['event_timestamp_scale'] == 1e-6
    assert len(metadata['timestamp_digest']) == 64


def test_prior_resume_matches_uninterrupted_and_rejects_changes(tmp_path):
    """Restored Python shuffle/crop RNG yields bitwise-identical epoch-two weights."""
    from PIL import Image
    root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(20)
    for name in ('dark0', 'dark1', 'bright0', 'bright1'):
        Image.fromarray(rng.integers(1, 240, (13, 15, 3), dtype=np.uint8)).save(tmp_path/f'{name}.png')
    np.save(tmp_path/'voxel.npy', rng.standard_normal((3, 13, 15)).astype(np.float32))
    item = {name: f'{name}.png' for name in ('dark0', 'dark1', 'bright0', 'bright1')}
    item['event_voxel'] = 'voxel.npy'
    manifest = {'held_out_scenes': ['lion'],
                'train': [{'scene': 'aux_a', **item}, {'scene': 'aux_b', **item}],
                'val': [{'scene': 'aux_val', **item}]}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    common = [sys.executable, str(root/'train_prior.py'), '--manifest', str(path), '--device', 'cpu',
              '--width', '4', '--event-bins', '3', '--crop-size', '8', '--seed', '30']
    env = {**__import__('os').environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}

    def run(extra, success=True):
        result = subprocess.run(common + extra, env=env, capture_output=True, text=True, timeout=90)
        if success:
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode != 0
        return result

    continuous = tmp_path/'continuous'
    resumed = tmp_path/'resumed'
    run(['--output', str(continuous), '--epochs', '2'])
    run(['--output', str(resumed), '--epochs', '1'])
    checkpoint_before = (resumed/'last.pt').read_bytes()
    rejected = run(['--output', str(resumed), '--epochs', '2'], success=False)
    assert 'refusing to overwrite' in rejected.stderr
    assert (resumed/'last.pt').read_bytes() == checkpoint_before
    run(['--output', str(resumed), '--epochs', '2', '--resume', str(resumed/'last.pt')])
    a = torch.load(continuous/'last.pt', map_location='cpu', weights_only=True)
    b = torch.load(resumed/'last.pt', map_location='cpu', weights_only=True)
    assert a['epoch'] == b['epoch'] == 2
    assert a['losses'] == b['losses']
    assert a['rng']['python'] == b['rng']['python']
    assert torch.equal(a['rng']['torch'], b['rng']['torch'])
    for name in a['state_dict']:
        assert torch.equal(a['state_dict'][name], b['state_dict'][name]), name
    # Same scene names, different pair definitions: no silent resume on new data.
    manifest['train'][0]['dark0'] = 'dark1.png'
    path.write_text(json.dumps(manifest))
    rejected = run(['--output', str(resumed), '--epochs', '3', '--resume', str(resumed/'last.pt')], success=False)
    assert 'manifest SHA256' in rejected.stderr
    assert (resumed/'last.pt').is_file()

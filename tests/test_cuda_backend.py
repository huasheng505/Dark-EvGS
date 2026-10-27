"""Actual CUDA kernel checks, skipped explicitly on CPU-only machines."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dark_evgs.backend import GaussianScene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='NVIDIA GPU required')


def make_scene():
    camera = SimpleNamespace(R=np.eye(3), T=np.zeros(3), fovx=.8, fovy=.7,
                             width=20, height=16, cx=10., cy=8.)
    model = GaussianScene(sh_degree=1, device='cuda')
    model.initialize_random([camera], 2, seed=4, bounds=([-.2,-.1,2.], [.2,.1,2.4]))
    with torch.no_grad():
        model._xyz[:] = model._xyz.new_tensor([[-.1,-.03,2.1], [.12,.04,2.3]])
        model._scaling.fill_(np.log(.12))
        model._opacity.fill_(-.4)
    model.training_setup(SimpleNamespace(iterations=20))
    return model,camera


@pytest.mark.parametrize('principal', [(10.,8.),(8.7,6.9)])
def test_cuda_forward_backward_matches_reference(principal):
    import diff_gaussian_rasterization  # Missing extension on a GPU host is a failure.
    model,camera = make_scene()
    camera.cx,camera.cy = principal
    reference = GaussianScene(sh_degree=1)
    reference.restore(model.capture())
    expected = reference.render(camera,'reference')['render']
    actual = model.render(camera,'cuda')
    difference = actual['render'].detach().cpu()-expected.detach()
    assert difference.square().mean().sqrt() < .01
    assert difference.abs().max() < .02
    losses=[]
    for _ in range(8):
        model.optimizer.zero_grad(set_to_none=True)
        output=model.render(camera,'cuda')
        loss=output['render'].square().mean()
        losses.append(float(loss.detach()))
        loss.backward()
        assert output['viewspace_points'].grad is not None
        for attribute in ['_xyz','_features_dc','_opacity','_scaling']:
            gradient=getattr(model,attribute).grad
            assert gradient is not None and torch.isfinite(gradient).all()
            assert gradient.abs().sum()>0,attribute
        model.add_densification_stats(output['viewspace_points'],output['visibility_filter'])
        model.optimizer.step()
    assert losses[-1]<losses[0]


def test_cuda_checkpoint_densify_and_ply_roundtrip(tmp_path):
    import diff_gaussian_rasterization
    model,camera=make_scene()
    output=model.render(camera,'cuda')
    output['render'].square().mean().backward()
    model.add_densification_stats(output['viewspace_points'],output['visibility_filter'])
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    model.densify_and_prune(0,.005,model.spatial_lr_scale,max_points=4)
    assert len(model.get_xyz)==4
    checkpoint_path=tmp_path/'training.pt'
    torch.save(model.capture(),checkpoint_path)
    saved=torch.load(checkpoint_path,map_location='cuda',weights_only=True)
    restored=GaussianScene(sh_degree=1,device='cuda')
    restored.restore(saved,SimpleNamespace(iterations=20))
    for key,value in model.state_dict().items():
        assert torch.equal(value,restored.state_dict()[key]),key
    for original_group,restored_group in zip(model.optimizer.param_groups,restored.optimizer.param_groups):
        original_state=model.optimizer.state[original_group['params'][0]]
        restored_state=restored.optimizer.state[restored_group['params'][0]]
        assert original_state.keys()==restored_state.keys()
        for key,value in original_state.items():
            other=restored_state[key]
            assert torch.equal(value.cpu(),other.cpu()) if torch.is_tensor(value) else value==other
    assert torch.allclose(model.render(camera,'cuda')['render'],restored.render(camera,'cuda')['render'])
    path=tmp_path/'gpu.ply'
    model.save_ply(path)
    reloaded=GaussianScene(device='cuda')
    reloaded.load_ply(path)
    reloaded.active_sh_degree=model.active_sh_degree
    assert torch.allclose(model.render(camera,'cuda')['render'],reloaded.render(camera,'cuda')['render'])

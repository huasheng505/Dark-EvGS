from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dark_evgs.backend import GaussianScene, infer_sh_degree


def camera(width=12, height=10):
    return SimpleNamespace(R=np.eye(3), T=np.zeros(3), fovx=0.8, fovy=0.7,
                           width=width, height=height, cx=width/2, cy=height/2)


def scene(points=4):
    model = GaussianScene(sh_degree=1)
    model.initialize_random([camera()], points, bounds=([-0.15, -0.1, 1.8], [0.15, 0.1, 2.2]))
    model.training_setup(SimpleNamespace(iterations=20))
    return model


def test_reference_has_real_gradients_and_optimizer_reduces_loss():
    model = scene()
    model.optimizer.param_groups[0]["lr"] = 0.001
    target = torch.full((3, 10, 12), 0.1)
    losses = []
    for _ in range(8):
        output = model.render(camera(), backend="reference")
        loss = (output["render"]-target).square().mean()
        losses.append(loss.item())
        loss.backward()
        assert output["viewspace_points"].grad is not None
        assert model._xyz.grad.isfinite().all()
        assert model._features_dc.grad.abs().sum() > 0
        assert model._scaling.grad.abs().sum() > 0
        assert output["visibility_filter"].all()
        model.add_densification_stats(output["viewspace_points"], output["visibility_filter"])
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
    assert losses[-1] < losses[0]
    assert (model.denom == 8).all()


def test_ply_roundtrip_all_coefficients_and_checkpoint_resume(tmp_path):
    model = scene()
    with torch.no_grad():
        model._features_rest.normal_()
    model.oneupSHdegree()
    path = tmp_path / "point_cloud.ply"
    model.save_ply(path)
    assert infer_sh_degree(path) == 1
    reloaded = GaussianScene()
    reloaded.load_ply(path)
    for name, attr in model._groups.items():
        assert torch.equal(getattr(model, attr), getattr(reloaded, attr)), name
    output = model.render(camera(), backend="reference")
    output["render"].sum().backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    checkpoint = model.capture()
    other = GaussianScene()
    other.restore(checkpoint, SimpleNamespace(iterations=20))
    assert torch.equal(other.render(camera(), "reference")["render"], model.render(camera(), "reference")["render"])
    assert other.active_sh_degree == model.active_sh_degree
    for ga, gb in zip(model.optimizer.param_groups, other.optimizer.param_groups):
        assert torch.equal(model.optimizer.state[ga["params"][0]]["exp_avg"], other.optimizer.state[gb["params"][0]]["exp_avg"])


def test_densification_preserves_retained_adam_rows_and_enforces_budget():
    model = scene(4)
    model.render(camera(), "reference")["render"].sum().backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    old_xyz = model._xyz.detach().clone()
    old_moment = model.optimizer.state[model._xyz]["exp_avg"].clone()
    with torch.no_grad():
        model._scaling.fill_(-8)  # All four are small clone candidates.
        model.xyz_gradient_accum.fill_(1)
        model.denom.fill_(1)
        model._opacity[1] = -20  # prune point 1
    result = model.densify_and_prune(0.1, 0.005, extent=1, max_points=5)
    assert result["after"] == 5
    assert torch.equal(model._xyz[:3], old_xyz[[0, 2, 3]])
    state = model.optimizer.state[model._xyz]
    assert torch.equal(state["exp_avg"][:3], old_moment[[0, 2, 3]])
    assert not state["exp_avg"][3:].any()
    assert len(model.denom) == 5
    model.render(camera(), "reference")["render"].sum().backward()
    model.optimizer.step()
    assert model._xyz.isfinite().all()


def test_render_guard_and_cuda_device_error():
    model = scene()
    with pytest.raises(ValueError, match="8 million"):
        model.render(camera(2000, 2000), "reference")
    with pytest.raises(RuntimeError, match="CUDA model"):
        model.render(camera(), "cuda")


def test_original_author_checkpoint_can_load():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/"data/lion/output/point_cloud/iteration_30000/point_cloud.ply"
    if not path.exists():
        pytest.skip("author sample not present")
    model = GaussianScene()
    model.load_ply(path)
    assert model.max_sh_degree == 3
    assert len(model.get_xyz) > 100
    assert torch.isfinite(model.get_features).all()


def test_depth_order_and_background_compositing():
    model = scene(2)
    with torch.no_grad():
        model._xyz[:] = torch.tensor([[0., 0., 2.], [0., 0., 3.]])
        model._scaling[:] = -0.5
        model._opacity[:] = 5
        model._features_dc[:] = 0
        model._features_dc[0, 0, 0] = 1
        model._features_dc[1, 0, 2] = 1
    output = model.render(camera(11, 11), "reference", background=torch.ones(3))["render"]
    assert output[0, 5, 5] > output[2, 5, 5]
    assert torch.isfinite(output).all()


def test_projection_matrices_match_reference_offcenter_camera():
    """Analytic test of the CUDA matrix convention, not CUDA execution."""
    model = scene()
    frame = camera(14, 10)
    angle = 0.2
    frame.R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]])
    frame.T = np.array([0.1, -0.2, 0.4])
    frame.cx, frame.cy = 6.3, 4.2
    view, projection = model.camera_matrices(frame)
    homogeneous = torch.cat((model.get_xyz, torch.ones(len(model.get_xyz), 1)), dim=1)
    clip = homogeneous@(projection@view).T
    ndc = clip[:, :2]/clip[:, 3:4]
    # Rasterizer ndc2Pix(v,S) = ((v + 1)*S - 1)/2.
    pixels = ((ndc+1)*torch.tensor([frame.width, frame.height])-1)/2
    actual = model.render(frame, "reference")["viewspace_points"][:, :2]
    assert torch.allclose(pixels, actual, atol=1e-5)


def test_initial_scale_ignores_self_distance_roundoff():
    model = GaussianScene(sh_degree=0)
    model.initialize_random([camera()], 1000, seed=9,
                            bounds=([-1, -1, 1.5], [1, 1, 3]))
    exact = torch.cdist(model.get_xyz, model.get_xyz,
                        compute_mode="donot_use_mm_for_euclid_dist")
    exact.fill_diagonal_(float("inf"))
    expected = exact.min(1).values
    assert torch.allclose(model.get_scaling[:, 0], expected, atol=2e-4)


def test_resume_next_update_and_stochastic_split_are_identical():
    model = scene(4)
    def step(field):
        field.optimizer.zero_grad(set_to_none=True)
        result = field.render(camera(), "reference")
        loss = (result["render"]-0.1).square().mean()
        loss.backward()
        field.add_densification_stats(result["viewspace_points"], result["visibility_filter"])
        field.optimizer.step()
        return loss.detach()
    step(model)
    restored = GaussianScene()
    restored.restore(model.capture(), SimpleNamespace(iterations=20))
    random_state = torch.get_rng_state()
    loss = step(model)
    model.densify_and_prune(0, 0.005, extent=1, max_points=6)
    torch.set_rng_state(random_state)
    other_loss = step(restored)
    restored.densify_and_prune(0, 0.005, extent=1, max_points=6)
    assert torch.equal(loss, other_loss)
    for attr in model._groups.values():
        assert torch.equal(getattr(model, attr), getattr(restored, attr)), attr
    step(model)
    step(restored)
    for attr in model._groups.values():
        assert torch.equal(getattr(model, attr), getattr(restored, attr)), attr

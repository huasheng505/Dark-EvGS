import torch
import pytest
from dark_evgs.losses import triplet_loss, log_intensity


def test_exact_supervision_zero_and_gradients():
    a = torch.full((3,8,8),0.2,requires_grad=True)
    b = torch.full((3,8,8),0.4,requires_grad=True)
    events = log_intensity(b.detach())-log_intensity(a.detach())
    exact = triplet_loss(a,b,a.detach(),b.detach(),events)
    assert exact['total'].item() == 0
    losses = triplet_loss(a,b,torch.full_like(a,.7),torch.full_like(b,.8),torch.zeros(1,8,8))
    assert losses['total'].item() == pytest.approx(
        losses['holistic'].item()+.25*losses['event'].item()+.25*losses['mixed'].item())
    losses['total'].backward()
    assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0
    assert torch.isfinite(a.grad).all()


def test_black_images_finite_and_no_broadcasting():
    image = torch.zeros(3,4,4,requires_grad=True)
    losses = triplet_loss(image,image,image,image,torch.zeros(4,4))
    losses['total'].backward()
    assert torch.isfinite(image.grad).all()
    with pytest.raises(ValueError, match='Event map shape'):
        triplet_loss(image,image,image,image,torch.zeros(3,4,4))

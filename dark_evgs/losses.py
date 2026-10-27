"""Differentiable triplet supervision, paper equations (6)--(9).

Events are monochrome, so RGB is gamma-linearized then converted to luminance.
The paper does not specify this RGB-to-event reduction: it is an explicit
reproduction choice. Continuous log differences avoid the zero gradient of a
hard event-trigger quantizer. Event targets are contrast * signed event counts.
"""

import torch
from torch.nn import functional as F


def log_intensity(image, gamma=2.2, epsilon=1e-5):
    if image.shape[-3] != 3:
        raise ValueError("Expected RGB tensor (...,3,H,W)")
    if gamma <= 0 or epsilon <= 0:
        raise ValueError("gamma and epsilon must be positive")
    weights = image.new_tensor([0.2126, 0.7152, 0.0722])
    shape = [1] * image.ndim
    shape[-3] = 3
    linear = image.clamp_min(0).pow(gamma)
    luminance = (linear * weights.view(shape)).sum(dim=-3, keepdim=True)
    return (luminance + epsilon).log()


def triplet_loss(render0, render1, pseudo0, pseudo1, events,
                 lambda_event=0.25, lambda_mix=0.25, gamma=2.2, epsilon=1e-5):
    if render0.shape != render1.shape or render0.shape != pseudo0.shape or pseudo0.shape != pseudo1.shape:
        raise ValueError("Rendered and pseudo RGB images must have identical shapes")
    if lambda_event < 0 or lambda_mix < 0:
        raise ValueError("Loss weights must be nonnegative")
    predicted = log_intensity(render1, gamma, epsilon) - log_intensity(render0, gamma, epsilon)
    if events.ndim == predicted.ndim - 1:
        events = events.unsqueeze(-3)
    if events.shape != predicted.shape:
        raise ValueError(f"Event map shape {events.shape} != predicted shape {predicted.shape}")
    if not all(torch.isfinite(x).all() for x in (render0, render1, pseudo0, pseudo1, events)):
        raise ValueError("Non-finite supervision or rendering")
    pseudo_delta = log_intensity(pseudo1.detach(), gamma, epsilon) - log_intensity(pseudo0.detach(), gamma, epsilon)
    holistic = 0.5 * (F.mse_loss(render0, pseudo0.detach()) + F.mse_loss(render1, pseudo1.detach()))
    event = F.mse_loss(predicted, events)
    mixed = F.mse_loss(predicted, pseudo_delta)
    return {"total": holistic + lambda_event * event + lambda_mix * mixed,
            "holistic": holistic, "event": event, "mixed": mixed}

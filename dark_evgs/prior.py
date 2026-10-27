"""Independent compact CTCM approximation, not the unpublished author model.

The paper specifies channel attention and an exposure parameter but does not
publish the complete CTCM or its fine-tuned weights. This implementation is a
trainable replacement. EvLowLight checkpoints are NOT architecture-compatible.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

FORMAT = 'dark-evgs-independent-ctcm-v1'


class CTMB(nn.Module):
    """Channel attention from spatially normalized Q/K (C x C attention)."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.depthwise = nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3)
        self.temperature = nn.Parameter(torch.ones(1))
        self.project = nn.Conv2d(channels, channels, 1)
        self.ffn = nn.Sequential(nn.Conv2d(channels, channels * 2, 1), nn.GELU(),
                                 nn.Conv2d(channels * 2, channels, 1))

    def forward(self, features):
        batch, channels, height, width = features.shape
        q, k, v = self.depthwise(self.qkv(self.norm(features))).chunk(3, dim=1)
        q, k, v = [x.reshape(batch, channels, height * width) for x in (q, k, v)]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-1, -2) * self.temperature.exp()).softmax(dim=-1)
        corrected = self.project((attention @ v).reshape(batch, channels, height, width))
        features = features + corrected
        return features + self.ffn(self.norm(features))


def normalize_voxel(voxel):
    """Normalize nonzero signed voxel entries; zero/background stays zero."""
    voxel = torch.as_tensor(voxel, dtype=torch.float32)
    mask = voxel != 0
    if mask.any():
        values = voxel[mask]
        # RMS normalization preserves polarity, unlike mean subtraction.
        voxel = voxel / values.square().mean().sqrt().clamp_min(1.0)
    return voxel


class CTCM(nn.Module):
    """Frame/event fusion + CTMB + exposure decoder; returns two RGB images."""
    def __init__(self, event_bins=5, width=32):
        super().__init__()
        if event_bins < 2 or width < 4:
            raise ValueError('event_bins >= 2 and width >= 4 required')
        self.config = {'event_bins': event_bins, 'width': width}
        self.frame_encoder = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.GELU(),
                                           nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU())
        self.event_encoder = nn.Sequential(nn.Conv2d(event_bins, width, 3, padding=1), nn.GELU(),
                                           nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU())
        self.fusion = nn.Sequential(nn.Conv2d(width * 3, width, 1), nn.GELU(), CTMB(width), CTMB(width))
        self.decoder = nn.Sequential(nn.Conv2d(width * 2, width, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(width, 3, 3, padding=1))
        self.exposure = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))

    def forward(self, frame0, frame1, event_voxel):
        if frame0.ndim != 4 or frame0.shape != frame1.shape or frame0.shape[1] != 3:
            raise ValueError('frames must have identical N x 3 x H x W shapes')
        if event_voxel.shape != (frame0.shape[0], self.config['event_bins'], *frame0.shape[-2:]):
            raise ValueError('voxel must have shape N x event_bins x H x W matching frames')
        f0, f1 = self.frame_encoder(frame0), self.frame_encoder(frame1)
        event = self.event_encoder(event_voxel)
        # The second branch views time in reverse, hence the negated event feature input.
        fused0 = self.fusion(torch.cat((f0, f1, event), 1))
        reversed_event = self.event_encoder(-event_voxel.flip(1))
        fused1 = self.fusion(torch.cat((f1, f0, reversed_event), 1))
        results = []
        for frame, feature, fused in ((frame0, f0, fused0), (frame1, f1, fused1)):
            residual = self.decoder(torch.cat((feature, fused), 1))
            residual = F.interpolate(residual, size=frame.shape[-2:], mode='bilinear', align_corners=False)
            # Bounded log-exposure avoids overflow; RGB offsets can correct color tone.
            exposure = 4.0 * torch.tanh(self.exposure(fused)).unsqueeze(-1).unsqueeze(-1)
            base = torch.logit(frame.clamp(1e-4, 1.0 - 1e-4))
            results.append(torch.sigmoid(base + exposure + residual))
        return tuple(results)


def load_prior(path, device='cpu'):
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if checkpoint.get('format') != FORMAT:
        raise ValueError('Not an independent CTCM checkpoint; official EvLowLight weights need an explicit adapter')
    model = CTCM(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device), checkpoint

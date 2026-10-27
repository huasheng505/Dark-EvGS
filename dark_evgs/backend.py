"""Gaussian model, checkpoint I/O, and differentiable rendering.

``cuda`` uses the external diff-gaussian-rasterization implementation. The
``reference`` renderer is a small, deliberately slow PyTorch implementation for
gradient checks and CPU smoke tests, not a performance/quality reproduction.
No synthetic camera poses or bright training targets are introduced here.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from plyfile import PlyData, PlyElement

from utils.sh_utils import C0, eval_sh


def infer_sh_degree(path):
    """Infer the SH degree from an original 3DGS Gaussian PLY."""
    vertex = PlyData.read(str(path))["vertex"]
    nrest = sum(name.startswith("f_rest_") for name in vertex.data.dtype.names)
    degree = int(math.isqrt(nrest // 3 + 1)) - 1
    if nrest != 3 * ((degree + 1) ** 2 - 1) or not 0 <= degree <= 4:
        raise ValueError("Unsupported or malformed spherical-harmonic coefficients")
    return degree


def quaternion_matrix(quaternion):
    q = F.normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ), dim=-1).reshape(-1, 3, 3)


def _arg(args, name, default):
    return getattr(args, name, default)


class GaussianScene(nn.Module):
    _groups = {"xyz": "_xyz", "f_dc": "_features_dc", "f_rest": "_features_rest",
               "opacity": "_opacity", "scaling": "_scaling", "rotation": "_rotation"}

    def __init__(self, sh_degree=3, device="cpu"):
        super().__init__()
        if not 0 <= sh_degree <= 4:
            raise ValueError("SH degree must be in [0, 4]")
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.spatial_lr_scale = 1.0
        self.percent_dense = 0.01
        self.optimizer = None
        device = torch.device(device)
        self._xyz = nn.Parameter(torch.empty(0, 3, device=device))
        self._features_dc = nn.Parameter(torch.empty(0, 1, 3, device=device))
        self._features_rest = nn.Parameter(torch.empty(0, (sh_degree+1)**2-1, 3, device=device))
        self._opacity = nn.Parameter(torch.empty(0, 1, device=device))
        self._scaling = nn.Parameter(torch.empty(0, 3, device=device))
        self._rotation = nn.Parameter(torch.empty(0, 4, device=device))
        self.register_buffer("max_radii2D", torch.empty(0, device=device))
        self.register_buffer("xyz_gradient_accum", torch.empty(0, 1, device=device))
        self.register_buffer("denom", torch.empty(0, 1, device=device))

    @property
    def get_xyz(self): return self._xyz
    @property
    def get_features(self): return torch.cat((self._features_dc, self._features_rest), dim=1)
    @property
    def get_scaling(self): return self._scaling.exp()
    @property
    def get_rotation(self): return F.normalize(self._rotation, dim=-1)
    @property
    def get_opacity(self): return self._opacity.sigmoid()

    def _reset_statistics(self):
        n = len(self._xyz)
        self.max_radii2D = self._xyz.new_zeros(n)
        self.xyz_gradient_accum = self._xyz.new_zeros(n, 1)
        self.denom = self._xyz.new_zeros(n, 1)

    @torch.no_grad()
    def initialize_random(self, cameras, num_points=1000, seed=0, bounds=None):
        """Initialize a random cloud using train-camera rays, never SfM points.

        The paper specifies 1000 random points but not their volume. This
        implementation estimates a look-at center by least-squares intersection
        of the optical axes and samples a cube about it (documented assumption).
        For a poorly constrained trajectory, pass explicit world-space bounds.
        """
        if not cameras or num_points < 1:
            raise ValueError("Initialization needs cameras and at least one point")
        centers = np.stack([-np.asarray(c.R) @ np.asarray(c.T) for c in cameras])
        rays = np.stack([np.asarray(c.R)[:, 2] for c in cameras])
        projectors = np.eye(3)[None] - rays[:, :, None]*rays[:, None, :]
        center = np.linalg.lstsq(projectors.sum(0),
                                np.einsum("nij,nj->i", projectors, centers), rcond=None)[0]
        radius = np.linalg.norm(centers - centers.mean(0), axis=1).max()
        distance = np.linalg.norm(centers - center, axis=1)
        ill_conditioned = np.linalg.cond(projectors.sum(0)) > 1e3
        if not np.isfinite(center).all() or ill_conditioned:
            if bounds is None:
                warnings.warn("Camera optical axes do not constrain a reliable look-at center; "
                              "using a camera-span initialization heuristic. Supply explicit world-space "
                              "initialization bounds for forward-facing scenes.", stacklevel=2)
            center = centers.mean(0) + rays.mean(0) / max(np.linalg.norm(rays.mean(0)), 1e-6) * max(radius*2, 1.0)
            distance = np.linalg.norm(centers-center, axis=1)
        extent = max(float(radius)*1.1, float(np.median(distance))*0.1, 1e-3)
        half = max(float(np.median(distance))*0.35, extent*0.1)
        lo, hi = (center-half, center+half) if bounds is None else bounds
        lo, hi = np.asarray(lo), np.asarray(hi)
        if (lo.shape != (3,) or hi.shape != (3,) or not np.isfinite(lo).all()
                or not np.isfinite(hi).all() or not np.all(hi > lo)):
            raise ValueError("bounds must contain finite increasing 3D corners")
        rng = np.random.default_rng(seed)
        xyz = rng.uniform(lo, hi, (num_points, 3)).astype(np.float32)
        device = self._xyz.device
        xyz = torch.as_tensor(xyz, device=device)
        # Exact small-cloud distances avoid requiring simple-knn for 1000 points.
        if num_points > 1:
            nearest = []
            offset = 0
            for block in xyz.split(512):
                d = torch.cdist(block, xyz)
                # MM-based cdist can produce positive self-distance roundoff.
                # Remove self by index, never by a numerical zero threshold.
                rows = torch.arange(len(block), device=device)
                d[rows, rows+offset] = float("inf")
                nearest.append(d.min(dim=1).values)
                offset += len(block)
            scale = torch.cat(nearest).clamp_min(1e-4)
        else:
            scale = xyz.new_full((1,), half*0.1)
        rgb = torch.as_tensor(rng.uniform(0.25, 0.75, (num_points, 1, 3)), dtype=torch.float32, device=device)
        self._xyz = nn.Parameter(xyz)
        self._features_dc = nn.Parameter((rgb-0.5)/C0)
        self._features_rest = nn.Parameter(xyz.new_zeros(num_points, (self.max_sh_degree+1)**2-1, 3))
        self._opacity = nn.Parameter(xyz.new_full((num_points, 1), math.log(0.1/0.9)))
        self._scaling = nn.Parameter(scale.log()[:, None].expand(-1, 3).clone())
        rotation = xyz.new_zeros(num_points, 4)
        rotation[:, 0] = 1
        self._rotation = nn.Parameter(rotation)
        self.spatial_lr_scale = extent
        self.active_sh_degree = 0
        self._reset_statistics()
        return extent

    def training_setup(self, args):
        self.percent_dense = _arg(args, "percent_dense", 0.01)
        self._training_args = SimpleNamespace(
            position_lr_init=_arg(args, "position_lr_init", 1.6e-4),
            position_lr_final=_arg(args, "position_lr_final", 1.6e-6),
            position_lr_delay_mult=_arg(args, "position_lr_delay_mult", 0.01),
            position_lr_delay_steps=_arg(args, "position_lr_delay_steps", 0),
            position_lr_max_steps=_arg(args, "position_lr_max_steps", _arg(args, "iterations", 30000)))
        rates = {"xyz": self._training_args.position_lr_init*self.spatial_lr_scale,
                 "f_dc": _arg(args, "feature_lr", 0.0025),
                 "f_rest": _arg(args, "feature_lr", 0.0025)/20,
                 "opacity": _arg(args, "opacity_lr", 0.05),
                 "scaling": _arg(args, "scaling_lr", 0.005),
                 "rotation": _arg(args, "rotation_lr", 0.001)}
        self.optimizer = torch.optim.Adam([
            {"params": [getattr(self, attr)], "name": name, "lr": rates[name]}
            for name, attr in self._groups.items()], lr=0.0, eps=1e-15)
        self._reset_statistics()

    def update_learning_rate(self, iteration):
        args = self._training_args
        t = min(max(iteration / max(args.position_lr_max_steps, 1), 0), 1)
        lr = math.exp((1-t)*math.log(args.position_lr_init) + t*math.log(args.position_lr_final))
        if args.position_lr_delay_steps > 0:
            f = min(max(iteration/args.position_lr_delay_steps, 0), 1)
            lr *= args.position_lr_delay_mult + (1-args.position_lr_delay_mult)*math.sin(math.pi*f/2)
        lr *= self.spatial_lr_scale
        self.optimizer.param_groups[0]["lr"] = lr
        return lr

    def oneupSHdegree(self):
        self.active_sh_degree = min(self.active_sh_degree+1, self.max_sh_degree)

    def _camera(self, camera):
        device, dtype = self._xyz.device, self._xyz.dtype
        R = torch.as_tensor(np.asarray(camera.R), device=device, dtype=dtype)
        T = torch.as_tensor(np.asarray(camera.T), device=device, dtype=dtype)
        w, h = int(camera.width), int(camera.height)
        fx, fy = w/(2*math.tan(camera.fovx/2)), h/(2*math.tan(camera.fovy/2))
        cx, cy = getattr(camera, "cx", None), getattr(camera, "cy", None)
        cx, cy = (w/2 if cx is None else float(cx)), (h/2 if cy is None else float(cy))
        return R, T, w, h, fx, fy, cx, cy

    def _colors(self, R, T):
        center = -R @ T
        direction = F.normalize(self._xyz-center, dim=-1)
        return (eval_sh(self.active_sh_degree, self.get_features.transpose(1, 2), direction)+0.5).clamp_min(0)

    def camera_matrices(self, camera):
        """World-to-view and view-to-clip matrices in column-vector convention.

        Exposed for analytic camera checks without pretending to execute CUDA.
        CUDA uses the contiguous transposes of these matrices.
        """
        R, T, w, h, fx, fy, cx, cy = self._camera(camera)
        view = torch.eye(4, device=self._xyz.device, dtype=self._xyz.dtype)
        view[:3, :3], view[:3, 3] = R.T, T
        proj = view.new_zeros(4, 4)
        proj[0, 0], proj[1, 1] = 2*fx/w, 2*fy/h
        proj[0, 2], proj[1, 2] = 2*cx/w-1, 2*cy/h-1
        proj[2, 2], proj[2, 3], proj[3, 2] = 100/(100-0.01), -100*0.01/(100-0.01), 1
        return view, proj

    def render(self, camera, backend="cuda", background=None):
        if not len(self._xyz):
            raise ValueError("Cannot render an empty Gaussian scene")
        if background is None:
            background = self._xyz.new_zeros(3)
        background = torch.as_tensor(background, dtype=self._xyz.dtype, device=self._xyz.device)
        if backend == "cuda":
            return self._render_cuda(camera, background)
        if backend == "reference":
            return self._render_reference(camera, background)
        raise ValueError("backend must be cuda or reference")

    def _render_cuda(self, camera, background):
        if not self._xyz.is_cuda:
            raise RuntimeError("CUDA rendering requires a CUDA model and NVIDIA GPU")
        try:
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        except ImportError as error:
            raise RuntimeError("Install a CUDA-compatible diff-gaussian-rasterization extension; see REPRODUCTION.md") from error
        R, T, w, h, fx, fy, cx, cy = self._camera(camera)
        view, proj = self.camera_matrices(camera)
        screen = torch.zeros_like(self._xyz, requires_grad=torch.is_grad_enabled())
        settings = dict(image_height=h, image_width=w, tanfovx=math.tan(camera.fovx/2),
                        tanfovy=math.tan(camera.fovy/2), bg=background, scale_modifier=1.0,
                        viewmatrix=view.T.contiguous(), projmatrix=(proj@view).T.contiguous(),
                        sh_degree=self.active_sh_degree, campos=-R@T, prefiltered=False, debug=False)
        if "antialiasing" in getattr(GaussianRasterizationSettings, "_fields", ()):
            settings["antialiasing"] = False
        rasterizer = GaussianRasterizer(GaussianRasterizationSettings(**settings))
        result = rasterizer(means3D=self._xyz, means2D=screen, shs=None,
                            colors_precomp=self._colors(R, T), opacities=self.get_opacity,
                            scales=self.get_scaling, rotations=self.get_rotation, cov3D_precomp=None)
        image, radii = result[0], result[1]
        return dict(render=image, viewspace_points=screen, visibility_filter=radii>0, radii=radii)

    def _render_reference(self, camera, background):
        R, T, w, h, fx, fy, cx, cy = self._camera(camera)
        if len(self._xyz)*w*h > 8_000_000:
            raise ValueError("Reference renderer is limited to 8 million Gaussian-pixels; use CUDA or a smaller smoke test")
        xyz = self._xyz@R + T
        z = xyz[:, 2]
        safez = z.clamp_min(0.01)
        xy = torch.stack((fx*xyz[:, 0]/safez+cx-0.5, fy*xyz[:, 1]/safez+cy-0.5), dim=1)
        screen = torch.cat((xy, z[:, None]), dim=1)
        if screen.requires_grad:
            screen.retain_grad()
        rot = quaternion_matrix(self._rotation)
        cov = (rot*self.get_scaling[:, None, :]) @ (rot*self.get_scaling[:, None, :]).transpose(1, 2)
        cov = R.T[None]@cov@R[None]
        jac = self._xyz.new_zeros(len(z), 2, 3)
        jac[:, 0, 0], jac[:, 1, 1] = fx/safez, fy/safez
        jac[:, 0, 2], jac[:, 1, 2] = -fx*xyz[:, 0]/safez.square(), -fy*xyz[:, 1]/safez.square()
        cov2 = jac@cov@jac.transpose(1, 2)+torch.eye(2, device=z.device)[None]*0.3
        inv = torch.linalg.inv(cov2)
        radii = 3*torch.linalg.eigvalsh(cov2)[:, -1].clamp_min(0).sqrt()
        visible = (z>0.01)&(z<100)&(xy[:, 0]+radii>=0)&(xy[:, 0]-radii<w)&(xy[:, 1]+radii>=0)&(xy[:, 1]-radii<h)
        radii = torch.where(visible, radii, torch.zeros_like(radii))
        yy, xx = torch.meshgrid(torch.arange(h, device=z.device, dtype=z.dtype),
                                torch.arange(w, device=z.device, dtype=z.dtype), indexing="ij")
        grid = torch.stack((xx, yy), dim=-1)
        trans = z.new_ones(h, w)
        image = z.new_zeros(3, h, w)
        colors = self._colors(R, T)
        for i in torch.argsort(z.detach()).tolist():
            if not bool(visible[i]):
                continue
            delta = grid-screen[i, :2]
            power = torch.einsum("hwi,ij,hwj->hw", delta, inv[i], delta)
            alpha = (self.get_opacity[i, 0]*torch.exp(-0.5*power)).clamp(max=0.99)
            alpha = torch.where(power<=9, alpha, torch.zeros_like(alpha))
            image = image + colors[i, :, None, None]*(trans*alpha)[None]
            trans = trans*(1-alpha)
        image = image+trans[None]*background[:, None, None]
        # Preserve a well-defined zero gradient for an all-invisible smoke scene.
        image = image + screen.sum()*0 + self.get_features.sum()*0
        return dict(render=image, viewspace_points=screen, visibility_filter=visible, radii=radii)

    @torch.no_grad()
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        if viewspace_point_tensor.grad is None:
            return
        self.xyz_gradient_accum[update_filter] += torch.linalg.vector_norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    @torch.no_grad()
    def _change_points(self, keep, extra):
        """Change point count while preserving retained Adam moments."""
        for group in self.optimizer.param_groups:
            name, old = group["name"], group["params"][0]
            extension = extra[name]
            new = nn.Parameter(torch.cat((old[keep], extension), dim=0))
            old_state = self.optimizer.state.pop(old, None)
            if old_state is not None:
                state = {}
                for key, value in old_state.items():
                    state[key] = (torch.cat((value[keep], torch.zeros_like(extension)), dim=0)
                                  if torch.is_tensor(value) and value.shape == old.shape else value)
                self.optimizer.state[new] = state
            group["params"][0] = new
            setattr(self, self._groups[name], new)
        self._reset_statistics()

    @torch.no_grad()
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size=None, max_points=1_000_000):
        if max_points < 1:
            raise ValueError("max_points must be positive")
        n = len(self._xyz)
        gradients = (self.xyz_gradient_accum/self.denom.clamp_min(1)).flatten()
        large = self.get_scaling.max(dim=1).values > self.percent_dense*extent
        keep = self.get_opacity.flatten() >= min_opacity
        if max_screen_size:
            keep &= (self.max_radii2D <= max_screen_size) & (self.get_scaling.max(dim=1).values <= 0.1*extent)
        if not keep.any():
            keep[self.get_opacity.flatten().argmax()] = True
        if int(keep.sum()) > max_points:
            valid = keep.nonzero().flatten()
            keep.zero_()
            keep[valid[self.get_opacity.flatten()[valid].topk(max_points).indices]] = True
        candidates = ((gradients>=max_grad)&keep).nonzero().flatten()
        # Clone costs +1, split replaces one parent with two children (+1).
        capacity = max(0, max_points-int(keep.sum()))
        if len(candidates)>capacity:
            candidates = candidates[gradients[candidates].topk(capacity).indices]
        clone, split = candidates[~large[candidates]], candidates[large[candidates]]
        indices = torch.cat((clone, split.repeat_interleave(2)))
        extra = {name: getattr(self, attr)[indices].clone() for name, attr in self._groups.items()}
        if len(split):
            ids = split.repeat_interleave(2)
            noise = torch.randn_like(self._xyz[ids])*self.get_scaling[ids]
            noise = (quaternion_matrix(self._rotation[ids])@noise[:, :, None]).squeeze(-1)
            extra["xyz"][len(clone):] += noise
            extra["scaling"][len(clone):] -= math.log(1.6)
            keep[split] = False
        self._change_points(keep, extra)
        return dict(before=n, after=len(self._xyz), cloned=len(clone), split=len(split))

    @torch.no_grad()
    def reset_opacity(self):
        self._opacity.copy_(torch.logit(self.get_opacity.clamp(max=0.01)))
        if self.optimizer is not None:
            state = self.optimizer.state.get(self._opacity, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key].zero_()

    def capture(self):
        return dict(model=deepcopy(self.state_dict()), active_sh_degree=self.active_sh_degree,
                    max_sh_degree=self.max_sh_degree, spatial_lr_scale=self.spatial_lr_scale,
                    optimizer=None if self.optimizer is None else deepcopy(self.optimizer.state_dict()))

    def restore(self, state, args=None):
        device = self._xyz.device
        self.max_sh_degree = state["max_sh_degree"]
        tensors = state["model"]
        for attr in self._groups.values():
            setattr(self, attr, nn.Parameter(tensors[attr].to(device).clone()))
        self._reset_statistics()
        self.load_state_dict({key: value.to(device) for key, value in tensors.items()})
        self.active_sh_degree = state["active_sh_degree"]
        self.spatial_lr_scale = state["spatial_lr_scale"]
        statistics = {name: getattr(self, name).clone() for name in ("max_radii2D", "xyz_gradient_accum", "denom")}
        if args is not None:
            self.training_setup(args)
            if state["optimizer"] is not None:
                self.optimizer.load_state_dict(state["optimizer"])
            for name, value in statistics.items():
                setattr(self, name, value)

    def save_ply(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        dc = self._features_dc.detach().transpose(1, 2).flatten(1).cpu().numpy()
        rest = self._features_rest.detach().transpose(1, 2).flatten(1).cpu().numpy()
        parts = [xyz, np.zeros_like(xyz), dc, rest, self._opacity.detach().cpu().numpy(),
                 self._scaling.detach().cpu().numpy(), self._rotation.detach().cpu().numpy()]
        names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
                 + [f"f_rest_{i}" for i in range(rest.shape[1])] + ["opacity"]
                 + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
        vertex = np.empty(len(xyz), dtype=[(name, "f4") for name in names])
        values = np.concatenate(parts, axis=1)
        for i, name in enumerate(names):
            vertex[name] = values[:, i]
        PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))

    def load_ply(self, path):
        self.max_sh_degree = infer_sh_degree(path)
        vertex = PlyData.read(str(path))["vertex"].data
        n = len(vertex)
        def columns(names):
            return torch.as_tensor(np.stack([vertex[name] for name in names], axis=1).copy(),
                                   dtype=self._xyz.dtype, device=self._xyz.device)
        self._xyz = nn.Parameter(columns(("x", "y", "z")))
        self._features_dc = nn.Parameter(columns([f"f_dc_{i}" for i in range(3)])[:, None])
        rest = [f"f_rest_{i}" for i in range(3*((self.max_sh_degree+1)**2-1))]
        self._features_rest = nn.Parameter(columns(rest).reshape(n, 3, -1).transpose(1, 2).contiguous()
                                           if rest else self._xyz.new_zeros(n, 0, 3))
        self._opacity = nn.Parameter(columns(["opacity"]))
        self._scaling = nn.Parameter(columns([f"scale_{i}" for i in range(3)]))
        self._rotation = nn.Parameter(columns([f"rot_{i}" for i in range(4)]))
        self.active_sh_degree = self.max_sh_degree
        self.optimizer = None
        self._reset_statistics()

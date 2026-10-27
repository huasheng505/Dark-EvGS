"""Validated event loading and interval accumulation, without a CUDA dependency.

Timestamp units are never inferred from inter-event spacing.  Epoch timestamps can
be recognized by magnitude; relative timestamps need an explicit conversion to
seconds.  Windows use (t0, t1], so adjacent windows do not count boundary events
twice.  ``density`` filtering is an optional reproducible approximation, not the
unreleased implementation/parameters of the paper's y-noise filter.
"""
from pathlib import Path
import numpy as np


def seconds_scale(values, explicit=None):
    """Return a unit multiplier; accept seconds/ms/us/ns for epoch-valued data."""
    if explicit is not None:
        scale = float(explicit)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('timestamp_scale must be finite and positive')
        return scale
    values = np.asarray(values)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Timestamps must be nonempty and finite')
    magnitude = float(np.median(np.abs(values)))
    # Unix epoch ranges: approximately years 2001--2286; no relative-time guess.
    for lower, upper, scale in ((1e9, 1e10, 1.), (1e12, 1e13, 1e-3),
                                (1e15, 1e16, 1e-6), (1e18, 1e19, 1e-9)):
        if lower <= magnitude < upper:
            return scale
    raise ValueError('Ambiguous timestamp units: supply timestamp_scale (e.g. 1e-6 for microseconds)')


class EventStream:
    def __init__(self, path, timestamp_scale=None):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            missing = {'x', 'y', 't', 'p'} - set(data.files)
            if missing:
                raise ValueError(f'{self.path}: missing event arrays {sorted(missing)}')
            arrays = {key: np.asarray(data[key]) for key in ('x', 'y', 't', 'p')}
        shapes = {v.shape for v in arrays.values()}
        if len(shapes) != 1 or arrays['t'].ndim != 1 or not arrays['t'].size:
            raise ValueError('Event arrays must have identical nonempty one-dimensional shapes')
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Event arrays contain nonfinite values')
        for key in ('x', 'y'):
            values = arrays[key]
            if np.any(values < 0) or np.any(values != np.floor(values)):
                raise ValueError(f'Event {key} coordinates must be nonnegative integers')
        raw_t = arrays['t']
        if np.any(raw_t[1:] < raw_t[:-1]):
            raise ValueError('Events must be sorted by timestamp; sort paired arrays before loading')
        self.timestamp_scale = seconds_scale(raw_t, timestamp_scale)
        self.t = raw_t.astype(np.float64) * self.timestamp_scale
        self.x = arrays['x'].astype(np.int32)
        self.y = arrays['y'].astype(np.int32)
        p = arrays['p']
        if np.isin(p, [0, 1]).all():
            p = np.where(p == 0, -1, 1)
        elif not np.isin(p, [-1, 1]).all():
            raise ValueError('Event polarities must use either 0/1 or -1/+1 encoding')
        self.p = p.astype(np.float32)
        self._filter_cache = {}
        self.time_range = (float(self.t[0]), float(self.t[-1]))
        self.width = int(self.x.max()) + 1
        self.height = int(self.y.max()) + 1

    def __len__(self):
        return len(self.t)

    def _window(self, t0, t1, height, width):
        if not np.isfinite([t0, t1]).all() or t1 <= t0:
            raise ValueError('Event interval must satisfy finite t0 < t1')
        if height <= 0 or width <= 0:
            raise ValueError('Event image dimensions must be positive')
        if self.height > height or self.width > width:
            raise ValueError(f'Events require at least {self.width}x{self.height}; requested {width}x{height}. '
                             'Undistort/crop event coordinates consistently with the frames first.')
        if t1 < self.t[0] or t0 > self.t[-1]:
            raise ValueError('Requested interval does not overlap the event timestamps; check units/origin')
        start, stop = np.searchsorted(self.t, [t0, t1], side='right')
        return slice(int(start), int(stop))

    def _density_keep(self, window, height, width, radius, min_neighbors, density_window_us):
        """Exact centered spatiotemporal support, including the event itself.

        Search keys combine pixel ID and relative timestamp with sufficient padding
        to prevent cross-pixel comparisons. Support can extend outside the rendered
        frame interval. This is a documented density-filter proxy, not the authors'
        unpublished y-noise implementation.
        """
        if radius < 0 or int(radius) != radius or min_neighbors < 1:
            raise ValueError('density_radius must be a nonnegative integer and min_neighbors >= 1')
        if not np.isfinite(density_window_us) or density_window_us <= 0:
            raise ValueError('density_window_us must be finite and positive')
        radius = int(radius)
        cache_key = (window.start, window.stop, height, width, radius, min_neighbors, density_window_us)
        if cache_key in self._filter_cache:
            return self._filter_cache[cache_key]
        query_t = self.t[window]
        if not query_t.size:
            return np.zeros(0, dtype=bool)
        half = float(density_window_us) * 0.5e-6
        left = int(np.searchsorted(self.t, query_t[0] - half, side='left'))
        right = int(np.searchsorted(self.t, query_t[-1] + half, side='right'))
        origin = self.t[left]
        stride = float(self.t[right-1] - origin) + 4 * half + 1.
        pixel = self.y[left:right].astype(np.int64) * width + self.x[left:right]
        keys = np.sort(pixel * stride + (self.t[left:right] - origin) + 2 * half)
        x, y = self.x[window], self.y[window]
        offsets = query_t - origin + 2 * half
        counts = np.zeros(len(query_t), dtype=np.int32)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                valid = (x+dx >= 0) & (x+dx < width) & (y+dy >= 0) & (y+dy < height)
                target = (y[valid].astype(np.int64)+dy)*width + x[valid]+dx
                center = target * stride + offsets[valid]
                counts[valid] += (np.searchsorted(keys, center+half, side='right')
                                  - np.searchsorted(keys, center-half, side='left')).astype(np.int32)
        keep = counts >= min_neighbors
        if len(self._filter_cache) >= 256:
            self._filter_cache.pop(next(iter(self._filter_cache)))
        self._filter_cache[cache_key] = keep
        return keep

    def accumulate(self, t0, t1, height, width, contrast=0.2,
                   filter_mode='none', density_radius=1, min_neighbors=2, density_window_us=2000.):
        """Return float32 ``contrast * sum(polarity)`` at each native sensor pixel.

        ``density`` counts spatial neighbors within a centered temporal window
        of ``density_window_us`` (2000 microseconds by default), including self.
        It is a proxy with no implied equivalence to the paper's y-filter code.
        An empty *overlapping* interval is valid and returns zeros (not a mask).
        """
        if not np.isfinite(contrast) or contrast <= 0:
            raise ValueError('contrast must be finite and positive')
        window = self._window(t0, t1, height, width)
        x, y, p = self.x[window], self.y[window], self.p[window]
        if filter_mode == 'density':
            keep = self._density_keep(window, height, width, density_radius, min_neighbors, density_window_us)
            x, y, p = x[keep], y[keep], p[keep]
        elif filter_mode != 'none':
            raise ValueError('filter_mode must be none or density')
        output = np.zeros((height, width), dtype=np.float32)
        np.add.at(output, (y, x), p * float(contrast))
        return output

    def voxel(self, t0, t1, bins, height, width, filter_mode='none', density_radius=1,
              min_neighbors=2, density_window_us=2000.):
        """Bilinearly interpolate signed unit events over temporal bins (B,H,W)."""
        if bins < 1 or int(bins) != bins:
            raise ValueError('bins must be a positive integer')
        bins = int(bins)
        window = self._window(t0, t1, height, width)
        x, y, p, times = self.x[window], self.y[window], self.p[window], self.t[window]
        if filter_mode == 'density':
            keep = self._density_keep(window, height, width, density_radius, min_neighbors, density_window_us)
            x, y, p, times = x[keep], y[keep], p[keep], times[keep]
        elif filter_mode != 'none':
            raise ValueError('filter_mode must be none or density')
        output = np.zeros((bins, height, width), dtype=np.float32)
        scaled = np.clip((times - t0) / (t1 - t0) * (bins - 1), 0, bins - 1)
        lower = np.floor(scaled).astype(np.int64)
        fraction = (scaled - lower).astype(np.float32)
        np.add.at(output, (lower, y, x), p * (1 - fraction))
        upper = lower + 1
        valid = upper < bins
        np.add.at(output, (upper[valid], y[valid], x[valid]), p[valid] * fraction[valid])
        return output

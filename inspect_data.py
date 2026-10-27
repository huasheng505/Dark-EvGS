#!/usr/bin/env python3
"""Audit source data, poses and synchronized event coverage without CUDA."""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
from dark_evgs.data import SceneDataset
from dark_evgs.events import EventStream


def inspect_scene(source, timestamps=None, timestamp_scale=None):
    source = Path(source).resolve()
    result = {'scene': source.name, 'source': str(source), 'ready': False, 'warnings': [], 'errors': []}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            dataset = SceneDataset(source, split='all', timestamps=timestamps,
                                   timestamp_scale=timestamp_scale)
        result['warnings'] = [str(w.message) for w in caught]
        frames = dataset.frames
        result.update(frames=len(frames), train_frames=len(dataset.train), test_frames=len(dataset.test),
                      frame_time_range=[frames[0].timestamp, frames[-1].timestamp],
                      image_sizes=sorted({(f.width, f.height) for f in frames}),
                      frame_timestamp_scale=dataset.timestamp_scale)
    except (ValueError, FileNotFoundError, KeyError, OSError) as error:
        result['errors'].append(str(error))
        frames = None
    try:
        events = EventStream(source/'event.npz', timestamp_scale=timestamp_scale)
        result.update(events=len(events), event_time_range=list(events.time_range),
                      event_coordinate_extent=[events.width, events.height],
                      event_timestamp_scale=events.timestamp_scale)
        if frames is not None:
            if any(f.width < events.width or f.height < events.height for f in frames):
                result['errors'].append('Event coordinates extend outside calibrated frame dimensions; '
                                        'consistent event/frame calibration is required')
            start, end = events.time_range
            inside = sum(start <= f.timestamp <= end for f in frames)
            result['frames_inside_event_span'] = inside
            if inside < 2:
                result['errors'].append('Fewer than two frame timestamps overlap event coverage; '
                                        'check timestamp units and clock origin')
            elif inside != len(frames):
                result['warnings'].append(f'{len(frames)-inside} frame timestamps lie outside event coverage; '
                                          'their interval supervision may be incomplete')
            # Native pixel event accumulation is a functional alignment check.
            first, second = frames[0], frames[1]
            if max(first.timestamp, start) < min(second.timestamp, end):
                counts = events.accumulate(first.timestamp, second.timestamp,
                                           first.height, first.width, contrast=1.)
                result['first_interval_signed_sum'] = float(counts.sum())
                result['first_interval_nonzero_pixels'] = int(np.count_nonzero(counts))
    except (ValueError, FileNotFoundError, KeyError, OSError) as error:
        result['errors'].append(str(error))
    result['ready'] = not result['errors']
    result['scope'] = 'data completeness only; RGB/event pixel calibration is not verified by range checks'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='*', help='Scene folders; defaults to all data/* folders')
    parser.add_argument('--timestamps', help='Explicit filename,timestamp manifest for one scene')
    parser.add_argument('--timestamp-scale', type=float, help='Multiplier converting timestamps to seconds')
    args = parser.parse_args()
    sources = args.sources or sorted(str(p) for p in Path('data').iterdir() if p.is_dir())
    if args.timestamps and len(sources) != 1:
        parser.error('--timestamps requires exactly one source scene')
    results = [inspect_scene(source, args.timestamps, args.timestamp_scale) for source in sources]
    print(json.dumps(results, indent=2))
    return 0 if all(result['ready'] for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())

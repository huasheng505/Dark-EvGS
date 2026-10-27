"""CPU checks for supervision alignment and interval semantics."""
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import numpy as np
from PIL import Image
from dark_evgs.data import SceneDataset
from dark_evgs.events import EventStream


class EventTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'events.npz'

    def write(self, **overrides):
        data = dict(t=np.array([0., 1., 2., 3.]), x=np.array([0, 0, 1, 1]),
                    y=np.array([0, 0, 0, 0]), p=np.array([0, 1, 0, 1]))
        data.update(overrides)
        np.savez(self.path, **data)
        return self.path

    def test_polarity_boundaries_and_empty_window(self):
        stream = EventStream(self.write(), timestamp_scale=1.)
        np.testing.assert_allclose(stream.accumulate(0., 2., 1, 2), [[.2, -.2]])
        np.testing.assert_allclose(stream.accumulate(2., 3., 1, 2), [[0., .2]])
        self.assertEqual(stream.accumulate(.2, .8, 1, 2).sum(), 0.)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            stream.accumulate(4., 5., 1, 2)

    def test_voxel_preserves_signed_counts(self):
        stream = EventStream(self.write(p=np.array([-1, 1, -1, 1])), timestamp_scale=1.)
        voxel = stream.voxel(0., 3., 5, 1, 2)
        np.testing.assert_allclose(voxel.sum(0), [[1., 0.]])
        self.assertEqual(voxel.shape, (5, 1, 2))
        np.testing.assert_allclose(stream.voxel(0., 3., 1, 1, 2)[0], [[1., 0.]])

    def test_validation_and_units(self):
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            EventStream(self.write())
        stream = EventStream(self.write(t=np.array([1730000000000000, 1730000000100000,
                                                    1730000000200000, 1730000000300000])))
        self.assertEqual(stream.timestamp_scale, 1e-6)
        with self.assertRaisesRegex(ValueError, 'sorted'):
            EventStream(self.write(t=np.array([0., 2., 1., 3.])), timestamp_scale=1.)
        with self.assertRaisesRegex(ValueError, 'polarities'):
            EventStream(self.write(p=np.array([0, 1, 2, 3])), timestamp_scale=1.)
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            EventStream(self.write(x=np.array([0, 0, -1, 1])), timestamp_scale=1.)

    def test_density_filter_uses_temporal_neighbors_outside_render_window(self):
        stream = EventStream(self.write(t=np.array([0., .0005, .005, .0055, .01]),
                                       x=np.array([0, 1, 0, 1, 0]), y=np.zeros(5),
                                       p=np.ones(5)), timestamp_scale=1.)
        filtered = stream.accumulate(0., .011, 1, 2, filter_mode='density',
                                     density_radius=1, min_neighbors=2, density_window_us=2000.)
        np.testing.assert_allclose(filtered, [[.2, .4]])
        # Event at .0055 supports .005 even though it lies outside this render interval.
        narrow = stream.accumulate(.0049, .0051, 1, 2, filter_mode='density',
                                   density_radius=1, min_neighbors=2, density_window_us=2000.)
        np.testing.assert_allclose(narrow, [[.2, 0.]])
        voxels = stream.voxel(0., .011, 3, 1, 2, filter_mode='density')
        np.testing.assert_allclose(voxels.sum(0), [[1., 2.]])

    def test_native_dimensions_and_density_filter(self):
        stream = EventStream(self.write(), timestamp_scale=1.)
        with self.assertRaisesRegex(ValueError, 'require at least'):
            stream.accumulate(0., 2., 1, 1)
        filtered = stream.accumulate(0., 2., 1, 2, filter_mode='density',
                                     density_radius=0, min_neighbors=2)
        np.testing.assert_allclose(filtered, [[0., 0.]])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root/'low_light_frames').mkdir()
        (self.root/'output').mkdir()
        cameras = []
        for i in range(10):
            name = f'{i:05d}.png'
            Image.new('RGB', (4, 3), (10, 20, 30)).save(self.root/'low_light_frames'/name)
            cameras.append(dict(img_name=f'{i:06d}.png', width=4, height=3,
                                position=[i, 0, 0], rotation=np.eye(3).tolist(), fx=3., fy=3.))
        (self.root/'output/cameras.json').write_text(json.dumps(cameras))
        (self.root/'timestamps.txt').write_text('\n'.join(str(i/10) for i in range(10)))

    def dataset(self, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return SceneDataset(self.root, timestamp_scale=1., **kw)

    def test_split_pose_convention_and_numeric_filename_mapping(self):
        dataset = self.dataset(split='all')
        self.assertEqual(len(dataset.frames), 10)
        self.assertEqual([f.name for f in dataset.test], ['000000.png', '000005.png'])
        self.assertEqual(len(dataset.train), 8)
        frame = dataset.frames[1]
        self.assertEqual(frame.image_path.name, '00001.png')
        np.testing.assert_allclose(frame.T, [-1, 0, 0])
        self.assertEqual(frame.timestamp, .1)

    def test_ambiguous_timestamp_mapping_rejected(self):
        (self.root/'timestamps.txt').write_text('\n'.join(str(i/10) for i in range(20)))
        with self.assertRaisesRegex(ValueError, 'more timestamps'):
            self.dataset()
        (self.root/'timestamps.txt').write_text('0 1 2 3\n')
        with self.assertRaisesRegex(ValueError, 'event export'):
            self.dataset()

    def test_static_camera_loading_never_invents_frame_timestamps(self):
        (self.root/'timestamps.txt').write_text('unusable event export 1 2 3\n')
        dataset = self.dataset(split='all', require_timestamps=False)
        self.assertEqual(len(dataset.frames), 10)
        self.assertTrue(all(frame.timestamp is None for frame in dataset.frames))
        self.assertIsNone(dataset.timestamp_scale)
        self.assertIsNone(dataset.timestamps_path)
        self.assertEqual([f.name for f in dataset.test], ['000000.png', '000005.png'])
        with self.assertRaisesRegex(ValueError, 'event export'):
            self.dataset(split='all')
        (self.root/'timestamps.txt').unlink()
        self.assertEqual(len(self.dataset(require_timestamps=False).frames), 8)
        with self.assertRaisesRegex(FileNotFoundError, 'timestamps'):
            self.dataset()

    def test_explicit_named_manifest_and_missing_pose(self):
        manifest = self.root/'named.csv'
        manifest.write_text('\n'.join(f'{i:05d}.png,{i/10}' for i in reversed(range(10))))
        dataset = self.dataset(timestamps=manifest)
        self.assertEqual(len(dataset.frames), 8)
        (self.root/'output/cameras.json').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'camera poses'):
            self.dataset()


if __name__ == '__main__':
    unittest.main()

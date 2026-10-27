"""Prepare a scene-disjoint pilot prior manifest from supplied repository scenes.

This uses public scene pairs, not the authors' unpublished 500 auxiliary pairs.
It never reads the target scene's bright images. See docs/COLAB_RESULTS.md.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import warnings

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('data'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-scene',default='baseball')
    p.add_argument('--val-scene',default='panda')
    p.add_argument('--target-scene',default='lion')
    p.add_argument('--event-bins',type=int,default=5)
    p.add_argument('--timestamp-scale',type=float,default=1e-6)
    p.add_argument('--event-timestamp-scale',type=float,default=1e-6)
    args=p.parse_args()
    import numpy as np
    from PIL import Image
    from dark_evgs.data import SceneDataset
    from dark_evgs.events import EventStream
    names=[args.train_scene,args.val_scene,args.target_scene]
    if len(set(names))!=3 or any(Path(name).name!=name for name in names):
        raise ValueError('Train, validation and target must be distinct scene folder names')
    if args.event_bins<2:
        raise ValueError('At least two event bins are required')
    args.output=args.output.resolve()
    if args.output.is_relative_to(args.data_root.resolve()):
        raise ValueError('Write auxiliary outputs outside the original scene data')
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=args.output.name+'.staging-',dir=args.output.parent))
    manifest={'held_out_scenes':[args.target_scene],'train':[],'val':[]}
    try:
        for split,scene in [('train',args.train_scene),('val',args.val_scene)]:
            source=args.data_root.resolve()/scene
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning)
                dataset=SceneDataset(source,split='all',timestamp_scale=args.timestamp_scale)
            events=EventStream(source/'event.npz',timestamp_scale=args.event_timestamp_scale)
            for index,(first,second) in enumerate(zip(dataset.frames[:-1],dataset.frames[1:])):
                bright=[source/'gt_bright_frames'/frame.name for frame in (first,second)]
                for path in bright:
                    with Image.open(path) as image:
                        if image.size!=(first.width,first.height):
                            raise ValueError(f'Auxiliary bright image resolution mismatch: {path}')
                voxel=events.voxel(first.timestamp,second.timestamp,args.event_bins,first.height,first.width)
                name=f'{scene}_{index:04d}.npz'
                np.savez_compressed(staging/name,voxel=voxel)
                manifest[split].append({'scene':scene,'dark0':str(first.image_path),'dark1':str(second.image_path),
                                       'bright0':str(bright[0]),'bright1':str(bright[1]),'event_voxel':name})
            print(json.dumps({'scene':scene,'split':split,'pairs':len(manifest[split])}),flush=True)
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        staging.rename(args.output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(f'Manifest: {args.output/"manifest.json"}')


if __name__=='__main__':
    main()

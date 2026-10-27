"""Exercise prior->pseudo->GS train->resume->inference->metrics on synthetic data.

This verifies executable plumbing and gradients, NOT real-data paper quality.
All commands run in subprocesses and their original exit codes are preserved.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir', type=Path, required=True, help='new/empty output directory')
    p.add_argument('--device', default='cpu')
    p.add_argument('--backend', choices=['reference', 'cuda'], default='reference')
    args = p.parse_args()
    output = args.workdir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new/empty smoke workdir')
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    source = output/'smoke_target'
    (source/'low_light_frames').mkdir(parents=True)
    (source/'gt_bright_frames').mkdir()
    (source/'output').mkdir()
    cameras = []
    h,w = 12,16
    yy,xx = np.mgrid[:h,:w]
    for index in range(10):
        pixels = np.stack([40+5*xx+index,50+6*yy,70+2*xx+2*yy],axis=-1).astype(np.uint8)
        name = f'{index:06d}.png'
        Image.fromarray(pixels).save(source/'gt_bright_frames'/name)
        Image.fromarray((pixels*.1).astype(np.uint8)).save(source/'low_light_frames'/name)
        cameras.append({'img_name':name,'width':w,'height':h,'fx':20.,'fy':20.,
                        'rotation':np.eye(3).tolist(),'position':[(index-4.5)*.02,0,0]})
    (source/'output/cameras.json').write_text(json.dumps(cameras))
    (source/'timestamps.txt').write_text('\n'.join(str(i*.1) for i in range(10))+'\n')
    rng = np.random.default_rng(42)
    times = np.linspace(0,.9,400)
    np.savez(source/'event.npz',x=rng.integers(0,w,400),y=rng.integers(0,h,400),
             t=times,p=rng.integers(0,2,400))
    aux = output/'aux'
    aux.mkdir()
    records = []
    for scene in ['aux_train','aux_val']:
        item = {'scene':scene}
        for index in range(2):
            bright = rng.integers(60,180,(h,w,3),dtype=np.uint8)
            for mode in ['dark','bright']:
                name = f'{scene}_{mode}{index}.png'
                Image.fromarray((bright*.1).astype(np.uint8) if mode=='dark' else bright).save(aux/name)
                item[f'{mode}{index}'] = name
        item['event_voxel'] = f'{scene}_voxel.npy'
        np.save(aux/item['event_voxel'],rng.normal(size=(3,h,w)).astype(np.float32))
        records.append(item)
    (aux/'manifest.json').write_text(json.dumps({'held_out_scenes':['smoke_target'],
                                               'train':records[:1],'val':records[1:]}))
    environment = {**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}

    def run(script,*options):
        command = [sys.executable,str(root/script),*map(str,options)]
        subprocess.run(command,cwd=root,env=environment,check=True,timeout=120)

    run('train_prior.py','--manifest',aux/'manifest.json','--output',output/'prior',
        '--epochs',1,'--width',4,'--event-bins',3,'--crop-size',8,'--device',args.device)
    run('prepare_pseudo.py','--source',source,'--output',output/'pseudo','--checkpoint',output/'prior/last.pt',
        '--timestamp-scale',1,'--event-timestamp-scale',1,'--device',args.device)
    common = ['-s',source,'-m',output/'model','--pseudo-dir',output/'pseudo',
              '--timestamp-scale',1,'--event-timestamp-scale',1,'--device',args.device,'--backend',args.backend,
              '--num-points',16,'--sh-degree',1,'--log-every',1,'--save-every',4,
              '--densify-from',2,'--densify-until',5,'--densify-interval',2,
              '--densify-threshold',0,'--max-points',32,'--opacity-reset-interval',1000]
    run('train.py',*common,'--iterations',4)
    run('train.py',*common,'--iterations',6,'--resume',output/'model/checkpoints/latest.pt')
    run('infer.py','-m',output/'model','--device',args.device,'--backend',args.backend)
    renders = output/'model/test/ours_6/renders'
    run('evaluate.py','--renders',renders,'--ground-truth',source/'gt_bright_frames')
    metrics = json.loads((renders.parent/'metrics.json').read_text())
    assert metrics['count']==2
    import torch
    checkpoint = torch.load(output/'model/checkpoints/latest.pt',map_location='cpu',weights_only=True)
    assert checkpoint['iteration']==6
    assert checkpoint['model']['optimizer']['state']
    continuous = list(common)
    continuous[continuous.index('-m')+1] = output/'continuous'
    run('train.py',*continuous,'--iterations',6)
    uninterrupted = torch.load(output/'continuous/checkpoints/latest.pt',map_location='cpu',weights_only=True)
    comparison = 'exact' if args.backend == 'reference' and args.device == 'cpu' else 'render_rmse<1e-3_max<1e-2'
    differences = {key:float((value-uninterrupted['model']['model'][key]).abs().max()) if value.numel() else 0.
                   for key,value in checkpoint['model']['model'].items()}
    if comparison == 'exact':
        assert all(value==0 for value in differences.values()), f'continuation mismatch: {differences}'
    else:
        # CUDA rasterizer backward uses atomic accumulation. Tiny gradient
        # differences can change Adam/rotation updates and then split positions
        # even before a checkpoint. Compare held-out rendered behavior here;
        # test_cuda_backend separately verifies exact checkpoint/Adam restoration.
        from dark_evgs.backend import GaussianScene
        from dark_evgs.data import SceneDataset
        fields=[]
        for state in (checkpoint,uninterrupted):
            field=GaussianScene(device=args.device)
            field.restore(state['model'])
            fields.append(field)
        render_errors=[]
        with torch.no_grad():
            for camera in SceneDataset(source,split='test',timestamp_scale=1).frames:
                delta=fields[0].render(camera,args.backend)['render']-fields[1].render(camera,args.backend)['render']
                error={'name':camera.name,'rmse':float(delta.square().mean().sqrt()),'max_abs':float(delta.abs().max())}
                assert error['rmse']<1e-3 and error['max_abs']<1e-2,error
                render_errors.append(error)
    resumed_pairs=[json.loads(line)['pair'] for line in (output/'model/train_log.jsonl').read_text().splitlines()]
    continuous_pairs=[json.loads(line)['pair'] for line in (output/'continuous/train_log.jsonl').read_text().splitlines()]
    assert resumed_pairs==continuous_pairs,'Continuation did not preserve training-view sampling'
    assert torch.equal(checkpoint['rng'],uninterrupted['rng']),'Continuation CPU RNG differs'
    for image in renders.glob('*.png'):
        assert np.asarray(Image.open(image)).max()>0
    summary = {'scope':'synthetic functionality only; not paper reproduction metrics',
               'device':args.device,'backend':args.backend,'resume_comparison':comparison,
               'parameter_max_differences':differences,
               'render_differences':render_errors if comparison!='exact' else [],
               'stages':['prior training','train-view pseudo generation','GS training','resume','held-out inference','PSNR/SSIM'],
               'final_iteration':6,'test_views':2,'resume_matches_uninterrupted':True,'metrics':metrics['mean']}
    (output/'smoke_result.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()

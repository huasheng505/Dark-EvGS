"""Evaluate rendered views against matching bright GT; never used in field training."""
import argparse
import json
import math
from pathlib import Path


def evaluate(renders, ground_truth, device='cpu', with_lpips=False):
    import torch
    from dark_evgs.runtime import image_tensor
    from utils.loss_utils import ssim
    renders, ground_truth = Path(renders), Path(ground_truth)
    manifest = renders.parent/'render_manifest.json'
    if manifest.is_file():
        names = json.loads(manifest.read_text())['frame_names']
    else:
        names = sorted(p.name for p in renders.glob('*.png'))
    if not names or len(names) != len(set(names)):
        raise ValueError('Need a nonempty list of uniquely named renders')
    perceptual = None
    if with_lpips:
        import lpips
        perceptual = lpips.LPIPS(net='alex').to(device).eval()
    rows = []
    with torch.inference_mode():
        for name in names:
            expected = ground_truth/name
            # Match differing zero padding only when unambiguous, as in Baseball.
            if not expected.is_file() and Path(name).stem.isdecimal():
                candidates = [p for p in ground_truth.glob('*.png') if p.stem.isdecimal() and int(p.stem)==int(Path(name).stem)]
                if len(candidates)==1:
                    expected = candidates[0]
            a, b = image_tensor(renders/name, device=device), image_tensor(expected, device=device)
            if a.shape != b.shape:
                raise ValueError(f'{name}: render/GT shape mismatch; evaluate native-resolution renders')
            mse = float((a-b).square().mean())
            row = {'name':name, 'psnr':-10*math.log10(max(mse,1e-12)),
                   'ssim':float(ssim(a[None], b[None]))}
            if perceptual:
                row['lpips'] = float(perceptual(a[None]*2-1, b[None]*2-1).mean())
            rows.append(row)
    keys = ['psnr','ssim'] + (['lpips'] if with_lpips else [])
    return {'count':len(rows), 'mean':{key:sum(row[key] for row in rows)/len(rows) for key in keys},
            'per_view':rows, 'lpips_status':'computed' if with_lpips else 'not requested',
            'psnr_perfect_match_cap_db':120.0,
            'protocol':'RGB [0,1], full uncropped native resolution; mean per-view metrics; SSIM legacy 11x11 Gaussian'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--renders', type=Path, required=True)
    p.add_argument('--ground-truth', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--lpips', action='store_true', help='requires optional lpips and torchvision, pretrained AlexNet weights')
    args = p.parse_args()
    from dark_evgs.runtime import write_json
    result = evaluate(args.renders,args.ground_truth,args.device,args.lpips)
    write_json(args.output or args.renders.parent/'metrics.json', result)
    print(json.dumps({'count':result['count'],'mean':result['mean'],'lpips_status':result['lpips_status']},indent=2))


if __name__ == '__main__':
    main()

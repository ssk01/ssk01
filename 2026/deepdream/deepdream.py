"""Actual DeepDream: freeze ImageNet GoogLeNet; ascend gradients of image pixels.

Run: python deepdream.py --layer inception4d --steps 35
Dependencies: numpy, pillow, torch, torchvision, scikit-learn.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import googlenet, GoogLeNet_Weights


def digest(model):
    result = hashlib.sha256()
    for p in model.parameters():
        result.update(p.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def make_features(layer, device):
    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1).eval()
    names = ['conv1', 'maxpool1', 'conv2', 'conv3', 'maxpool2',
             'inception3a', 'inception3b', 'maxpool3', 'inception4a',
             'inception4b', 'inception4c', 'inception4d', 'inception4e']
    features = torch.nn.Sequential(*(getattr(model, name) for name in names[:names.index(layer)+1]))
    features.to(device).eval().requires_grad_(False)
    return features


def to_image(x):
    pixels = x.detach().clamp(0, 1).cpu()[0].permute(1, 2, 0).numpy()
    return Image.fromarray(np.uint8(np.round(255*pixels)))


def dream(image, features, steps, sizes, outdir, device):
    original = torch.tensor(np.asarray(image).copy(), dtype=torch.float32, device=device).permute(2, 0, 1)[None]/255
    learned_detail = torch.zeros_like(original)
    frames = [dict(file='original.jpg', step=0, octave=0, loss=None)]
    count = 0
    rng = np.random.default_rng(42)
    for octave, size in enumerate(sizes, 1):
        h = round(image.height*size/image.width)
        base = F.interpolate(original, size=(h, size), mode='bilinear', align_corners=False)
        x = (base + F.interpolate(learned_detail, size=(h, size), mode='bilinear', align_corners=False)).clamp(0,1)
        for iteration in range(1, steps+1):
            x = x.detach().requires_grad_(True)
            shift = tuple(int(v) for v in rng.integers(-8,9,size=2))
            jittered = torch.roll(x, shifts=shift, dims=(-2,-1))
            # Torchvision GoogLeNet's built-in input transform reduces to RGB [-1,1].
            activation = features(2*jittered-1)
            objective = activation.square().mean()
            grad, = torch.autograd.grad(objective, x)
            assert torch.isfinite(grad).all()
            with torch.no_grad():
                x = (x + 0.012*grad/(grad.abs().mean()+1e-8)).clamp(0,1)
            count += 1
            if iteration % 5 == 0 or iteration == steps:
                filename = f'{outdir.name}/frame-{count:03d}.jpg'
                to_image(x).resize((768,512), Image.Resampling.LANCZOS).save(ROOT/filename, quality=92)
                frames.append(dict(file=filename, step=count, octave=octave, loss=float(objective.detach().cpu())))
                print(f'{outdir.name}: step {count}, size {size}, activation energy {frames[-1]["loss"]:.4f}', flush=True)
        learned_detail = x.detach()-base
    to_image(x).save(outdir/'final.png')
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', default='inception4d', choices=['inception3a','inception4c','inception4d','inception4e'])
    parser.add_argument('--steps', type=int, default=35)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    outdir = ROOT/args.layer
    outdir.mkdir(exist_ok=True)
    if not (ROOT/'original.jpg').exists():
        from sklearn.datasets import load_sample_image
        Image.fromarray(load_sample_image('flower.jpg')).save(ROOT/'original.jpg', quality=95)
    image = Image.open(ROOT/'original.jpg').convert('RGB')
    features = make_features(args.layer,device)
    before = digest(features)
    started = time.monotonic()
    frames = dream(image, features, args.steps, [200,280,392,548,768],outdir,device)
    assert digest(features)==before, 'Weights changed unexpectedly'
    metadata = dict(layer=args.layer, device=str(device), weights='GoogLeNet_Weights.IMAGENET1K_V1',
                    weight_sha256=before, weights_unchanged=True, elapsed_seconds=time.monotonic()-started,
                    step_size=0.012, steps_per_octave=args.steps, sizes=[200,280,392,548,768], frames=frames,
                    image_source='scikit-learn load_sample_image flower.jpg',
                    method='Gradient ascent on mean squared hidden activations, random jitter and multiscale detail carry-over')
    (outdir/'run.json').write_text(json.dumps(metadata,indent=2))
    print('Finished',args.layer,round(metadata['elapsed_seconds'],1),'seconds; fixed weights verified.',flush=True)


if __name__=='__main__':
    main()

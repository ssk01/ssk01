"""Compare dataset-trained GoogLeNets and diagnose random-weight brightness."""
from pathlib import Path
import json
import hashlib
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import googlenet
import pytorch_caffe_models as pcm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
DEVICE = torch.device('cpu')  # Caffe LRN is not supported on all MPS versions.
IMAGE = torch.tensor(np.asarray(Image.open(ROOT.parent / 'original.jpg')).copy(), dtype=torch.float32).permute(2, 0, 1)[None] / 255
SIZES = [200, 280, 392, 548, 768]

def save(x, name):
    Image.fromarray((x.detach().clamp(0, 1)[0].permute(1, 2, 0).numpy()*255).round().astype('uint8')).save(OUT / (name+'.png'))

def stats(x):
    return {'mean':float(x.mean()), 'near_white_channels':float((x > .99).float().mean()), 'near_black_channels':float((x < .01).float().mean())}

def digest(model):
    return hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in model.state_dict().values())).hexdigest()

def dream(model, transform, initial, name):
    detail = torch.zeros_like(initial)
    rng = np.random.default_rng(42)
    before = digest(model)
    for size in SIZES:
        shape = (round(initial.shape[-2]*size/initial.shape[-1]), size)
        base = F.interpolate(initial, size=shape, mode='bilinear', align_corners=False)
        x = (base + F.interpolate(detail, size=shape, mode='bilinear', align_corners=False)).clamp(0, 1)
        for step in range(35):
            x = x.detach().requires_grad_(True)
            shift = tuple(int(v) for v in rng.integers(-8, 9, 2))
            a = model(transform(torch.roll(x, shift, (-2, -1))))
            objective = a.square().mean()
            g, = torch.autograd.grad(objective, x)
            assert torch.isfinite(g).all()
            x = (x + .012*g/g.abs().mean().clamp_min(1e-30)).detach().clamp(0, 1)
        detail = x-base
    assert digest(model) == before
    save(x, name)
    print(name, stats(x), flush=True)
    return {'weights_sha256':before, 'stats':stats(x)}

def random_model(seed):
    torch.manual_seed(seed)
    model = googlenet(weights=None, aux_logits=False, init_weights=False)
    names = ['conv1','maxpool1','conv2','conv3','maxpool2','inception3a','inception3b','maxpool3','inception4a','inception4b','inception4c','inception4d']
    return nn.Sequential(*(getattr(model,n) for n in names)).eval().requires_grad_(False)

def probe(model, x):
    x=x.clone().requires_grad_(True)
    value=model(2*x-1).square().mean()
    g,=torch.autograd.grad(value,x)
    grad=g/g.abs().mean().clamp_min(1e-30)
    return {'energy':float(value.detach()), 'mean_normalized_gradient':float(grad.mean()), 'radial_gradient':float(((x-.5)*grad).mean().detach()), 'euler_ratio':float(((x-.5)*g).sum().detach()/value.detach()) if value.detach()>0 else None}

results={'sizes':SIZES,'steps_per_size':35,'step_size':.012,'original':stats(IMAGE)}
signatures=[]
for name, loader in [('imagenet',pcm.googlenet_bvlc),('places205',pcm.googlenet_places205)]:
    model, native = loader()
    parts=[]
    for n,m in model.named_children():
        parts.append(m)
        if n=='inception_4d':break
    features=nn.Sequential(*parts).eval().requires_grad_(False)
    signatures.append(str(features))
    # Shared preprocessing avoids a mean-subtraction confound in the main comparison.
    common=lambda x:(255*x-x.new_tensor([123,117,104])[None,:,None,None]).flip(1)
    results[name]=dream(features,common,IMAGE,name)
    if name=='places205':results['places205_native']=dream(features,native,IMAGE,'places205-native')
assert signatures[0]==signatures[1]
results['identical_feature_architecture']=True
for seed in [42,43]:
    model=random_model(seed)
    trans=lambda x:2*x-1
    results[f'random_{seed}']=dream(model,trans,IMAGE,f'random-{seed}')
    results[f'probes_{seed}']={str(v):probe(model,torch.full((1,3,160,240),v)) for v in [.1,.3,.5,.7,.9]}
    results[f'probes_{seed}']['flower']=probe(model,F.interpolate(IMAGE,size=(160,240)))
    if seed==42:
        for name,initial in [('inverted',1-IMAGE),('gray',torch.full_like(IMAGE,.5))]:
            results[name]=dream(model,trans,initial,'random-'+name)
(OUT/'measurements.json').write_text(json.dumps(results,indent=2))
print('Finished')

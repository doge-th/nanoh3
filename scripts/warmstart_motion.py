"""Two-phase VAE training: appearance first, then motion at a small step size.

Why two phases. Every single-phase run oscillated between two attractors:
  - a good one: L1 0.03-0.16, motion exactly 0 (a still clip)
  - a bad one:  L1 0.6-1.7,  motion 10-20x the reference (motion, but the picture broke)
The jump happens in one optimisation step, so no weight on the temporal term
lands between them. Training appearance to convergence *first* and then adding
motion at a small learning size means the model never has to cross the gap in one
step.

usage: warmstart_motion.py [out_dir] [phase_a_steps] [phase_b_steps]
"""
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, '/Users/doge/.zcode/workspace/default/nanoh3')
from nanoh3.data.procedural import ProceduralClips
from nanoh3.models.vae import NanoVAE, VAEConfig

torch.manual_seed(0)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('runs/warmstart')
A_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 250
B_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 250
TW_B = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
LR_B = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-5
KL = 0.05

data = ProceduralClips(frames=9, size=64, seed=0)
train = torch.stack([data[i][0] for i in range(16)])
vae = NanoVAE(VAEConfig(base_channels=16, channel_multipliers=(1, 2, 2, 4, 4, 4), norm_mode='clip'))
start = time.perf_counter()


def score(n=16):
    with torch.no_grad():
        l1s, mot, ref = [], [], []
        for i in range(n):
            x = train[i : i + 1]
            la = vae.encode(x).chunk(2, dim=1)[0]
            r = vae.decode(la, (9, 64, 64))[0]
            l1s.append(float((r - x[0]).abs().mean()))
            mot.append(sum(float((r[:, f] - r[:, f - 1]).abs().mean()) for f in range(1, 9)) / 8)
            ref.append(sum(float((x[0, :, f] - x[0, :, f - 1]).abs().mean()) for f in range(1, 9)) / 8)
    return sum(l1s) / n, sum(mot) / n, sum(ref) / n


def run(steps, tw, lr, tag):
    opt = torch.optim.AdamW(vae.parameters(), lr=lr)
    for step in range(1, steps + 1):
        frac = step / steps
        lr_now = lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g['lr'] = lr_now
        idx = torch.randint(0, len(train), (4,))
        v = train[idx]
        post = vae.encode(v)
        lat = post.chunk(2, dim=1)[0]
        rec = vae.decode(lat, (9, 64, 64))
        l1 = F.l1_loss(rec, v)
        d_rec = rec[:, :, 1:] - rec[:, :, :-1]
        d_ref = v[:, :, 1:] - v[:, :, :-1]
        w = d_ref.abs().flatten(1).sum(1)
        per_clip = (w.view(-1, 1) * (d_rec - d_ref).pow(2).flatten(1)).sum(1)
        scale = d_ref.pow(2).flatten(1).mean(1).clamp(min=1e-8)
        raw = per_clip / w.clamp(min=1e-8) / scale
        temporal = torch.where(w > 1e-6, raw, torch.zeros_like(w)).mean()
        loss = l1 + tw * temporal + KL * vae.kl_divergence(post)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            L, M, R = score()
            flag = 'PASS' if (L < 0.25 and M / R >= 0.5) else ''
            print(
                f'  [{tag}] step {step:4d}/{steps} lr={lr_now:.2e} tw={tw:.1f} l1={L:.4f} '
                f'motion={M:.5f} ratio={M / R:.2f}x {flag} ({time.perf_counter() - start:.0f}s)',
                flush=True,
            )
    return score()


print(f'phase A: appearance only, {A_STEPS} steps @3e-4', flush=True)
L, M, R = run(A_STEPS, 0.0, 3e-4, 'A')
OUT.mkdir(parents=True, exist_ok=True)
torch.save({'vae': vae.state_dict()}, OUT / 'phaseA.pt')
print(f'  -> phase A done: l1={L:.4f} ratio={M / R:.2f}x', flush=True)

print(f'phase B: add motion tw={TW_B} @ {LR_B:.0e}', flush=True)
L, M, R = run(B_STEPS, TW_B, LR_B, 'B')
torch.save({'vae': vae.state_dict()}, OUT / 'phaseB.pt')
print(f'  -> phase B done: l1={L:.4f} ratio={M / R:.2f}x  saved to {OUT}', flush=True)
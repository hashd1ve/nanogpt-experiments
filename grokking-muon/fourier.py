"""
What solution does each optimizer find? -- Fourier analysis of the grokked embeddings.

Nanda et al. ("Progress measures for grokking"): a transformer that groks (a+b) mod p learns a
FOURIER-MULTIPLICATION algorithm -- the token embeddings concentrate their power on a few "key
frequencies" (roughly circular embeddings). Diagnostic: take the DFT of W_E along the token axis;
a Fourier-algorithm model puts its power on a few frequencies.

exp result: AdamW+wd groks at LOW effective rank (~0.54), Muon at HIGH rank (~0.86). Question:
does Muon use the same sparse Fourier algorithm (few frequencies) at higher rank, or a DISTRIBUTED
solution (many frequencies)? This retrains deterministically (per seed) and analyses the embeddings.

Usage: python -u fourier.py --steps 2500 --seeds 1337 2 3
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import torch.nn.functional as F
import harness as Hn
from model import GPTConfig, GPT
from muon import Muon

INK, MUTED, SURFACE, GRID = '#2b2b2b', '#8a8a8a', '#fcfcfb', '#e6e6e6'
C_ADAMW, C_MUON = '#D55E00', '#0072B2'


def train_arm(opt_kind, wd, lr, muon_lr, steps, p, frac, seed, dev):
    """Reproduce the training exactly (same harness pieces)."""
    torch.manual_seed(seed)
    Xtr, Ytr, Xva, Yva = Hn.make_data(p, frac, seed, dev)
    cfg = GPTConfig(block_size=3, vocab_size=p + 1, n_layer=1, n_head=4, n_embd=128, dropout=0.0, bias=False)
    model = GPT(cfg).to(dev)
    hidden, other = Hn.partition(model)
    if opt_kind == 'adamw':
        opt = torch.optim.AdamW([{'params': hidden, 'weight_decay': wd},
                                 {'params': other, 'weight_decay': wd}], lr=lr, betas=(0.9, 0.98))
        muon = None
    else:
        muon = Muon(hidden, lr=muon_lr)
        opt = torch.optim.AdamW([{'params': other, 'weight_decay': wd}], lr=lr, betas=(0.9, 0.98))
    for _ in range(steps):
        logits = model(Xtr)[0][:, -1, :]
        loss = F.cross_entropy(logits, Ytr)
        opt.zero_grad(set_to_none=True)
        if muon is not None:
            muon.zero_grad(set_to_none=True)
        loss.backward()
        if muon is not None and wd > 0:
            with torch.no_grad():
                for pr in hidden:
                    pr.mul_(1.0 - lr * wd)          # same pressure as AdamW (reference lr)
        opt.step()
        if muon is not None:
            muon.step()
    with torch.no_grad():
        va = (model(Xva)[0][:, -1, :].argmax(-1) == Yva).float().mean().item()
    return model, va


def fourier_stats(model, p):
    """DFT of the p number-token embeddings along the token axis."""
    WE = model.transformer.wte.weight.detach().float().cpu()[:p]      # (p, d)
    WE = WE - WE.mean(0, keepdim=True)                                # remove DC (freq 0)
    Fh = torch.fft.fft(WE, dim=0)                                     # (p, d) complex
    power = (Fh.abs() ** 2).sum(1)                                    # power per frequency
    half = (p - 1) // 2
    fp = power[1:half + 1]                                            # freqs 1..48 (real symmetry)
    fp = fp / fp.sum()
    part = float(1.0 / (fp ** 2).sum())                              # participation ratio = # effective freqs
    order = torch.argsort(fp, descending=True)
    cum = torch.cumsum(fp[order], 0)
    n90 = int((cum < 0.90).sum().item()) + 1                          # # freqs to reach 90% power
    top = [(int(order[i].item()) + 1, float(fp[order[i]].item())) for i in range(min(5, len(order)))]
    return fp.numpy(), part, n90, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=2500)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1337, 2, 3])
    ap.add_argument('--p', type=int, default=97)
    ap.add_argument('--frac', type=float, default=0.3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--muon_lr', type=float, default=0.01)
    ap.add_argument('--device', type=str, default='mps')
    ap.add_argument('--out', type=str, default='figures/g5_fourier.png')
    a = ap.parse_args()
    arms = [('AdamW wd=1', 'adamw', 1.0, C_ADAMW), ('Muon no wd', 'muon', 0.0, C_MUON)]
    results = {}
    print(f"=== Fourier spectrum of grokked embeddings (p={a.p}, frac={a.frac}, steps={a.steps}) ===")
    for lab, kind, wd, col in arms:
        spectra, parts, n90s, vas = [], [], [], []
        for s in a.seeds:
            model, va = train_arm(kind, wd, a.lr, a.muon_lr, a.steps, a.p, a.frac, s, a.device)
            fp, part, n90, top = fourier_stats(model, a.p)
            spectra.append(fp); parts.append(part); n90s.append(n90); vas.append(va)
            print(f"  {lab:12s} seed {s}: val={va:.3f} | effective_freqs(PR)={part:.1f} | "
                  f"n90={n90} | top5={[(f, round(w,3)) for f,w in top]}")
        results[lab] = (np.mean(spectra, 0), np.mean(parts), np.mean(n90s), np.std(n90s), col)
        print(f"  >> {lab}: mean PR={np.mean(parts):.1f}  mean n90={np.mean(n90s):.1f}+-{np.std(n90s):.1f}  "
              f"mean val={np.mean(vas):.3f}")

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=140)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for lab, (spec, pr, n90, n90s, col) in results.items():
        freqs = np.arange(1, len(spec) + 1)
        ax.plot(freqs, spec, color=col, lw=2.2, marker='o', ms=4, zorder=3,
                label=f'{lab}  (PR={pr:.1f}, n90={n90:.0f})')
    ax.set_xlabel('Fourier frequency (over tokens 0..p-1)', color=INK)
    ax.set_ylabel('fraction of embedding power', color=INK)
    ax.set_title('Same algorithm? Fourier spectrum of the grokked embeddings',
                 color=INK, fontsize=12.5, fontweight='bold', pad=12)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#cccccc')
    ax.grid(True, color=GRID, lw=0.7); ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=10, labelcolor=INK)
    fig.tight_layout(); fig.savefig(a.out, facecolor=SURFACE)
    print(f'\nsaved -> {a.out}')


if __name__ == '__main__':
    main()

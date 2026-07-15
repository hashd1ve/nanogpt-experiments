"""
Test CAUSAL (Mundo A, registro): ¿el modelo del mundo es causal o solo correlacional?
Ajustamos una sonda lineal del estado; en una posicion R con memoria m, empujamos el residual
a lo largo de la direccion-sonda (W[m'] - W[m]) para "reescribir" la memoria a m'; corremos el
forward y miramos si la prediccion del valor leido pasa de Val_m a **Val_m'**.

Exito = fraccion que voltea a m'. Control = direccion aleatoria de igual norma (no debe voltear).

Uso:  python -u experiments/worldmodels/causal.py --task register --seed 0
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from tasks import TASKS
from wm_common import load_checkpoint, residuals

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('--task', type=str, default='register')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--layer', type=int, default=-1)         # -1 = ultima capa
ap.add_argument('--device', type=str, default='cpu')
args = ap.parse_args()


def last_step(d):
    return max(int(os.path.basename(f)[len('ckpt_step'):-3]) for f in glob.glob(os.path.join(d, 'ckpt_step*.pt')))


def fit_probe(X, y, nclasses, epochs=300, lr=0.05):
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    W = nn.Linear(X.shape[1], nclasses)
    opt = torch.optim.Adam(W.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(); F.cross_entropy(W(Xn), y).backward(); opt.step()
    return W, mu.squeeze(0), sd.squeeze(0)


def targeted_forward(model, idx, l, pos, vecs):
    device = idx.device
    b, t = idx.size()
    pos_ids = torch.arange(t, device=device)
    x = model.transformer.wte(idx) + model.transformer.wpe(pos_ids)
    x = model.transformer.drop(x)
    for i, block in enumerate(model.transformer.h):
        x = block(x)
        if i == l:
            x = x.clone()
            x[torch.arange(b, device=device), pos] += vecs
    x = model.transformer.ln_f(x)
    return model.lm_head(x)


def main():
    ckdir = os.path.join(HERE, 'logs', args.task, f'seed{args.seed}')
    model, ck = load_checkpoint(os.path.join(ckdir, f'ckpt_step{last_step(ckdir)}.pt'), device=args.device)
    V, T = ck['task_args']['V'], ck['task_args']['T']
    R, VAL0 = V, V + 1
    L = ck['config']['n_layer']
    l = args.layer if args.layer >= 0 else L - 1

    data = TASKS['register'](n=256, T=T, V=V, seed=999)
    toks = data['tokens'].to(args.device)
    mem = data['states']['mem'][0]

    # sonda en posiciones de acarreo (R), split train/test por secuencia
    with torch.no_grad():
        res = residuals(model, toks)[l].cpu()
    mask = data['behav']
    ntr = int(toks.shape[0] * 0.6)
    Xtr = res[:ntr].reshape(-1, res.shape[-1])[mask[:ntr].reshape(-1)]
    ytr = mem[:ntr].reshape(-1)[mask[:ntr].reshape(-1)].long()
    W, mu, sd = fit_probe(Xtr, ytr, V)

    # una posicion R por secuencia de test (la ultima con valor siguiente)
    scale = res.reshape(-1, res.shape[-1])[mask.reshape(-1)].norm(dim=-1).mean().item()
    seqs, poss, ms, mps = [], [], [], []
    for i in range(ntr, toks.shape[0]):
        rpos = (toks[i] == R).nonzero().flatten()
        rpos = [int(p) for p in rpos if p + 1 < T]
        if not rpos:
            continue
        p = rpos[-1]
        m = int(mem[i, p]); mp = (m + 1) % V
        seqs.append(i); poss.append(p); ms.append(m); mps.append(mp)
    seqs = torch.tensor(seqs); poss = torch.tensor(poss)
    ms = torch.tensor(ms); mps = torch.tensor(mps)
    idx = toks[seqs]

    Wd = W.weight.detach()
    dirs = ((Wd[mps] - Wd[ms]) / sd)                 # direccion m->m' en espacio residual
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(dirs.shape, generator=g)
    rnd = rnd / (rnd.norm(dim=-1, keepdim=True) + 1e-8)

    print(f"{args.task} seed{args.seed} capa {l}: intervencion causal ({len(seqs)} casos), scale={scale:.2f}")
    with torch.no_grad():
        base = targeted_forward(model, idx, l, poss, torch.zeros_like(dirs))
        pb = base[torch.arange(len(seqs)), poss].argmax(-1)
        keep_true = (pb == VAL0 + ms).float().mean().item()
        print(f"  base (sin intervenir): predice el valor VERDADERO Val_m en {keep_true:.3f}")
        for a in (2, 4, 8, 12):
            fj = targeted_forward(model, idx, l, poss, a * scale * dirs)
            fr = targeted_forward(model, idx, l, poss, a * scale * rnd)
            pj = fj[torch.arange(len(seqs)), poss].argmax(-1)
            pr = fr[torch.arange(len(seqs)), poss].argmax(-1)
            flip_j = (pj == VAL0 + mps).float().mean().item()
            flip_r = (pr == VAL0 + mps).float().mean().item()
            print(f"  alpha={a:2d}: voltea a Val_m' sonda={flip_j:.3f}  aleatorio={flip_r:.3f}")


if __name__ == '__main__':
    main()

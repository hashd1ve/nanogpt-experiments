"""
Sonda LINEAL del estado latente VERDADERO desde el residual. Verificable: comparamos contra
el estado que nosotros generamos.

HONESTIDAD: probamos SOLO en las posiciones donde el estado debe ACARREARSE y no esta en el
token actual (mascara `behav`) -> ese es el test real del modelo del mundo (evita la fuga
trivial: en un token de escritura/valor, el estado ES el token). Controles: etiquetas barajadas
(nulo) y modelo sin entrenar.

probe_all(model, data, device) -> {estado: {capa: acc_carry, 'shuffle': acc0}}
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from wm_common import residuals


def linear_probe(Xtr, ytr, Xte, yte, nclasses, epochs=250, lr=0.05, device='cpu', seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    W = nn.Linear(Xtr.shape[1], nclasses).to(device)
    opt = torch.optim.Adam(W.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(W(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return (W(Xte).argmax(-1) == yte).float().mean().item()


def _split_masked(res_l, y, mask, frac=0.8, cap=24000):
    """Split por SECUENCIA (sin fuga) y quedarse solo con las posiciones de `mask`."""
    n, d = res_l.shape[0], res_l.shape[-1]
    ntr = int(n * frac)

    def collect(a, b, m):
        m = m.reshape(-1)
        return a.reshape(-1, d)[m], b.reshape(-1)[m]

    Xtr, ytr = collect(res_l[:ntr], y[:ntr], mask[:ntr])
    Xte, yte = collect(res_l[ntr:], y[ntr:], mask[ntr:])
    if Xtr.shape[0] > cap:
        Xtr, ytr = Xtr[:cap], ytr[:cap]
    return Xtr, ytr.long(), Xte, yte.long()


@torch.no_grad()
def _features(model, tokens, device):
    return [r.detach().cpu() for r in residuals(model, tokens.to(device))]


def probe_all(model, data, device='cpu'):
    feats = _features(model, data['tokens'], device)          # L x [n,T,d]
    L = len(feats)
    mask = data['behav']                                      # posiciones que exigen el estado
    out = {}
    for name, (states, ncls) in data['states'].items():
        out[name] = {}
        for l in range(L):
            Xtr, ytr, Xte, yte = _split_masked(feats[l], states, mask)
            out[name][l] = round(linear_probe(Xtr, ytr, Xte, yte, ncls, device=device), 4)
        # control nulo: etiquetas barajadas (ultima capa)
        Xtr, ytr, Xte, yte = _split_masked(feats[L-1], states, mask)
        out[name]['shuffle'] = round(
            linear_probe(Xtr, ytr[torch.randperm(ytr.shape[0])], Xte, yte, ncls, device=device), 4)
    return out

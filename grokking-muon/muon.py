"""
Muon — MomentUm Orthogonalized by Newton-schulz  (Keller Jordan, finales de 2024).
El optimizador estrella del "nanoGPT speedrun".

IDEA EN UNA FRASE
-----------------
AdamW/SGD aplican el gradiente (con momento) tal cual. Muon, para cada MATRIZ de
pesos 2D, primero ORTOGONALIZA la actualización. Intuición: el gradiente suele
"empujar" muy fuerte en unas pocas direcciones y flojo en el resto; ortogonalizar
reparte el empuje por igual en todas las direcciones de la matriz, así que cada
paso mueve al modelo de forma más completa y hacen falta menos pasos.

La ortogonalización exacta (SVD) es cara. Muon la APROXIMA con 5 iteraciones de
Newton-Schulz, que son solo multiplicaciones de matrices (baratas en la GPU).

REGLA DE ORO: Muon SOLO va sobre matrices 2D ocultas (atención + MLP).
Embeddings, LayerNorm y la capa de salida se dejan a AdamW: meter Muon ahí perjudica.
"""
import torch


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Aproxima la 'ortogonalización' de la matriz G (mismos ejes que G).

    Newton-Schulz empuja todos los valores singulares de G hacia ~1 sin calcular
    la SVD: solo matmuls. Los coeficientes (a,b,c) están elegidos para converger
    rápido en pocas iteraciones. Se trabaja en bfloat16: es una aproximación, no
    necesita precisión alta.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)             # normaliza para que NS sea estable
    transpose = G.size(0) > G.size(1)
    if transpose:                        # trabaja con la orientación "ancha"
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon para una sola GPU (versión didáctica, sin distribución ni torch.compile)."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum = group['lr'], group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)                                   # momento
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf # nesterov
                u = zeropower_via_newtonschulz5(g, group['ns_steps'])        # ORTOGONALIZA
                u = u * max(1.0, p.size(0) / p.size(1)) ** 0.5               # escala por forma
                p.add_(u, alpha=-lr)


def build_muon_optimizers(model, muon_lr=0.02, adamw_lr=1e-3,
                          weight_decay=0.1, betas=(0.9, 0.99)):
    """Reparte los parámetros del GPT: matrices 2D ocultas -> Muon; el resto -> AdamW.

    Devuelve (muon, adamw, n_muon, n_adamw).
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        es_matriz_oculta = (
            p.ndim == 2
            and 'wte' not in name       # embedding de tokens
            and 'wpe' not in name       # embedding de posición
            and 'lm_head' not in name   # capa de salida (atada al embedding)
        )
        (muon_params if es_matriz_oculta else adamw_params).append(p)

    muon = Muon(muon_params, lr=muon_lr)
    adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr, betas=betas,
                              weight_decay=weight_decay)
    return muon, adamw, len(muon_params), len(adamw_params)

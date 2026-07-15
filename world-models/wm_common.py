"""Utilidades compartidas: construir/cargar el GPT enano y extraer residuales por capa."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from model import GPTConfig, GPT


def build_model(vocab, T, n_layer=3, n_head=4, n_embd=128, device='cpu', seed=1337):
    torch.manual_seed(seed)
    cfg = GPTConfig(block_size=T, vocab_size=vocab, n_layer=n_layer, n_head=n_head,
                    n_embd=n_embd, dropout=0.0, bias=False)
    return GPT(cfg).to(device)


def load_checkpoint(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = GPTConfig(**ck['config'])
    model = GPT(cfg)
    model.load_state_dict(ck['model_state'])
    model.to(device).eval()
    return model, ck


@torch.no_grad()
def residuals(model, idx):
    """Lista de longitud L; residuals[l] = residual tras el bloque l, [B,T,d] (pre-ln_f)."""
    device = idx.device
    _, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)
    x = model.transformer.wte(idx) + model.transformer.wpe(pos)
    x = model.transformer.drop(x)
    outs = []
    for block in model.transformer.h:
        x = block(x)
        outs.append(x)
    return outs


@torch.no_grad()
def all_logits(model, idx):
    """Logits en TODAS las posiciones (model.forward solo da la ultima)."""
    r = residuals(model, idx)[-1]
    return model.lm_head(model.transformer.ln_f(r))

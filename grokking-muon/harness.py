"""
Immutable harness — "Grokking x Muon".

Classic grokking task (Power et al. 2022): c = (a + b) mod p. The input sequence is
[a, b, '='] (the '=' token has id p) and the model predicts c at the LAST position.
Under regularization pressure, train accuracy hits ~100% quickly while validation
accuracy stays flat for a long time and then suddenly GROKS (generalizes). We reuse the
SAME tiny GPT and the SAME Muon optimizer everywhere, so "Muon vs AdamW" is a clean
comparison on one architecture.

Design (built for a defensible result):
  - Fair AdamW-vs-Muon comparison: matching lr/steps is NOT enough (Muon travels further in
    weight space per step). We log the weight travel Sum||dW||_F and the relative travel, and
    plot val accuracy against distance travelled, not just against steps.
  - `--wd` (hidden matrices) and `--emb_wd` (embeddings / lm_head) are separated, so we can
    isolate WHERE weight decay acts (in modular addition the embeddings carry the algorithm;
    note the weight tying wte == lm_head).
  - Geometry probes during training: Frobenius / spectral norm, effective rank (entropy of the
    singular values), cosine between successive updates, logit margin. All SVDs run on CPU.
  - Formal metrics: mem_step (train >= 0.99), gen_step (val >= 0.95 AND stays), grok delay,
    AREA between the curves, travel-to-grok. No eyeballing the "click".

Fair play: `--wd lambda` applies the SAME decoupled shrink (lr_ref * lambda) in both optimizers.
Muon itself carries no weight decay, so for the Muon arm we apply the decoupled shrink by hand
using the SAME reference lr as AdamW -- muon.py stays untouched (ground truth).

Loop metric (minimize): grok_step. `fail` = did not grok within budget (a valid negative result).

Usage:
  python -u harness.py --opt adamw --wd 1.0 --emb_wd 1.0 --frac 0.3 --tag B_adamw_wd
  python -u harness.py --opt muon  --wd 0   --emb_wd 0   --frac 0.3 --tag C_muon_nowd
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import torch.nn.functional as F
from model import GPTConfig, GPT
from muon import Muon  # ground truth

GROK_THR, MEM_THR, STAY_THR = 0.95, 0.99, 0.90


def make_data(p, frac, seed, device):
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    X = torch.stack([a, b, torch.full_like(a, p)], dim=1)   # [a, b, '=']
    Y = (a + b) % p
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=g)
    n_tr = int(frac * p * p)
    tr, va = perm[:n_tr], perm[n_tr:]
    return X[tr].to(device), Y[tr].to(device), X[va].to(device), Y[va].to(device)


def partition(model):
    """Like muon.py: 2D hidden matrices (attn/mlp) vs the rest (embeddings/norms/lm_head)."""
    hidden, other = [], []
    for name, pr in model.named_parameters():
        if not pr.requires_grad:
            continue
        is_hidden = (pr.ndim == 2 and 'wte' not in name and 'wpe' not in name and 'lm_head' not in name)
        (hidden if is_hidden else other).append(pr)
    return hidden, other


@torch.no_grad()
def acc_loss_margin(model, X, Y):
    logits = model(X)[0][:, -1, :]
    loss = F.cross_entropy(logits, Y).item()
    pred = logits.argmax(-1)
    acc = (pred == Y).float().mean().item()
    correct = logits.gather(1, Y[:, None]).squeeze(1)
    tmp = logits.clone(); tmp.scatter_(1, Y[:, None], float('-inf'))
    margin = (correct - tmp.max(1).values).mean().item()
    return acc, loss, margin


@torch.no_grad()
def geometry(hidden):
    """Mean over hidden matrices: effective rank (entropy of sigma), Frobenius, spectral norm."""
    effs, fros, specs = [], [], []
    for pr in hidden:
        sv = torch.linalg.svdvals(pr.detach().float().cpu())
        s = sv / sv.sum().clamp(min=1e-30)
        s = s[s > 0]
        effs.append(float(torch.exp(-(s * torch.log(s)).sum())) / sv.numel())  # ratio in 0..1
        fros.append(float(pr.detach().norm()))
        specs.append(float(sv[0]))
    return float(np.mean(effs)), float(np.mean(fros)), float(np.mean(specs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p', type=int, default=97)
    ap.add_argument('--frac', type=float, default=0.3)
    ap.add_argument('--n_layer', type=int, default=1)
    ap.add_argument('--n_head', type=int, default=4)
    ap.add_argument('--n_embd', type=int, default=128)
    ap.add_argument('--opt', type=str, default='adamw', choices=['adamw', 'muon'])
    ap.add_argument('--lr', type=float, default=1e-3)          # AdamW lr (matrices if adamw; embeddings always)
    ap.add_argument('--muon_lr', type=float, default=0.01)     # Muon lr (matrices if opt=muon)
    ap.add_argument('--wd', type=float, default=1e-2)          # weight decay on HIDDEN matrices (both opts)
    ap.add_argument('--emb_wd', type=float, default=1e-2)      # weight decay on EMBEDDINGS / lm_head
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--geom_every', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--device', type=str, default='mps')
    ap.add_argument('--tag', type=str, default=None)
    ap.add_argument('--stop_after_grok', action='store_true')  # off by default: full budget
    args = ap.parse_args()
    dev = args.device
    tag = args.tag or f"{args.opt}_wd{args.wd:g}_ewd{args.emb_wd:g}_f{args.frac:g}_s{args.seed}"

    torch.manual_seed(args.seed)
    Xtr, Ytr, Xva, Yva = make_data(args.p, args.frac, args.seed, dev)
    cfg = GPTConfig(block_size=3, vocab_size=args.p + 1, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=0.0, bias=False)
    model = GPT(cfg).to(dev)
    hidden, other = partition(model)

    if args.opt == 'adamw':
        opt = torch.optim.AdamW([{'params': hidden, 'weight_decay': args.wd},
                                 {'params': other, 'weight_decay': args.emb_wd}],
                                lr=args.lr, betas=(0.9, 0.98))
        muon = None
    else:
        muon = Muon(hidden, lr=args.muon_lr)
        opt = torch.optim.AdamW([{'params': other, 'weight_decay': args.emb_wd}],
                                lr=args.lr, betas=(0.9, 0.98))

    cdir = os.path.join(HERE, 'results', 'curves'); os.makedirs(cdir, exist_ok=True)
    cpath = os.path.join(cdir, f"{tag}.csv")
    fcsv = open(cpath, 'w')
    fcsv.write("step,train_acc,val_acc,train_loss,val_loss,margin,cum_travel,rel_travel,"
               "eff_rank,fro,spec,upd_cos\n")

    steps_l, tr_l, va_l, travel_l = [], [], [], []
    cum_travel = 0.0
    prev_flat = None
    eff = fro = spec = ucos = float('nan')
    print(f"[grok] {tag} | opt={args.opt} p={args.p} frac={args.frac} lr={args.lr} muon_lr={args.muon_lr} "
          f"wd={args.wd} emb_wd={args.emb_wd} seed={args.seed} dev={dev} | {sum(q.numel() for q in model.parameters())/1e3:.0f}K")
    grok_step = None
    for step in range(args.steps + 1):
        if step % args.geom_every == 0:
            model.eval(); eff, fro, spec = geometry(hidden); model.train()
        if step % args.eval_every == 0:
            model.eval()
            tra, trl, _ = acc_loss_margin(model, Xtr, Ytr)
            vaa, val, mrg = acc_loss_margin(model, Xva, Yva)
            model.train()
            fcsv.write(f"{step},{tra:.4f},{vaa:.4f},{trl:.4f},{val:.4f},{mrg:.4f},"
                       f"{cum_travel:.5f},{cum_travel/(fro+1e-9):.5f},{eff:.4f},{fro:.4f},{spec:.4f},{ucos:.4f}\n")
            fcsv.flush()
            steps_l.append(step); tr_l.append(tra); va_l.append(vaa); travel_l.append(cum_travel)
            if grok_step is None and vaa >= GROK_THR:
                grok_step = step
                print(f"  GROK step {step}: val={vaa:.3f} travel={cum_travel:.2f}", flush=True)
            if step % (args.eval_every * 10) == 0:
                print(f"  step {step:6d} | train {tra:.3f} | val {vaa:.3f} | eff_rank {eff:.3f} | travel {cum_travel:.2f}", flush=True)
            if args.stop_after_grok and grok_step is not None and step >= grok_step + args.eval_every * 5:
                break
        if step == args.steps:
            break
        # ---- full-batch step ----
        logits = model(Xtr)[0][:, -1, :]
        loss = F.cross_entropy(logits, Ytr)
        opt.zero_grad(set_to_none=True)
        if muon is not None:
            muon.zero_grad(set_to_none=True)
        loss.backward()
        snap = [pr.detach().clone() for pr in hidden]              # to measure the real ||dW||
        if muon is not None and args.wd > 0:                       # decoupled wd on hidden matrices
            with torch.no_grad():                                  # SAME pressure as AdamW: shrink = lr_ref * wd
                for pr in hidden:                                  # (uses args.lr, NOT muon_lr -> apples to apples)
                    pr.mul_(1.0 - args.lr * args.wd)
        opt.step()
        if muon is not None:
            muon.step()
        with torch.no_grad():                                      # weight travel + update cosine
            upd = [pr.detach() - sp for pr, sp in zip(hidden, snap)]
            flat = torch.cat([u.reshape(-1) for u in upd])
            cum_travel += float(flat.norm())
            if prev_flat is not None:
                ucos = float(F.cosine_similarity(flat, prev_flat, dim=0))
            prev_flat = flat
    fcsv.close()

    # ---- formal metrics (post-hoc, robust) ----
    st, tr, va, tv = map(np.array, (steps_l, tr_l, va_l, travel_l))
    mem_step = int(st[np.argmax(tr >= MEM_THR)]) if (tr >= MEM_THR).any() else None
    gen_step = None
    for i in range(len(st)):
        if va[i] >= GROK_THR and (va[i:] >= STAY_THR).all():       # grokked AND stays
            gen_step = int(st[i]); break
    grokked = gen_step is not None
    delay = (gen_step - mem_step) if (grokked and mem_step is not None) else None
    gap = np.clip(tr - va, 0, None)                                # area between curves (trapezoid)
    area = float(np.sum((gap[:-1] + gap[1:]) / 2.0 * np.diff(st))) if len(st) > 1 else 0.0
    travel_to_grok = float(tv[st == gen_step][0]) if grokked else float(tv[-1])
    metric = str(gen_step) if grokked else 'fail'
    print(f"\n=== RESULT {tag} ===")
    print(f"mem_step={mem_step} gen_step={gen_step} grok_delay={delay} grokked={grokked}")
    print(f"area_gap={area:.1f} travel_to_grok={travel_to_grok:.2f} final_val={va[-1]:.3f} "
          f"eff_rank_fin={eff:.3f} fro_fin={fro:.2f}")
    print(f"\nMETRIC {metric}  (gen_step; lower=better; fail=did not grok/stay)")
    print(f"SECONDARY grokked={grokked},mem={mem_step},gen={gen_step},delay={delay},"
          f"area={area:.1f},travel={travel_to_grok:.2f},final_val={va[-1]:.3f},eff_rank={eff:.3f}")
    print(f"CURVE_CSV {cpath}")


if __name__ == '__main__':
    main()

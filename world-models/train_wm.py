"""
Entrena un transformer enano en una tarea de mundo, con checkpoints DENSOS, por semilla.
Reusa model.py + muon.py. Brujula conductual = behav_acc (acierto en las posiciones cuya
prediccion exige el estado latente).

Uso:
  python -u experiments/worldmodels/train_wm.py --task register --seed 0
  python -u experiments/worldmodels/train_wm.py --task gridworld --seed 0 --max_iters 3000
"""
import os, sys, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))       # experiments/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo
import torch
import torch.nn.functional as F
from muon import build_muon_optimizers
from tasks import TASKS
from wm_common import build_model, all_logits

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('--task', type=str, default='register', choices=list(TASKS))
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--n_layer', type=int, default=3)
ap.add_argument('--n_head', type=int, default=4)
ap.add_argument('--n_embd', type=int, default=128)
ap.add_argument('--T', type=int, default=128)
ap.add_argument('--batch_size', type=int, default=64)
ap.add_argument('--max_iters', type=int, default=2500)
ap.add_argument('--warmup', type=int, default=50)
ap.add_argument('--muon_lr', type=float, default=0.02)
ap.add_argument('--adamw_lr', type=float, default=1e-3)
ap.add_argument('--pool', type=int, default=8192)
ap.add_argument('--V', type=int, default=8)
ap.add_argument('--k', type=int, default=5)
ap.add_argument('--device', type=str, default='mps')
args = ap.parse_args()

device = args.device


def gen(nseq, seed):
    if args.task == 'register':
        return TASKS['register'](n=nseq, T=args.T, V=args.V, seed=seed)
    return TASKS['gridworld'](n=nseq, T=args.T, k=args.k, seed=seed)


def cosine_lr(it, base, mn):
    if it < args.warmup:
        return base * (it + 1) / (args.warmup + 1)
    if it > args.max_iters:
        return mn
    r = (it - args.warmup) / max(1, args.max_iters - args.warmup)
    return mn + 0.5 * (1 + math.cos(math.pi * r)) * (base - mn)


@torch.no_grad()
def behav_acc(model, ev):
    toks = ev['tokens'].to(device)
    logits = all_logits(model, toks)                  # [n,T,vocab]
    pred = logits[:, :-1].argmax(-1)                  # prediccion de t+1
    true = toks[:, 1:]
    mask = ev['behav'][:, :-1].to(device)             # posiciones que exigen estado
    if mask.sum() == 0:
        return 0.0
    return (pred[mask] == true[mask]).float().mean().item()


def log_steps(mx):
    base = [0, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1100, 1500, 2000]
    return sorted(set([s for s in base if s <= mx] + [mx]))


def main():
    train = gen(args.pool, seed=1000 + args.seed)
    ev = gen(1024, seed=999)                           # eval fijo (mismo para todas las semillas)
    vocab = train['vocab_size']
    pool_tok = train['tokens']
    outdir = os.path.join(HERE, 'logs', args.task, f'seed{args.seed}')
    os.makedirs(outdir, exist_ok=True)

    model = build_model(vocab, args.T, args.n_layer, args.n_head, args.n_embd, device, seed=args.seed)
    muon, adamw, nm, na = build_muon_optimizers(model, args.muon_lr, args.adamw_lr)
    opts = [muon, adamw]; bases = [args.muon_lr, args.adamw_lr]; mins = [args.muon_lr/10, args.adamw_lr/10]
    save = set(log_steps(args.max_iters))
    g = torch.Generator().manual_seed(args.seed)
    print(f"[{args.task} seed{args.seed}] {model.get_num_params()/1e6:.2f}M | vocab {vocab} | "
          f"{args.max_iters} iters | {len(save)} ckpts", flush=True)

    for it in range(args.max_iters + 1):
        for opt, base, mn in zip(opts, bases, mins):
            lr = cosine_lr(it, base, mn)
            for gr in opt.param_groups:
                gr['lr'] = lr
        if it in save:
            model.eval()
            ba = behav_acc(model, ev)
            state = {kk: vv.detach().cpu() for kk, vv in model.state_dict().items()}
            torch.save({'model_state': state, 'config': model.config.__dict__, 'step': it,
                        'behav_acc': ba, 'task': args.task,
                        'task_args': {'V': args.V, 'k': args.k, 'T': args.T}},
                       os.path.join(outdir, f'ckpt_step{it}.pt'))
            model.train()
            print(f"  step {it:4d} | behav_acc {ba:.3f}", flush=True)
        if it == args.max_iters:
            break
        ix = torch.randint(pool_tok.shape[0], (args.batch_size,), generator=g)
        xb = pool_tok[ix].to(device)
        _, loss = model(xb[:, :-1].contiguous(), xb[:, 1:].contiguous())   # next-token: predice t+1
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in opts:
            opt.step()
    print(f"OK -> {outdir}")


if __name__ == '__main__':
    main()

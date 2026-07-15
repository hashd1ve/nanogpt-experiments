"""
Universalidad: ¿distintas semillas construyen la MISMA geometria del modelo del mundo?
Como todas las semillas ven el MISMO eval (seed 999), las posiciones se corresponden 1:1,
asi que podemos medir similitud representacional con **CKA lineal** (invariante a la base).

Control: CKA(entrenado, SIN entrenar) debe ser bajo. Escalar: CKA medio par-a-par entre
semillas entrenadas por capa. Alto => universal.

Uso:  python -u experiments/worldmodels/universality.py --task register --seeds 0,1,2,3,4
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tasks import TASKS
from wm_common import load_checkpoint, build_model, residuals

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('--task', type=str, default='register')
ap.add_argument('--seeds', type=str, default='0,1,2,3,4')
ap.add_argument('--device', type=str, default='cpu')
args = ap.parse_args()

seeds = [int(s) for s in args.seeds.split(',')]


def cka(X, Y):
    """CKA lineal entre X[N,d1], Y[N,d2] (mismas N filas = mismas posiciones)."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xty = (Y.t() @ X).pow(2).sum()
    xx = (X.t() @ X).norm()
    yy = (Y.t() @ Y).norm()
    return (xty / (xx * yy + 1e-9)).item()


def last_step(ckdir):
    files = glob.glob(os.path.join(ckdir, 'ckpt_step*.pt'))
    return max(int(os.path.basename(f)[len('ckpt_step'):-3]) for f in files)


def gen_eval(task, ta):
    if task == 'register':
        return TASKS['register'](n=256, T=ta['T'], V=ta['V'], seed=999)
    return TASKS['gridworld'](n=256, T=ta['T'], k=ta['k'], seed=999)


@torch.no_grad()
def carry_feats(model, data, l):
    r = residuals(model, data['tokens'])[l]                # [n,T,d]
    mask = data['behav']                                   # posiciones de estado
    return r.reshape(-1, r.shape[-1])[mask.reshape(-1)]    # [Ncarry, d]


def main():
    ck0 = load_checkpoint(os.path.join(HERE, 'logs', args.task, f'seed{seeds[0]}',
                                       f'ckpt_step{last_step(os.path.join(HERE, "logs", args.task, f"seed{seeds[0]}"))}.pt'))[1]
    ta = ck0['task_args']; L = ck0['config']['n_layer']; vocab = ck0['config']['vocab_size']; T = ta['T']
    data = gen_eval(args.task, ta)

    # features de cada semilla entrenada + un modelo SIN entrenar (control)
    feats = {}  # seed -> {l: [N,d]}
    for s in seeds:
        ckdir = os.path.join(HERE, 'logs', args.task, f'seed{s}')
        m, _ = load_checkpoint(os.path.join(ckdir, f'ckpt_step{last_step(ckdir)}.pt'))
        feats[s] = {l: carry_feats(m, data, l) for l in range(L)}
    untr = build_model(vocab, T, n_layer=L, n_head=ck0['config']['n_head'],
                       n_embd=ck0['config']['n_embd'], seed=777).eval()
    feats['untrained'] = {l: carry_feats(untr, data, l) for l in range(L)}

    print(f"{args.task}: CKA medio par-a-par entre semillas {seeds} (y control sin-entrenar)")
    rows = []
    for l in range(L):
        vals = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                vals.append(cka(feats[seeds[i]][l], feats[seeds[j]][l]))
        mean_tr = sum(vals) / len(vals)
        # control: media CKA(entrenado, sin-entrenar)
        ctl = sum(cka(feats[s][l], feats['untrained'][l]) for s in seeds) / len(seeds)
        rows.append((l, round(mean_tr, 4), round(ctl, 4)))
        print(f"  capa {l}: CKA_entrenadas={mean_tr:.3f}  vs  CKA_vs_sin-entrenar={ctl:.3f}")

    out = os.path.join(HERE, 'logs', args.task, 'universality.csv')
    with open(out, 'w') as f:
        f.write('layer,cka_trained,cka_vs_untrained\n')
        for r in rows:
            f.write(','.join(map(str, r)) + '\n')
    print(f"CSV -> {out}")


if __name__ == '__main__':
    main()

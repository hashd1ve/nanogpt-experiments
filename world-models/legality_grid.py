"""
Brujula conductual CORRECTA para gridworld: el siguiente movimiento es aleatorio entre los
LEGALES, asi que exact-match no sirve (tope ~1/n_legales). Medimos **legalidad** = masa de
probabilidad que el modelo pone en movimientos legales dada la posicion verdadera. Un modelo
que USA su mapa -> ~1.0; sin mapa -> ~n_legales/4. Escribe legality en la develop.csv.

Uso:  python -u experiments/worldmodels/legality_grid.py --seeds 0,1,2,3,4
"""
import os, sys, glob, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from tasks import gridworld_task, _MOVES
from wm_common import load_checkpoint, all_logits

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--seeds', type=str, default='0,1,2,3,4')
args = ap.parse_args()


@torch.no_grad()
def legality(model, data, k):
    toks = data['tokens']
    xs, ys = data['states']['x'][0], data['states']['y'][0]
    legal = torch.zeros(*toks.shape, 4)                 # [n,T,4]
    for m, (dx, dy) in _MOVES.items():
        legal[..., m] = ((xs + dx >= 0) & (xs + dx < k) & (ys + dy >= 0) & (ys + dy < k)).float()
    p = F.softmax(all_logits(model, toks), dim=-1)      # [n,T,4]
    return (p * legal).sum(-1).mean().item()


def main():
    for s in args.seeds.split(','):
        ckdir = os.path.join(HERE, 'logs', 'gridworld', f'seed{int(s)}')
        files = sorted(glob.glob(os.path.join(ckdir, 'ckpt_step*.pt')),
                       key=lambda f: int(os.path.basename(f)[len('ckpt_step'):-3]))
        _, ck0 = load_checkpoint(files[0])
        k, T = ck0['task_args']['k'], ck0['task_args']['T']
        data = gridworld_task(n=256, T=T, k=k, seed=999)
        legs = {}
        for f in files:
            step = int(os.path.basename(f)[len('ckpt_step'):-3])
            m, _ = load_checkpoint(f)
            legs[step] = round(legality(m, data, k), 4)
        # augmentar develop.csv con la columna legality
        csvp = os.path.join(ckdir, 'develop.csv')
        with open(csvp) as fh:
            rows = list(csv.DictReader(fh))
        cols = list(rows[0].keys()) + (['legality'] if 'legality' not in rows[0] else [])
        for r in rows:
            r['legality'] = legs.get(int(float(r['step'])), '')
        with open(csvp, 'w') as fh:
            fh.write(','.join(cols) + '\n')
            for r in rows:
                fh.write(','.join(str(r[c]) for c in cols) + '\n')
        ordered = sorted(legs)
        print(f"seed{s}: legality {legs[ordered[0]]:.3f} -> {legs[ordered[-1]]:.3f}")


if __name__ == '__main__':
    main()

"""
Driver developmental para UNA (tarea, semilla): por checkpoint, sonda-verdad por capa +
behav_acc -> logs/{task}/seed{seed}/develop.csv. Materia prima de la pelicula del nacimiento.

Uso:  python -u experiments/worldmodels/develop_wm.py --task register --seed 0
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tasks import TASKS
from wm_common import load_checkpoint
from probe import probe_all

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('--task', type=str, default='register', choices=list(TASKS))
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--nprobe', type=int, default=256)
ap.add_argument('--device', type=str, default='cpu')
args = ap.parse_args()

CKDIR = os.path.join(HERE, 'logs', args.task, f'seed{args.seed}')


def gen_eval(task_args):
    if args.task == 'register':
        return TASKS['register'](n=args.nprobe, T=task_args['T'], V=task_args['V'], seed=999)
    return TASKS['gridworld'](n=args.nprobe, T=task_args['T'], k=task_args['k'], seed=999)


def main():
    files = glob.glob(os.path.join(CKDIR, 'ckpt_step*.pt'))
    steps = sorted(int(os.path.basename(f)[len('ckpt_step'):-3]) for f in files)
    _, ck0 = load_checkpoint(os.path.join(CKDIR, f'ckpt_step{steps[0]}.pt'))
    data = gen_eval(ck0['task_args'])
    L = ck0['config']['n_layer']
    names = list(data['states'].keys())
    print(f"{args.task} seed{args.seed}: {len(steps)} ckpts | estados {names} | {L} capas", flush=True)

    rows = []
    for s in steps:
        model, ck = load_checkpoint(os.path.join(CKDIR, f'ckpt_step{s}.pt'), device=args.device)
        pa = probe_all(model, data, device=args.device)
        row = {'step': s, 'behav_acc': ck['behav_acc']}
        for name in names:
            for l in range(L):
                row[f'probe_{name}_L{l}'] = pa[name][l]
            row[f'probe_{name}_shuffle'] = pa[name]['shuffle']
        rows.append(row)
        best = max(pa[names[0]][l] for l in range(L))
        print(f"  step {s:4d} | behav {ck['behav_acc']:.3f} | best probe({names[0]})={best:.3f} "
              f"| shuffle={row[f'probe_{names[0]}_shuffle']:.3f}", flush=True)

    out = os.path.join(CKDIR, 'develop.csv')
    cols = list(rows[0].keys())
    with open(out, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print(f"CSV -> {out}")


if __name__ == '__main__':
    main()

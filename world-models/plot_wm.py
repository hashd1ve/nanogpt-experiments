"""
La pelicula del nacimiento (una tarea, una semilla): behav_acc + sonda-verdad por capa vs paso.
Marca el NACIMIENTO de la representacion (mejor sonda cruza 0.5) y el salto conductual.
Paleta Okabe-Ito. Abrir el PNG con `open`.

Uso:  python -u experiments/worldmodels/plot_wm.py --task register --seed 0
"""
import os, sys, csv, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OI = ['#0072B2', '#D55E00', '#009E73', '#E69F00']
INK, BG = '#2b2b2b', '#fcfcfb'

ap = argparse.ArgumentParser()
ap.add_argument('--task', type=str, default='register')
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

CKDIR = os.path.join(HERE, 'logs', args.task, f'seed{args.seed}')


def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    d = {k: [float(r[k]) for r in rows] for k in rows[0]}
    return d


def cross(steps, series):
    """paso donde la curva cruza el punto medio de su propio rango (nacimiento robusto)."""
    lo, hi = min(series), max(series)
    thr = lo + 0.5 * (hi - lo)
    for s, v in zip(steps, series):
        if v >= thr:
            return s
    return None


def main():
    d = load_csv(os.path.join(CKDIR, 'develop.csv'))
    steps = d['step']
    xs = [max(s, 1) for s in steps]
    states = sorted({k.split('_')[1] for k in d if k.startswith('probe_') and 'shuffle' not in k})
    L = len({k for k in d if k.startswith(f'probe_{states[0]}_L')})

    def best_probe(name):
        return [max(d[f'probe_{name}_L{l}'][i] for l in range(L)) for i in range(len(steps))]

    bp = best_probe(states[0])
    behav = d.get('legality', d['behav_acc'])          # gridworld -> legalidad
    behav_label = 'legalidad' if 'legality' in d else 'behav_acc'
    birth = cross(steps, bp)
    behav_birth = cross(steps, behav)

    plt.rcParams.update({'axes.facecolor': BG, 'figure.facecolor': BG, 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                         'axes.edgecolor': INK})
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Nacimiento de un modelo del mundo — {args.task} (semilla {args.seed})",
                 fontsize=13, color=INK)

    def marks(a):
        if birth:
            a.axvline(max(birth, 1), ls=':', lw=1.5, color=OI[2], label=f'nace rep. ({birth})')
        if behav_birth:
            a.axvline(max(behav_birth, 1), ls='--', lw=1, color='#888', label=f'salto conducta ({behav_birth})')

    # (a) conducta vs representacion
    a = ax[0]
    a.plot(xs, behav, color=INK, lw=2, label=f'{behav_label} (conducta)')
    a.plot(xs, bp, color=OI[2], lw=2, label='mejor sonda-verdad')
    a.plot(xs, d[f'probe_{states[0]}_shuffle'], color=OI[1], lw=1, ls=':', label='sonda barajada (nulo)')
    a.set_xscale('log'); a.set_ylim(-0.02, 1.02); a.set_xlabel('step'); a.set_ylabel('accuracy')
    a.set_title('(a) ¿la representacion precede a la conducta?')
    marks(a); a.legend(fontsize=8, loc='center right')

    # (b) sonda por capa
    a = ax[1]
    for st in states:
        for l in range(L):
            a.plot(xs, d[f'probe_{st}_L{l}'], color=OI[l % 4], lw=2,
                   label=f'{st} capa {l}' if len(states) > 1 else f'capa {l}')
    a.set_xscale('log'); a.set_ylim(-0.02, 1.02); a.set_xlabel('step'); a.set_ylabel('probe acc')
    a.set_title('(b) sonda-verdad por capa'); marks(a); a.legend(fontsize=8, loc='center right')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(CKDIR, 'film.png')
    fig.savefig(out, dpi=130)
    print(f"PNG -> {out}")
    print(f"nace representacion (sonda>0.5) = step {birth} | salto conducta (behav>0.5) = step {behav_birth}")


if __name__ == '__main__':
    main()

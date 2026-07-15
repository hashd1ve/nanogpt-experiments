"""
Universalidad de la DINAMICA: superpone las curvas de nacimiento de todas las semillas y
muestra la distribucion de pasos-de-nacimiento (conducta y representacion). Lee tambien
universality.csv (CKA) si existe. Paleta Okabe-Ito. Abrir el PNG con `open`.

Uso:  python -u experiments/worldmodels/plot_ensemble.py --task register --seeds 0,1,2,3,4
"""
import os, sys, csv, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OI = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9']
INK, BG = '#2b2b2b', '#fcfcfb'

ap = argparse.ArgumentParser()
ap.add_argument('--task', type=str, default='register')
ap.add_argument('--seeds', type=str, default='0,1,2,3,4')
args = ap.parse_args()
seeds = [int(s) for s in args.seeds.split(',')]


def load_csv(p):
    with open(p) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def best_probe(d):
    states = sorted({k.split('_')[1] for k in d if k.startswith('probe_') and 'shuffle' not in k})
    L = len({k for k in d if k.startswith(f'probe_{states[0]}_L')})
    n = len(d['step'])
    # world-model probe = media sobre estados de (max sobre capas)
    out = []
    for i in range(n):
        per_state = [max(d[f'probe_{st}_L{l}'][i] for l in range(L)) for st in states]
        out.append(sum(per_state) / len(per_state))
    return out


def cross(steps, series):
    lo, hi = min(series), max(series)
    thr = lo + 0.5 * (hi - lo)
    for s, v in zip(steps, series):
        if v >= thr:
            return s
    return None


def main():
    data = {}
    for s in seeds:
        p = os.path.join(HERE, 'logs', args.task, f'seed{s}', 'develop.csv')
        if os.path.exists(p):
            data[s] = load_csv(p)
    steps = data[seeds[0]]['step']
    xs = [max(x, 1) for x in steps]

    plt.rcParams.update({'axes.facecolor': BG, 'figure.facecolor': BG, 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                         'axes.edgecolor': INK})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(f"Universalidad del nacimiento — {args.task} ({len(data)} semillas)", fontsize=13, color=INK)

    b_births, p_births = [], []
    for i, (s, d) in enumerate(data.items()):
        c = OI[i % len(OI)]
        behav = d.get('legality', d['behav_acc'])       # gridworld -> legalidad
        ax[0].plot(xs, behav, color=c, lw=1.5, alpha=0.8, label=f'seed{s}')
        bp = best_probe(d)
        ax[1].plot(xs, bp, color=c, lw=1.5, alpha=0.8, label=f'seed{s}')
        bb = cross(steps, behav); pb = cross(steps, bp)
        if bb: b_births.append(bb)
        if pb: p_births.append(pb)
    blab = 'legalidad' if 'legality' in data[list(data)[0]] else 'behav_acc'
    for a, t in ((ax[0], f'(a) conducta ({blab})'), (ax[1], '(b) sonda-verdad (mejor capa)')):
        a.set_xscale('log'); a.set_ylim(-0.02, 1.02); a.set_xlabel('step'); a.set_title(t)
        a.legend(fontsize=7)

    # (c) distribucion de pasos-de-nacimiento
    a = ax[2]
    a.scatter(p_births, [1]*len(p_births), color=OI[2], s=60, label='nace representacion', zorder=3)
    a.scatter(b_births, [0]*len(b_births), color=INK, s=60, label='salto conducta', zorder=3)
    a.set_yticks([0, 1]); a.set_yticklabels(['conducta', 'represent.'])
    a.set_xscale('log'); a.set_xlabel('step de nacimiento'); a.set_title('(c) ¿universal en el tiempo?')
    a.legend(fontsize=8)
    # CKA si existe
    ckap = os.path.join(HERE, 'logs', args.task, 'universality.csv')
    if os.path.exists(ckap):
        with open(ckap) as f:
            rr = list(csv.DictReader(f))
        txt = ' | '.join(f"L{r['layer']}: CKA={float(r['cka_trained']):.2f}" for r in rr)
        ctl = ' | '.join(f"L{r['layer']}:{float(r['cka_vs_untrained']):.2f}" for r in rr)
        a.text(0.02, -0.32, f"CKA entre semillas: {txt}\n(control vs sin-entrenar: {ctl})",
               transform=a.transAxes, fontsize=7.5, color=INK)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = os.path.join(HERE, 'logs', args.task, 'ensemble.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f"PNG -> {out}")
    if b_births:
        print(f"conducta nace en pasos {sorted(b_births)} | representacion en {sorted(p_births)}")


if __name__ == '__main__':
    main()

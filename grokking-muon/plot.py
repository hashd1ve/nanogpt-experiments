"""
Flexible plotter for grokking curves. Reads results/curves/<tag>.csv.

Columns: step, train_acc, val_acc, train_loss, val_loss, margin, cum_travel, rel_travel,
         eff_rank, fro, spec, upd_cos

Examples:
  # val accuracy vs steps (log-x) for several arms
  python plot.py --tags A B C D --y val_acc --x step --logx --out figures/g1_val_step.png
  # THE fair comparison: val accuracy vs distance travelled by the weights
  python plot.py --tags A B C D --y val_acc --x cum_travel --out figures/g2_val_travel.png
  # effective rank vs steps
  python plot.py --tags A B C D --y eff_rank --x step --logx --out figures/g3_effrank.png
"""
import os, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, SURFACE, GRID = '#2b2b2b', '#8a8a8a', '#fcfcfb', '#e6e6e6'
PAL = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9', '#000000', '#999999']
YLAB = {'val_acc': 'validation accuracy', 'train_acc': 'train accuracy',
        'eff_rank': 'effective rank / r_max', 'fro': 'Frobenius norm (mean)',
        'spec': 'spectral norm (mean)', 'margin': 'logit margin', 'val_loss': 'validation loss',
        'train_loss': 'train loss', 'rel_travel': 'relative travel  Sum||dW||/||W||'}
XLAB = {'step': 'optimization step', 'cum_travel': 'distance travelled by the weights  Sum||dW||_F',
        'rel_travel': 'cumulative relative travel'}


def load(tag):
    return np.genfromtxt(os.path.join(HERE, 'results', 'curves', f'{tag}.csv'), delimiter=',', names=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', default=None)
    ap.add_argument('--y', default='val_acc')
    ap.add_argument('--x', default='step')
    ap.add_argument('--logx', action='store_true')
    ap.add_argument('--trainval', action='store_true', help='train (dashed) + val (solid) per tag')
    ap.add_argument('--title', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    labels = a.labels or a.tags
    fig, ax = plt.subplots(figsize=(8.6, 5.3), dpi=140)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    if a.y in ('val_acc', 'train_acc'):
        ax.axhline(1.0, color=GRID, lw=1, zorder=1)

    for i, (tag, lab) in enumerate(zip(a.tags, labels)):
        d = load(tag)
        x = d[a.x].copy()
        if a.logx:
            x = np.clip(x, 1, None)
        col = PAL[i % len(PAL)]
        if a.trainval:
            ax.plot(x, d['train_acc'], color=col, lw=1.5, ls=(0, (4, 2)), zorder=2, label=f'{lab} - train')
            ax.plot(x, d['val_acc'], color=col, lw=2.4, zorder=3, label=f'{lab} - val')
        else:
            ax.plot(x, d[a.y], color=col, lw=2.3, zorder=3, label=lab)

    if a.logx:
        ax.set_xscale('log')
    ax.set_xlabel(XLAB.get(a.x, a.x), color=INK)
    ax.set_ylabel(YLAB.get(a.y, a.y), color=INK)
    if a.y in ('val_acc', 'train_acc'):
        ax.set_ylim(-0.02, 1.05)
    ax.set_title(a.title or f'{YLAB.get(a.y, a.y)} vs {XLAB.get(a.x, a.x)}',
                 color=INK, fontsize=12.5, fontweight='bold', pad=12)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#cccccc')
    ax.grid(True, which='both', color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc='best')
    fig.tight_layout(); fig.savefig(a.out, facecolor=SURFACE)
    print(f'saved -> {a.out}')


if __name__ == '__main__':
    main()

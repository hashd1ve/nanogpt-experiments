"""
Phase diagram (Grokking x Muon): grok step and final effective rank vs weight decay,
for AdamW and Muon. Reads the metrics straight from results/curves/<tag>.csv (self-contained).

python phase.py --out figures/g4_phase.png
"""
import os, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, SURFACE, GRID = '#2b2b2b', '#8a8a8a', '#fcfcfb', '#e6e6e6'
C_ADAMW, C_MUON = '#D55E00', '#0072B2'
GROK_THR, STAY_THR, BUDGET = 0.95, 0.90, 3000

# (wd, tag) per optimizer. wd=0 is placed on the left as a category.
ADAMW = [(0, 'A_adamw_nowd'), (0.03, 'aw_wd.03'), (0.1, 'aw_wd.1'),
         (0.3, 'aw_wd.3'), (1.0, 'B_adamw_wd'), (3.0, 'aw_wd3')]
MUON = [(0, 'C_muon_nowd'), (0.1, 'mu_wd.1'), (1.0, 'D_muon_wd'), (3.0, 'mu_wd3')]


def read(tag):
    """Return (grok_step or None, final_eff_rank) computed from the curve CSV."""
    d = np.genfromtxt(os.path.join(HERE, 'results', 'curves', f'{tag}.csv'), delimiter=',', names=True)
    st, va, eff = d['step'], d['val_acc'], d['eff_rank']
    grok = None
    for i in range(len(st)):
        if va[i] >= GROK_THR and (va[i:] >= STAY_THR).all():
            grok = int(st[i]); break
    return grok, float(eff[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

    wds = [0, 0.03, 0.1, 0.3, 1.0, 3.0]
    xpos = {w: i for i, w in enumerate(wds)}
    xlabels = ['0', '0.03', '0.1', '0.3', '1', '3']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=140)
    for ax in (ax1, ax2):
        fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
        ax.set_xticks(range(len(wds))); ax.set_xticklabels(xlabels)
        ax.set_xlabel('weight decay lambda (hidden matrices + embeddings)', color=INK)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color('#cccccc')
        ax.grid(True, color=GRID, lw=0.7); ax.tick_params(colors=MUTED)

    # --- panel 1: grok step vs wd ---
    for data, col, lab in [(ADAMW, C_ADAMW, 'AdamW'), (MUON, C_MUON, 'Muon')]:
        xs, ys, xno = [], [], []
        for w, tag in data:
            grok, _ = read(tag)
            (xno.append(xpos[w]) if grok is None else (xs.append(xpos[w]) or ys.append(grok)))
        if xs:
            ax1.plot(xs, ys, color=col, lw=2.4, marker='o', ms=8, zorder=3, label=lab)
        if xno:
            ax1.scatter(xno, [BUDGET] * len(xno), color=col, marker='x', s=90, zorder=4,
                        label=f'{lab}: never groks')
    ax1.axhline(BUDGET, color=MUTED, ls=':', lw=1)
    ax1.text(0.05, BUDGET, ' budget (did not grok)', color=MUTED, fontsize=8, va='bottom')
    ax1.set_ylabel('grokking step (lower = earlier)', color=INK)
    ax1.set_ylim(0, BUDGET * 1.08)
    ax1.set_title('Muon groks at EVERY wd (incl. 0); AdamW only with strong wd',
                  color=INK, fontsize=11.5, fontweight='bold', pad=10)
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK, loc='center right')

    # --- panel 2: final effective rank vs wd ---
    for data, col, lab in [(ADAMW, C_ADAMW, 'AdamW'), (MUON, C_MUON, 'Muon')]:
        xs = [xpos[w] for w, _ in data]
        ys = [read(tag)[1] for _, tag in data]
        ax2.plot(xs, ys, color=col, lw=2.4, marker='o', ms=8, zorder=3, label=lab)
    ax2.set_ylabel('final effective rank / r_max', color=INK)
    ax2.set_title('AdamW collapses the rank under wd; Muon keeps it high',
                  color=INK, fontsize=11.5, fontweight='bold', pad=10)
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK, loc='lower left')

    fig.tight_layout(); fig.savefig(a.out, facecolor=SURFACE)
    print(f'saved -> {a.out}')


if __name__ == '__main__':
    main()

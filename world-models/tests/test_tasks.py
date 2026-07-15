"""Sanity de la verdad-terreno. `python experiments/worldmodels/tests/test_tasks.py`."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tasks import register_task, gridworld_task, _MOVES


def check_register():
    d = register_task(n=64, T=96, V=8, seed=1)
    toks, (mem, V) = d['tokens'], d['states']['mem']
    R, VAL0 = d['meta']['R'], d['meta']['VAL0']
    ok = True
    for i in range(toks.shape[0]):
        m = None
        for t in range(toks.shape[1]):
            tok = int(toks[i, t])
            if tok < V:                       # Wv
                m = tok
            elif tok == R:                    # R: mem no cambia
                pass
            else:                             # Val_v: debe igualar mem
                if tok - VAL0 != m:
                    ok = False
            if int(mem[i, t]) != m:           # estado devuelto == recomputado
                ok = False
            # behav: tras R, el siguiente token es Val_mem
            if tok == R and t + 1 < toks.shape[1]:
                if int(toks[i, t+1]) != VAL0 + m:
                    ok = False
    # ¿hay señal aprendible? proporcion de posiciones behav
    frac = d['behav'].float().mean().item()
    print(f"register: verdad-terreno consistente={ok} | frac behav={frac:.2f} | vocab={d['vocab_size']}")
    return ok and frac > 0.05


def check_gridworld():
    d = gridworld_task(n=64, T=96, k=5, seed=1)
    toks, (xs, k) = d['tokens'], d['states']['x']
    ys = d['states']['y'][0]
    ok = True
    for i in range(toks.shape[0]):
        x, y = k // 2, k // 2
        for t in range(toks.shape[1]):
            dx, dy = _MOVES[int(toks[i, t])]
            if not (0 <= x + dx < k and 0 <= y + dy < k):
                ok = False                    # movimiento ilegal generado
            x, y = x + dx, y + dy
            if int(xs[i, t]) != x or int(ys[i, t]) != y:
                ok = False
            if not (0 <= x < k and 0 <= y < k):
                ok = False
    print(f"gridworld: verdad-terreno consistente={ok} | vocab={d['vocab_size']} | k={k}")
    return ok


if __name__ == '__main__':
    a, b = check_register(), check_gridworld()
    print("==>", "TAREAS OK" if (a and b) else "FALLO EN VERDAD-TERRENO")
    sys.exit(0 if (a and b) else 1)

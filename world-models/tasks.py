"""
Tareas sinteticas con ESTADO LATENTE VERDADERO por posicion. Esa es la clave: como
controlamos el proceso generador, la sonda es VERIFICABLE (no sugerente).

Cada tarea genera:
  tokens : LongTensor [n, T]           secuencias de entrada (next-token LM)
  states : dict nombre -> (LongTensor [n, T], n_clases)   estado latente verdadero por posicion
  behav  : BoolTensor [n, T]           posiciones cuya PREDICCION (t->t+1) exige el estado
  vocab_size, meta

Mundo A (registro/flip-flop): estado = valor en memoria.
Mundo B (gridworld): estado = (x, y) del agente.
"""
import torch


# ============================ Mundo A: registro ============================
def register_task(n=2048, T=128, V=8, p_read=0.35, seed=0):
    """
    Tokens: Wv (escribe v) para v in 0..V-1  |  R (lee)  |  Vv (valor v, aparece tras R).
    vocab = [W0..W{V-1}, R, Val0..Val{V-1}]  -> tamaño 2V+1.
    Tras 'R' el token siguiente es el valor en memoria => predecirlo exige el registro.
    state 'mem' = valor en memoria en cada posicion (0..V-1). behav = (token == R).
    """
    W0, R, VAL0 = 0, V, V + 1          # offsets: Wv=v ; R=V ; Val_v = V+1+v
    vocab = 2 * V + 1
    g = torch.Generator().manual_seed(seed)
    toks = torch.zeros(n, T, dtype=torch.long)
    mem = torch.zeros(n, T, dtype=torch.long)
    behav = torch.zeros(n, T, dtype=torch.bool)
    for i in range(n):
        seq, st = [], []
        m = int(torch.randint(V, (1,), generator=g))   # memoria inicial aleatoria
        seq.append(W0 + m); st.append(m)               # arranca escribiendo m
        while len(seq) < T:
            if float(torch.rand(1, generator=g)) < p_read:
                seq.append(R); st.append(m)            # token R (mem no cambia)
                if len(seq) < T:
                    seq.append(VAL0 + m); st.append(m) # el valor leido
            else:
                m = int(torch.randint(V, (1,), generator=g))
                seq.append(W0 + m); st.append(m)       # escribe nuevo valor
        t = torch.tensor(seq[:T]); s = torch.tensor(st[:T])
        toks[i], mem[i] = t, s
        behav[i] = (t == R)
    return {'tokens': toks, 'states': {'mem': (mem, V)}, 'behav': behav,
            'vocab_size': vocab, 'meta': {'V': V, 'R': R, 'VAL0': VAL0, 'name': 'register'}}


# ============================ Mundo B: gridworld ============================
_MOVES = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0)}  # N,S,E,O  (dx,dy)


def gridworld_task(n=2048, T=128, k=5, seed=0):
    """
    Tokens = movimientos {N=0,S=1,E=2,O=3} (vocab 4). Solo se generan movimientos LEGALES
    (no salir de la rejilla k x k). Estado latente = (x,y). Predecir el proximo movimiento
    legal exige conocer la posicion. behav = todas las posiciones (cada next-move debe ser legal).
    """
    vocab = 4
    g = torch.Generator().manual_seed(seed)
    toks = torch.zeros(n, T, dtype=torch.long)
    xs = torch.zeros(n, T, dtype=torch.long)
    ys = torch.zeros(n, T, dtype=torch.long)
    for i in range(n):
        x, y = k // 2, k // 2
        seq, sx, sy = [], [], []
        for _ in range(T):
            legal = [mv for mv, (dx, dy) in _MOVES.items()
                     if 0 <= x + dx < k and 0 <= y + dy < k]
            mv = legal[int(torch.randint(len(legal), (1,), generator=g))]
            dx, dy = _MOVES[mv]
            x, y = x + dx, y + dy
            seq.append(mv); sx.append(x); sy.append(y)
        toks[i] = torch.tensor(seq)
        xs[i] = torch.tensor(sx); ys[i] = torch.tensor(sy)
    behav = torch.ones(n, T, dtype=torch.bool)
    return {'tokens': toks, 'states': {'x': (xs, k), 'y': (ys, k)}, 'behav': behav,
            'vocab_size': vocab, 'meta': {'k': k, 'name': 'gridworld'}}


TASKS = {'register': register_task, 'gridworld': gridworld_task}

# Diseño — Ver nacer un modelo del mundo verificable en un transformer enano

**Fecha:** 2026-07-15 · **Tesis-madre:** interpretabilidad *verificable* = verdad-terreno
controlada + tiempo (checkpoints densos) + ensemble de semillas. Los tres superpoderes que
un enano en un portátil regala y un lab grande no puede permitirse baratos.

## La pregunta
¿Cuándo y cómo un transformer enano construye un **modelo del mundo** (representación interna
del estado latente que controlamos), es **causal**, y es **universal** entre semillas?

Linaje: Othello-GPT (Li et al. 2023 → Nanda, "emergent world representations"): un GPT construye
el estado del tablero, linealmente decodificable y causal — pero enseñado en el modelo YA
convergido. Hueco novedoso y barato: la **dinámica** (¿en qué paso nace? ¿transición de fase?) y
la **universalidad** (¿misma geometría entre semillas?).

## Dos mundos (dificultad creciente)
- **A · Registro (flip-flop):** tokens de escritura `Wv`, lectura `R` (seguida del valor en
  memoria). Estado latente = valor en memoria ∈ {0..V-1}. Predecir el valor tras `R` EXIGE el
  registro. (cf. flip-flop LM, Liu et al. 2023.)
- **B · Gridworld:** agente en rejilla k×k, movimientos legales {N,S,E,O}. Estado latente =
  posición (x,y). Predecir solo movimientos legales EXIGE el mapa. (Othello-lite.)

## Instrumento (verificable, superficie inmutable)
- **Sonda-verdad** `probe_acc(capa, paso)`: sonda LINEAL entrenada a decodificar el estado latente
  *verdadero* desde el residual. Accuracy held-out. Controles: etiqueta barajada, modelo sin
  entrenar (el nulo).
- **Conducta** `behav_acc`: en A, acierto del valor tras `R`; en B, fracción de top-1 que es legal.
- **Nacimiento**: paso donde `probe_acc` cruza umbral / máxima pendiente; cruzar con el codo de loss.
- **Causalidad**: intervenir el residual con la dirección-sonda de un estado alterno → ¿cambia la
  conducta hacia ese estado? Éxito vs dirección aleatoria de igual norma.
- **Universalidad**: N semillas; transferencia cruzada de la sonda (probe de semilla i aplicada a
  j) y similitud representacional (CKA/Procrustes). Escalar: accuracy de transferencia cruzada.

## Superpoderes, todos a la vez
TIEMPO (checkpoints densos → nacimiento) · VERDAD (sonda contra estado real → verificable, no
sugerente) · SEMILLAS (ensemble → universalidad).

## Fases
0. `tasks.py` — generadores con estado verdadero por posición (+ sanity de determinismo).
1. `train_wm.py` — entrenar enano (reusa `model.py`+`muon.py`), checkpoints densos, por semilla.
2. `probe.py` — sonda lineal por capa/paso + controles + `behav_acc`.
3. `develop_wm.py` + `plot_wm.py` — la película del nacimiento (A y B).
4. `causal.py` — test de intervención.
5. `universality.py` — transferencia cruzada + CKA sobre el ensemble.

## Método (heredado)
Una variable/experimento; anti-autoengaño (control barajado, dirección aleatoria, sin-entrenar);
anotar negativos; abrir PNG con `open`; paleta Okabe-Ito; todo lo pesado (ensembles) en paralelo.

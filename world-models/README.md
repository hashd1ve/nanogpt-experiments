# World-Model Birth — a verifiable world model, and how its *shape of emergence* depends on the world

**TL;DR.** We train tiny transformers on two synthetic tasks whose *true latent state we control*, so a
linear probe of that state is **verifiable, not suggestive**. In both, an internal world model forms and
becomes linearly decodable to **~0.99** (shuffled-label control at chance). The interesting part is the
**shape of its birth**: for a **discrete/algorithmic** latent (a written register), the world model
appears in a **sharp phase transition** (~step 50–75), representation and behavior born together; for a
**continuous/additive** latent (an agent's (x,y) position on a grid), it emerges **gradually** (probe
0.37→0.99 over hundreds of steps). The representation is **causal** (steering the probe direction rewrites
the model's behavior: 44% vs 4% for a matched-norm random direction) and largely **universal across seeds**
(CKA 0.7–0.9 between seeds, ~0.01–0.12 vs an untrained net; birth timing near-identical across 5 seeds).

> Tiny-scale, controlled *nano* study (0.6M params, 5 seeds/task). It is a clean isolation of one variable
> — the geometry of the task's latent state — not a claim about frontier models. The two individual
> halves ("discrete → sharp", "continuous → gradual") each have prior art; our contribution is the
> **paired, controlled, ground-truth-probed, seeded contrast**. See [Prior art & novelty](#prior-art--novelty)
> and [Limitations](#limitations) — both stated up front and honestly.

---

## The question

Language-model interpretability usually has **no ground truth**: on natural language you never know the
"correct" latent, so probe results are *suggestive*, not verifiable. The one setting where you *can* verify
is a **synthetic task whose generative process you own**. There you get three things a frontier lab can't
cheaply have: **time** (dense checkpoints → watch structure form), **ground truth** (probe against the real
latent), and **seeds** (an ensemble → measure universality).

Prior work (Othello-GPT; Nanda's linear probes; Chess-GPT) established that a transformer *builds* a
probeable world model — on **converged** models. We ask the developmental question: **when and how does the
world model get born, and does the *shape* of its birth depend on the structure of the world?**

## Setup

- **Two worlds** (`tasks.py`), each with the true latent state at every position (verified, `tests/`):
  - **A · register (flip-flop).** Tokens write/read a value; latent state = **value in memory** ∈ {0..V−1}.
    Predicting the value after a `read` *requires* carrying the register. Discrete/algorithmic latent.
  - **B · gridworld.** An agent takes legal moves {N,S,E,W} on a 5×5 grid; latent state = **(x, y)**.
    Predicting only-legal moves *requires* tracking position. Continuous/additive latent.
- **Model.** The same nanoGPT used across this repo (`model.py`), tiny (3 layers, `n_embd=128`, ~0.6M),
  trained with the lab's Muon (`muon.py`). Dense log-spaced checkpoints; 5 seeds/task.
- **Instrument** (`probe.py`, immutable). A **linear probe** of the true latent, trained *only on
  carry positions* — positions where the state must be carried and is **not** in the current token — so we
  measure the world model, not trivial token leakage. Controls: **shuffled labels** (null), **untrained
  model** (null), split by sequence (no leakage).
- **Behavior.** A `behav_acc` (register: value after `read`) and, for gridworld, **legality** (probability
  mass on legal moves — the right compass, since the next move is random-among-legal so exact-match caps at
  ~1/#legal). Causal test in `causal.py`; universality (CKA) in `universality.py`.

## Results

**1 · A verifiable world model is born.** The probe of the true latent reaches **0.99** (register memory;
grid x/y), with the shuffled-label control pinned at chance (~0.13 for V=8, ~0.20 for k=5). Not suggestive
— verified against ground truth.

**2 · The structure of the world dictates the shape of the birth** (the headline). Same architecture, same
recipe, qualitatively different dynamics:

| world | latent | birth of the world model | behavior |
|---|---|---|---|
| **register** | discrete / algorithmic | **sharp phase transition** (~step 50–75); representation and behavior jump together | `behav_acc` 0 → 1.0 |
| **gridworld** | continuous / additive (position) | **gradual** (probe 0.37 → 0.99 over hundreds of steps) | legality 0.82 → 0.999 |

→ See `figures/register_seed0_film.png` and `figures/gridworld_seed0_film.png`.

**3 · The representation is causal, not correlational.** Steering the residual along the probe direction
`(m → m′)` rewrites the register and flips the model's read-out to the rewritten value: **44% (layer 1) vs
4%** for a matched-norm random direction (~11×); 55% vs 7% at layer 2; weak at layer 0 — the memory
consolidates in the middle layer.

**4 · Universal — geometry *and* timing.** Across 5 independent seeds, **CKA = 0.70/0.81/0.91** (register,
layers 0/1/2) and **0.87/0.87/0.79** (gridworld); vs an untrained net, **~0.01–0.12** (near zero). The
**birth step is near-identical across seeds**. Different seeds converge to essentially the same world-model
geometry (a quantified, verifiable instance of the universality hypothesis) — see `figures/*_ensemble.png`.

## Prior art & novelty

We were adversarial about novelty (a literature sweep is archived in
[`results/prior_art_raw_evidence.txt`](./results/prior_art_raw_evidence.txt)). Honest positioning:

- **Both halves are individually established.** *Discrete → sharp:* the Quantization Model of neural scaling
  (Michaud et al. 2023), grokking progress measures (Nanda et al. 2023), induction-head phase change
  (Olsson et al. 2022). *Continuous → gradual:* "Convergent World Representations and Divergent Tasks"
  (2026) shows a continuous/spatial latent emerging with a loss plateau and cross-seed CKA.
- **The apparatus is now standard.** Ground-truth linear probes + a seed ensemble + developmental tracking
  is accepted practice (e.g. TRACE 2025). So the novelty is **not** the method.
- **What appears to be the residual, unfilled gap:** the *paired*, controlled contrast — discrete-algorithmic
  vs continuous-additive latent as the **single manipulated variable**, measured with ground-truth probes
  across seeds, with a causal check — tying the **shape** of emergence to the **geometry** of the latent.
  Nearby work circles it (Quiet Feature Learning 2025; "structural information" 2026; a June-2026 lexical-vs-
  compositional probing note) but none isolates *this* variable this way.
- **Honest caveat on universality:** for group-theoretic tasks, the developmental *order* across seeds has
  been reported as arbitrary (Chughtai et al. 2023). Our "universal timing" is in a different regime and
  must be scoped carefully, not oversold.

**Status: not fundamental** (the halves are known); **plausibly a workshop-scale contribution** (mechanistic
interpretability / world-models workshop) *if* extended — see below.

## Limitations & what would make it a real contribution

- **Two tasks = two points, not a law.** To claim "structure determines emergence-shape" you need a **family
  of tasks** interpolating discrete↔continuous, a **sharpness metric** (transition width in log-steps)
  controlled for **probe capacity** and for the **metric-artifact** critique (Schaeffer et al. 2023 — use
  the continuous probe accuracy, not a thresholded metric), and ideally a **mechanism** predicting the shape.
- **Gridworld's behavioral metric** is legality (mass on legal moves), because the next move is stochastic
  among legal moves; exact-match is not meaningful here.
- **Causal steering** is strongest at middle/late layers; a layer-0 intervention is weak.
- Tiny scale, one architecture, one optimizer. Whether the discrete/continuous dichotomy survives scale is
  untested.

## Files

`tasks.py` (worlds + ground truth) · `train_wm.py` (train + dense checkpoints) · `probe.py` (linear probe
of the true latent) · `causal.py` (intervention) · `universality.py` (CKA) · `develop_wm.py` /
`plot_wm.py` / `plot_ensemble.py` (the films) · `legality_grid.py` · `wm_common.py` · `model.py`, `muon.py`
(vendored). Raw curves in `results/curves/`, figures in `figures/`.

## References

Li et al. 2023, *Emergent World Representations* (Othello-GPT) · Nanda, *Othello-GPT has a linear emergent
world representation* · Karvonen 2024, *Chess-GPT* · Liu et al. 2023, *Flip-Flop Language Modeling* ·
Michaud et al. 2023, *The Quantization Model of Neural Scaling* · Nanda et al. 2023, *Progress Measures for
Grokking* · Hoogland et al. 2024, *The Developmental Landscape of In-Context Learning* (SLT/LLC) · Kornblith
et al. 2019 (CKA) · Chughtai, Chan & Nanda 2023, *A Toy Model of Universality*. Full swept evidence with
URLs in `results/prior_art_raw_evidence.txt`.

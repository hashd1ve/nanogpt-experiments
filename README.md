# nanogpt-experiments

Small, self-contained experiments on a tiny [nanoGPT](https://github.com/karpathy/nanoGPT), aimed at
**understanding how LLMs actually learn** — training dynamics, optimizers, generalization, and the
structure of the weights — at a scale where everything is cheap, reproducible, and fully instrumented.

**Each folder is one experiment.** It ships with its own `README`, the code, the charts, the raw
result curves, and a note on what is and isn't established.

### Method

Every experiment follows the same discipline:

- **One variable at a time**, measured with a single comparable metric.
- **Immutable harness**: the task, data split, evaluation and metric are ground truth and never edited
  to make a number look better.
- **Controls against self-deception** (matched-norm baselines, fairness normalizations, multiple seeds).
- **Honest negatives**: results that *didn't* work are recorded, and limitations are stated up front.

Runs on Apple Silicon (MPS), CUDA, or CPU.

### Experiments

| experiment | one-liner |
|---|---|
| [**grokking-muon**](./grokking-muon) | Muon groks modular addition *without* weight decay — and via a different (distributed-Fourier, high-rank) mechanism than AdamW+wd (sparse-Fourier, low-rank). |
| [**world-models**](./world-models) | A transformer's world model is *born* in a sharp phase transition for a **discrete** latent (a register) but **gradually** for a **continuous** one (grid position) — and the representation is causal and universal across seeds, verified against ground-truth linear probes. |

*(more to come — this is a growing meta-repo.)*

### Credits

Built on [nanoGPT](https://github.com/karpathy/nanoGPT) (Andrej Karpathy, MIT) and
[Muon](https://kellerjordan.github.io/posts/muon/) (Keller Jordan). See each experiment's README for
task-specific references.

### License & citation

Original content and code: **CC BY 4.0** — free to use and adapt, **as long as you give
credit** (see [LICENSE](./LICENSE)). Please cite this repository if you use the findings,
figures, or code — a ready-made citation is in [CITATION.cff](./CITATION.cff) (GitHub's
"Cite this repository" button). Vendored nanoGPT and Muon remain MIT — see
[THIRD_PARTY.md](./THIRD_PARTY.md).

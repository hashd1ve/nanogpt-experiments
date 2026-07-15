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

*(more to come — this is a growing meta-repo.)*

### Credits

Built on [nanoGPT](https://github.com/karpathy/nanoGPT) (Andrej Karpathy, MIT) and
[Muon](https://kellerjordan.github.io/posts/muon/) (Keller Jordan). See each experiment's README for
task-specific references.

### License

MIT — see [LICENSE](./LICENSE).

# Collective learning

Minimal experiments on coupled routing in mixture-of-experts Transformers.

The current model replaces independent expert routing with a learned interaction term

```text
p0 = softmax(Q h)
p  = softmax(Q h + lambda C p0)
C  = Q M Q^T
```

where `M` is learned jointly with the Transformer. The first experiment compares `lambda = 0` and `lambda > 0` with the same architecture and parameter count on WikiText-2.

See `draft/main.tex` for the model and experimental rationale.

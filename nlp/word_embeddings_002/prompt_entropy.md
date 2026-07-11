User: how would i measure entropy 

GitHub Copilot: Use Shannon entropy. For a discrete variable with class/token probabilities $p_1,\dots,p_n$:

$$
H(X) = -\sum_{i=1}^{n} p_i \log_b p_i
$$

- If $b=2$, entropy is in bits.
- If $b=e$, entropy is in nats.

Quick workflow:
1. Estimate probabilities from counts: $p_i = \frac{c_i}{\sum_j c_j}$.
2. Plug into the formula.
3. Interpret:
- $H=0$ means fully certain (all mass on one outcome).
- Max entropy is $\log_b n$ when all outcomes are equally likely.

Python example:
```python
import numpy as np

counts = np.array([50, 30, 20], dtype=float)
p = counts / counts.sum()
H_bits = -(p * np.log2(p)).sum()
print(H_bits)
```

For a classifier output (like softmax), use the predicted probability vector exactly the same way per sample:
$$
H(\hat{y}) = -\sum_k \hat{p}_k \log_2 \hat{p}_k
$$
Higher entropy = less confident prediction.

If you want, I can also show how to compute entropy across your notebook predictions and plot confidence distribution.

User: sure

GitHub Copilot: I’ll quickly inspect the current notebook structure to find your prediction variable names, then I’ll give you a drop-in cell that computes entropy and plots confidence.

Retrieving Notebook summary.

I found the latest notebook structure; next I’ll read the model-evaluation cells near your current section so the entropy code matches your exact variable names.

Read [](file:///Users/aricode0/devML/nlp/word_embeddings_002/word_embeddings_classification.ipynb#1330-1330), lines 1330 to 1455

Read [](file:///Users/aricode0/devML/nlp/word_embeddings_002/word_embeddings_classification.ipynb#1465-1465), lines 1465 to 1546

Read [](file:///Users/aricode0/devML/nlp/word_embeddings_002/word_embeddings_classification.ipynb#1568-1568), lines 1568 to 1732

I’ve identified the evaluation section and now I’ll inspect how each model payload is stored so entropy can be computed from test probabilities, not just sample sentences.

Searched for regex `threshold_sweep|best_threshold|classification_results` (`**/nlp/word_embeddings_002/word_embeddings_classification.ipynb`), 39 results

Searched for regex `y_prob|y_pred|predict\(|test_ds|eval_stats|probs` (`**/nlp/word_embeddings_002/word_embeddings_classification.ipynb`), 96 results

Perfect. Since your pipeline already stores test probabilities in each model result, run this new cell right under your Evaluation section:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def binary_entropy_bits(p_pos, eps=1e-12):
    # Clip to avoid log(0)
    p = np.clip(np.asarray(p_pos, dtype=float), eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

rows = []
entropy_by_model = {}

for name, payload in classification_results.items():
    y_prob = payload["eval"]["y_prob"]  # test-set P(positive)
    H = binary_entropy_bits(y_prob)     # per-sample entropy in bits, range [0, 1]
    entropy_by_model[name] = H

    rows.append({
        "model": name,
        "n_samples": len(H),
        "mean_entropy_bits": float(H.mean()),
        "median_entropy_bits": float(np.median(H)),
        "p90_entropy_bits": float(np.percentile(H, 90)),
        "low_conf_rate_H_gt_0.90": float(np.mean(H > 0.90)),  # very uncertain
        "high_conf_rate_H_lt_0.30": float(np.mean(H < 0.30)), # confident
    })

summary = pd.DataFrame(rows).sort_values("mean_entropy_bits")
display(summary)

# Plot entropy distributions
n = len(entropy_by_model)
fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=True)
if n == 1:
    axes = [axes]

for ax, (name, H) in zip(axes, entropy_by_model.items()):
    ax.hist(H, bins=30, alpha=0.8)
    ax.axvline(H.mean(), linestyle="--", linewidth=2, label=f"mean={H.mean():.3f}")
    ax.set_title(f"{name} entropy (bits)")
    ax.set_xlabel("Entropy H")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()

plt.suptitle("Prediction Uncertainty Distribution (Test Set)")
plt.tight_layout()
plt.show()
```

How to read it:
1. Entropy near 0 means confident predictions.
2. Entropy near 1 means uncertain predictions (binary case max is 1 bit).
3. Lower mean entropy generally means a sharper, more confident model.

If you want, I can give you one more cell to compare entropy against correctness, so you can see whether high-entropy samples are actually where errors happen.
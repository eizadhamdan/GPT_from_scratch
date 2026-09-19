# Bigram Model

This file documents the character-level bigram language model implemented in [bigram.py](bigram.py). It focuses on the model itself: how text is encoded, how the next-character distribution is learned, and how generated text is produced from the trained weights.

The implementation is intentionally small and readable. It demonstrates the core language-modeling workflow in a compact form: tokenize text, build a training set, compute a next-token loss, optimize the parameters, and generate new output autoregressively.

---

## Table of contents

1. [Overview](#overview)
2. [Quick start](#quick-start)
3. [Data pipeline](#data-pipeline)
4. [Model design](#model-design)
5. [Training loop](#training-loop)
6. [Generation](#generation)
7. [Evaluation](#evaluation)
8. [Tensor shape reference](#tensor-shape-reference)
9. [Limitations](#limitations)
10. [Summary](#summary)

---

## Overview

The model predicts the next character from the previous character and then uses that learned mapping to generate text one character at a time.

| Property | Value |
|---|---|
| Framework | PyTorch |
| Tokenization | Character-level |
| Model | Single embedding lookup table |
| Context used | One previous character |
| Loss | Cross-entropy |
| Optimizer | AdamW |
| Task | Autoregressive next-token prediction |

This is intentionally simple: it has no attention mechanism, no transformer blocks, and no long-range memory beyond the immediate previous character.

---

## Quick start

### Requirements

- Python
- PyTorch
- A text file such as `input.txt` in the working directory

### Run

```bash
python bigram.py
```

The script prints training and validation loss as it trains and then generates text from the learned model.

---

## Data pipeline

### Loading text

```python
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
```

The text is loaded into memory as a single string and converted into tokens.

### Tokenization

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(indices):
    return "".join([itos[i] for i in indices])
```

This is a character-level tokenizer. Each unique character becomes a token ID. The model then works with integer IDs instead of raw text.

### Train/validation split

```python
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

The data is split into a training set and a validation set, and the sequence order is preserved. This follows the standard workflow for language-model evaluation without shuffling the temporal structure.

### Batching

```python
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y
```

Each row in `x` is a short character sequence, and each row in `y` is the same sequence shifted by one position. This creates a supervised next-character prediction task.

---

## Model design

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
```

This model uses a single embedding table with shape `(vocab_size, vocab_size)`. For each character, it retrieves a row of logits that represents the distribution over possible next characters.

### Logits and probabilities

A logit is an unnormalized score. Softmax converts those scores into probabilities:

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

The objective is cross-entropy, which compares the predicted next-token distribution with the observed target token.

---

## Training loop

```python
model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

The loop follows the standard PyTorch pattern:

1. sample a mini-batch
2. compute predictions and the loss
3. clear gradients
4. perform backpropagation
5. update the parameters

### What the model learns

The embedding table is adjusted so that each character maps to the empirical distribution of characters that typically follow it. Over many updates, the table becomes a learned transition model over the vocabulary.

---

## Generation

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, loss = self(idx)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

Generation is autoregressive. The model predicts the next token, samples one value from the distribution, appends it to the current context, and repeats the process.

This is the same high-level pattern used in larger language models, but the bigram model only conditions on the immediately previous character.

---

## Evaluation

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

This helper measures the model loss on both the training and validation sets without building a graph for gradient computation. It gives a stable estimate of how well the model generalizes to held-out examples.

---

## Tensor shape reference

| Tensor | Shape | Meaning |
|---|---|---|
| `data` | `(N,)` | Encoded text as integer IDs |
| `train_data` / `val_data` | `(0.9N,)` / `(0.1N,)` | Training and validation splits |
| `ix` | `(B,)` | Random starting positions |
| `x`, `y` | `(B, T)` | Input window and shifted target window |
| `token_embedding_table.weight` | `(V, V)` | Learned transition table |
| `logits` | `(B, T, V)` | Raw scores for the next character |
| `logits` after flattening | `(B*T, V)` | Cross-entropy input |
| `targets` after flattening | `(B*T,)` | Correct target token IDs |
| `loss` | scalar | Mean cross-entropy over the batch |
| `idx_next` | `(B, 1)` | One sampled token per sequence |

Here, `B` is the batch size, `T` is the sequence length, and `V` is the vocabulary size.

---

## Limitations

The bigram model is intentionally limited, and that is the point of the exercise.

1. It only conditions on the immediately previous character.
2. It has no long-range memory.
3. It cannot capture broader syntax or semantics.
4. The parameter count grows with $V^2$, which becomes expensive for large vocabularies.
5. Generated text can look plausible at a local level while still lacking global coherence.

This is a compact baseline for next-character prediction, not a full-context language model.

---

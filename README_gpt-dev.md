# GPT from Scratch Notebook

This notebook is a compact walkthrough of a minimal GPT-style language model. It focuses on the core building blocks behind transformer-based language modeling: tokenization, embeddings, causal attention, and autoregressive generation.

The file [gpt-dev.ipynb](gpt-dev.ipynb) is designed as a step-by-step technical notebook. It begins with raw text, converts it into token IDs, and then builds up the sequence-modeling concepts needed for GPT-style architectures.

---

## Overview

The notebook covers the main stages of a small language model:

- raw text is converted into integer token IDs
- the sequence is split into training and validation examples
- context windows are created for next-token prediction
- a simple embedding model is used to map tokens to probabilities
- self-attention is introduced to allow context-aware aggregation
- generation is performed autoregressively one token at a time

The emphasis is on clarity and conceptual structure rather than scale or optimization.

---

## Notebook structure

### 1. Tokenization and vocabulary construction

The notebook reads text from disk and builds a vocabulary of unique characters.

```python
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```

It then creates the mappings between characters and integer IDs:

```python
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
```

This is the foundational step in language modeling: the model does not work with strings directly, only with integers that represent token IDs.

### 2. Encoding and decoding

```python
def encode(s):
    return [stoi[c] for c in s]


def decode(indices):
    return ''.join([itos[i] for i in indices])
```

This makes the notebook's data flow explicit: text is encoded before training and decoded again when generating output.

### 3. Sequence preparation

The encoded corpus is converted into a tensor and split into train and validation sections:

```python
data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

This keeps the sequence order intact and reflects standard practice for language-model evaluation.

### 4. Context windows and next-token prediction

The central objective is simple:

> predict the next token from the preceding context.

The notebook uses a fixed window size:

```python
block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]
```

This creates the basic supervised setup used by autoregressive language models.

### 5. Batching

The notebook creates minibatches of short token windows:

```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
```

This allows many training examples to be processed together while preserving sequence structure.

---

## Bigram baseline

The notebook begins with a minimal bigram model:

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

This model maps each token to a vector of logits over the vocabulary. Each position produces a distribution over the possible next token.

The key idea is that the model only conditions on the immediately previous token. This is intentionally simple and highlights the limitations of a unigram-like or bigram-like setup.

### Loss

The notebook uses cross-entropy:

```python
loss = F.cross_entropy(logits, targets)
```

This is the standard objective for next-token prediction: the model learns to assign higher probability to the correct next token.

### Generation

The notebook implements autoregressive generation like this:

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

This loop repeatedly:

1. reads the current context
2. predicts the next token distribution
3. samples a token from that distribution
4. appends it to the sequence
5. repeats until the target length is reached

---

## Why the notebook matters

This notebook is useful because it makes the internal mechanics of GPT-style models visible without hiding them behind a black box.

It shows how the following ideas fit together:

- tokenization and vocabulary mapping
- embedding lookup tables
- next-token prediction
- loss computation with cross-entropy
- batching and sequence handling
- causal attention
- weighted aggregation over prior tokens
- normalization and stability tricks

The notebook is not intended as a production-scale model. It is a teaching tool for understanding the foundational concept of transformer-style language modeling.

---

## Matrix multiplication and weighted aggregation

A key conceptual step in the notebook is showing that matrix multiplication can act like a weighted sum over prior tokens.

```python
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x
```

This is conceptually important because attention is also a weighted aggregation mechanism. The model chooses which previous tokens are relevant and combines them according to learned weights.

---

## Self-attention

The notebook introduces a single self-attention head:

```python
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)
q = query(x)
wei = q @ k.transpose(-2, -1)
```

Each token is converted into three vectors:

- `query`: what the token is looking for
- `key`: what the token offers
- `value`: the content that is actually passed forward

The attention weights are normalized with softmax, and each token output becomes a weighted sum of the value vectors from earlier positions.

### Causal masking

The notebook enforces a lower-triangular mask so that future tokens are not visible to the current position:

```python
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
```

This is the core autoregressive constraint in GPT-style architectures.

---

## Scaling and normalization

The notebook also introduces a scaled attention formulation:

```python
wei = q @ k.transpose(-2, -1) * head_size**-0.5
```

This keeps attention values in a stable range and avoids overly sharp distributions.

A simplified layer normalization appears as well:

```python
class LayerNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
```

This stabilizes activation values and is a standard ingredient in transformer blocks.

---

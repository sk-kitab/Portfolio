---
title: "Attention Is All You Need — Code & Paper Analysis"
date: "2026-04-14"
excerpt: "A mapping between a minimal GPT-style implementation and Vaswani et al. (2017), plus key conceptual explanations for positional encoding and self-attention."
category: "Deep Learning"
readTime: "12 min read"
---

If you’ve ever implemented a tiny “GPT-ish” model and wondered how it lines up with the original Transformer paper, this post is the bridge.

## TL;DR

- **Embeddings + softmax**: your token embeddings and final linear head are exactly the paper’s “embeddings and softmax” story.
- **Position matters**: Transformers need positional information because there’s no recurrence; you can use sinusoidal encodings (paper) or learned embeddings (common in practice).
- **Self-attention is the core**: \(softmax(QK^T/\sqrt{d_k})V\) is the mechanism that mixes context.
- **A minimal `gpt.py`** often skips the actual Transformer blocks (attention + FFN + residual + LayerNorm). That’s why it can *look* like a language model but not behave like a real one.

## 1) `gpt.py` → Paper mapping (quick, practical)

### Token embeddings → Paper §3.4 “Embeddings and Softmax”

```py
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # line 67
tok_emb = self.token_embedding_table(idx)  # (B, T, C)         # line 75
```

Paper §3.4: **“we use learned embeddings to convert the input tokens and output tokens to vectors of dimension d_model.”**  
`token_embedding_table` is exactly this: a lookup table mapping each integer token ID to a dense vector of size `n_embd` (playing the role of \(d_{model}\)).

### Positional information → Paper §3.5 “Positional Encoding”

```py
self.position_embedding_table = nn.Embedding(block_size, n_embd)  # line 68
pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # line 76
x = tok_emb + pos_emb  # line 77
```

Transformers process tokens in parallel, so they have no built-in sense of order. The paper injects position information using **positional encodings**, then **adds** them to token embeddings (same dimension → sum is valid).

The paper’s classic (fixed) sinusoidal encoding is:

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

In practice, **learned positional embeddings** are extremely common. The paper tested them too (Table 3, row (E)) and found **“nearly identical results.”** Your code uses the learned variant via `nn.Embedding(block_size, n_embd)` and then does the key operation: `tok_emb + pos_emb`.

### Linear head + softmax → Paper §3.4 (prediction distribution)

```py
self.lm_head = nn.Linear(n_embd, vocab_size)  # line 69
logits = self.lm_head(x)  # line 78
probs = F.softmax(logits, dim=-1)  # line 98 (applied after selecting last step)
idx_next = torch.multinomial(probs, num_samples=1)  # line 100
```

Paper §3.4: **“we use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.”**

`lm_head` is the linear projection from hidden states to vocab logits, `softmax` turns logits into probabilities, and `multinomial` samples a next token.

### Autoregressive generation loop → Paper §3.1 “Decoder”

```py
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, loss = self(idx)
        logits = logits[:, -1, :]             # line 96: last time step
        probs = F.softmax(logits, dim=-1)     # line 98
        idx_next = torch.multinomial(probs, num_samples=1)  # line 100
        idx = torch.cat((idx, idx_next), dim=1)  # line 102: append
    return idx
```

Paper §3.1: the decoder generates **one token at a time** and is **auto-regressive**, consuming the previously generated tokens as input for the next step.  
`torch.cat` appends the sampled token to the context, implementing exactly that loop.

### Cross-entropy loss → Paper §5 “Training”

```py
B, T, C = logits.shape
logits = logits.view(B*T, C)
targets = targets.view(B*T)
loss = F.cross_entropy(logits, targets)
```

The paper’s training objective is next-token prediction; it additionally uses **label smoothing** (§5.4). This code uses standard cross-entropy without label smoothing, but the core objective matches.

### Optimizer choice → Paper §5.3 “Optimizer”

```py
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```

Paper §5.3 uses **Adam** with specific \(\beta_1, \beta_2\) and a warmup/decay learning-rate schedule (Equation 3).  
This code uses **AdamW** (a modern Adam variant with decoupled weight decay) with a constant learning rate.

### Batching & data windows → Paper §5.1 “Training Data and Batching”

```py
ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([data[i:i+block_size] for i in ix])
y = torch.stack([data[i+1:i+block_size+1] for i in ix])
```

Paper §5.1 discusses batching by approximate length for translation. Here, `get_batch` samples many fixed-length windows in parallel and shifts targets by 1 token, matching the next-token prediction setup.

## 2) What a minimal `gpt.py` usually *doesn’t* include (and why it matters)

If your file is basically **embeddings → linear head → sampling**, it can generate tokens, but it’s missing the parts that make Transformers *Transformer-y*:

| Missing component | Paper section |
|---|---|
| Scaled Dot-Product Attention: \(softmax(QK^T/\sqrt{d_k})V\) | §3.2.1 |
| Multi-Head Attention (parallel heads, concat + output projection) | §3.2.2 |
| Causal masking (prevent attending to future tokens) | §3.2.3 |
| Position-wise feed-forward network (two linear layers + ReLU) | §3.3 |
| Residual connections \(LayerNorm(x + Sublayer(x))\) | §3.1 |
| Layer normalization | §3.1 |

The big takeaway: **self-attention + FFN blocks** are where representation learning happens. Without them, you don’t really have a Transformer—just the outer shell.

## 3) Positional encoding, explained like a blog

Transformers process tokens in parallel. Without recurrence or convolution, they have no inherent notion of token order. **Positional encoding** injects order information into token representations before attention is applied.

The paper’s sinusoidal positional encoding:

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

Why sinusoids?
- The paper hypothesizes this enables learning **relative position**: for a fixed offset \(k\), \(PE_{pos+k}\) can be expressed as a linear function of \(PE_{pos}\).
- Different dimensions encode different frequencies, giving both short-range and long-range position signals.

Your code uses a **learned** positional embedding table instead (`nn.Embedding(block_size, n_embd)`), which the paper reports performs similarly (Table 3(E)). The key idea remains: **add position vectors to token embeddings** so position is present in all downstream computations.

## 4) Self-attention (the one formula that changed everything)

**Self-attention** lets each token build a context-aware representation by “looking at” other tokens in the same sequence.

### Mechanics

1) Start from token representations \(X\), then project into queries, keys, values:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

2) Compute “who should I pay attention to?” and mix values:

```text
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
```

The \(/ \sqrt{d_k}\) scaling prevents large dot products from pushing softmax into saturation (tiny gradients), which the paper explains in §3.2.1.

### Multi-head attention

Instead of one attention, the paper runs \(h\) attentions in parallel with different learned projections:

```text
head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W_O
```

This lets different heads learn different relations (syntax, long-range dependencies, etc.) simultaneously.

## 5) Why GPT is “decoder-only”

The original Transformer (paper) uses an **encoder-decoder** structure:

```text
Encoder stack:   bidirectional self-attention + FFN
Decoder stack:   causal (masked) self-attention + (optional) cross-attention + FFN
```

### Encoder vs Decoder (intuition)
- **Encoder**: each position can attend to all positions (left and right). Great for “understanding” tasks.
- **Decoder**: uses **causal masking** so position \(t\) can only attend to positions \(\le t\), preserving autoregressive generation.

### Why GPT drops the encoder
GPT is trained for **next-token prediction** (language modeling). For that setting:
- There is no separate “source sequence” to encode (unlike translation).
- Generation must be **left-to-right**, so causal masking is the right inductive bias.
- Cross-attention to an encoder output is unnecessary without a distinct input sequence.

So GPT is effectively a **stack of Transformer decoder blocks** (masked self-attention + FFN + residual/LayerNorm), trained on next-token prediction at scale.


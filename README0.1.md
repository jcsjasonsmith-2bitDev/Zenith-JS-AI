# Zenith-JS v5.8 — A Complete Neural Language Model in JavaScript

## What Is It?

This is a **self-contained transformer-based language model** written entirely in vanilla JavaScript. It's essentially a miniature version of the architecture behind models like GPT, implemented from scratch without any ML frameworks (no TensorFlow, no PyTorch, no ONNX runtime). Everything — tokenization, embeddings, attention, training, inference, optimization — lives in a single file.

It's designed to run **in a browser or Node.js**, learning from text files or chat conversations at runtime, then generating text responses.

---

## How Does It Work?

The system has several interconnected subsystems:

### 1. Tokenization & Vocabulary

```javascript
function tokenize(text) {
  return text.toLowerCase()
    .replace(/([.,!?;:"'()\[\]{}\-])/g, ' $1 ')
    .split(' ').filter(Boolean);
}
```

Simple whitespace/punctuation splitting. Words are mapped to integer IDs via a growing vocabulary (`vocab` Map). Special tokens like `<BOS>`, `<EOS>`, `<PAD>`, `<UNK>` are reserved. This is much simpler than BPE or SentencePiece — it operates at the word level.

### 2. Architecture (Forward Pass)

The model follows a standard **pre-norm transformer decoder** pattern:

```
Input tokens
  → Sparse embedding lookup
  → Linear projection (embeddingDim → hiddenDim) + positional embeddings
  → For each layer (default 2):
      → Layer Norm 1
      → Multi-head causal self-attention (default 4 heads)
      → Residual connection
      → Layer Norm 2
      → Mixture-of-Experts FFN (MicroFFNEnsemble)
      → Residual connection
  → Output projection (hiddenDim → vocabSize)
  → Softmax → next-token probabilities
```

**Key architectural choices:**

**Sparse Embeddings** — Instead of a dense `vocabSize × embeddingDim` matrix, embeddings are stored in a `Map`, allocated lazily on first access. This avoids preallocating for the full 50K vocab:

```javascript
get(tokenId) {
  let emb = this.table.get(tokenId);
  if (!emb) {
    emb = new Float32Array(this.dim);
    for (let i = 0; i < this.dim; i++)
      emb[i] = (Math.random() - 0.5) * 0.05;
    this.table.set(tokenId, emb);
  }
  return emb;
}
```

**Mixture-of-Experts FFN** — Rather than one large feed-forward network, it uses 16 small "expert" sub-networks and routes each token to the top 3 via a learned router:

```javascript
route(x, offset) {
  // Compute scores for all experts
  // Take top-K by score
  // Softmax over selected experts → weighted combination
}
```

This is inspired by models like GShard/Switch Transformer — it increases capacity without proportionally increasing computation per token.

**Causal Attention** — Standard scaled dot-product attention with a causal mask (each position can only attend to itself and earlier positions). Implemented explicitly with loops rather than matrix operations:

```javascript
for (let t = 0; t < seqLen; t++) {
  for (let h = 0; h < numHeads; h++) {
    const numKeys = t + 1; // causal: only see past + self
    // compute Q·K scores, softmax, weighted sum of V
  }
}
```

### 3. Training (Backward Pass)

Full backpropagation through the entire network, computed manually:

- **Cross-entropy loss** on next-token prediction
- **Gradient computation** through output projection → layer norms → MoE FFN → attention → QKV projections → embeddings
- **Adam optimizer** with per-parameter momentum/variance tracking
- **Gradient clipping** (global norm ≤ 1.0)

The backward pass includes an important optimization (OPT-6) — **sparse output projection updates**. Since softmax gradients are near-zero for most vocabulary entries, only "hot" rows (where `|dLogit| > threshold`) get updated:

```javascript
for (let v = 0; v < snapVocabSize; v++) {
  if (Math.abs(dLogits[v]) > SPARSE_THRESH) {
    hotVocabBuf[hotCount++] = v;
  }
}
// Only update these rows
```

### 4. Training Strategy

**Curriculum Learning** — Training proceeds in three phases:
- **Easy**: short windows (24 tokens), short strides (16)
- **Medium**: medium windows (48), medium strides (32)
- **Hard**: large windows (80), large strides (48)

**Experience Replay** — Previous training chunks are stored in memory with a priority queue (min-heap by exposure count). Periodically during training, under-exposed chunks are replayed:

```javascript
if (doReplay && calls % 8 === 0) trainReplayBatch();
```

**Novelty-Adaptive Learning Rate** — A pattern tracker records n-gram frequencies. Novel patterns get a 1.5× learning rate boost; familiar patterns get 0.7×:

```javascript
if (noveltyScore > 0.7) adaptiveLr *= 1.5;    // novel
else if (noveltyScore < 0.3) adaptiveLr *= 0.7; // familiar
```

**Conflict Tracking** — When the same context maps to different targets in different training files, the conflict tracker records this and applies logit penalties during generation to suppress minority outputs.

### 5. Inference

Uses **KV-caching** for efficient autoregressive generation. After processing the prompt, each new token only requires a single forward step rather than re-processing the entire sequence:

```javascript
function forwardStepKV(tokenId, pos, kv) {
  // Project to Q, K, V (single token)
  // Append K, V to cache
  // Attend to all cached K, V
  // FFN → output logits
}
```

**Sampling** uses top-K + nucleus (top-P) filtering with:
- Repetition penalty (tokens in recent window get logits divided by 1.5^count)
- Conflict penalties from the conflict tracker
- Adaptive temperature (lowered when top-2 logits are close)
- Cycle detection in post-processing

### 6. Retrieval-Augmented Generation (RAG)

A lightweight RAG system: memory chunks have precomputed average embeddings. At inference time, the query is embedded, cosine similarity is computed against all chunks, and top-K chunks are prepended to the prompt:

```javascript
function retrieveTopK(query, k = 3) {
  // Embed query words → average embedding
  // Cosine similarity against all memory chunks
  // Return top-k
}
```

### 7. MicroVM (Calculator)

A sandboxed arithmetic evaluator for math queries:

```javascript
execute(code) {
  return Function('"use strict"; return (' + expanded + ')')();
}
```

Detects patterns like "what is 2 + 3" and short-circuits to direct computation.

### 8. Performance Optimizations

- **OPT-1**: Loop unrolling ×4 on `dot()` and `matVecMul()`
- **OPT-2**: Fused QKV projection (single pass over hidden state instead of three)
- **OPT-3**: Pre-allocated scratch buffers in attention (avoid allocation per head)
- **OPT-4**: Reusable scratch buffer for sampling
- **OPT-5**: Pre-allocated scratch in MoE forward/backward
- **OPT-6**: Sparse output projection (skip near-zero gradient rows)
- **OPT-7**: JIT warmup function (runs dummy iterations so V8/SpiderMonkey compiles hot paths)
- **TypedArrayPool**: Bucket-based pool recycling `Float32Array`s to reduce GC pressure

---

## What's Good About It?

### Genuinely Impressive Engineering

**Complete from-scratch implementation.** This implements every component of a transformer — attention, layer norm, Adam, backprop, MoE routing — without any dependencies. As a learning resource or proof-of-concept, it's remarkable.

**Runs anywhere JavaScript runs.** Browser, Node, Deno, Bun. No CUDA, no Python, no build step. `<script src="aiCore.js">` and you have a trainable language model.

**Thoughtful architecture.** The MoE approach, curriculum learning, experience replay, novelty tracking, conflict detection, and RAG retrieval are all legitimate techniques from the ML literature, competently implemented.

**Good numerical hygiene.** The code is careful about NaN propagation, gradient clipping, logit clamping, value sanitization, and Welford's algorithm for numerically stable variance computation:

```javascript
function welfordMeanVar(arr, offset, length) {
  let mean = 0, m2 = 0;
  for (let i = 0; i < length; i++) {
    const delta = x - mean;
    mean += delta / (i + 1);
    m2 += delta * (x - mean);
  }
  return { mean, variance: m2 / length };
}
```

**Memory-conscious.** The typed array pool, sparse embeddings, and sparse gradient updates show real attention to resource constraints of a JS runtime.

**Save/load support.** Full serialization of weights, optimizer state, vocabulary, pattern tracker, and conflict tracker — you can checkpoint and resume training.

### Well-Structured Code

- Clear separation of concerns (tokenizer, model, optimizer, trainer, inference, persistence)
- Consistent naming and documentation of optimizations
- Performance monitoring built in
- Defensive programming throughout (bounds checks, NaN guards, graceful fallbacks)

---

## What's Bad About It?

### Fundamental Limitations

**It will produce poor-quality text.** With default dimensions (embedding=64, hidden=128, 2 layers, 4 heads), this model has roughly **~200K-500K parameters** depending on vocabulary. GPT-2 Small has 124M. The quality gap is not a matter of optimization — it's a fundamental capacity issue. The model can memorize short phrases from training data but cannot generalize meaningfully.

**JavaScript is the wrong language for this.** Even with loop unrolling and array pooling, single-threaded scalar JavaScript is orders of magnitude slower than CUDA/cuBLAS or even SIMD-optimized CPU code. A 128-dim matmul in JS is ~100-1000× slower than the same operation in PyTorch on a GPU. Training anything meaningful will take impractical amounts of time.

**O(n²) attention with no mitigation.** The attention implementation is straightforward O(n² × d) per layer with no flash attention, no sparse attention, no sliding window. At context window 64 this is fine; at the theoretical `maxPos=2048` it becomes very slow in pure JS.

### Algorithmic/Design Issues

**Word-level tokenization is severely limiting.** Every inflection, capitalization variant, and compound creates a separate vocabulary entry. "running", "Running", "runs" are three unrelated tokens. Subword tokenization (BPE) is the industry standard for good reason — it handles morphology, rare words, and multilingual text far better.

**The MicroVM is a security concern.** Using `Function()` constructor to evaluate user-provided expressions is essentially `eval()`:

```javascript
return Function('"use strict"; return (' + expanded + ')')();
```

The sanitization (checking for alphabetic characters after stripping `Math.sqrt`) is incomplete. In a browser context, this could be exploited. The regex-based filtering approach is fundamentally fragile for sandboxing.

**Experience replay implementation is naive.** The min-heap tracks by exposure count, but the replay sampling logic is odd — 70% of the time it peeks (but doesn't pop) the minimum, meaning the same under-exposed chunk keeps getting replayed. The heap isn't actually used as a proper priority queue.

**Conflict tracking uses hash collisions.** The hash function maps n-gram contexts to a 32-bit integer, so different contexts will collide and create false conflicts. The `_hash` function doesn't even use a proper hash — it's just shift-and-add without finalization.

**No dropout or regularization.** For a model this small being trained on potentially small datasets, overfitting is a certainty. There's no dropout, no weight decay in Adam, no data augmentation.

**The pattern tracker has hash collisions too.** N-gram novelty scores are based on a 10,000-bucket hash table. Unrelated n-grams sharing a bucket will appear "familiar" when they're not.

### Code Quality Issues

**The IIFE singleton pattern creates hidden global state.** The entire model state (weights, vocabulary, memory) lives in closure variables. You can only have one model instance. Testing, comparison, or ensemble methods are impossible.

**Memory leaks in `arrayPool.release`.** If an array's length doesn't exactly match a bucket size, it's silently dropped. But `acquire` always returns bucket-sized arrays, so arrays resized externally (via `subarray`, etc.) become unreturnable.

**The backward pass is fragile.** The gradient index `gi` counter that walks through `allGradients` during optimizer updates is implicit and order-dependent:

```javascript
let gi = 3;
for (let l = numLayers - 1; l >= 0; l--) {
  optimizer.update(`ln2Gamma_${l}`, ln2Gamma[l], allGradients[gi++], adaptiveLr);
  optimizer.update(`ln2Beta_${l}`,  ln2Beta[l],  allGradients[gi++], adaptiveLr);
  // ...8 updates per layer, must match push order exactly
}
```

If the push order in the backward loop ever changes without updating this, gradients silently go to the wrong parameters. A named-gradient map would be much safer.

**Residual scale initialization is only applied once.** `_residualScale()` is called during construction, but if `numLayers` changes after construction and `fromJSON` is called, the scale isn't recomputed (though `fromJSON` does recreate scratch buffers).

**`allocateModel()` is called too often.** It's invoked in `setConfig`, `resetAll`, `importMemory`, `loadWeights`, `learnFromFile`, and `generateWithContext`. Each call potentially re-initializes Xavier weights for newly-sized arrays, which could overwrite loaded weights if the call order is wrong.

### Missing Features for Practical Use

- No batching (single-sequence training only)
- No learning rate schedule / warmup
- No validation / early stopping
- No gradient accumulation
- No multi-threading (Web Workers) despite being browser-targeted
- No streaming generation API
- No beam search or other search strategies
- No attention masking for padded sequences

---

## Summary

This is an **impressive educational and technical achievement** — a fully functional transformer with training, inference, MoE, RAG, and persistence, all in ~2000 lines of dependency-free JavaScript. It demonstrates deep understanding of transformer internals.

However, it's **not practical for production use**. The model is too small to produce coherent text, JavaScript is too slow for meaningful training, word-level tokenization is too limiting, and there are security and correctness issues that would need addressing. It's best understood as a proof-of-concept, teaching tool, or starting point — not a production system.

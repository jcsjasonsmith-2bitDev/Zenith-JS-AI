/* =========================================================================
   aiCore.js — Zenith-JS v5.8 (Performance Optimizations)

   Optimizations vs v5.7:
   - OPT-1: Loop unrolling x4 on dot() and matVecMul()
   - OPT-2: Fused QKV projection in forwardFull() and forwardStepKV()
   - OPT-3: Pre-allocated scratch buffers in CausalAttention (forward+backward)
   - OPT-4: Pre-allocated scratch buffer in sampleLogits()
   - OPT-5: Pre-allocated scratch buffers in MicroFFNEnsemble (forward+backward)
   - OPT-6: Sparse output projection update in backwardAndUpdate()
   - OPT-7: JIT warmup function added to public API
   - All v5.7 crash fixes and correctness fixes retained
   ========================================================================= */

const aiCore = (() => {
  "use strict";

  // ========================================================================
  // CONSTANTS
  // ========================================================================
  const CONSTANTS = {
    YIELD_INTERVAL_MS:        8,
    GRAD_CLIP_NORM:           1.0,
    BUCKET_SIZE:              8,
    NUM_MICRO_FFN:            16,
    ACTIVE_MICRO_FFN:         3,
    GRADIENT_THRESHOLD:       0.001,
    GRADIENT_RECOMPUTE_EVERY: 10,
    REPLAY_RATIO:             0.3,
    REPLAY_BATCH_SIZE:        4,
    CONSOLIDATION_PASSES:     3,
    MIN_MEMORY_FOR_REPLAY:    3,
    NGRAM_SIZE:               3,
    NOVELTY_HIGH_THRESHOLD:   0.7,
    NOVELTY_LOW_THRESHOLD:    0.3,
    NOVEL_LR_MULTIPLIER:      1.5,
    FAMILIAR_LR_MULTIPLIER:   0.7,
    PATTERN_HASH_SIZE:        10000,
    FAST_WEIGHT_BLEND:        0.15,
    SLOW_WEIGHT_BLEND:        0.85,
    CHUNK_OVERLAP:            8,
    MIN_GENERATION_TOKENS:    3,
    REPETITION_PENALTY:       1.5,
    REPETITION_WINDOW:        20,
    CONFLICT_LOGIT_PENALTY:   2.0,
    LN_EPSILON:               1e-4,
    LOGIT_CLAMP:              50,
    HIDDEN_CLAMP:             1e3,
    GRAD_VALUE_CLAMP:         10,
    SPARSE_GRAD_THRESHOLD:    1e-4,
  };

  // ========================================================================
  // PERFORMANCE MONITOR
  // ========================================================================
  const PerfMonitor = {
    trainCalls: 0, forwardCalls: 0, backwardCalls: 0,
    totalTimeMs: 0, gradientClips: 0, nanGradients: 0,
    vmExecutions: 0, vmErrors: 0, lossSum: 0, lossCount: 0,
    lastLoss: 0, minLoss: Infinity, maxLoss: -Infinity,
    sessionStartTime: 0, cacheHits: 0, cacheMisses: 0,
    novelPatterns: 0, familiarPatterns: 0, nanSkips: 0,

    startSession() {
      this.sessionStartTime = performance.now();
      console.log('[PerfMonitor] Session started');
    },

    endSession() {
      const elapsed = performance.now() - this.sessionStartTime;
      const hitRate = this.cacheHits + this.cacheMisses > 0
        ? (this.cacheHits / (this.cacheHits + this.cacheMisses) * 100).toFixed(1)
        : '0.0';
      console.log(
        `[PerfMonitor] Session ended: ${(elapsed / 1000).toFixed(2)}s, ` +
        `${this.trainCalls} calls, avgLoss: ${this.getAvgLoss().toFixed(4)}, ` +
        `clips: ${this.gradientClips}, nanGrads: ${this.nanGradients}, ` +
        `activeGradRate: ${hitRate}%`
      );
    },

    recordTrain(loss, seqLen, timeMs) {
      this.trainCalls++;
      this.forwardCalls++;
      this.backwardCalls++;
      this.totalTimeMs += timeMs;
      if (Number.isFinite(loss)) {
        this.lossSum  += loss;
        this.lossCount++;
        this.lastLoss  = loss;
        if (loss < this.minLoss) this.minLoss = loss;
        if (loss > this.maxLoss) this.maxLoss = loss;
      }
      if (this.trainCalls % 50 === 0) {
        console.log(
          `[Train #${this.trainCalls}] Loss: ${loss.toFixed(4)}, ` +
          `Avg: ${this.getAvgLoss().toFixed(4)}, ` +
          `NaNGrads: ${this.nanGradients}, ` +
          `Novel: ${this.novelPatterns}, Familiar: ${this.familiarPatterns}`
        );
      }
    },

    recordNovel()     { this.novelPatterns++; },
    recordFamiliar()  { this.familiarPatterns++; },
    recordCacheHit()  { this.cacheHits++; },
    recordCacheMiss() { this.cacheMisses++; },
    recordGradClip()  { this.gradientClips++; },
    recordNanGrad()   { this.nanGradients++; },
    recordNanSkip()   { this.nanSkips++; },
    recordVM()        { this.vmExecutions++; },
    recordVMError()   { this.vmErrors++; },
    getAvgLoss()      { return this.lossCount > 0 ? this.lossSum / this.lossCount : 0; },

    getStats() {
      return {
        trainCalls:       this.trainCalls,
        forwardCalls:     this.forwardCalls,
        backwardCalls:    this.backwardCalls,
        totalTime:        this.totalTimeMs,
        avgLoss:          this.getAvgLoss(),
        lastLoss:         this.lastLoss,
        minLoss:          this.minLoss === Infinity  ? 0 : this.minLoss,
        maxLoss:          this.maxLoss === -Infinity ? 0 : this.maxLoss,
        gradientClips:    this.gradientClips,
        nanGradients:     this.nanGradients,
        nanSkips:         this.nanSkips,
        vmExecutions:     this.vmExecutions,
        vmErrors:         this.vmErrors,
        cacheHits:        this.cacheHits,
        cacheMisses:      this.cacheMisses,
        activeGradRate:   this.cacheHits + this.cacheMisses > 0
          ? this.cacheHits / (this.cacheHits + this.cacheMisses) : 0,
        novelPatterns:    this.novelPatterns,
        familiarPatterns: this.familiarPatterns,
      };
    },

    reset() {
      this.trainCalls = 0; this.forwardCalls = 0; this.backwardCalls = 0;
      this.totalTimeMs = 0; this.gradientClips = 0; this.nanGradients = 0;
      this.vmExecutions = 0; this.vmErrors = 0; this.lossSum = 0;
      this.lossCount = 0; this.lastLoss = 0; this.minLoss = Infinity;
      this.maxLoss = -Infinity; this.cacheHits = 0; this.cacheMisses = 0;
      this.novelPatterns = 0; this.familiarPatterns = 0; this.nanSkips = 0;
    }
  };

  // ========================================================================
  // PATTERN TRACKER
  // ========================================================================
  class PatternTracker {
    constructor(hashSize = CONSTANTS.PATTERN_HASH_SIZE) {
      this.hashSize      = hashSize;
      this.patternCounts = new Uint32Array(hashSize);
      this.totalPatterns = 0;
      this.ngramSize     = CONSTANTS.NGRAM_SIZE;
    }

    hash(tokens) {
      let h = 0;
      for (let i = 0; i < tokens.length; i++) {
        h = ((h << 5) - h + tokens[i]) | 0;
      }
      return Math.abs(h) % this.hashSize;
    }

    extractNgrams(tokenIds) {
      const ngrams = [];
      for (let i = 0; i <= tokenIds.length - this.ngramSize; i++) {
        ngrams.push(tokenIds.slice(i, i + this.ngramSize));
      }
      return ngrams;
    }

    recordPatterns(tokenIds) {
      const ngrams = this.extractNgrams(tokenIds);
      for (const ngram of ngrams) {
        const h = this.hash(ngram);
        this.patternCounts[h]++;
        this.totalPatterns++;
      }
    }

    getNoveltyScore(tokenIds) {
      if (tokenIds.length < this.ngramSize) return 1.0;
      const ngrams = this.extractNgrams(tokenIds);
      if (ngrams.length === 0) return 1.0;
      let totalScore = 0;
      for (const ngram of ngrams) {
        const h     = this.hash(ngram);
        const count = this.patternCounts[h];
        totalScore += count === 0 ? 1.0 : 1.0 - Math.min(count / 10, 1.0);
      }
      return totalScore / ngrams.length;
    }

    clear() { this.patternCounts.fill(0); this.totalPatterns = 0; }

    toJSON() {
      return {
        patternCounts: Array.from(this.patternCounts),
        totalPatterns: this.totalPatterns
      };
    }

    fromJSON(data) {
      if (data && data.patternCounts) {
        this.patternCounts = new Uint32Array(data.patternCounts);
      }
      this.totalPatterns = (data && data.totalPatterns) || 0;
    }
  }

  const patternTracker = new PatternTracker();

  // ========================================================================
  // CONFLICT TRACKER
  // ========================================================================
  class ConflictTracker {
    constructor(maxConflicts = 1000) {
      this.conflicts    = new Map();
      this.maxConflicts = maxConflicts;
    }

    _hash(context) {
      let h = 0;
      for (let i = 0; i < context.length; i++) {
        h = ((h << 5) - h + context[i]) | 0;
      }
      return h;
    }

    record(contextIds, targetId, fileName) {
      const h = this._hash(contextIds);
      if (!this.conflicts.has(h)) {
        this.conflicts.set(h, { context: Array.from(contextIds), outputs: new Map() });
      }
      const entry = this.conflicts.get(h);
      if (!entry.outputs.has(targetId)) {
        entry.outputs.set(targetId, { count: 0, sources: new Set() });
      }
      const output = entry.outputs.get(targetId);
      output.count++;
      output.sources.add(fileName);
      if (this.conflicts.size > this.maxConflicts) {
        const firstKey = this.conflicts.keys().next().value;
        this.conflicts.delete(firstKey);
      }
    }

    getPenalties(contextIds, vs) {
      const penalties = new Float32Array(vs);
      const h         = this._hash(contextIds);
      const entry     = this.conflicts.get(h);
      if (!entry || entry.outputs.size < 2) return penalties;
      let maxCount   = 0;
      let majorityId = -1;
      for (const [tid, info] of entry.outputs.entries()) {
        if (info.count > maxCount) { maxCount = info.count; majorityId = tid; }
      }
      const totalCount = Array.from(entry.outputs.values())
        .reduce((s, x) => s + x.count, 0);
      for (const [tid, info] of entry.outputs.entries()) {
        if (tid === majorityId) continue;
        if (tid < vs) {
          penalties[tid] = CONSTANTS.CONFLICT_LOGIT_PENALTY *
            (info.count / totalCount);
        }
      }
      return penalties;
    }

    clear() { this.conflicts.clear(); }

    toJSON() {
      const entries = [];
      for (const [h, entry] of this.conflicts.entries()) {
        entries.push({
          hash: h, context: entry.context,
          outputs: Array.from(entry.outputs.entries()).map(([t, info]) => ({
            target: t, count: info.count, sources: Array.from(info.sources)
          }))
        });
      }
      return entries;
    }

    fromJSON(entries) {
      this.conflicts.clear();
      if (!entries) return;
      for (const e of entries) {
        const outputs = new Map();
        for (const o of e.outputs) {
          outputs.set(o.target, { count: o.count, sources: new Set(o.sources) });
        }
        this.conflicts.set(e.hash, { context: e.context, outputs });
      }
    }
  }

  const conflictTracker = new ConflictTracker();

  // ========================================================================
  // TOKENIZER
  // ========================================================================
  function tokenize(text) {
    if (typeof text !== 'string' || !text) return [];
    return text
      .toLowerCase()
      .replace(/([.,!?;:"'()\[\]{}\-])/g, ' $1 ')
      .replace(/\s+/g, ' ')
      .trim()
      .split(' ')
      .filter(Boolean);
  }

  // ========================================================================
  // MICRO VM
  // ========================================================================
  const MicroVM = {
    execute(code) {
      try {
        const trimmed  = (code || '').trim();
        const expanded = trimmed.replace(
          /\bsqrt\s*\(\s*([^)]+)\s*\)/g, 'Math.sqrt($1)'
        );
        const stripped = expanded.replace(/Math\.sqrt\(/g, '');
        if (/[a-zA-Z_$]/.test(stripped)) return null;
        // eslint-disable-next-line no-new-func
        return Function('"use strict"; return (' + expanded + ')')();
      } catch { return null; }
    },
    calc(expr) { return this.execute(expr); },
    reset()    {}
  };

  const ExecutionEngine = {
    enabled: true,
    needsExecution(text) {
      if (!this.enabled || typeof text !== 'string') return false;
      const lower = text.toLowerCase().trim();
      return /^what\s+is\s+\d/.test(lower) ||
             /^calculate\s+\d/.test(lower)  ||
             /^\d+\s*[+\-*/]\s*\d+/.test(lower);
    },
    extractCode(prompt) {
      const lower = prompt.toLowerCase().trim();
      let match = lower.match(/(\d+\.?\d*)\s*%\s+(?:of\s+)?(\d+\.?\d*)/);
      if (match) return `${match[2]} * ${match[1]} / 100`;
      match = lower.match(/(?:what\s+is|calculate)\s+([\d\s+\-*/().]+)/);
      if (match) return match[1].trim();
      return null;
    },
    autoExecute(prompt) {
      if (!this.needsExecution(prompt)) return null;
      const code = this.extractCode(prompt);
      if (!code) return null;
      const result = MicroVM.execute(code);
      if (result === null || typeof result !== 'number' || !isFinite(result)) return null;
      PerfMonitor.recordVM();
      return {
        result,
        formatted: Number.isInteger(result)
          ? String(result)
          : result.toFixed(4).replace(/\.?0+$/, '')
      };
    },
    isExecToken(w) {
      return ['<EXEC_START>', '<EXEC_END>', '<EXEC_RESULT>', '<VERIFY>'].includes(w);
    }
  };

  // ========================================================================
  // TYPED ARRAY POOL
  // ========================================================================
  class TypedArrayPool {
    constructor() {
      this.pools = new Map();
      this.stats = { allocations: 0, reuses: 0 };
    }

    acquire(size) {
      const bucket = this._bucket(size);
      const pool   = this.pools.get(bucket);
      if (pool && pool.length > 0) {
        this.stats.reuses++;
        const arr = pool.pop();
        arr.fill(0);
        return arr;
      }
      this.stats.allocations++;
      return new Float32Array(bucket);
    }

    release(arr) {
      if (!arr || !(arr instanceof Float32Array)) return;
      const bucket = this._bucket(arr.length);
      if (arr.length !== bucket) return;
      let pool = this.pools.get(bucket);
      if (!pool) { pool = []; this.pools.set(bucket, pool); }
      if (pool.length < 64) pool.push(arr);
    }

    _bucket(size) {
      if (size <= 64)    return 64;
      if (size <= 128)   return 128;
      if (size <= 192)   return 192;
      if (size <= 256)   return 256;
      if (size <= 384)   return 384;
      if (size <= 512)   return 512;
      if (size <= 768)   return 768;
      if (size <= 1024)  return 1024;
      if (size <= 1536)  return 1536;
      if (size <= 2048)  return 2048;
      if (size <= 3072)  return 3072;
      if (size <= 4096)  return 4096;
      if (size <= 6144)  return 6144;
      if (size <= 8192)  return 8192;
      if (size <= 12288) return 12288;
      if (size <= 16384) return 16384;
      if (size <= 24576) return 24576;
      if (size <= 32768) return 32768;
      return Math.ceil(size / 4096) * 4096;
    }

    getStats() {
      return {
        ...this.stats,
        poolSizes: Array.from(this.pools.entries())
          .map(([k, v]) => ({ size: k, count: v.length }))
      };
    }

    clear() { this.pools.clear(); this.stats = { allocations: 0, reuses: 0 }; }
  }

  const arrayPool = new TypedArrayPool();

  // ========================================================================
  // SPARSE EMBEDDING TABLE
  // ========================================================================
  class SparseEmbeddings {
    constructor(embeddingDim) {
      this.dim            = embeddingDim;
      this.table          = new Map();
      this.gradients      = new Map();
      this.accessCounts   = new Map();
      this.exposureCounts = new Map();
    }

    get(tokenId) {
      let emb = this.table.get(tokenId);
      if (!emb) {
        emb = new Float32Array(this.dim);
        for (let i = 0; i < this.dim; i++) {
          emb[i] = (Math.random() - 0.5) * 0.05;
        }
        this.table.set(tokenId, emb);
        this.exposureCounts.set(tokenId, 0);
      }
      this.accessCounts.set(tokenId, (this.accessCounts.get(tokenId) || 0) + 1);
      return emb;
    }

    set(tokenId, embedding)  { this.table.set(tokenId, embedding); }

    recordExposure(tokenId) {
      this.exposureCounts.set(tokenId,
        (this.exposureCounts.get(tokenId) || 0) + 1);
    }

    getExposure(tokenId)          { return this.exposureCounts.get(tokenId) || 0; }
    isNovel(tokenId, threshold=5) { return this.getExposure(tokenId) < threshold; }

    getGradient(tokenId) {
      let grad = this.gradients.get(tokenId);
      if (!grad) {
        grad = new Float32Array(this.dim);
        this.gradients.set(tokenId, grad);
      }
      return grad;
    }

    clearGradients() {
      for (const grad of this.gradients.values()) grad.fill(0);
    }

    applyGradients(lr, optimizer) {
      for (const [tokenId, grad] of this.gradients.entries()) {
        const emb = this.table.get(tokenId);
        if (!emb) continue;
        let hasUpdate = false;
        for (let i = 0; i < this.dim; i++) {
          if (Math.abs(grad[i]) > 1e-8) { hasUpdate = true; break; }
        }
        if (hasUpdate) {
          const effectiveLr = lr * (this.isNovel(tokenId)
            ? CONSTANTS.NOVEL_LR_MULTIPLIER
            : CONSTANTS.FAMILIAR_LR_MULTIPLIER);
          optimizer.updateInPlace(`emb_${tokenId}`, emb, grad, effectiveLr);
        }
      }
    }

    size() { return this.table.size; }

    toJSON() {
      const entries = [];
      for (const [id, emb] of this.table.entries()) {
        entries.push([id, Array.from(emb), this.exposureCounts.get(id) || 0]);
      }
      return entries;
    }

    fromJSON(entries) {
      this.table.clear();
      this.exposureCounts.clear();
      if (!entries) return;
      for (const entry of entries) {
        const [id, arr, exposure] = entry;
        this.table.set(id, new Float32Array(arr));
        this.exposureCounts.set(id, exposure || 0);
      }
    }
  }

  // ========================================================================
  // CAUSAL ATTENTION — OPT-3: Pre-allocated scratch buffers
  // ========================================================================
  class CausalAttention {
    constructor(numHeads, headDim) {
      this.numHeads    = numHeads;
      this.headDim     = headDim;
      this.scale       = 1 / Math.sqrt(headDim);
      this.lastEntropy = 0;
      this._scratch        = new Float32Array(2048);
      this._scratchDScores = new Float32Array(2048);
    }

    forward(Q, K, V, seqLen, qkvDim) {
      const output     = arrayPool.acquire(seqLen * qkvDim);
      let totalEntropy = 0;
      let entropyCount = 0;

      for (let t = 0; t < seqLen; t++) {
        for (let h = 0; h < this.numHeads; h++) {
          const qOffset = t * qkvDim + h * this.headDim;
          const numKeys = t + 1;
          const scores  = this._scratch;
          let maxScore  = -Infinity;

          for (let s = 0; s < numKeys; s++) {
            const kOffset = s * qkvDim + h * this.headDim;
            let d = 0;
            for (let dd = 0; dd < this.headDim; dd++) {
              d += Q[qOffset + dd] * K[kOffset + dd];
            }
            scores[s] = d * this.scale;
            if (scores[s] > maxScore) maxScore = scores[s];
          }

          if (!isFinite(maxScore)) maxScore = 0;
          let sumExp = 0;
          for (let s = 0; s < numKeys; s++) {
            scores[s] = Math.exp(Math.max(-CONSTANTS.LOGIT_CLAMP,
              Math.min(CONSTANTS.LOGIT_CLAMP, scores[s] - maxScore)));
            sumExp += scores[s];
          }
          const invSum = 1 / (sumExp + 1e-12);
          let entropy  = 0;
          for (let s = 0; s < numKeys; s++) {
            scores[s] *= invSum;
            if (scores[s] > 1e-10) entropy -= scores[s] * Math.log(scores[s]);
          }
          totalEntropy += entropy;
          entropyCount++;

          const outOffset = t * qkvDim + h * this.headDim;
          for (let dd = 0; dd < this.headDim; dd++) {
            let sum = 0;
            for (let s = 0; s < numKeys; s++) {
              sum += scores[s] * V[s * qkvDim + h * this.headDim + dd];
            }
            output[outOffset + dd] = sum;
          }
        }
      }

      this.lastEntropy = entropyCount > 0 ? totalEntropy / entropyCount : 0;
      return output;
    }

    backward(dOutput, Q, K, V, seqLen, qkvDim) {
      const dQ = arrayPool.acquire(seqLen * qkvDim);
      const dK = arrayPool.acquire(seqLen * qkvDim);
      const dV = arrayPool.acquire(seqLen * qkvDim);

      for (let t = 0; t < seqLen; t++) {
        const numKeys = t + 1;
        for (let h = 0; h < this.numHeads; h++) {
          const qOffset = t * qkvDim + h * this.headDim;
          const scores  = this._scratch;
          const dScores = this._scratchDScores;
          let maxScore  = -Infinity;

          for (let s = 0; s < numKeys; s++) {
            const kOffset = s * qkvDim + h * this.headDim;
            let d = 0;
            for (let dd = 0; dd < this.headDim; dd++) {
              d += Q[qOffset + dd] * K[kOffset + dd];
            }
            scores[s] = d * this.scale;
            if (scores[s] > maxScore) maxScore = scores[s];
          }

          if (!isFinite(maxScore)) maxScore = 0;
          let sumExp = 0;
          for (let s = 0; s < numKeys; s++) {
            scores[s] = Math.exp(Math.max(-CONSTANTS.LOGIT_CLAMP,
              Math.min(CONSTANTS.LOGIT_CLAMP, scores[s] - maxScore)));
            sumExp += scores[s];
          }
          const invSum = 1 / (sumExp + 1e-12);
          for (let s = 0; s < numKeys; s++) scores[s] *= invSum;

          const outOffset = t * qkvDim + h * this.headDim;

          for (let s = 0; s < numKeys; s++) {
            const vOffset = s * qkvDim + h * this.headDim;
            let dw = 0;
            for (let dd = 0; dd < this.headDim; dd++) {
              dV[vOffset + dd] += scores[s] * dOutput[outOffset + dd];
              dw += dOutput[outOffset + dd] * V[vOffset + dd];
            }
            dScores[s] = dw;
          }

          let sumDS = 0;
          for (let s = 0; s < numKeys; s++) sumDS += scores[s] * dScores[s];
          for (let s = 0; s < numKeys; s++) {
            dScores[s] = scores[s] * (dScores[s] - sumDS) * this.scale;
          }

          for (let s = 0; s < numKeys; s++) {
            const kOffset = s * qkvDim + h * this.headDim;
            const ds      = dScores[s];
            for (let dd = 0; dd < this.headDim; dd++) {
              dQ[qOffset + dd] += ds * K[kOffset + dd];
              dK[kOffset + dd] += ds * Q[qOffset + dd];
            }
          }
        }
      }

      return { dQ, dK, dV };
    }
  }

  // ========================================================================
  // MICRO-FFN ENSEMBLE — OPT-5: Pre-allocated scratch buffers
  // ========================================================================
  class MicroFFNEnsemble {
    constructor(hiddenDim, ffDim,
                numExperts    = CONSTANTS.NUM_MICRO_FFN,
                activeExperts = CONSTANTS.ACTIVE_MICRO_FFN,
                modelNumLayers = 1) {
      this.hiddenDim       = hiddenDim;
      this.ffDim           = ffDim;
      this.expertDim       = Math.max(1, Math.floor(ffDim / numExperts));
      this.numExperts      = numExperts;
      this.activeExperts   = activeExperts;
      this._modelNumLayers = modelNumLayers;

      this.routerW = new Float32Array(numExperts * hiddenDim);
      this._initXavier(this.routerW, hiddenDim, numExperts, 1.0);

      this.experts = [];
      const rs = this._residualScale();
      for (let e = 0; e < numExperts; e++) {
        this.experts[e] = {
          W1: new Float32Array(this.expertDim * hiddenDim),
          b1: new Float32Array(this.expertDim),
          W2: new Float32Array(hiddenDim * this.expertDim),
          b2: new Float32Array(hiddenDim)
        };
        this._initXavier(this.experts[e].W1, hiddenDim, this.expertDim, 1.0);
        this._initXavier(this.experts[e].W2, this.expertDim, hiddenDim, rs);
      }

      this.lastRouting     = null;
      this.lastActivations = null;

      this._actScratch       = new Float32Array(this.expertDim);
      this._dActScratch      = new Float32Array(this.expertDim);
      this._expertOutScratch = new Float32Array(hiddenDim);
    }

    _residualScale() {
      return 1.0 / Math.sqrt(2 * Math.max(this._modelNumLayers, 1));
    }

    _initXavier(arr, fanIn, fanOut, scaleMul = 1.0) {
      const std   = Math.sqrt(2 / (fanIn + fanOut)) * scaleMul;
      const clamp = 3 * std;
      for (let i = 0; i < arr.length; i++) {
        let v;
        do {
          v = (Math.random() - 0.5) * 2 * std;
        } while (Math.abs(v) > clamp || v === 0);
        arr[i] = v;
      }
    }

    route(x, offset) {
      const logits  = new Float32Array(this.numExperts);
      for (let e = 0; e < this.numExperts; e++) {
        let score = 0;
        for (let i = 0; i < this.hiddenDim; i++) {
          score += this.routerW[e * this.hiddenDim + i] * x[offset + i];
        }
        logits[e] = score;
      }
      const indices = Array.from({ length: this.numExperts }, (_, i) => i);
      indices.sort((a, b) => logits[b] - logits[a]);
      const topK     = indices.slice(0, this.activeExperts);
      const maxLogit = isFinite(logits[topK[0]]) ? logits[topK[0]] : 0;
      let sumExp     = 0;
      const weights  = new Float32Array(this.activeExperts);
      for (let i = 0; i < this.activeExperts; i++) {
        weights[i] = Math.exp(Math.max(-CONSTANTS.LOGIT_CLAMP,
          Math.min(CONSTANTS.LOGIT_CLAMP, logits[topK[i]] - maxLogit)));
        sumExp += weights[i];
      }
      for (let i = 0; i < this.activeExperts; i++) {
        weights[i] /= (sumExp + 1e-12);
      }
      return { experts: topK, weights, logits };
    }

    forward(input, seqLen) {
      const output         = arrayPool.acquire(seqLen * this.hiddenDim);
      this.lastRouting     = [];
      this.lastActivations = [];

      for (let t = 0; t < seqLen; t++) {
        const inOff   = t * this.hiddenDim;
        const outOff  = t * this.hiddenDim;
        const routing = this.route(input, inOff);
        this.lastRouting[t]  = routing;
        const tokenActs      = [];

        for (let e = 0; e < routing.experts.length; e++) {
          const expertIdx = routing.experts[e];
          const weight    = routing.weights[e];
          const expert    = this.experts[expertIdx];
          const act       = this._actScratch;

          for (let i = 0; i < this.expertDim; i++) {
            let sum = expert.b1[i];
            for (let j = 0; j < this.hiddenDim; j++) {
              sum += expert.W1[i * this.hiddenDim + j] * input[inOff + j];
            }
            act[i] = sum > 0 ? sum : 0;
          }

          const actSnapshot = new Float32Array(act);
          tokenActs.push({ expertIdx, act: actSnapshot, weight });

          for (let i = 0; i < this.hiddenDim; i++) {
            let sum = expert.b2[i] * weight;
            for (let j = 0; j < this.expertDim; j++) {
              sum += expert.W2[i * this.expertDim + j] * act[j] * weight;
            }
            output[outOff + i] += sum;
          }
        }
        this.lastActivations[t] = tokenActs;
      }
      return output;
    }

    backward(dOutput, input, seqLen) {
      if (!this.lastActivations || !this.lastRouting) {
        const dInput = arrayPool.acquire(seqLen * this.hiddenDim);
        return {
          dInput,
          dRouterW: arrayPool.acquire(this.numExperts * this.hiddenDim),
          dExperts: this.experts.map(() => ({
            dW1: arrayPool.acquire(this.expertDim * this.hiddenDim),
            db1: arrayPool.acquire(this.expertDim),
            dW2: arrayPool.acquire(this.hiddenDim * this.expertDim),
            db2: arrayPool.acquire(this.hiddenDim)
          }))
        };
      }

      const dInput   = arrayPool.acquire(seqLen * this.hiddenDim);
      const dRouterW = arrayPool.acquire(this.numExperts * this.hiddenDim);

      const dExperts = this.experts.map(() => ({
        dW1: arrayPool.acquire(this.expertDim * this.hiddenDim),
        db1: arrayPool.acquire(this.expertDim),
        dW2: arrayPool.acquire(this.hiddenDim * this.expertDim),
        db2: arrayPool.acquire(this.hiddenDim)
      }));

      const GC = CONSTANTS.GRAD_VALUE_CLAMP;

      for (let t = 0; t < seqLen; t++) {
        const inOff       = t * this.hiddenDim;
        const outOff      = t * this.hiddenDim;
        const activations = this.lastActivations[t];
        const routing     = this.lastRouting[t];
        if (!activations || !routing) continue;

        const { experts: topK, weights } = routing;
        const expertOuts = [];

        for (let e = 0; e < activations.length; e++) {
          const { expertIdx, act, weight } = activations[e];
          const expert   = this.experts[expertIdx];
          const dExp     = dExperts[expertIdx];
          const dAct     = this._dActScratch;
          const expertOut = this._expertOutScratch;

          dAct.fill(0);
          expertOut.fill(0);

          for (let i = 0; i < this.hiddenDim; i++) {
            let s = expert.b2[i];
            for (let j = 0; j < this.expertDim; j++) {
              s += expert.W2[i * this.expertDim + j] * act[j];
            }
            expertOut[i] = s;
          }
          expertOuts.push(new Float32Array(expertOut));

          for (let i = 0; i < this.hiddenDim; i++) {
            const dOut = dOutput[outOff + i];
            if (!isFinite(dOut)) continue;
            dExp.db2[i] += dOut * weight;
            for (let j = 0; j < this.expertDim; j++) {
              const dW2val = dOut * act[j] * weight;
              dExp.dW2[i * this.expertDim + j] +=
                Math.max(-GC, Math.min(GC, dW2val));
              const backW2 = expert.W2[i * this.expertDim + j] * dOut * weight;
              dAct[j] += isFinite(backW2) ? backW2 : 0;
            }
          }

          for (let j = 0; j < this.expertDim; j++) {
            if (act[j] <= 0) dAct[j] = 0;
            dExp.db1[j] += dAct[j];
            for (let k = 0; k < this.hiddenDim; k++) {
              const dW1val = dAct[j] * input[inOff + k];
              dExp.dW1[j * this.hiddenDim + k] +=
                Math.max(-GC, Math.min(GC, dW1val));
              const backW1 = expert.W1[j * this.hiddenDim + k] * dAct[j];
              dInput[inOff + k] += isFinite(backW1) ? backW1 : 0;
            }
          }
        }

        const numActive = activations.length;
        const dLdWeight = new Float32Array(numActive);
        for (let e = 0; e < numActive; e++) {
          let dp = 0;
          for (let i = 0; i < this.hiddenDim; i++) {
            dp += dOutput[outOff + i] * expertOuts[e][i];
          }
          dLdWeight[e] = isFinite(dp) ? dp : 0;
        }

        let sumWdL = 0;
        for (let e = 0; e < numActive; e++) sumWdL += weights[e] * dLdWeight[e];
        for (let e = 0; e < numActive; e++) {
          const expertIdx = topK[e];
          const dLogit    = weights[e] * (dLdWeight[e] - sumWdL);
          if (!isFinite(dLogit)) continue;
          for (let k = 0; k < this.hiddenDim; k++) {
            dRouterW[expertIdx * this.hiddenDim + k] +=
              Math.max(-GC, Math.min(GC, dLogit * input[inOff + k]));
            const rBack = this.routerW[expertIdx * this.hiddenDim + k] * dLogit;
            dInput[inOff + k] += isFinite(rBack) ? rBack : 0;
          }
        }
      }

      this.lastRouting     = null;
      this.lastActivations = null;
      return { dInput, dRouterW, dExperts };
    }

    applyGradients(gradients, lr, opt) {
      opt.updateInPlace('routerW', this.routerW, gradients.dRouterW, lr);
      for (let e = 0; e < this.numExperts; e++) {
        const dExp = gradients.dExperts[e];
        opt.updateInPlace(`expert_${e}_W1`, this.experts[e].W1, dExp.dW1, lr);
        opt.updateInPlace(`expert_${e}_b1`, this.experts[e].b1, dExp.db1, lr);
        opt.updateInPlace(`expert_${e}_W2`, this.experts[e].W2, dExp.dW2, lr);
        opt.updateInPlace(`expert_${e}_b2`, this.experts[e].b2, dExp.db2, lr);
        arrayPool.release(dExp.dW1);
        arrayPool.release(dExp.db1);
        arrayPool.release(dExp.dW2);
        arrayPool.release(dExp.db2);
      }
      arrayPool.release(gradients.dRouterW);
    }

    toJSON() {
      return {
        hiddenDim: this.hiddenDim,
        ffDim:     this.ffDim,
        routerW:   Array.from(this.routerW),
        experts:   this.experts.map(e => ({
          W1: Array.from(e.W1), b1: Array.from(e.b1),
          W2: Array.from(e.W2), b2: Array.from(e.b2)
        }))
      };
    }

    fromJSON(data) {
      this.routerW = new Float32Array(data.routerW);
      for (let e = 0; e < data.experts.length && e < this.numExperts; e++) {
        this.experts[e] = {
          W1: new Float32Array(data.experts[e].W1),
          b1: new Float32Array(data.experts[e].b1),
          W2: new Float32Array(data.experts[e].W2),
          b2: new Float32Array(data.experts[e].b2)
        };
      }
      this._actScratch       = new Float32Array(this.expertDim);
      this._dActScratch      = new Float32Array(this.expertDim);
      this._expertOutScratch = new Float32Array(this.hiddenDim);
    }
  }

  // ========================================================================
  // SELECTIVE GRADIENT TRACKER
  // ========================================================================
  class SelectiveGradientTracker {
    constructor() {
      this.hotParams = new Map();
      this.stepCount = 0;
      this.threshold = CONSTANTS.GRADIENT_THRESHOLD;
    }

    recordGradient(name, gradient) {
      let maxGrad = 0;
      for (let i = 0; i < gradient.length; i++) {
        const ag = Math.abs(gradient[i]);
        if (isFinite(ag) && ag > maxGrad) maxGrad = ag;
      }
      if (maxGrad > this.threshold) {
        this.hotParams.set(name, { maxGrad, lastUpdate: this.stepCount });
        PerfMonitor.recordCacheHit();
      } else {
        PerfMonitor.recordCacheMiss();
      }
    }

    step() {
      this.stepCount++;
      if (this.stepCount % 100 === 0) {
        for (const [name, info] of this.hotParams.entries()) {
          if (this.stepCount - info.lastUpdate > 20) this.hotParams.delete(name);
        }
      }
    }

    reset() { this.hotParams.clear(); this.stepCount = 0; }
  }

  const gradientTracker = new SelectiveGradientTracker();

  // ========================================================================
  // ADAM OPTIMIZER
  // ========================================================================
  class AdamOptimizer {
    constructor() {
      this.beta1  = 0.9;
      this.beta2  = 0.999;
      this.eps    = 1e-8;
      this.t      = 0;
      this.b1p    = 1;
      this.b2p    = 1;
      this.states = new Map();
      this._outProjRowStates = new Map();
      this._hiddenDimRef     = 0;
    }

    register(name, param) {
      if (name === 'outputProj') return;
      if (!this.states.has(name) || this.states.get(name).m.length !== param.length) {
        this.states.set(name, {
          m: new Float32Array(param.length),
          v: new Float32Array(param.length)
        });
      }
    }

    step() {
      this.t++;
      this.b1p *= this.beta1;
      this.b2p *= this.beta2;
      gradientTracker.step();
    }

    updateInPlace(name, param, grad, lr) {
      if (name === 'outputProj') {
        const H = this._hiddenDimRef || param.length;
        const V = Math.floor(param.length / H);
        for (let v = 0; v < V; v++) {
          this._updateOutputProjRow(v, param, grad, v * H, H, lr);
        }
        gradientTracker.recordGradient(name, grad);
        return;
      }

      let state = this.states.get(name);
      if (!state || state.m.length !== param.length) {
        state = {
          m: new Float32Array(param.length),
          v: new Float32Array(param.length)
        };
        this.states.set(name, state);
      }

      const { m, v } = state;
      const bc1      = 1 - this.b1p;
      const bc2      = 1 - this.b2p;
      const stepSize = lr * Math.sqrt(bc2) / (bc1 + 1e-12);

      for (let i = 0; i < param.length; i++) {
        const gi = grad[i];
        if (!Number.isFinite(gi)) continue;
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * gi;
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * gi * gi;
        const update = stepSize * m[i] / (Math.sqrt(v[i]) + this.eps);
        if (Number.isFinite(update)) param[i] -= update;
      }

      gradientTracker.recordGradient(name, grad);
    }

    update(name, param, grad, lr) { this.updateInPlace(name, param, grad, lr); }

    _updateOutputProjRow(vocabId, param, grad, offset, H, lr) {
      let rowState = this._outProjRowStates.get(vocabId);
      if (!rowState) {
        rowState = { m: new Float32Array(H), v: new Float32Array(H) };
        this._outProjRowStates.set(vocabId, rowState);
      }
      const { m, v } = rowState;
      const bc1      = 1 - this.b1p;
      const bc2      = 1 - this.b2p;
      const stepSize = lr * Math.sqrt(bc2) / (bc1 + 1e-12);
      for (let i = 0; i < H; i++) {
        const gi = grad[offset + i];
        if (!Number.isFinite(gi)) continue;
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * gi;
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * gi * gi;
        const update = stepSize * m[i] / (Math.sqrt(v[i]) + this.eps);
        if (Number.isFinite(update)) param[offset + i] -= update;
      }
    }

    setHiddenDim(h) { this._hiddenDimRef = h; }

    reset() {
      this.t = 0; this.b1p = 1; this.b2p = 1;
      for (const state of this.states.values()) {
        state.m.fill(0); state.v.fill(0);
      }
      this._outProjRowStates.clear();
      gradientTracker.reset();
    }

    getState() {
      const states = {};
      for (const [name, { m, v }] of this.states) {
        states[name] = { m: Array.from(m), v: Array.from(v) };
      }
      const outProjRows = {};
      for (const [id, { m, v }] of this._outProjRowStates) {
        outProjRows[id] = { m: Array.from(m), v: Array.from(v) };
      }
      return { t: this.t, b1p: this.b1p, b2p: this.b2p, states, outProjRows };
    }

    loadState(saved) {
      if (!saved) return;
      this.t   = saved.t   || 0;
      this.b1p = saved.b1p || 1;
      this.b2p = saved.b2p || 1;
      if (saved.states) {
        for (const [name, data] of Object.entries(saved.states)) {
          this.states.set(name, {
            m: new Float32Array(data.m),
            v: new Float32Array(data.v)
          });
        }
      }
      if (saved.outProjRows) {
        for (const [id, data] of Object.entries(saved.outProjRows)) {
          this._outProjRowStates.set(Number(id), {
            m: new Float32Array(data.m),
            v: new Float32Array(data.v)
          });
        }
      }
    }
  }

  const optimizer = new AdamOptimizer();

  // ========================================================================
  // NUMERICAL UTILITIES — OPT-1: Unrolled dot() and matVecMul()
  // ========================================================================
  function welfordMeanVar(arr, offset, length) {
    if (length === 0) return { mean: 0, variance: 0 };
    let mean = 0, m2 = 0;
    for (let i = 0; i < length; i++) {
      const x     = arr[offset + i];
      const delta = x - mean;
      mean += delta / (i + 1);
      m2   += delta * (x - mean);
    }
    return { mean, variance: length > 1 ? m2 / length : 0 };
  }

  function layerNormForward(x, xOff, gamma, beta, out, outOff,
                            xhat, xhatOff, dim) {
    const { mean, variance } = welfordMeanVar(x, xOff, dim);
    const invStd = 1 / Math.sqrt(variance + CONSTANTS.LN_EPSILON);
    for (let i = 0; i < dim; i++) {
      const xh          = (x[xOff + i] - mean) * invStd;
      xhat[xhatOff + i] = xh;
      out[outOff + i]   = gamma[i] * xh + beta[i];
    }
    return { mean, invStd };
  }

  function layerNormBackward(dOut, dOutOff, xhat, xhatOff, gamma,
                             invStd, dIn, dInOff, dGamma, dBeta, dim) {
    let sumDyGamma = 0, sumDyGammaXhat = 0;
    for (let i = 0; i < dim; i++) {
      const dy  = dOut[dOutOff + i];
      const xh  = xhat[xhatOff + i];
      dGamma[i] += dy * xh;
      dBeta[i]  += dy;
      const dyg  = dy * gamma[i];
      sumDyGamma     += dyg;
      sumDyGammaXhat += dyg * xh;
    }
    const invN = 1 / dim;
    for (let i = 0; i < dim; i++) {
      const dyg = dOut[dOutOff + i] * gamma[i];
      const xh  = xhat[xhatOff + i];
      dIn[dInOff + i] = invStd * (
        dyg - sumDyGamma * invN - xh * sumDyGammaXhat * invN
      );
    }
  }

  // OPT-1: Unrolled x4
  function matVecMul(matrix, vec, vecOff, rows, cols, out, outOff) {
    const limit = cols - (cols % 4);
    for (let i = 0; i < rows; i++) {
      let sum      = 0;
      const rowOff = i * cols;
      let j        = 0;
      for (; j < limit; j += 4) {
        sum += matrix[rowOff + j]     * vec[vecOff + j]     +
               matrix[rowOff + j + 1] * vec[vecOff + j + 1] +
               matrix[rowOff + j + 2] * vec[vecOff + j + 2] +
               matrix[rowOff + j + 3] * vec[vecOff + j + 3];
      }
      for (; j < cols; j++) sum += matrix[rowOff + j] * vec[vecOff + j];
      out[outOff + i] = sum;
    }
  }

  // OPT-1: Unrolled x4
  function dot(a, aOff, b, bOff, length) {
    let sum     = 0;
    const limit = length - (length % 4);
    let i       = 0;
    for (; i < limit; i += 4) {
      sum += a[aOff + i]     * b[bOff + i]     +
             a[aOff + i + 1] * b[bOff + i + 1] +
             a[aOff + i + 2] * b[bOff + i + 2] +
             a[aOff + i + 3] * b[bOff + i + 3];
    }
    for (; i < length; i++) sum += a[aOff + i] * b[bOff + i];
    return sum;
  }

  function clipGlobalNorm(gradients, maxNorm) {
    let sumSq = 0;
    for (const grad of gradients) {
      if (!grad) continue;
      for (let i = 0; i < grad.length; i++) {
        if (!isFinite(grad[i])) {
          grad[i] = 0;
          PerfMonitor.recordNanGrad();
        } else {
          sumSq += grad[i] * grad[i];
        }
      }
    }
    const norm = Math.sqrt(sumSq);
    if (!isFinite(norm) || norm <= maxNorm) return;
    PerfMonitor.recordGradClip();
    const scale = maxNorm / (norm + 1e-8);
    for (const grad of gradients) {
      if (!grad) continue;
      for (let i = 0; i < grad.length; i++) grad[i] *= scale;
    }
  }

  function initXavier(arr, fanIn, fanOut, scaleMul = 1.0) {
    const std   = Math.sqrt(2 / (fanIn + fanOut)) * scaleMul;
    const clamp = 3 * std;
    for (let i = 0; i < arr.length; i++) {
      let v;
      do { v = (Math.random() - 0.5) * 2 * std; } while (Math.abs(v) > clamp);
      arr[i] = v !== 0 ? v : std * 0.01;
    }
  }

  function sanitizeArray(arr, length) {
    for (let i = 0; i < length; i++) {
      if (!isFinite(arr[i])) arr[i] = 0;
    }
  }

  function clampArray(arr, length, bound) {
    for (let i = 0; i < length; i++) {
      if      (arr[i] >  bound) arr[i] =  bound;
      else if (arr[i] < -bound) arr[i] = -bound;
    }
  }

  // ========================================================================
  // ASYNC UTILITIES
  // ========================================================================
  let lastYieldTime = 0;

  async function maybeYield() {
    const now = performance.now();
    if (now - lastYieldTime > CONSTANTS.YIELD_INTERVAL_MS) {
      await new Promise(r => setTimeout(r, 0));
      lastYieldTime = performance.now();
    }
  }

  function yieldToBrowser() {
    return new Promise(r => setTimeout(r, 0));
  }

  // ========================================================================
  // MODEL STATE
  // ========================================================================
  let embeddingDim       = 64;
  let hiddenDim          = 128;
  let contextWindow      = 64;
  let chunkWords         = 64;
  let numHeads           = 4;
  let numLayers          = 2;
  let ffDim              = 512;
  let learningRate       = 0.0005;
  let maxVocabSize       = 50000;
  let maxPos             = 2048;
  let maxSeqLen          = 128;
  let trainLossPositions = 8;
  let currentFileName    = '';

  let _ffnBuiltHiddenDim = 0;
  let _ffnBuiltFfDim     = 0;

  function getQkvDim() {
    const effectiveHeads = resolveNumHeads(numHeads, hiddenDim);
    return effectiveHeads * Math.floor(hiddenDim / effectiveHeads);
  }

  const vocab    = new Map();
  const id2word  = [];
  const wordFreq = new Map();
  let vocabSize  = 0;
  const memory   = [];

  let embeddings       = null;
  let posEmbeddings    = null;
  let inputProjection  = null;
  let outputProjection = null;

  let attentionLayers = [];
  let ffnLayers       = [];
  let ln1Gamma = [], ln1Beta = [];
  let ln2Gamma = [], ln2Beta = [];
  let Wq = [], Wk = [], Wv = [], WattnOut = [];

  let isTraining         = false;
  let shouldStopTraining = false;

  // OPT-4: Module-scope scratch for sampleLogits
  let _sampleScratch     = null;
  let _sampleScratchSize = 0;

  function stopTraining()      { shouldStopTraining = true; }
  function getTrainingStatus() { return { isTraining, shouldStopTraining }; }

  // ========================================================================
  // MODEL INITIALISATION
  // ========================================================================
  function initSpecialTokens() {
    for (const token of [
      '<PAD>', '<BOS>', '<EOS>', '<UNK>',
      '<USER>', '<ASSISTANT>', '<SEP>'
    ]) {
      if (!vocab.has(token)) {
        vocab.set(token, vocabSize);
        id2word[vocabSize] = token;
        wordFreq.set(token, 1);
        vocabSize++;
      }
    }
  }

  function resolveNumHeads(requested, hidden) {
    for (let h = requested; h >= 1; h--) {
      if (hidden % h === 0) return h;
    }
    return 1;
  }

  function allocateModel() {
    try {
      initSpecialTokens();
      ffDim = hiddenDim * 4;

      const effectiveHeads = resolveNumHeads(numHeads, hiddenDim);
      if (effectiveHeads !== numHeads) {
        console.warn(
          `[Model] numHeads(${numHeads}) does not divide hiddenDim(${hiddenDim}). ` +
          `Auto-corrected to numHeads=${effectiveHeads}.`
        );
      }
      const headDim = Math.floor(hiddenDim / effectiveHeads);
      const qkvDim  = effectiveHeads * headDim;

      optimizer.setHiddenDim(hiddenDim);

      if (!embeddings || embeddings.dim !== embeddingDim) {
        const oldTable    = embeddings ? embeddings.table          : null;
        const oldExposure = embeddings ? embeddings.exposureCounts : null;
        embeddings = new SparseEmbeddings(embeddingDim);
        if (oldTable) {
          for (const [id, emb] of oldTable.entries()) {
            if (emb.length === embeddingDim) {
              embeddings.set(id, new Float32Array(emb));
              if (oldExposure && oldExposure.has(id))
                embeddings.exposureCounts.set(id, oldExposure.get(id));
            }
          }
        }
      }

      if (!posEmbeddings || posEmbeddings.length !== maxPos * hiddenDim) {
        const old = posEmbeddings;
        posEmbeddings = new Float32Array(maxPos * hiddenDim);
        if (old) {
          posEmbeddings.set(old.subarray(0,
            Math.min(old.length, posEmbeddings.length)));
        }
        const startI = old ? old.length : 0;
        for (let i = startI; i < posEmbeddings.length; i++) {
          posEmbeddings[i] = (Math.random() - 0.5) * 0.02;
        }
      }

      if (!inputProjection ||
          inputProjection.length !== hiddenDim * embeddingDim) {
        inputProjection = new Float32Array(hiddenDim * embeddingDim);
        initXavier(inputProjection, embeddingDim, hiddenDim);
      }

      const neededSize = vocabSize * hiddenDim;
      if (!outputProjection || outputProjection.length < neededSize) {
        const newProj      = new Float32Array(neededSize);
        if (outputProjection) {
          const oldHiddenDim = _ffnBuiltHiddenDim > 0
            ? _ffnBuiltHiddenDim : hiddenDim;
          const oldVocabRows = Math.floor(outputProjection.length / oldHiddenDim);
          const copyRows     = Math.min(oldVocabRows, vocabSize);
          const copyCols     = Math.min(oldHiddenDim, hiddenDim);
          for (let v = 0; v < copyRows; v++) {
            for (let h = 0; h < copyCols; h++) {
              newProj[v * hiddenDim + h] =
                outputProjection[v * oldHiddenDim + h];
            }
          }
          for (let v = copyRows; v < vocabSize; v++) {
            for (let h = 0; h < hiddenDim; h++) {
              newProj[v * hiddenDim + h] = (Math.random() - 0.5) * 0.02;
            }
          }
        } else {
          for (let v = 0; v < vocabSize; v++) {
            for (let h = 0; h < hiddenDim; h++) {
              newProj[v * hiddenDim + h] = (Math.random() - 0.5) * 0.02;
            }
          }
        }
        outputProjection = newProj;
      }

      const qkvSize       = qkvDim * hiddenDim;
      const attnOutSize   = hiddenDim * qkvDim;
      const residualScale = 1.0 / Math.sqrt(2 * Math.max(numLayers, 1));

      const ffnDimsChanged =
        hiddenDim !== _ffnBuiltHiddenDim ||
        ffDim     !== _ffnBuiltFfDim;

      for (let l = 0; l < numLayers; l++) {
        if (!attentionLayers[l] ||
            attentionLayers[l].numHeads !== effectiveHeads ||
            attentionLayers[l].headDim  !== headDim) {
          attentionLayers[l] = new CausalAttention(effectiveHeads, headDim);
        }

        if (!ffnLayers[l] || ffnDimsChanged) {
          ffnLayers[l] = new MicroFFNEnsemble(
            hiddenDim, ffDim,
            CONSTANTS.NUM_MICRO_FFN,
            CONSTANTS.ACTIVE_MICRO_FFN,
            numLayers
          );
        }

        if (!Wq[l] || Wq[l].length !== qkvSize) {
          Wq[l] = new Float32Array(qkvSize);
          initXavier(Wq[l], hiddenDim, qkvDim);
        }
        if (!Wk[l] || Wk[l].length !== qkvSize) {
          Wk[l] = new Float32Array(qkvSize);
          initXavier(Wk[l], hiddenDim, qkvDim);
        }
        if (!Wv[l] || Wv[l].length !== qkvSize) {
          Wv[l] = new Float32Array(qkvSize);
          initXavier(Wv[l], hiddenDim, qkvDim);
        }
        if (!WattnOut[l] || WattnOut[l].length !== attnOutSize) {
          WattnOut[l] = new Float32Array(attnOutSize);
          initXavier(WattnOut[l], qkvDim, hiddenDim, residualScale);
        }

        if (!ln1Gamma[l] || ln1Gamma[l].length !== hiddenDim) {
          ln1Gamma[l] = new Float32Array(hiddenDim);
          ln1Gamma[l].fill(1);
        }
        if (!ln1Beta[l]  || ln1Beta[l].length  !== hiddenDim)
          ln1Beta[l]  = new Float32Array(hiddenDim);
        if (!ln2Gamma[l] || ln2Gamma[l].length !== hiddenDim) {
          ln2Gamma[l] = new Float32Array(hiddenDim);
          ln2Gamma[l].fill(1);
        }
        if (!ln2Beta[l]  || ln2Beta[l].length  !== hiddenDim)
          ln2Beta[l]  = new Float32Array(hiddenDim);
      }

      _ffnBuiltHiddenDim = hiddenDim;
      _ffnBuiltFfDim     = ffDim;

      // OPT-4: Keep sample scratch in sync with vocab size
      if (_sampleScratchSize < vocabSize) {
        _sampleScratch     = new Float32Array(vocabSize);
        _sampleScratchSize = vocabSize;
      }

      registerOptimizer();
      console.log(
        `[Model] Allocated: vocab=${vocabSize}, layers=${numLayers}, ` +
        `emb=${embeddings.size()}, heads=${effectiveHeads}, qkvDim=${qkvDim}`
      );
    } catch (e) {
      const msg = e instanceof Error ? e.message : JSON.stringify(e);
      console.error('[Model] Allocation failed:', msg, e);
      throw e;
    }
  }

  function registerOptimizer() {
    optimizer.register('inputProj', inputProjection);
    optimizer.register('posEmb',    posEmbeddings);
    for (let l = 0; l < numLayers; l++) {
      optimizer.register(`Wq_${l}`,       Wq[l]);
      optimizer.register(`Wk_${l}`,       Wk[l]);
      optimizer.register(`Wv_${l}`,       Wv[l]);
      optimizer.register(`WattnOut_${l}`, WattnOut[l]);
      optimizer.register(`ln1Gamma_${l}`, ln1Gamma[l]);
      optimizer.register(`ln1Beta_${l}`,  ln1Beta[l]);
      optimizer.register(`ln2Gamma_${l}`, ln2Gamma[l]);
      optimizer.register(`ln2Beta_${l}`,  ln2Beta[l]);
    }
  }

  // ========================================================================
  // FORWARD PASS — OPT-2: Fused QKV projection
  // ========================================================================
  function forwardFull(tokenIds) {
    const seqLen = tokenIds.length;
    if (seqLen <= 0 || seqLen > maxSeqLen)
      throw new Error(`Invalid seqLen: ${seqLen}`);

    const headDim = attentionLayers[0]
      ? attentionLayers[0].headDim
      : Math.floor(hiddenDim / resolveNumHeads(numHeads, hiddenDim));
    const qkvDim  = attentionLayers[0]
      ? attentionLayers[0].numHeads * headDim
      : getQkvDim();

    const cache = {
      seqLen,
      vocabSize,
      tokenIds:     Array.from(tokenIds),
      hiddenStates: [],
      Q:            [],
      K:            [],
      V:            [],
      attnOutRaw:   [],
      res1:         [],
      ln1Xhat:      [],
      ln2Xhat:      [],
      ln1Stats:     [],
      ln2Stats:     [],
      ffnInput:     [],
      noveltyScore: patternTracker.getNoveltyScore(tokenIds),
      attentionEntropy: 0,
      qkvDim
    };

    let h = arrayPool.acquire(seqLen * hiddenDim);

    for (let t = 0; t < seqLen; t++) {
      const tokenId = tokenIds[t];
      const emb     = embeddings.get(tokenId);
      const hOff    = t * hiddenDim;
      for (let i = 0; i < hiddenDim; i++) {
        let sum = posEmbeddings[t * hiddenDim + i];
        for (let j = 0; j < embeddingDim; j++) {
          sum += inputProjection[i * embeddingDim + j] * emb[j];
        }
        h[hOff + i] = sum;
      }
    }

    sanitizeArray(h, seqLen * hiddenDim);
    clampArray(h, seqLen * hiddenDim, CONSTANTS.HIDDEN_CLAMP);
    cache.hiddenStates.push(new Float32Array(h));

    let totalEntropy = 0;

    for (let l = 0; l < numLayers; l++) {
      const Q = arrayPool.acquire(seqLen * qkvDim);
      const K = arrayPool.acquire(seqLen * qkvDim);
      const V = arrayPool.acquire(seqLen * qkvDim);

      // OPT-2: Fused QKV — one pass over h per token
      const fusedLimit = hiddenDim - (hiddenDim % 4);
      for (let t = 0; t < seqLen; t++) {
        const hOff = t * hiddenDim;
        const qOff = t * qkvDim;
        for (let r = 0; r < qkvDim; r++) {
          const wOff = r * hiddenDim;
          let sq = 0, sk = 0, sv = 0;
          let j  = 0;
          for (; j < fusedLimit; j += 4) {
            const h0 = h[hOff + j],     h1 = h[hOff + j + 1],
                  h2 = h[hOff + j + 2], h3 = h[hOff + j + 3];
            sq += Wq[l][wOff + j]     * h0 + Wq[l][wOff + j + 1] * h1 +
                  Wq[l][wOff + j + 2] * h2 + Wq[l][wOff + j + 3] * h3;
            sk += Wk[l][wOff + j]     * h0 + Wk[l][wOff + j + 1] * h1 +
                  Wk[l][wOff + j + 2] * h2 + Wk[l][wOff + j + 3] * h3;
            sv += Wv[l][wOff + j]     * h0 + Wv[l][wOff + j + 1] * h1 +
                  Wv[l][wOff + j + 2] * h2 + Wv[l][wOff + j + 3] * h3;
          }
          for (; j < hiddenDim; j++) {
            const hv = h[hOff + j];
            sq += Wq[l][wOff + j] * hv;
            sk += Wk[l][wOff + j] * hv;
            sv += Wv[l][wOff + j] * hv;
          }
          Q[qOff + r] = sq;
          K[qOff + r] = sk;
          V[qOff + r] = sv;
        }
      }

      cache.Q.push(Q);
      cache.K.push(K);
      cache.V.push(V);

      const attnOutRaw = attentionLayers[l].forward(Q, K, V, seqLen, qkvDim);
      totalEntropy    += attentionLayers[l].lastEntropy;
      cache.attnOutRaw.push(attnOutRaw);

      const attnProj = arrayPool.acquire(seqLen * hiddenDim);
      for (let t = 0; t < seqLen; t++) {
        matVecMul(WattnOut[l], attnOutRaw, t * qkvDim,
                  hiddenDim, qkvDim, attnProj, t * hiddenDim);
      }

      const res1Arr = arrayPool.acquire(seqLen * hiddenDim);
      for (let i = 0; i < seqLen * hiddenDim; i++) {
        res1Arr[i] = h[i] + attnProj[i];
      }
      arrayPool.release(attnProj);
      cache.res1.push(res1Arr);

      const ln1Out  = arrayPool.acquire(seqLen * hiddenDim);
      const ln1Xhat = arrayPool.acquire(seqLen * hiddenDim);
      const ln1Stats= [];
      for (let t = 0; t < seqLen; t++) {
        const off = t * hiddenDim;
        ln1Stats.push(layerNormForward(
          res1Arr, off, ln1Gamma[l], ln1Beta[l],
          ln1Out, off, ln1Xhat, off, hiddenDim));
      }
      cache.ln1Xhat.push(ln1Xhat);
      cache.ln1Stats.push(ln1Stats);

      const ffnInputArr = arrayPool.acquire(seqLen * hiddenDim);
      ffnInputArr.set(ln1Out.subarray(0, seqLen * hiddenDim));
      cache.ffnInput.push(ffnInputArr);

      const ffnOut = ffnLayers[l].forward(ln1Out, seqLen);

      const res2 = arrayPool.acquire(seqLen * hiddenDim);
      for (let i = 0; i < seqLen * hiddenDim; i++) {
        res2[i] = res1Arr[i] + ffnOut[i];
      }
      arrayPool.release(ffnOut);
      arrayPool.release(ln1Out);

      const newH    = arrayPool.acquire(seqLen * hiddenDim);
      const ln2Xhat = arrayPool.acquire(seqLen * hiddenDim);
      const ln2Stats= [];
      for (let t = 0; t < seqLen; t++) {
        const off = t * hiddenDim;
        ln2Stats.push(layerNormForward(
          res2, off, ln2Gamma[l], ln2Beta[l],
          newH, off, ln2Xhat, off, hiddenDim));
      }
      cache.ln2Xhat.push(ln2Xhat);
      cache.ln2Stats.push(ln2Stats);
      arrayPool.release(res2);

      sanitizeArray(newH, seqLen * hiddenDim);
      clampArray(newH, seqLen * hiddenDim, CONSTANTS.HIDDEN_CLAMP);

      arrayPool.release(h);
      h = newH;
      cache.hiddenStates.push(new Float32Array(h));
    }

    cache.finalHidden      = h;
    cache.attentionEntropy = totalEntropy / Math.max(numLayers, 1);
    return cache;
  }

  // ========================================================================
  // BACKWARD PASS — OPT-6: Sparse output projection update
  // ========================================================================
  function backwardAndUpdate(cache, lr) {
    const {
      seqLen, tokenIds, hiddenStates,
      Q, K, V, attnOutRaw,
      res1, ln1Xhat, ln1Stats,
      ffnInput, ln2Xhat, ln2Stats,
      finalHidden, noveltyScore, qkvDim
    } = cache;

    const snapVocabSize = cache.vocabSize;

    let adaptiveLr = lr;
    if (noveltyScore > CONSTANTS.NOVELTY_HIGH_THRESHOLD) {
      adaptiveLr *= CONSTANTS.NOVEL_LR_MULTIPLIER;
      PerfMonitor.recordNovel();
    } else if (noveltyScore < CONSTANTS.NOVELTY_LOW_THRESHOLD) {
      adaptiveLr *= CONSTANTS.FAMILIAR_LR_MULTIPLIER;
      PerfMonitor.recordFamiliar();
    }

    const startT = 0;
    let totalLoss = 0;
    let lossTerms = 0;

    const logits      = arrayPool.acquire(snapVocabSize);
    const probs       = arrayPool.acquire(snapVocabSize);
    const dLogits     = arrayPool.acquire(snapVocabSize);
    const dH          = arrayPool.acquire(seqLen * hiddenDim);
    const gOutputProj = arrayPool.acquire(snapVocabSize * hiddenDim);

    // OPT-6: Reusable index buffer for hot vocab rows
    const hotVocabBuf = new Int32Array(snapVocabSize);

    for (let t = startT; t < seqLen - 1; t++) {
      const targetId = tokenIds[t + 1];
      const hOff     = t * hiddenDim;

      const ctxSize = Math.min(3, t + 1);
      const ctx     = tokenIds.slice(t - ctxSize + 1, t + 1);
      conflictTracker.record(ctx, targetId, currentFileName);

      for (let v = 0; v < snapVocabSize; v++) {
        let sum      = 0;
        const rowOff = v * hiddenDim;
        for (let i = 0; i < hiddenDim; i++) {
          sum += outputProjection[rowOff + i] * finalHidden[hOff + i];
        }
        logits[v] = Math.max(-CONSTANTS.LOGIT_CLAMP,
          Math.min(CONSTANTS.LOGIT_CLAMP, sum));
      }

      let maxLogit = -Infinity;
      for (let v = 0; v < snapVocabSize; v++) {
        if (logits[v] > maxLogit) maxLogit = logits[v];
      }
      let sumExp = 0;
      for (let v = 0; v < snapVocabSize; v++) {
        probs[v] = Math.exp(logits[v] - maxLogit);
        sumExp  += probs[v];
      }
      const invSumExp = 1 / (sumExp + 1e-12);
      for (let v = 0; v < snapVocabSize; v++) probs[v] *= invSumExp;

      const targetProb = probs[targetId];
      if (!isFinite(targetProb) || targetProb <= 0) continue;

      totalLoss += -Math.log(Math.max(targetProb, 1e-12));
      lossTerms++;

      for (let v = 0; v < snapVocabSize; v++) dLogits[v] = probs[v];
      dLogits[targetId] -= 1;

      // OPT-6: Collect hot indices
      const SPARSE_THRESH = CONSTANTS.SPARSE_GRAD_THRESHOLD;
      let hotCount = 0;
      for (let v = 0; v < snapVocabSize; v++) {
        if (Math.abs(dLogits[v]) > SPARSE_THRESH) {
          hotVocabBuf[hotCount++] = v;
        }
      }

      // OPT-6: Update gOutputProj only for hot rows, unrolled x4
      const hdLimit = hiddenDim - (hiddenDim % 4);
      for (let hi = 0; hi < hotCount; hi++) {
        const v      = hotVocabBuf[hi];
        const dL     = dLogits[v];
        const rowOff = v * hiddenDim;
        let i        = 0;
        for (; i < hdLimit; i += 4) {
          gOutputProj[rowOff + i]     += dL * finalHidden[hOff + i];
          gOutputProj[rowOff + i + 1] += dL * finalHidden[hOff + i + 1];
          gOutputProj[rowOff + i + 2] += dL * finalHidden[hOff + i + 2];
          gOutputProj[rowOff + i + 3] += dL * finalHidden[hOff + i + 3];
        }
        for (; i < hiddenDim; i++) {
          gOutputProj[rowOff + i] += dL * finalHidden[hOff + i];
        }
      }

      // OPT-6: Accumulate dH only from hot rows, unrolled x4
      for (let hi = 0; hi < hotCount; hi++) {
        const v      = hotVocabBuf[hi];
        const dL     = dLogits[v];
        const rowOff = v * hiddenDim;
        let i        = 0;
        for (; i < hdLimit; i += 4) {
          dH[hOff + i]     += outputProjection[rowOff + i]     * dL;
          dH[hOff + i + 1] += outputProjection[rowOff + i + 1] * dL;
          dH[hOff + i + 2] += outputProjection[rowOff + i + 2] * dL;
          dH[hOff + i + 3] += outputProjection[rowOff + i + 3] * dL;
        }
        for (; i < hiddenDim; i++) {
          dH[hOff + i] += outputProjection[rowOff + i] * dL;
        }
      }
    }

    arrayPool.release(logits);
    arrayPool.release(probs);
    arrayPool.release(dLogits);

    const avgLoss = lossTerms > 0 ? totalLoss / lossTerms : 0;

    if (!isFinite(avgLoss) || lossTerms === 0) {
      PerfMonitor.recordNanSkip();
      console.warn('[Backward] NaN/Inf loss or zero terms — skipping update');
      arrayPool.release(dH);
      arrayPool.release(gOutputProj);
      arrayPool.release(finalHidden);
      for (let l = 0; l < numLayers; l++) {
        arrayPool.release(ln1Xhat[l]);
        arrayPool.release(ln2Xhat[l]);
        arrayPool.release(Q[l]);
        arrayPool.release(K[l]);
        arrayPool.release(V[l]);
        arrayPool.release(attnOutRaw[l]);
        arrayPool.release(res1[l]);
        arrayPool.release(ffnInput[l]);
      }
      return 0;
    }

    const gInputProj = arrayPool.acquire(hiddenDim * embeddingDim);
    const gPosEmb    = arrayPool.acquire(maxPos * hiddenDim);
    const allGradients = [gOutputProj, gInputProj, gPosEmb];
    const GC = CONSTANTS.GRAD_VALUE_CLAMP;

    for (let l = numLayers - 1; l >= 0; l--) {
      const hPrev = hiddenStates[l];

      const dRes2     = arrayPool.acquire(seqLen * hiddenDim);
      const gLn2Gamma = arrayPool.acquire(hiddenDim);
      const gLn2Beta  = arrayPool.acquire(hiddenDim);

      for (let t = 0; t < seqLen; t++) {
        const off = t * hiddenDim;
        layerNormBackward(
          dH, off, ln2Xhat[l], off,
          ln2Gamma[l], ln2Stats[l][t].invStd,
          dRes2, off, gLn2Gamma, gLn2Beta, hiddenDim
        );
      }
      arrayPool.release(ln2Xhat[l]);
      allGradients.push(gLn2Gamma, gLn2Beta);

      const ffnGrads  = ffnLayers[l].backward(dRes2, ffnInput[l], seqLen);
      const dFfnInput = ffnGrads.dInput;
      ffnLayers[l].applyGradients(ffnGrads, adaptiveLr, optimizer);
      arrayPool.release(ffnInput[l]);

      for (let i = 0; i < seqLen * hiddenDim; i++) dH[i] += dRes2[i];
      clampArray(dH, seqLen * hiddenDim, GC);
      arrayPool.release(dRes2);

      const dRes1     = arrayPool.acquire(seqLen * hiddenDim);
      const gLn1Gamma = arrayPool.acquire(hiddenDim);
      const gLn1Beta  = arrayPool.acquire(hiddenDim);

      for (let t = 0; t < seqLen; t++) {
        const off = t * hiddenDim;
        layerNormBackward(
          dFfnInput, off, ln1Xhat[l], off,
          ln1Gamma[l], ln1Stats[l][t].invStd,
          dRes1, off, gLn1Gamma, gLn1Beta, hiddenDim
        );
      }
      arrayPool.release(dFfnInput);
      arrayPool.release(ln1Xhat[l]);
      allGradients.push(gLn1Gamma, gLn1Beta);

      for (let i = 0; i < seqLen * hiddenDim; i++) dH[i] += dRes1[i];
      clampArray(dH, seqLen * hiddenDim, GC);

      const dAttnOutRaw = arrayPool.acquire(seqLen * qkvDim);
      const gWattnOut   = arrayPool.acquire(hiddenDim * qkvDim);
      const aRaw        = attnOutRaw[l];

      for (let t = 0; t < seqLen; t++) {
        const hOff = t * hiddenDim;
        const qOff = t * qkvDim;

        for (let i = 0; i < hiddenDim; i++) {
          const dOut = dRes1[hOff + i];
          if (!isFinite(dOut) || Math.abs(dOut) < 1e-10) continue;
          const wRow = i * qkvDim;
          for (let j = 0; j < qkvDim; j++) {
            gWattnOut[wRow + j] += dOut * aRaw[qOff + j];
          }
        }

        for (let j = 0; j < qkvDim; j++) {
          let sum = 0;
          for (let i = 0; i < hiddenDim; i++) {
            sum += WattnOut[l][i * qkvDim + j] * dRes1[hOff + i];
          }
          dAttnOutRaw[qOff + j] = isFinite(sum) ? sum : 0;
        }
      }

      allGradients.push(gWattnOut);
      arrayPool.release(dRes1);
      arrayPool.release(attnOutRaw[l]);
      arrayPool.release(res1[l]);

      const { dQ, dK, dV } = attentionLayers[l].backward(
        dAttnOutRaw, Q[l], K[l], V[l], seqLen, qkvDim
      );
      arrayPool.release(dAttnOutRaw);

      const gWq = arrayPool.acquire(qkvDim * hiddenDim);
      const gWk = arrayPool.acquire(qkvDim * hiddenDim);
      const gWv = arrayPool.acquire(qkvDim * hiddenDim);

      for (let t = 0; t < seqLen; t++) {
        const hOff = t * hiddenDim;
        const qOff = t * qkvDim;
        for (let r = 0; r < qkvDim; r++) {
          const wOff = r * hiddenDim;
          const dq   = isFinite(dQ[qOff + r]) ? dQ[qOff + r] : 0;
          const dk   = isFinite(dK[qOff + r]) ? dK[qOff + r] : 0;
          const dv   = isFinite(dV[qOff + r]) ? dV[qOff + r] : 0;
          for (let j = 0; j < hiddenDim; j++) {
            gWq[wOff + j] += Math.max(-GC, Math.min(GC, dq * hPrev[hOff + j]));
            gWk[wOff + j] += Math.max(-GC, Math.min(GC, dk * hPrev[hOff + j]));
            gWv[wOff + j] += Math.max(-GC, Math.min(GC, dv * hPrev[hOff + j]));
            const dhContrib = Wq[l][wOff + j] * dq +
                              Wk[l][wOff + j] * dk +
                              Wv[l][wOff + j] * dv;
            dH[hOff + j] += isFinite(dhContrib) ? dhContrib : 0;
          }
        }
      }
      clampArray(dH, seqLen * hiddenDim, GC);

      allGradients.push(gWq, gWk, gWv);
      arrayPool.release(dQ);
      arrayPool.release(dK);
      arrayPool.release(dV);
      arrayPool.release(Q[l]);
      arrayPool.release(K[l]);
      arrayPool.release(V[l]);
    }

    embeddings.clearGradients();
    for (let t = 0; t < seqLen; t++) {
      const tokenId = tokenIds[t];
      const emb     = embeddings.get(tokenId);
      const gEmb    = embeddings.getGradient(tokenId);
      const hOff    = t * hiddenDim;
      embeddings.recordExposure(tokenId);

      for (let i = 0; i < hiddenDim; i++) {
        const dh = dH[hOff + i];
        if (!isFinite(dh)) continue;
        gPosEmb[t * hiddenDim + i] += dh;
        for (let j = 0; j < embeddingDim; j++) {
          gInputProj[i * embeddingDim + j] +=
            Math.max(-GC, Math.min(GC, dh * emb[j]));
        }
      }

      for (let j = 0; j < embeddingDim; j++) {
        let sum = 0;
        for (let i = 0; i < hiddenDim; i++) {
          const dh = dH[hOff + i];
          sum += isFinite(dh) ? inputProjection[i * embeddingDim + j] * dh : 0;
        }
        gEmb[j] += Math.max(-GC, Math.min(GC, sum));
      }
    }

    arrayPool.release(finalHidden);
    arrayPool.release(dH);

    for (const g of allGradients) {
      if (g) sanitizeArray(g, g.length);
    }

    clipGlobalNorm(allGradients, CONSTANTS.GRAD_CLIP_NORM);
    optimizer.step();

    optimizer.update('outputProj', outputProjection, allGradients[0], adaptiveLr);
    optimizer.update('inputProj',  inputProjection,  allGradients[1], adaptiveLr);
    optimizer.update('posEmb',     posEmbeddings,    allGradients[2], adaptiveLr);
    embeddings.applyGradients(adaptiveLr, optimizer);

    let gi = 3;
    for (let l = numLayers - 1; l >= 0; l--) {
      optimizer.update(`ln2Gamma_${l}`,   ln2Gamma[l],   allGradients[gi++], adaptiveLr);
      optimizer.update(`ln2Beta_${l}`,    ln2Beta[l],    allGradients[gi++], adaptiveLr);
      optimizer.update(`ln1Gamma_${l}`,   ln1Gamma[l],   allGradients[gi++], adaptiveLr);
      optimizer.update(`ln1Beta_${l}`,    ln1Beta[l],    allGradients[gi++], adaptiveLr);
      optimizer.update(`WattnOut_${l}`,   WattnOut[l],   allGradients[gi++], adaptiveLr);
      optimizer.update(`Wq_${l}`,         Wq[l],         allGradients[gi++], adaptiveLr);
      optimizer.update(`Wk_${l}`,         Wk[l],         allGradients[gi++], adaptiveLr);
      optimizer.update(`Wv_${l}`,         Wv[l],         allGradients[gi++], adaptiveLr);
    }

    for (const g of allGradients) arrayPool.release(g);

    patternTracker.recordPatterns(tokenIds);
    return avgLoss;
  }

  // ========================================================================
  // PRIORITY REPLAY HEAP
  // ========================================================================
  class MinHeap {
    constructor(key) { this._heap = []; this._key = key; }

    push(item) { this._heap.push(item); this._bubbleUp(this._heap.length - 1); }

    pop() {
      const top  = this._heap[0];
      const last = this._heap.pop();
      if (this._heap.length > 0) { this._heap[0] = last; this._sinkDown(0); }
      return top;
    }

    peek()    { return this._heap[0]; }
    size()    { return this._heap.length; }
    isEmpty() { return this._heap.length === 0; }

    _bubbleUp(i) {
      const heap = this._heap, key = this._key;
      while (i > 0) {
        const parent = (i - 1) >> 1;
        if (key(heap[parent]) <= key(heap[i])) break;
        [heap[parent], heap[i]] = [heap[i], heap[parent]];
        i = parent;
      }
    }

    _sinkDown(i) {
      const heap = this._heap, key = this._key, n = heap.length;
      while (true) {
        let s = i;
        const l = 2 * i + 1, r = 2 * i + 2;
        if (l < n && key(heap[l]) < key(heap[s])) s = l;
        if (r < n && key(heap[r]) < key(heap[s])) s = r;
        if (s === i) break;
        [heap[s], heap[i]] = [heap[i], heap[s]];
        i = s;
      }
    }
  }

  const replayHeap = new MinHeap(m => m.exposure || 0);

  function addToMemory(chunk) {
    memory.push(chunk);
    replayHeap.push(chunk);
  }

  // ========================================================================
  // EXPERIENCE REPLAY
  // ========================================================================
  function trainReplayBatch() {
    if (memory.length < CONSTANTS.MIN_MEMORY_FOR_REPLAY) return;
    const unkId = vocab.get('<UNK>') ?? 3;

    for (let s = 0; s < CONSTANTS.REPLAY_BATCH_SIZE; s++) {
      let chunk;
      if (Math.random() < 0.7 && !replayHeap.isEmpty()) {
        chunk = replayHeap.peek();
      } else {
        chunk = memory[Math.floor(Math.random() * memory.length)];
      }
      if (!chunk || !chunk.tokens || chunk.tokens.length < 2) continue;
      const ids = chunk.tokens.map(w => vocab.get(w) ?? unkId);
      try {
        const c = forwardFull(ids);
        backwardAndUpdate(c, learningRate * 0.5);
        chunk.exposure = (chunk.exposure || 0) + 1;
      } catch (err) {
        const msg = err instanceof Error ? err.message : JSON.stringify(err);
        console.warn('[replay] Skipping chunk due to error:', msg);
      }
    }
  }

  // ========================================================================
  // TRAINING
  // ========================================================================
  function trainSequence(seqIds) {
    if (!seqIds || seqIds.length < 2) return 0;
    const t0 = performance.now();
    try {
      const cache = forwardFull(seqIds);
      const loss  = backwardAndUpdate(cache, learningRate);
      PerfMonitor.recordTrain(loss, seqIds.length, performance.now() - t0);
      return loss;
    } catch (e) {
      const msg = e instanceof Error ? e.message : JSON.stringify(e);
      console.error('[trainSequence] Error:', msg, e);
      throw e;
    }
  }

  // ========================================================================
  // LEARN FROM FILE
  // ========================================================================
  async function learnFromFile(fileName, fileText, options = {}) {
    if (typeof fileText !== 'string' || !fileText.trim()) return false;
    if (typeof fileName !== 'string') fileName = 'unknown';

    const {
      progressCallback = null,
      passes           = 5,
      useCurriculum    = true
    } = options;

    if (isTraining) { console.warn('[learnFromFile] Already training!'); return false; }

    isTraining         = true;
    shouldStopTraining = false;
    currentFileName    = fileName;
    PerfMonitor.startSession();

    try {
      const t0 = performance.now();
      const words = tokenize(fileText);
      if (!words.length) return false;

      const existingBefore = vocabSize;
      for (const w of words) {
        wordFreq.set(w, (wordFreq.get(w) || 0) + 1);
        if (!vocab.has(w) && vocabSize < maxVocabSize) {
          vocab.set(w, vocabSize);
          id2word[vocabSize] = w;
          vocabSize++;
        }
      }
      console.log(
        `[learnFromFile] +${vocabSize - existingBefore} tokens. Vocab: ${vocabSize}`
      );

      allocateModel();
      await yieldToBrowser();

      const memBefore   = memory.length;
      const totalChunks = Math.ceil(words.length / chunkWords);
      for (let c = 0; c < totalChunks; c++) {
        const chunkTokens = words.slice(c * chunkWords, (c + 1) * chunkWords);
        const chunkEmb    = new Float32Array(embeddingDim);
        let count = 0;
        for (const w of chunkTokens) {
          const id = vocab.get(w);
          if (id != null) {
            const emb = embeddings.get(id);
            for (let i = 0; i < embeddingDim; i++) chunkEmb[i] += emb[i];
            count++;
          }
        }
        if (count > 0) for (let i = 0; i < embeddingDim; i++) chunkEmb[i] /= count;
        addToMemory({
          chunkId:   memory.length,
          fileName,
          chunkText: chunkTokens.join(' '),
          tokens:    chunkTokens,
          embedding: chunkEmb,
          exposure:  0
        });
      }
      console.log(`[learnFromFile] +${memory.length - memBefore} chunks`);

      const unkId          = vocab.get('<UNK>') ?? 3;
      const ids            = words.map(w => vocab.get(w) ?? unkId);
      const initialNovelty = patternTracker.getNoveltyScore(ids);
      console.log(`[learnFromFile] Initial novelty: ${initialNovelty.toFixed(3)}`);

      let calls      = 0;
      const doReplay = memory.length >= CONSTANTS.MIN_MEMORY_FOR_REPLAY;
      const overlap  = CONSTANTS.CHUNK_OVERLAP;

      if (useCurriculum) {
        const phases = [
          { name: 'easy',   stride: 16, window: 24,
            passes: Math.max(1, Math.ceil(passes * 0.3)) },
          { name: 'medium', stride: 32, window: 48,
            passes: Math.max(1, Math.ceil(passes * 0.4)) },
          { name: 'hard',   stride: 48, window: 80,
            passes: Math.max(1, passes
              - Math.ceil(passes * 0.3)
              - Math.ceil(passes * 0.4)) },
        ];
        const totalPasses = phases.reduce((s, p) => s + p.passes, 0);
        let globalPass = 0;

        for (const phase of phases) {
          for (let p = 0; p < phase.passes && !shouldStopTraining; p++) {
            for (let i = 0; i < ids.length && !shouldStopTraining;
                 i += phase.stride - overlap) {
              const chunk = ids.slice(i, Math.min(i + phase.window, ids.length));
              if (chunk.length > 1) {
                try { trainSequence(chunk); }
                catch (e) {
                  const msg = e instanceof Error ? e.message : JSON.stringify(e);
                  console.warn('[learnFromFile] Skipping chunk:', msg);
                }
                calls++;
                if (doReplay && calls % 8 === 0) trainReplayBatch();
              }
              await maybeYield();
            }
            globalPass++;
            if (progressCallback) {
              progressCallback({
                phase: phase.name, pass: globalPass, total: totalPasses
              });
            }
          }
        }
      } else {
        for (let p = 0; p < passes && !shouldStopTraining; p++) {
          for (let i = 0; i < ids.length && !shouldStopTraining;
               i += 32 - overlap) {
            const chunk = ids.slice(i, Math.min(i + 48, ids.length));
            if (chunk.length > 1) {
              try { trainSequence(chunk); }
              catch (e) {
                const msg = e instanceof Error ? e.message : JSON.stringify(e);
                console.warn('[learnFromFile] Skipping chunk:', msg);
              }
              calls++;
              if (doReplay && calls % 8 === 0) trainReplayBatch();
            }
            await maybeYield();
          }
          if (progressCallback) {
            progressCallback({ phase: 'train', pass: p + 1, total: passes });
          }
        }
      }

      PerfMonitor.endSession();
      const finalNovelty = patternTracker.getNoveltyScore(ids);
      const elapsed      = (performance.now() - t0) / 1000;
      console.log(
        `[learnFromFile] Done: ${calls} calls, ${elapsed.toFixed(2)}s, ` +
        `Loss: ${PerfMonitor.getAvgLoss().toFixed(4)}, ` +
        `Novelty: ${initialNovelty.toFixed(3)} → ${finalNovelty.toFixed(3)}`
      );
      return true;
    } catch (e) {
      const msg = e instanceof Error ? e.message : JSON.stringify(e);
      console.error('[learnFromFile] Error:', msg, e);
      return false;
    } finally {
      isTraining = false;
    }
  }

  async function learnFromChat(user, assistant) {
    const u = (user      || '').trim();
    const a = (assistant || '').trim();
    if (!u && !a) return false;
    const text = `<USER> ${u} <ASSISTANT> ${a} <EOS>`;
    return learnFromFile('chat', text, { passes: 2, useCurriculum: false });
  }

  // ========================================================================
  // INFERENCE — OPT-2: Fused QKV in forwardStepKV
  // ========================================================================
  function initKVCache(maxLen) {
    const hd     = attentionLayers[0]
      ? attentionLayers[0].headDim
      : Math.floor(hiddenDim / resolveNumHeads(numHeads, hiddenDim));
    const nH     = attentionLayers[0]
      ? attentionLayers[0].numHeads
      : resolveNumHeads(numHeads, hiddenDim);
    const qkvDim = nH * hd;
    return {
      maxLen, len: 0, qkvDim,
      K: Array.from({ length: numLayers }, () => new Float32Array(maxLen * qkvDim)),
      V: Array.from({ length: numLayers }, () => new Float32Array(maxLen * qkvDim))
    };
  }

  function forwardStepKV(tokenId, pos, kv) {
    const headDim = attentionLayers[0]
      ? attentionLayers[0].headDim
      : Math.floor(hiddenDim / resolveNumHeads(numHeads, hiddenDim));
    const nH      = attentionLayers[0]
      ? attentionLayers[0].numHeads
      : resolveNumHeads(numHeads, hiddenDim);
    const qkvDim  = kv.qkvDim;
    const scale   = 1 / Math.sqrt(headDim);

    const emb = embeddings.get(tokenId);
    let x     = new Float32Array(hiddenDim);

    for (let i = 0; i < hiddenDim; i++) {
      let sum = posEmbeddings[pos * hiddenDim + i];
      for (let j = 0; j < embeddingDim; j++) {
        sum += inputProjection[i * embeddingDim + j] * emb[j];
      }
      x[i] = sum;
    }
    sanitizeArray(x, hiddenDim);
    clampArray(x, hiddenDim, CONSTANTS.HIDDEN_CLAMP);

    const q        = new Float32Array(qkvDim);
    const k        = new Float32Array(qkvDim);
    const v        = new Float32Array(qkvDim);
    const attnOut  = new Float32Array(qkvDim);
    const attnProj = new Float32Array(hiddenDim);
    const ffnOut   = new Float32Array(hiddenDim);

    for (let l = 0; l < numLayers; l++) {
      // OPT-2: Fused QKV — reads x once
      const fusedLimit = hiddenDim - (hiddenDim % 4);
      for (let r = 0; r < qkvDim; r++) {
        const wOff = r * hiddenDim;
        let sq = 0, sk = 0, sv = 0;
        let j  = 0;
        for (; j < fusedLimit; j += 4) {
          const x0 = x[j], x1 = x[j + 1], x2 = x[j + 2], x3 = x[j + 3];
          sq += Wq[l][wOff + j]     * x0 + Wq[l][wOff + j + 1] * x1 +
                Wq[l][wOff + j + 2] * x2 + Wq[l][wOff + j + 3] * x3;
          sk += Wk[l][wOff + j]     * x0 + Wk[l][wOff + j + 1] * x1 +
                Wk[l][wOff + j + 2] * x2 + Wk[l][wOff + j + 3] * x3;
          sv += Wv[l][wOff + j]     * x0 + Wv[l][wOff + j + 1] * x1 +
                Wv[l][wOff + j + 2] * x2 + Wv[l][wOff + j + 3] * x3;
        }
        for (; j < hiddenDim; j++) {
          const xv = x[j];
          sq += Wq[l][wOff + j] * xv;
          sk += Wk[l][wOff + j] * xv;
          sv += Wv[l][wOff + j] * xv;
        }
        q[r] = sq; k[r] = sk; v[r] = sv;
      }

      kv.K[l].set(k, pos * qkvDim);
      kv.V[l].set(v, pos * qkvDim);

      attnOut.fill(0);
      for (let h = 0; h < nH; h++) {
        const qBase   = h * headDim;
        const numKeys = pos + 1;
        const scores  = new Float32Array(numKeys);
        let maxScore  = -Infinity;

        for (let s = 0; s < numKeys; s++) {
          let d = 0;
          for (let dd = 0; dd < headDim; dd++) {
            d += q[qBase + dd] * kv.K[l][s * qkvDim + h * headDim + dd];
          }
          scores[s] = d * scale;
          if (scores[s] > maxScore) maxScore = scores[s];
        }

        if (!isFinite(maxScore)) maxScore = 0;
        let sumExp = 0;
        for (let s = 0; s < numKeys; s++) {
          scores[s] = Math.exp(Math.max(-CONSTANTS.LOGIT_CLAMP,
            Math.min(CONSTANTS.LOGIT_CLAMP, scores[s] - maxScore)));
          sumExp += scores[s];
        }
        const invSum = 1 / (sumExp + 1e-12);
        for (let s = 0; s < numKeys; s++) scores[s] *= invSum;

        for (let dd = 0; dd < headDim; dd++) {
          let sum = 0;
          for (let s = 0; s < numKeys; s++) {
            sum += scores[s] * kv.V[l][s * qkvDim + h * headDim + dd];
          }
          attnOut[qBase + dd] = sum;
        }
      }

      matVecMul(WattnOut[l], attnOut, 0, hiddenDim, qkvDim, attnProj, 0);

      const res1 = new Float32Array(hiddenDim);
      for (let i = 0; i < hiddenDim; i++) res1[i] = x[i] + attnProj[i];

      const { mean: m1, variance: v1 } = welfordMeanVar(res1, 0, hiddenDim);
      const inv1   = 1 / Math.sqrt(v1 + CONSTANTS.LN_EPSILON);
      const ln1Out = new Float32Array(hiddenDim);
      for (let i = 0; i < hiddenDim; i++) {
        ln1Out[i] = ln1Gamma[l][i] * ((res1[i] - m1) * inv1) + ln1Beta[l][i];
      }

      const routing = ffnLayers[l].route(ln1Out, 0);
      ffnOut.fill(0);
      for (let e = 0; e < routing.experts.length; e++) {
        const expertIdx = routing.experts[e];
        const weight    = routing.weights[e];
        const expert    = ffnLayers[l].experts[expertIdx];
        const act       = new Float32Array(ffnLayers[l].expertDim);
        for (let i = 0; i < ffnLayers[l].expertDim; i++) {
          let sum = expert.b1[i];
          for (let j = 0; j < ffnLayers[l].hiddenDim; j++) {
            sum += expert.W1[i * ffnLayers[l].hiddenDim + j] * ln1Out[j];
          }
          act[i] = sum > 0 ? sum : 0;
        }
        for (let i = 0; i < ffnLayers[l].hiddenDim; i++) {
          let sum = expert.b2[i] * weight;
          for (let j = 0; j < ffnLayers[l].expertDim; j++) {
            sum += expert.W2[i * ffnLayers[l].expertDim + j] * act[j] * weight;
          }
          ffnOut[i] += sum;
        }
      }

      const res2 = new Float32Array(hiddenDim);
      for (let i = 0; i < hiddenDim; i++) res2[i] = res1[i] + ffnOut[i];

      const { mean: m2, variance: v2 } = welfordMeanVar(res2, 0, hiddenDim);
      const inv2 = 1 / Math.sqrt(v2 + CONSTANTS.LN_EPSILON);
      const newX = new Float32Array(hiddenDim);
      for (let i = 0; i < hiddenDim; i++) {
        newX[i] = ln2Gamma[l][i] * ((res2[i] - m2) * inv2) + ln2Beta[l][i];
      }

      sanitizeArray(newX, hiddenDim);
      clampArray(newX, hiddenDim, CONSTANTS.HIDDEN_CLAMP);
      x = newX;
    }

    const logitsOut = new Float32Array(vocabSize);
    matVecMul(outputProjection, x, 0, vocabSize, hiddenDim, logitsOut, 0);
    for (let i = 0; i < vocabSize; i++) {
      logitsOut[i] = Math.max(-CONSTANTS.LOGIT_CLAMP,
        Math.min(CONSTANTS.LOGIT_CLAMP, logitsOut[i]));
    }

    kv.len = pos + 1;
    return logitsOut;
  }

  // ========================================================================
  // SAMPLING — OPT-4: Reuse module-scope scratch buffer
  // ========================================================================
  function sampleLogits(logits, temp = 0.8, topK = 20, topP = 0.85,
                        recentTokens = [], contextIds = []) {
    const n = vocabSize;

    if (_sampleScratchSize < n) {
      _sampleScratch     = new Float32Array(n);
      _sampleScratchSize = n;
    }
    const scaled = _sampleScratch;
    for (let i = 0; i < n; i++) scaled[i] = logits[i];

    if (recentTokens.length > 0) {
      const window = recentTokens.slice(-CONSTANTS.REPETITION_WINDOW);
      const counts = new Map();
      for (const id of window) counts.set(id, (counts.get(id) || 0) + 1);
      for (const [id, cnt] of counts.entries()) {
        if (id < n) scaled[id] /= Math.pow(CONSTANTS.REPETITION_PENALTY, cnt);
      }
    }

    if (contextIds.length > 0) {
      const penalties = conflictTracker.getPenalties(contextIds, n);
      for (let i = 0; i < n; i++) scaled[i] -= penalties[i];
    }

    let effectiveTemp = temp;
    {
      let max1 = -Infinity, max2 = -Infinity;
      for (let i = 0; i < n; i++) {
        if      (scaled[i] > max1) { max2 = max1; max1 = scaled[i]; }
        else if (scaled[i] > max2)   max2 = scaled[i];
      }
      if (max1 - max2 < 0.5) effectiveTemp = Math.max(0.3, temp * 0.5);
    }

    const invTemp = 1 / Math.max(0.01, effectiveTemp);
    for (let i = 0; i < n; i++) scaled[i] *= invTemp;

    const indices = Array.from({ length: n }, (_, i) => i);
    indices.sort((a, b) => scaled[b] - scaled[a]);
    let keptCount = Math.min(topK, n);

    if (topP < 1) {
      const maxS = scaled[indices[0]];
      let sumE   = 0;
      const tempProbs = new Float32Array(keptCount);
      for (let i = 0; i < keptCount; i++) {
        tempProbs[i] = Math.exp(Math.max(-CONSTANTS.LOGIT_CLAMP,
          Math.min(CONSTANTS.LOGIT_CLAMP, scaled[indices[i]] - maxS)));
        sumE += tempProbs[i];
      }
      let cumulative   = 0;
      let nucleusCount = 0;
      for (let i = 0; i < keptCount && cumulative < topP; i++) {
        cumulative += tempProbs[i] / (sumE + 1e-12);
        nucleusCount++;
      }
      keptCount = Math.max(1, nucleusCount);
    }

    const maxKept   = scaled[indices[0]];
    const keptProbs = new Float32Array(keptCount);
    let sumProbs    = 0;
    for (let i = 0; i < keptCount; i++) {
      keptProbs[i] = Math.exp(Math.max(-CONSTANTS.LOGIT_CLAMP,
        Math.min(CONSTANTS.LOGIT_CLAMP, scaled[indices[i]] - maxKept)));
      sumProbs += keptProbs[i];
    }

    const r    = Math.random() * (sumProbs + 1e-12);
    let cumul  = 0;
    for (let i = 0; i < keptCount; i++) {
      cumul += keptProbs[i];
      if (r <= cumul) return indices[i];
    }
    return indices[0];
  }

  function postProcess(tokens) {
    if (!tokens.length) return '';
    const filtered = tokens.filter(t =>
      !ExecutionEngine.isExecToken(t) && !/^\d+\.\d{2,}$/.test(t));
    const deduped = [];
    let prev = null;
    for (const tok of filtered) {
      if (tok !== prev) { deduped.push(tok); prev = tok; }
    }
    const collapsed = removeCycles(deduped, 6);
    let text = collapsed.join(' ')
      .replace(/\s+([.,!?;:])/g, '$1')
      .replace(/(^|\.\s+|\!\s+|\?\s+)([a-z])/g,
        (_, pre, ch) => pre + ch.toUpperCase())
      .trim();
    if (text && !/[.!?]$/.test(text)) text += '.';
    return text;
  }

  function removeCycles(tokens, maxCycle = 6) {
    if (tokens.length <= maxCycle * 2) return tokens;
    for (let cycleLen = 2; cycleLen <= maxCycle; cycleLen++) {
      const tail = tokens.slice(-cycleLen * 3);
      if (tail.length < cycleLen * 2) continue;
      const pattern = tail.slice(0, cycleLen).join('|');
      const rest    = tail.slice(cycleLen).join('|');
      if (rest.startsWith(pattern)) {
        return tokens.slice(0, tokens.length - cycleLen * 2);
      }
    }
    return tokens;
  }

  function retrieveTopK(query, k = 3) {
    if (typeof query !== 'string' || !memory.length) return [];
    const words    = tokenize(query);
    const queryEmb = new Float32Array(embeddingDim);
    let count = 0, queryNorm = 0;
    for (const w of words) {
      const id = vocab.get(w);
      if (id != null) {
        const emb = embeddings.get(id);
        for (let i = 0; i < embeddingDim; i++) queryEmb[i] += emb[i];
        count++;
      }
    }
    if (count === 0) return [];
    for (let i = 0; i < embeddingDim; i++) {
      queryEmb[i] /= count;
      queryNorm   += queryEmb[i] * queryEmb[i];
    }
    queryNorm = Math.sqrt(queryNorm);
    return memory.map(m => {
      let dp = 0, memNorm = 0;
      for (let i = 0; i < embeddingDim; i++) {
        dp      += queryEmb[i] * m.embedding[i];
        memNorm += m.embedding[i] * m.embedding[i];
      }
      return {
        chunk: m,
        score: dp / (queryNorm * Math.sqrt(memNorm) + 1e-9)
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
  }

  function generateWithContext(prompt, opts = {}) {
    if (typeof prompt !== 'string') prompt = '';
    const {
      maxTokens   = 50,
      temperature = 0.8,
      topK        = 20,
      topP        = 0.85,
      useRAG      = false,
      useVM       = true
    } = opts;

    const safeMaxTokens   = Math.max(1,    Math.min(Number.isFinite(maxTokens)   ? maxTokens   : 50,  512));
    const safeTemperature = Math.max(0.01, Math.min(Number.isFinite(temperature) ? temperature : 0.8,  10));
    const safeTopK        = Math.max(1,    Math.min(Number.isFinite(topK)        ? topK        : 20,  vocabSize));
    const safeTopP        = Math.max(0.01, Math.min(Number.isFinite(topP)        ? topP        : 0.85, 1));

    allocateModel();

    if (useVM && ExecutionEngine.enabled) {
      const result = ExecutionEngine.autoExecute(prompt);
      if (result) return `The answer is ${result.formatted}.`;
    }

    let ragContext = '';
    if (useRAG && memory.length > 0) {
      ragContext = retrieveTopK(prompt, 3)
        .map(x => x.chunk.chunkText).join(' ') + ' ';
    }

    const fullPrompt = (ragContext + prompt).trim();
    const words      = fullPrompt ? tokenize(fullPrompt) : [];
    const unkId = vocab.get('<UNK>') ?? 3;
    const bosId = vocab.get('<BOS>') ?? 1;

    let ids = [bosId, ...words.map(w => vocab.get(w) ?? unkId)];
    if (ids.length > contextWindow) ids = ids.slice(-contextWindow);

    const kv = initKVCache(Math.min(maxPos, ids.length + safeMaxTokens + 8));

    let logits = null;
    for (let p = 0; p < ids.length; p++) {
      logits = forwardStepKV(ids[p], p, kv);
    }
    if (!logits) logits = forwardStepKV(bosId, 0, kv);

    const generated    = [];
    const generatedIds = [];
    let pos = ids.length - 1;

    for (let i = 0; i < safeMaxTokens; i++) {
      const recentCtx = ids.concat(generatedIds).slice(-3);
      const nextId    = sampleLogits(
        logits, safeTemperature, safeTopK, safeTopP,
        generatedIds, recentCtx
      );
      const nextWord  = id2word[nextId] || '<UNK>';

      if (!nextWord.startsWith('<') || nextWord === '<UNK>') {
        generated.push(nextWord);
        generatedIds.push(nextId);
      }

      pos++;
      if (pos >= kv.maxLen) break;
      logits = forwardStepKV(nextId, pos, kv);

      if (['.', '!', '?', '<EOS>'].includes(nextWord) &&
          generated.length >= CONSTANTS.MIN_GENERATION_TOKENS) break;
    }

    return postProcess(generated);
  }

  function askQuestion(prompt, opts) {
    if (typeof prompt !== 'string') return '';
    return generateWithContext(prompt, { ...opts, useRAG: true });
  }

  // ========================================================================
  // PERSISTENCE
  // ========================================================================
  function exportMemory() {
    return JSON.stringify({
      memory:          memory.map(m => ({ ...m, embedding: Array.from(m.embedding) })),
      id2word:         Array.from(id2word),
      vocabSize,
      wordFreq:        Array.from(wordFreq.entries()),
      patternTracker:  patternTracker.toJSON(),
      conflictTracker: conflictTracker.toJSON()
    });
  }

  function importMemory(json) {
    try {
      const data = JSON.parse(json);
      if (data.memory) {
        memory.length    = 0;
        replayHeap._heap = [];
        data.memory.forEach(m => {
          const chunk = { ...m, embedding: new Float32Array(m.embedding) };
          memory.push(chunk);
          replayHeap.push(chunk);
        });
      }
      if (data.id2word) {
        id2word.length = 0;
        id2word.push(...data.id2word);
        vocab.clear();
        data.id2word.forEach((w, i) => vocab.set(w, i));
      }
      if (typeof data.vocabSize === 'number') vocabSize = data.vocabSize;
      if (data.wordFreq) {
        wordFreq.clear();
        data.wordFreq.forEach(([k, v]) => wordFreq.set(k, v));
      }
      if (data.patternTracker)  patternTracker.fromJSON(data.patternTracker);
      if (data.conflictTracker) conflictTracker.fromJSON(data.conflictTracker);
      allocateModel();
      return true;
    } catch (e) {
      const msg = e instanceof Error ? e.message : JSON.stringify(e);
      console.error('[importMemory] Failed:', msg, e);
      return false;
    }
  }

  function saveWeights() {
    return JSON.stringify({
      version: 'zenith-js-5.8',
      config: {
        embeddingDim, hiddenDim, numHeads, numLayers,
        maxPos, maxSeqLen, trainLossPositions, vocabSize,
        ffDim, learningRate, chunkWords, contextWindow
      },
      id2word:          Array.from(id2word),
      embeddings:       embeddings.toJSON(),
      posEmbeddings:    Array.from(posEmbeddings),
      inputProjection:  Array.from(inputProjection),
      outputProjection: Array.from(outputProjection),
      Wq:               Wq.map(w => Array.from(w)),
      Wk:               Wk.map(w => Array.from(w)),
      Wv:               Wv.map(w => Array.from(w)),
      WattnOut:         WattnOut.map(w => Array.from(w)),
      ffnLayers:        ffnLayers.map(f => f.toJSON()),
      ln1Gamma:         ln1Gamma.map(g => Array.from(g)),
      ln1Beta:          ln1Beta.map(b => Array.from(b)),
      ln2Gamma:         ln2Gamma.map(g => Array.from(g)),
      ln2Beta:          ln2Beta.map(b => Array.from(b)),
      optimizerState:   optimizer.getState(),
      patternTracker:   patternTracker.toJSON(),
      conflictTracker:  conflictTracker.toJSON()
    });
  }

  function loadWeights(json) {
    try {
      const data = JSON.parse(json);
      if (data.config) {
        embeddingDim       = data.config.embeddingDim       ?? embeddingDim;
        hiddenDim          = data.config.hiddenDim          ?? hiddenDim;
        numHeads           = data.config.numHeads           ?? numHeads;
        numLayers          = data.config.numLayers          ?? numLayers;
        maxPos             = data.config.maxPos             ?? maxPos;
        maxSeqLen          = data.config.maxSeqLen          ?? maxSeqLen;
        vocabSize          = data.config.vocabSize          ?? vocabSize;
        ffDim              = data.config.ffDim              ?? hiddenDim * 4;
        learningRate       = data.config.learningRate       ?? learningRate;
        trainLossPositions = data.config.trainLossPositions ?? trainLossPositions;
        chunkWords         = data.config.chunkWords         ?? chunkWords;
        contextWindow      = data.config.contextWindow      ?? contextWindow;
      }
      if (data.id2word) {
        id2word.length = 0;
        id2word.push(...data.id2word);
        vocab.clear();
        data.id2word.forEach((w, i) => vocab.set(w, i));
      }
      _ffnBuiltHiddenDim = 0;
      _ffnBuiltFfDim     = 0;
      allocateModel();
      if (data.embeddings)       embeddings.fromJSON(data.embeddings);
      if (data.posEmbeddings)    posEmbeddings    = new Float32Array(data.posEmbeddings);
      if (data.inputProjection)  inputProjection  = new Float32Array(data.inputProjection);
      if (data.outputProjection) outputProjection = new Float32Array(data.outputProjection);
      if (data.Wq)       Wq       = data.Wq.map(w => new Float32Array(w));
      if (data.Wk)       Wk       = data.Wk.map(w => new Float32Array(w));
      if (data.Wv)       Wv       = data.Wv.map(w => new Float32Array(w));
      if (data.WattnOut) WattnOut = data.WattnOut.map(w => new Float32Array(w));
      if (data.ffnLayers) {
        data.ffnLayers.forEach((f, l) => {
          if (ffnLayers[l]) ffnLayers[l].fromJSON(f);
        });
      }
      if (data.ln1Gamma) ln1Gamma = data.ln1Gamma.map(g => new Float32Array(g));
      if (data.ln1Beta)  ln1Beta  = data.ln1Beta.map(b => new Float32Array(b));
      if (data.ln2Gamma) ln2Gamma = data.ln2Gamma.map(g => new Float32Array(g));
      if (data.ln2Beta)  ln2Beta  = data.ln2Beta.map(b => new Float32Array(b));
      if (data.optimizerState)  optimizer.loadState(data.optimizerState);
      if (data.patternTracker)  patternTracker.fromJSON(data.patternTracker);
      if (data.conflictTracker) conflictTracker.fromJSON(data.conflictTracker);
      return true;
    } catch (e) {
      const msg = e instanceof Error ? e.message : JSON.stringify(e);
      console.error('[loadWeights] Failed:', msg, e);
      return false;
    }
  }

  function setConfig(cfg = {}) {
    if (typeof cfg !== 'object' || cfg === null) return;
    if (typeof cfg.embeddingDim       === 'number' && cfg.embeddingDim > 0)
      embeddingDim       = cfg.embeddingDim;
    if (typeof cfg.hiddenDim          === 'number' && cfg.hiddenDim > 0)
      hiddenDim          = cfg.hiddenDim;
    if (typeof cfg.contextWindow      === 'number' && cfg.contextWindow > 0)
      contextWindow      = cfg.contextWindow;
    if (typeof cfg.chunkWords         === 'number' && cfg.chunkWords > 0)
      chunkWords         = cfg.chunkWords;
    if (typeof cfg.numHeads           === 'number' && cfg.numHeads > 0)
      numHeads           = cfg.numHeads;
    if (typeof cfg.numLayers          === 'number' && cfg.numLayers > 0)
      numLayers          = cfg.numLayers;
    if (typeof cfg.learningRate       === 'number' && cfg.learningRate > 0)
      learningRate       = cfg.learningRate;
    if (typeof cfg.maxSeqLen          === 'number' && cfg.maxSeqLen > 1)
      maxSeqLen          = cfg.maxSeqLen;
    if (typeof cfg.trainLossPositions === 'number' && cfg.trainLossPositions > 0)
      trainLossPositions = cfg.trainLossPositions;
    ffDim = hiddenDim * 4;
    allocateModel();
  }

  function getConfig() {
    return {
      embeddingDim, hiddenDim, contextWindow, chunkWords,
      numHeads, numLayers, learningRate, maxSeqLen,
      trainLossPositions, vocabSize, ffDim,
      qkvDim: getQkvDim()
    };
  }

  function resetAll() {
    shouldStopTraining = true;
    isTraining         = false;
    vocab.clear();
    id2word.length = 0;
    wordFreq.clear();
    vocabSize = 0;
    memory.length    = 0;
    replayHeap._heap = [];
    embeddings        = null;
    posEmbeddings     = null;
    inputProjection   = null;
    outputProjection  = null;
    attentionLayers   = [];
    ffnLayers         = [];
    Wq = []; Wk = []; Wv = []; WattnOut = [];
    ln1Gamma = []; ln1Beta  = [];
    ln2Gamma = []; ln2Beta  = [];
    _ffnBuiltHiddenDim = 0;
    _ffnBuiltFfDim     = 0;
    _sampleScratch     = null;
    _sampleScratchSize = 0;
    optimizer.reset();
    patternTracker.clear();
    conflictTracker.clear();
    PerfMonitor.reset();
    arrayPool.clear();
    gradientTracker.reset();
    allocateModel();
    console.log('[resetAll] Model fully reset.');
  }

  // ========================================================================
  // OPT-7: JIT WARMUP
  // ========================================================================
  function warmupJIT() {
    console.log('[Warmup] Warming up JIT compiler...');
    const dummyIds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const kv       = initKVCache(20);
    for (let i = 0; i < 30; i++) {
      try {
        const cache = forwardFull(dummyIds);
        backwardAndUpdate(cache, 0);
      } catch(e) { /* intentionally ignored during warmup */ }
      try { forwardStepKV(1, 0, kv); } catch(e) { /* ignore */ }
    }
    console.log('[Warmup] JIT warmup complete.');
  }

  allocateModel();

  // ========================================================================
  // PUBLIC API
  // ========================================================================
  return {
    learnFromFile,
    learnFromChat,
    trainSequence,
    generateWithContext,
    askQuestion,
    retrieveTopK,
    exportMemory,
    importMemory,
    saveWeights,
    loadWeights,
    setConfig,
    getConfig,
    stopTraining,
    getTrainingStatus,
    tokenize,
    resetAll,
    warmupJIT,

    getPerformanceStats() {
      return {
        ...PerfMonitor.getStats(),
        memory: {
          embeddingTableSize: embeddings ? embeddings.size() : 0,
          arrayPool:          arrayPool.getStats()
        }
      };
    },

    resetPerformanceStats() { PerfMonitor.reset(); },
    resetOptimizer()        { optimizer.reset(); },

    vm: {
      execute:    c => MicroVM.execute(c),
      calc:       e => MicroVM.calc(e),
      reset:      () => MicroVM.reset(),
      isEnabled:  () => ExecutionEngine.enabled,
      setEnabled: e  => { ExecutionEngine.enabled = !!e; }
    },

    _internal: {
      vocab, id2word, memory, wordFreq,
      embeddings, optimizer, arrayPool, gradientTracker,
      attentionLayers, ffnLayers,
      patternTracker, conflictTracker,
      MicroVM, ExecutionEngine,
      get vocabSize() { return vocabSize; },
      get qkvDim()    { return getQkvDim(); },
      allocateModel, PerfMonitor, CONSTANTS
    }
  };
})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = aiCore;
}

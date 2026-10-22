# FlashMap XDNA2 static-graph boundary

This note fixes the smallest FlashMap inference graph worth qualifying on the
Windows XDNA2 NPU. It is bound to the exact Qwen3.8-Flash-Next 5K training
recipe and to Gufo's existing external-draft verifier. It does not introduce a
serving option, an ONNX Runtime dependency, or a claim that an NPU checkpoint
has passed qualification.

## Bound geometry

The converted stage-2 flow map must have this exact geometry:

| Item | Value |
| --- | --- |
| target hidden size | 2,560 |
| target vocabulary | 248,320 |
| completed target-layer taps | `[1, 12, 23, 34, 45]` |
| concatenated target context | 12,800 |
| proposal block | 7 tokens |
| DiT width / layers | 2,048 / 6 |
| attention | 16 heads x 128 |
| MLP width | 8,192 |
| time-conditioning width | 256 |
| draft jump | one, `s = 0` to `t = 1` |

The training configuration names input layers `[2, 13, 24, 35, 46]`; release
conversion deliberately subtracts one and therefore exposes the completed
layer outputs above. Layer 47 is verifier-only and is not drafter context.

## First graph

The first artifact is a single-request, one-jump draft graph. Compile one graph
per fixed context bucket `S`, beginning at 128 and then qualifying 256, 512,
1,024, 2,048 and 4,096 only as needed. `S` includes the transient anchor row.

Inputs, with batch fixed to one, are:

| Name | Shape | Logical dtype | Meaning |
| --- | --- | --- | --- |
| `noise` | `[1, 7, 2560]` | BF16 | deterministic Gaussian keyed by request seed and absolute slot position |
| `rope_cos` | `[1, 7, 1, 64]` | FP32 | precomputed draft-position rotary cosine |
| `rope_sin` | `[1, 7, 1, 64]` | FP32 | precomputed draft-position rotary sine |
| `ctx_k_0` ... `ctx_k_5` | each `[1, 16, S, 128]` | BF16 | rotated, normalized context keys |
| `ctx_v_0` ... `ctx_v_5` | each `[1, 16, S, 128]` | BF16 | context values |
| `ctx_mask` | `[1, 1, 7, S]` | FP32 | exactly `0.0` for valid context and finite `-1.0e9` for padding; every query must retain at least one valid key |

Separate per-layer K/V inputs avoid a dynamic `Gather` over the layer axis.
The reference exporter may use FP32 external tensors when required by the
installed Windows ML compiler, but the compiled activation and cache contract
remains BF16. Such adapter casts are part of the model identity.

The first graph outputs `draft_logits [1, 7, Vd]`, where `Vd` is a pinned draft
vocabulary. Start with `Vd = 65,536`, the established FlashMap serving size,
and retain a full-vocabulary control. Host code performs stable argmax and maps
the local indices through a pinned `d2t[Vd]` table. Keeping argmax and token
mapping out of the first graph prevents a small unsupported operator from
silently moving the graph to the CPU.

The restricted vocabulary is a performance choice, not a target-quality
choice. A target token outside it merely ends acceptance at that slot. The
conversion artifact must bind the ordered `d2t` table, its digest and the
response-token corpus used to rank it.

## Required one-jump folds

The graph is exported from the optimized one-jump form already implemented by
FlashMap serving, not from the training forward:

1. Evaluate the `s = 0`, `t = 1` time and jump embeddings at conversion time.
   Bake every layer's self-attention, cross-attention and MLP AdaLN
   `(1 + scale, shift, gate)` and the final AdaLN `(1 + scale, shift)` as
   initializers.
2. Bake the uniform-posterior branch at `s = 0` into
   `first_hidden [2048]`.
3. Factor the exact target embedding covariance and fold the target embedding,
   `noise_norm` and `x_embed` into `noise_weight [2048, 2560]` and
   `noise_eigenvalues [2560]`. Runtime initialization is:

   ```text
   rms = rsqrt(reduce_sum(noise * noise * noise_eigenvalues) + eps)
   hidden = first_hidden + rms * matmul(noise, transpose(noise_weight))
   ```

4. Fold the DiT head `[2560, 2048]` into the selected target LM-head rows,
   producing one `head_weight [Vd, 2048]` and `head_bias [Vd]`.
5. Supply rotary tables as inputs. Do not put random generation, trigonometry,
   eigendecomposition or Cholesky decomposition in the compiled graph.

The resulting operator family is deliberately ordinary: `MatMul`/`Gemm`,
`Add`, `Mul`, `ReduceSum`, `Rsqrt`, normalization primitives,
`Reshape`/`Transpose`/`Split`, `Softmax`, `Gelu` and explicit casts. There is no
dynamic slice, scatter, control flow or random operator.

The unfused release is approximately 474.9 million trainable parameters, or
0.885 GiB in BF16, before target embedding and LM-head weights. With a 65,536
row folded head, the static one-jump graph is approximately 603.7 million
parameters: 1.124 GiB in BF16, or roughly 0.56 GiB if its major linear weights
are W8. The full-vocabulary control is approximately 978.2 million parameters
and 1.822 GiB in BF16.

## Context state

Each session logically owns:

```text
context_kv: [6, 2, 1, 16, max_context, 128] BF16
context_length: scalar INT32
noise_seed: scalar UINT64
```

The K/V cache costs 49,152 bytes per context token; 4,096 tokens therefore
cost 192 MiB. The cache must remain in execution-provider-local or IO-bound
memory. Copying the complete cache into or out of the NPU for each draft is a
hard performance failure, even if the isolated graph is fast.

The first compiler probe consumes captured, already-projected K/V and does not
update state. Production integration adds two bounded operations:

- `context_append_N`: target taps `[1, N, 12800]` plus target-position rotary
  tables produce only K/V deltas `[6, 2, 1, 16, N, 128]`;
- `anchor_kv`: target embedding `[1, 1, 2560]` produces the transient anchor
  row used by the current draft.

The host or an OGA wrapper appends deltas into persistent pages. No graph emits
or copies the whole cache. The anchor row is not persistent until target
verification commits it. Rejected verification rows are never appended, so
rollback remains a context-length update.

If generic Windows ML IO binding cannot retain this cache without a full-copy
round trip, stop that route and qualify an OGA/custom state wrapper. OGA is not
the first export target: AMD's architectural-change operator-compilation path
is experimental and assumes ONNX Runtime GenAI model-builder conventions.

## Quantization order

1. Export a float graph and use Vitis AI's automatic BF16 lowering as the
   fidelity and operator-placement baseline.
2. Produce an A16W8 QDQ graph with symmetric, per-output-channel INT8 weights
   for context projection, fused QKV/KV, attention outputs, MLPs, folded noise
   and the folded head.
3. Retain BF16 activations and K/V. Keep normalization, softmax, rotary,
   residual arithmetic, biases and baked AdaLN constants in BF16/FP32.
4. Calibrate any compiler-required activation ranges from the target-bound 5K
   corpus. Do not introduce A8 activation quantization in the first candidate.
5. Compare X2 and X1 compiler backends only after both satisfy the same
   provider and quality gates.

Quantized draft logits need not equal BF16 logits numerically. Initially retain
the existing differential harness's stricter exact-token gate. A later variant
may differ in proposals only after end-to-end target verification proves exact
output and its accepted-prefix distribution remains worthwhile.

## Exact-greedy quality

For proposals `d1 ... d7`, the GPU target scores
`[anchor, d1, ... d7]` once and produces authoritative greedy tokens
`y1 ... y8`. Gufo commits only the longest prefix for which `di == yi`, then
commits the exact target token `y(m+1)`; if all seven drafts match, the bonus is
`y8`. Unused proposal state is rolled back.

Therefore NPU compilation or quantization drift can only change proposal
acceptance and speed. It cannot change greedy output: no unmatched NPU token is
ever committed. NPU output remains ineligible until it passes through
`Session::DecodeDraftStep`.

## Qualification gates

A candidate fails before serving integration if any of these holds:

- ONNX Runtime profiling assigns any required node to a provider other than
  `VitisAIExecutionProvider`, or records no provider events;
- the cache cannot remain provider-local and incurs a whole-cache copy;
- warm draft latency including all input/output synchronization is not lower
  than the retained GPU drafter path;
- A16W8 materially reduces accepted tokens per target pass;
- target output tokens, output hash, final logits, sampler history or rollback
  state differ from ordinary greedy decode;
- cancellation cannot generation-fence a stale NPU completion;
- concurrent NPU work reduces target GPU throughput or creates WDDM stalls.

Record compile/session time separately from warm execution. Also retain draft
p50/p95, K/V append time, transfer time, target verification time, acceptance
histogram, target cycles and end-to-end tokens per second.

## Staged implementation

1. Convert a trained stage-2 checkpoint and bind its target, feature ABI,
   checkpoint, folded weights, draft-vocabulary table and static shapes.
2. Export the `S = 128` BF16 baseline from captured projected K/V. Replay the
   same cases through CPU and NPU with the existing fail-closed differential
   harness.
3. Export A16W8 QDQ and compare token agreement, acceptance and warm latency.
4. Add context/anchor projection and prove persistent cache IO without a full
   copy. Expand context buckets only after the preceding bucket passes.
5. Wire a synchronous, single-session greedy backend behind
   `DecodeDraftStep`; preserve ordinary autoregressive fallback and exact
   output checks.
6. Batch ready sessions and overlap request A's NPU draft with request B's GPU
   target verification only after synchronous qualification.
7. Treat sampled decoding and multi-jump FlashMap as separate designs.

The Windows runtime probe and differential harness remain optional developer
tools. No production dependency or public serving option is justified until a
real checkpoint clears every gate above.

# FlashMap capture residency and cross-record batching

## Measured bottleneck

The 5,000-record production capture published 79 immutable GFM shards.  On the
75 adjacent shard intervals without a recovery pause, completion time regresses
against the newly published shard size as:

```text
seconds = 86.658 + 0.030762 * decimal_megabytes
correlation = 0.922
```

The median interval is 104.106 seconds for a 559.185 MB shard.  The current
orchestrator starts one capture process per shard, and each process loads the
same roughly 111 GB target.  The fitted fixed term therefore accounts for
about 83% of healthy wall time.  Loading the model once and processing the same
immutable shard jobs in a resident process is the first optimization gate.  A
simple projection reduces healthy 5K capture from roughly 2.2 hours to 24--30
minutes before changing any target arithmetic.

`tools/qwen-flash/flashmap_capture_profile.py` reproduces the estimate from
published-file metadata without opening GFM payloads or live partials.

## Safe resident-shard prototype

A resident capture command should accept an immutable job manifest containing,
in order, each input JSONL digest, output directory, shard ordinal, record
range, and expected target/producer identities.  It should:

1. load and identity-check the target once;
2. for each job, require an absent output and create a fresh unique partial;
3. execute the existing single-record capture path without changing its
   arithmetic or feature ABI;
4. durably finalize and publish each shard with no replacement before opening
   the next job; and
5. preserve the active partial and every prior published shard on failure.

This changes process lifetime only.  A fixture must prove each resident shard
is byte-identical to a frozen one-process-per-shard run, including record order,
tokens, BF16 features, headers, and all digests except the deliberately unique
shard UUID/timestamp fields.  A semantic comparator should normalize only those
two header fields before comparing bytes.

## Why the existing `ForwardBatch` is not the capture batch

`Executor::ForwardBatch` has the correct state ownership: every item has a
distinct `Session`, and recurrent SSM, attention/indexer cache, PLE history,
n-gram history, positions, controls, cancellation, and rollback storage remain
private.  It is nevertheless decode-shaped:

- at most eight tokens per session;
- at most eight sessions;
- it always runs the vocabulary projection for every row;
- it has no layer-tap output contract; and
- it uses batched projection/MoE kernels whose floating-point reduction shapes
  need an explicit capture parity qualification.

Typical FlashMap records are hundreds of tokens and the executor already has a
4,096-row prefill scratch allocation.  Feeding them as repeated eight-token
decode batches would multiply launches and compute unnecessary logits, so it
is not a safe performance optimization.

## Prefill-shaped cross-record prototype gate

Only after resident capture is measured should a separate
`ForwardPrefillCaptureBatch` be considered.  The bounded design is:

- 1--8 unique autoregressive sessions;
- arbitrary positive rows per session with total rows no greater than
  `max_batch`;
- explicit per-session offsets and private control/state updates;
- no vocabulary head except an optional final row per record;
- raw tap residuals copied in concatenated row order, followed by the existing
  target head mixer and scatter back to record-major output;
- no graph capture, checkpoint, speculative rollback, MTP, or vision in the
  first prototype; and
- output records committed strictly in original manifest order.

Qualification must run the same 1, 2, 4, and 8-record fixtures through solo
and batched paths from fresh sessions.  It must require exact input and output
token IDs, exact feature shapes/order, finite values, and report max absolute,
max relative, and RMS feature error.  Acceptance is byte equality where the
kernel route is identical; otherwise the bound must be fixed in advance from
the target's existing numerical contracts and must not be widened after a
failure.  Final-row argmax tokens must be exact.  A reordered-input fixture and
one record crossing the 4,096-row boundary must prove session and position
independence.

No cross-record executor change should ship until that fixture passes on both
the production `gfx1151` path and the intended `gfx942` cloud producer.  The
resident-process optimization does not depend on this riskier arithmetic
change and should be completed first.

## Resident implementation

`flashmap_5k_orchestrate.py capture --resident-capture` now publishes a
canonical, digest-named job manifest and starts one capture process for the
entire first contiguous run of missing shards.  `--max-new-shards` can bound
that run.  The manifest binds the exact input bytes and SHA-256, output path,
shard ordinal, record range, record/token totals, expected GFM byte count,
target identity, capture executable identity and Git commit, feature ABI, and
maximum context.

The capture executable preflights every input and requires every output shard
directory to be absent before initializing HIP or loading the model.  It loads
the model once, then calls the unchanged single-record
`CaptureFlashMapFeatures` path and unchanged GFM serializer for each job.  Each
shard is durably published without replacement and independently revalidated
by the orchestrator before success is reported.  Input identity, totals, byte
count, and exclusive partial-directory ownership are rechecked before each
publication.

Resident mode deliberately rejects an existing partial.  Recover that one
shard with the existing single-shard `--retry-partial-shard` path, normally
bounded by `--max-new-shards 1`, and then resume resident mode.  Previously
published shards and the active partial remain untouched after any failure.
This keeps recovery compatible with the proven per-shard validator without
adding a second partial-resume implementation to the long-lived process.

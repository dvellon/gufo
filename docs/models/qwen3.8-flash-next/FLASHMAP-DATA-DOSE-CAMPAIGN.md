# FlashMap data-dose campaign

This campaign measures the value of additional unique target traces without
serializing response generation, capture, training, and evaluation.  It keeps
the existing 5,000-record corpus as chunk zero, publishes later training data
in immutable 5,000-record chunks, and publishes cumulative prefix milestones
from 10,000 through 50,000 records.

`tools/qwen-flash/flashmap_chunk_campaign.py` owns only prompt membership and
milestone identity.  The existing generator, capture planner, and capture
orchestrator continue to own their established byte and producer contracts.
The training importer must additionally bind the campaign, chunk, and
milestone manifest digests.

## Required topology

- Freeze an independent holdout corpus before adding future training chunks.
- Initialize the campaign with the existing 5K corpus as `chunk-00000`.
- Add one complete 5K prompt corpus at a time.  Addition fails if an ID or
  exact canonical message digest occurs in any earlier chunk or the holdout.
- Publish a milestone only after its complete ordered chunk prefix exists.
- Never append to or replace a chunk, holdout, campaign, or milestone file.
- Generate and capture each chunk separately.  A response, plan, trace, or
  importer failure never changes prompt membership.

The default milestones are 10K, 15K, 20K, 25K, 30K, 35K, 40K, 45K, and 50K.
Clean training anchors should normally be 10K, 20K, 40K, and the final chosen
dose.  Continuation runs at intermediate milestones are useful operational
feedback but are not equivalent to clean dataset-size comparisons.

## Storage and transfer

A captured 5K dataset is approximately 45.7 GB.  The inexpensive 80 GB
staging host therefore cannot retain multiple captured chunks.  Staging should
retain only small campaign manifests and any already-authorized durable source
packet.  The generator/capture GPU should send each validated, content-bound
chunk directly to the trainer over the private VPC.  The trainer has enough
local and scratch storage to retain immutable received chunks and construct
cumulative importer manifests without copying their payloads.

Transfer publication is two-phase:

1. copy into a new absent partial destination;
2. validate bytes, record counts, producer identity, target identity, feature
   ABI, plan content, and campaign chunk identity;
3. atomically publish without replacement;
4. only then allow the corresponding milestone to consume the chunk.

Partial transfers remain preserved and are never treated as published input.

## Corpus expansion boundary

The current pinned source normalizer selects only 15,000 candidates, so it
cannot safely supply 50,000 unique training prompts plus an independent
holdout.  Expanding it is a separate review boundary: increase the authenticated
candidate inventory or add pinned, license-compatible sources, assign revised
category quotas, and publish each resulting 5K corpus through the ordinary
prompt-corpus tool.  The campaign tool accepts such independently verified
corpora but deliberately does not download data or weaken the pinned source
inventory itself.

## Example lifecycle

Initialize once, using already-published corpora and the exact frozen
generation identities:

```text
python -B tools/qwen-flash/flashmap_chunk_campaign.py init \
  EXISTING-5K-CORPUS HOLDOUT-CORPUS CAMPAIGN \
  --target-sha256 TARGET \
  --generator-sha256 GENERATOR \
  --generator-git GENERATOR-GIT \
  --generation-config-sha256 CONFIG
```

For every newly reviewed 5K corpus:

```text
python -B tools/qwen-flash/flashmap_chunk_campaign.py add-chunk \
  CAMPAIGN NEXT-5K-CORPUS
```

After enough chunks exist, publish the corresponding immutable prefix:

```text
python -B tools/qwen-flash/flashmap_chunk_campaign.py publish-milestone \
  CAMPAIGN 10000
```

Run `validate` before handing a campaign or milestone to generation, capture,
import, or training.  Do not infer completion merely from a directory name.

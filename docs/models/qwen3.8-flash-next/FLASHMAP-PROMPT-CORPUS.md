# FlashMap 5,000-prompt corpus

`tools/qwen-flash/flashmap_prompt_sources.py` authenticates and normalizes the
six pinned source configurations. `tools/qwen-flash/flashmap_prompt_corpus.py`
then publishes a deterministic, license-aware prompt source for
`qwen38_flash_next_flashmap_generate`. Neither tool calls a model or reuses an
upstream answer.

## Reviewed mix

The checked-in recipe template proposes exactly 5,000 prompts:

| Category | Records | Pinned source | Why it is present |
| --- | ---: | --- | --- |
| General instruction | 1,750 | Smol-Magpie-Ultra | Broad, curated instruction following |
| Constraint following | 500 | Smol-constraints | Verifiable formatting and lexical constraints |
| Editing | 250 | Smol-rewrite | Controlled transformations rather than only question answering |
| Summarization | 250 | Smol-summarize | Grounded compression tasks |
| Code | 1,250 | BigCode Self-OSS StarCoder2 50K | Execution-filtered code instructions |
| Mathematical reasoning | 1,000 | OpenR1-Math-220K `default` | Verified hard reasoning prompts |

The four SmolTalk subsets are the new, Apache-2.0-licensed components described
by the [SmolTalk dataset card](https://huggingface.co/datasets/HuggingFaceTB/smoltalk/tree/5feaf2fd3ffca7c237fc38d1861bc30365d48ffa).
The code source is pinned to the
[ODC-By-1.0 BigCode dataset](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k/tree/356bb069eee815daa6e23e9a282eeefe1490ad44),
and the reasoning source is pinned to the Apache-2.0
[OpenR1-Math dataset](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/tree/e4e141ec9dea9f8326f4d347be56105859b2bd68).
These links are provenance, not permission to float to a later `main` revision.

The mix is deliberately prompt-only. The exact target model generates every
answer, so teacher style, stale answers, and foreign tokenization cannot leak
into the drafter target. Code and reasoning prompts remain substantial without
dominating the ordinary instruction distribution.

## Normalized-source contract

Each source is a newline-terminated canonical JSONL file beside the final
recipe. A row has exactly two fields:

```json
{"messages":[{"content":"Write a bounded ring buffer.","role":"user"}],"record_id":"upstream-stable-id"}
```

`messages` may contain leading `system` and `developer` messages followed by
exactly one final `user` message. Upstream assistant answers are never copied.
Numbers, tool calls, attachments, and unreviewed role aliases are not accepted.
For the proposed sources, the normalization decisions are:

- SmolTalk: retain the prompt prefix through the first user turn; retain a
  preceding system message if one exists; reject rows requiring an assistant
  turn to provide context. Use the upstream stable row identifier, not the row
  ordinal.
- BigCode: use `instruction` as the sole user content and its stable `id` as
  `record_id`. Do not concatenate the seed code, generated prompt, or response.
- OpenR1-Math: use `uuid` as `record_id`; user content is the dataset card's
  reviewed instruction, `Please reason step by step, and put your final answer
  within \\boxed{}.`, followed by a blank line and `problem`. Do not copy any
  generated solution.

The checked-in normalizer implements those decisions as three narrow adapters.
It authenticates every remote object against the pinned repository commit, LFS
SHA-256, and byte length before reading it. It SHA-256-ranks Parquet row groups
and records, fetches only the selected prompt columns by HTTP range, and writes
the prompt prefix only. This avoids downloading the multi-gigabyte source
corpora and ensures assistant responses never enter a retained artifact.

The normalizer writes to a new path and records every raw object identity,
accessed row group, transferred-byte count, adapter name, row count, rejection
count, and final normalized digest in `source-manifest.json`. It also emits the
canonical, digest-bound preparation recipe. The all-zero digests in
`flashmap_prompt_corpus_recipe.template.json` are intentional placeholders; the
template is not an executable approval packet.

The materialization host needs `requests` and `pyarrow`. Create and validate a
source bundle with:

```powershell
python -B tools/qwen-flash/flashmap_prompt_sources.py prepare `
  build/flashmap-qwen38-normalized-sources-v1
python -B tools/qwen-flash/flashmap_prompt_sources.py validate `
  build/flashmap-qwen38-normalized-sources-v1
```

## Deterministic publication

After normalization is reviewed:

1. Independently review the emitted recipe's source revisions, license
   identifiers, quotas, and normalized-input digests.
2. Compute the recipe digest and run:

```powershell
$recipe = 'build/flashmap-qwen38-normalized-sources-v1/flashmap_prompt_corpus_recipe.json'
$sha = (Get-FileHash -Algorithm SHA256 $recipe).Hash.ToLowerInvariant()
python tools/qwen-flash/flashmap_prompt_corpus.py prepare `
  $recipe build/flashmap-qwen38-prompts-v1 --recipe-sha256 $sha
```

The selector SHA-256-ranks every eligible prompt using the reviewed seed,
source identity, source revision, record identifier, and canonical messages.
It removes exact cross-source message duplicates before enforcing each quota.
Selection is independent of input row order. Insufficient unique eligible rows,
digest drift, malformed input, an unknown field, or a quota mismatch fails
closed. It never overwrites an output directory.

The result contains only:

- `prompts.jsonl`, whose rows exactly match the response exporter's
  `id/messages/metadata/source_id` input contract; and
- `recipe.json`, the exact reviewed recipe bytes; and
- `manifest.json`, self-binding the recipe, output, tool, selection algorithm,
  source revisions, licenses, categories, and counts.

Validate a retained result with:

```powershell
python tools/qwen-flash/flashmap_prompt_corpus.py validate `
  build/flashmap-qwen38-prompts-v1
```

`prompts.jsonl` is then passed to the response exporter with its exact SHA-256.
The exporter still applies the target tokenizer and context envelope, and stops
if even one prompt does not fit. Inspect category counts and the response
export's `stop` versus `length` distribution before accepting the training run;
a 128-token length stop is an exact target prefix, but it is not evidence that
the source task was completed.

The first retained materialization is summarized in
[`artifacts/flashmap-prompt-corpus-v1.json`](artifacts/flashmap-prompt-corpus-v1.json).
It authenticated 20 pinned Parquet objects, transferred 132,753,462 bytes of
selected metadata and prompt columns, and produced exactly 5,000 unique prompts
with SHA-256 `81c962011ca55e1214495c41e23c0c69f1ca6819157264ce732e4fef3509a5f5`.

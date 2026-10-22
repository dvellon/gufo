# FlashMap 50K prompt inventory

`tools/qwen-flash/flashmap_prompt_inventory.py` extends the reviewed 5K prompt
source without changing its upstream datasets, revisions, adapters, licenses, or
raw-object identities.  It is the source boundary for the clean data-dose
campaign.

The inventory contains:

- the existing reviewed 5K corpus as immutable chunk zero;
- nine new, disjoint 5K training corpora, yielding 50K training prompts total;
- one fixed 1K holdout corpus; and
- one separate fixed 1K final-test corpus.

The expanded parent selection contains 52K prompts.  The tool requires the
original 5K identities and canonical messages to be an exact subset, removes
them, and partitions the remaining 47K by a fixed SHA-256 ranking.  Category
mixes remain the reviewed 35/10/5/5/25/20 percent distribution.  Every
partition is published through `flashmap_prompt_corpus.py`, and the resulting
root is published atomically without replacement.  Validation proves the 11
published partitions are mutually disjoint and are exactly the expanded corpus
minus the original 5K.

## Source and license boundary

The inventory reuses only these pinned sources:

| Source | Revision | License |
| --- | --- | --- |
| `HuggingFaceTB/smoltalk` | `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa` | Apache-2.0 |
| `bigcode/self-oss-instruct-sc2-exec-filter-50k` | `356bb069eee815daa6e23e9a282eeefe1490ad44` | ODC-By-1.0 |
| `open-r1/OpenR1-Math-220k` | `e4e141ec9dea9f8326f4d347be56105859b2bd68` | Apache-2.0 |

The authenticated Parquet object digests and byte sizes remain those in
`flashmap_prompt_sources.py`.  The campaign profile raises only the number of
deterministically ranked prompt candidates read from those objects.  Assistant
answers are still excluded by the same three reviewed adapters.

## Materialization

Create the expanded normalized bundle on a networked preparation host with
`requests` and `pyarrow`:

```text
python -B tools/qwen-flash/flashmap_prompt_inventory.py normalize \
  build/flashmap-qwen38-campaign-sources-v1
python -B tools/qwen-flash/flashmap_prompt_inventory.py validate-source \
  build/flashmap-qwen38-campaign-sources-v1
```

Independently review the emitted source manifest and its content digest.  Then
bind that digest and the existing 5K corpus manifest digest when building:

```text
python -B tools/qwen-flash/flashmap_prompt_inventory.py build \
  build/flashmap-qwen38-campaign-sources-v1 \
  build/flashmap-qwen38-prompts-v1 \
  build/flashmap-qwen38-prompt-inventory-v1 \
  --source-manifest-sha256 SOURCE_MANIFEST_CONTENT_SHA256 \
  --exclusion-manifest-sha256 72040f19e4fd2bc8ab8eae68ae6beb244da0d61291b57c30655f710a6999af32
python -B tools/qwen-flash/flashmap_prompt_inventory.py validate \
  build/flashmap-qwen38-prompt-inventory-v1
```

The existing local normalized-source cache contains only 15K candidates and
cannot satisfy this plan.  No expanded download has been performed yet.  The
remaining empirical gate is that every pinned subset must yield its exact
bounded candidate quota; normalization stops without publishing if any subset
is too small after schema, character-envelope, identity, and exact-message
deduplication checks.

Initialize `flashmap_chunk_campaign.py` with the existing 5K corpus and the
new `holdout` corpus, then add `chunks/chunk-00001` through
`chunks/chunk-00009` in order.  The `test` corpus is never training input and
must remain unopened for model selection.

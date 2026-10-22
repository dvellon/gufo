# FlashMap exact-target response export

`qwen38_flash_next_flashmap_generate` creates the authoritative token export
consumed by `tools/qwen-flash/flashmap_5k_plan.py`. It uses the target's Qwen
chat template and tokenizer directly. It does not render text through an HTTP
API or retokenize a decoded response.

The input is newline-terminated JSONL. Every row has exactly these fields:

```json
{"id":"sample-0001","messages":[{"content":"Write a merge sort.","role":"user"}],"metadata":{"kind":"code"},"source_id":"dataset-row-1"}
```

The reviewed 5,000-prompt source is prepared by
`tools/qwen-flash/flashmap_prompt_corpus.py`; see
`FLASHMAP-PROMPT-CORPUS.md`. Its published `prompts.jsonl` already satisfies
this schema and supplies the exact source digest required below.

`messages` accepts only `system`, `developer`, and `user`; the final message
must be a user message. Metadata uses the planner's portable JSON subset: it
may contain objects, arrays, strings, booleans, and null, but no numbers.

The checked-in 128-token candidate configuration is canonical JSON. Its digest
is deliberately supplied again on the command line, as are the source,
generator executable, target artifact, and producer Git identities. A mismatch
stops before generation. Obtain the executable digest after the final build.

```powershell
$generator = 'build/release-rocm101-upstream/src/models/qwen38_flash_next/qwen38_flash_next_flashmap_generate.exe'
$config = 'tools/qwen-flash/flashmap_generation_config_128.json'
& $generator `
  --model C:/models/Qwen3.8-Flash-Next-00001-of-00004.gguf `
  --input prompts.jsonl --input-sha256 SOURCE_SHA256 `
  --output responses.jsonl `
  --generation-config $config `
  --generation-config-sha256 9361849986d68bbabce49b3cb9e9c4e72580cd26a708608a28be46c0176ff9e8 `
  --target-sha256 TARGET_SHA256 --generator-sha256 GENERATOR_SHA256 `
  --producer-git PRODUCER_COMMIT --expected-records 5000
```

Generation is target-only greedy decoding with no MTP sidecar, penalties,
sampling, or grammar. Stop tokens are not included in `response_tokens`.
`metadata.termination` is `stop` or `length`, so a budget-truncated answer is
not mistaken for a completed answer.

Each response row is flushed durably to `responses.jsonl.partial`. `--resume`
accepts only an existing canonical prefix whose source digests, rendered prompt
tokens, target, executable, Git commit, and generation configuration still
match. A rebuilt executable therefore cannot silently continue an older run.
After the expected record count and a final source-file hash check, publication
renames the partial file without replacement.

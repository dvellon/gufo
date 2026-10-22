# FlashMap prompt-token derivation

`qwen38_flash_next_flashmap_generate --derive-prompt-tokens` is the canonical
CPU-only bridge from campaign message JSONL to the token-input JSONL consumed
by the MI325X llama.cpp response exporter. It deliberately lives in the Gufo
generator so derivation calls the same `ParsePromptRecord` and `RenderPrompt`
path as Gufo generation. It does not initialize HIP, upload model weights, or
generate response tokens.

The mode reads tokenizer and chat-template metadata from the first bound GGUF
shard. The caller must bind all of the following exact identities:

- source JSONL SHA-256 and record count;
- generation configuration and chat-template SHA-256;
- tokenizer GGUF metadata-shard SHA-256 and complete model artifact identity;
- executable SHA-256;
- producer Git commit and tree; and
- partition name.

Output is canonical newline-terminated JSONL, published without replacement.
Every row has exactly `id`, `metadata`, `prompt_tokens`, and `source_id`, so it
is accepted directly by `llama-flashmap-token-exporter`. Metadata carries the
original user metadata plus every source, tokenizer, configuration, binary,
and partition identity. Each campaign partition is derived independently;
valid production labels are `chunk-00000` through `chunk-00009`, `validation`,
and `test`.

Before bulk derivation, run the one-record focused check with
`flashmap_prompt_token_selftest.py`. The self-test invokes the native deriver,
uses `--verify-prompt-token-parity` to compare its token IDs and source binding
against an exact previously generated Gufo fixture output, and proves that a
second derivation cannot replace the published fixture. Python only
orchestrates the native modes; it never renders a template or tokenizes text.

Do not substitute a Python/Hugging Face tokenizer, decoded response text, or a
different chat-template implementation. Freeze the final source commit/tree,
source hash, executable hash, fixture response hash, and every per-partition
output hash before enabling paid bulk generation.

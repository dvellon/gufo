#!/usr/bin/env python3
"""Build a deterministic, provenance-bound FlashMap prompt corpus.

This tool intentionally does not download or reinterpret upstream datasets.  Each
source is first normalized to the tiny prompt-only JSONL contract documented in
FLASHMAP-PROMPT-CORPUS.md, then bound by SHA-256 in a canonical recipe.  That
separates source-specific parsing review from selection and publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


FORMAT = "gufo.flashmap.prompt-corpus.v1"
RECIPE_FORMAT = "gufo.flashmap.prompt-corpus-recipe.v1"
SELECTION_ALGORITHM = "sha256-rank-with-global-exact-message-dedup-v1"
OUTPUT_FIELDS = {"id", "messages", "metadata", "source_id"}
SOURCE_ROW_FIELDS = {"messages", "record_id"}
MESSAGE_FIELDS = {"content", "role"}
RECIPE_FIELDS = {"expected_records", "format", "seed", "sources"}
SOURCE_FIELDS = {
    "category",
    "dataset_id",
    "dataset_revision",
    "dataset_url",
    "input",
    "input_sha256",
    "license_spdx",
    "license_url",
    "max_prompt_chars",
    "min_prompt_chars",
    "quota",
    "split",
    "subset",
}
MANIFEST_FIELDS = {
    "format",
    "manifest_content_sha256",
    "prompt_bytes",
    "prompt_file",
    "prompt_sha256",
    "recipe_file",
    "recipe_sha256",
    "records",
    "selection_algorithm",
    "sources",
    "tool_sha256",
}
MANIFEST_SOURCE_FIELDS = {
    "category",
    "dataset_id",
    "dataset_revision",
    "dataset_url",
    "duplicates_excluded",
    "input_sha256",
    "license_spdx",
    "license_url",
    "records",
    "split",
    "subset",
}
MAX_JSONL_LINE_BYTES = 16 * 1024 * 1024
MAX_RECIPE_BYTES = 1024 * 1024
ALLOWED_ROLES = {"system", "developer", "user"}


class CorpusError(RuntimeError):
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CorpusError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_digest(manifest: dict[str, object]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_content_sha256", None)
    return _sha256_bytes(_canonical(unsigned))


def _hex(value: object, field: str, digits: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != digits
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CorpusError(f"{field} must be {digits} lowercase hexadecimal digits")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CorpusError(f"{field} must be a non-empty NUL-free string")
    return value


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CorpusError(f"{field} must be a positive integer")
    return value


def _load_strict_json(path: Path, maximum: int) -> tuple[dict[str, object], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise CorpusError(f"cannot read {path}") from error
    if not payload or len(payload) > maximum or not payload.endswith(b"\n"):
        raise CorpusError(f"{path} must be bounded, non-empty and newline-terminated")
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, CorpusError) as error:
        raise CorpusError(f"{path} is not strict JSON") from error
    if not isinstance(value, dict) or _canonical(value) != payload:
        raise CorpusError(f"{path} is not a canonical JSON object")
    return value, payload


def _messages(value: object, *, context: str) -> tuple[list[dict[str, str]], bytes]:
    if not isinstance(value, list) or not value:
        raise CorpusError(f"{context} messages must be a non-empty array")
    result: list[dict[str, str]] = []
    for ordinal, message in enumerate(value):
        if not isinstance(message, dict) or set(message) != MESSAGE_FIELDS:
            raise CorpusError(f"{context} message {ordinal} has unexpected fields")
        role = _string(message.get("role"), f"{context} message role")
        content = _string(message.get("content"), f"{context} message content")
        if role not in ALLOWED_ROLES:
            raise CorpusError(f"{context} message role {role!r} is not prompt-only")
        if any(
            ord(character) < 32 and character not in "\n\r\t"
            for character in content
        ):
            raise CorpusError(f"{context} message contains a control character")
        if not content.strip():
            raise CorpusError(f"{context} message content is whitespace-only")
        result.append({"content": content, "role": role})
    if result[-1]["role"] != "user":
        raise CorpusError(f"{context} final message must have user role")
    if sum(message["role"] == "user" for message in result) != 1:
        raise CorpusError(f"{context} must contain exactly one final user message")
    return result, _canonical(result).rstrip(b"\n")


@dataclass(frozen=True)
class Source:
    ordinal: int
    category: str
    dataset_id: str
    dataset_revision: str
    dataset_url: str
    input_path: Path
    input_sha256: str
    license_spdx: str
    license_url: str
    max_prompt_chars: int
    min_prompt_chars: int
    quota: int
    split: str
    subset: str


@dataclass(frozen=True)
class Candidate:
    source: Source
    record_id: str
    messages: list[dict[str, str]]
    messages_sha256: str
    rank_sha256: str


def _parse_recipe(
    path: Path, reviewed_sha256: str
) -> tuple[dict[str, object], bytes, list[Source]]:
    reviewed_sha256 = _hex(reviewed_sha256, "reviewed recipe SHA-256", 64)
    recipe, payload = _load_strict_json(path, MAX_RECIPE_BYTES)
    if _sha256_bytes(payload) != reviewed_sha256:
        raise CorpusError("recipe identity differs from the reviewed SHA-256")
    if set(recipe) != RECIPE_FIELDS or recipe.get("format") != RECIPE_FORMAT:
        raise CorpusError("recipe has unexpected fields or format")
    seed = _hex(recipe.get("seed"), "seed", 64)
    expected = _positive_integer(recipe.get("expected_records"), "expected_records")
    raw_sources = recipe.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise CorpusError("sources must be a non-empty array")
    sources: list[Source] = []
    categories: set[str] = set()
    inputs: set[Path] = set()
    for ordinal, raw in enumerate(raw_sources):
        if not isinstance(raw, dict) or set(raw) != SOURCE_FIELDS:
            raise CorpusError(f"source {ordinal} has unexpected fields")
        category = _string(raw.get("category"), "category")
        if category in categories:
            raise CorpusError(f"category {category!r} occurs more than once")
        categories.add(category)
        relative_input = Path(_string(raw.get("input"), "input"))
        if relative_input.is_absolute() or relative_input.name != str(relative_input):
            raise CorpusError("source input must be a filename beside the recipe")
        input_path = (path.parent / relative_input).resolve()
        if input_path in inputs:
            raise CorpusError("source input is reused")
        inputs.add(input_path)
        minimum = _positive_integer(raw.get("min_prompt_chars"), "min_prompt_chars")
        maximum = _positive_integer(raw.get("max_prompt_chars"), "max_prompt_chars")
        if minimum > maximum or maximum > 1_000_000:
            raise CorpusError("source prompt character envelope is invalid")
        source = Source(
            ordinal=ordinal,
            category=category,
            dataset_id=_string(raw.get("dataset_id"), "dataset_id"),
            dataset_revision=_hex(raw.get("dataset_revision"), "dataset_revision", 40),
            dataset_url=_string(raw.get("dataset_url"), "dataset_url"),
            input_path=input_path,
            input_sha256=_hex(raw.get("input_sha256"), "input_sha256", 64),
            license_spdx=_string(raw.get("license_spdx"), "license_spdx"),
            license_url=_string(raw.get("license_url"), "license_url"),
            max_prompt_chars=maximum,
            min_prompt_chars=minimum,
            quota=_positive_integer(raw.get("quota"), "quota"),
            split=_string(raw.get("split"), "split"),
            subset=_string(raw.get("subset"), "subset"),
        )
        if not source.dataset_url.startswith(
            "https://"
        ) or not source.license_url.startswith("https://"):
            raise CorpusError("dataset and license URLs must use https")
        sources.append(source)
    if sum(source.quota for source in sources) != expected:
        raise CorpusError("source quotas do not add up to expected_records")
    recipe["seed"] = seed
    return recipe, payload, sources


def _read_candidates(source: Source, seed: str) -> list[Candidate]:
    if _sha256_file(source.input_path) != source.input_sha256:
        raise CorpusError(f"normalized source identity differs for {source.category}")
    candidates: list[Candidate] = []
    record_ids: set[str] = set()
    with source.input_path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.endswith(b"\n") or len(raw_line) > MAX_JSONL_LINE_BYTES:
                raise CorpusError(
                    f"{source.category} line {line_number} is not bounded JSONL"
                )
            try:
                raw = json.loads(raw_line, object_pairs_hook=_unique_object)
            except (UnicodeDecodeError, json.JSONDecodeError, CorpusError) as error:
                raise CorpusError(
                    f"{source.category} line {line_number} is not strict JSON"
                ) from error
            if (
                not isinstance(raw, dict)
                or set(raw) != SOURCE_ROW_FIELDS
                or _canonical(raw) != raw_line
            ):
                raise CorpusError(
                    f"{source.category} line {line_number} is not canonical "
                    "normalized JSON"
                )
            record_id = _string(raw.get("record_id"), "record_id")
            if record_id in record_ids:
                raise CorpusError(f"{source.category} repeats record_id {record_id!r}")
            record_ids.add(record_id)
            messages, messages_bytes = _messages(
                raw.get("messages"), context=source.category
            )
            prompt_chars = sum(len(message["content"]) for message in messages)
            if not source.min_prompt_chars <= prompt_chars <= source.max_prompt_chars:
                continue
            messages_sha256 = _sha256_bytes(messages_bytes)
            rank = _sha256_bytes(
                b"\x00".join(
                    (
                        seed.encode("ascii"),
                        source.category.encode("utf-8"),
                        source.dataset_id.encode("utf-8"),
                        source.dataset_revision.encode("ascii"),
                        record_id.encode("utf-8"),
                        messages_sha256.encode("ascii"),
                    )
                )
            )
            candidates.append(
                Candidate(source, record_id, messages, messages_sha256, rank)
            )
    return candidates


def _select(sources: list[Source], seed: str) -> tuple[list[Candidate], dict[str, int]]:
    candidates = [
        candidate
        for source in sources
        for candidate in _read_candidates(source, seed)
    ]
    duplicate_counts = {source.category: 0 for source in sources}
    owners: dict[str, Candidate] = {}
    for candidate in sorted(
        candidates, key=lambda item: (item.messages_sha256, item.rank_sha256)
    ):
        owner = owners.get(candidate.messages_sha256)
        if owner is None:
            owners[candidate.messages_sha256] = candidate
        else:
            duplicate_counts[candidate.source.category] += 1
    selected: list[Candidate] = []
    for source in sources:
        eligible = sorted(
            (candidate for candidate in owners.values() if candidate.source == source),
            key=lambda item: (item.rank_sha256, item.record_id),
        )
        if len(eligible) < source.quota:
            raise CorpusError(
                f"{source.category} has {len(eligible)} unique eligible prompts; "
                f"quota is {source.quota}"
            )
        selected.extend(eligible[: source.quota])
    return selected, duplicate_counts


def _output_row(candidate: Candidate, ordinal: int) -> dict[str, object]:
    source = candidate.source
    source_id = (
        f"hf:{source.dataset_id}@{source.dataset_revision}/"
        f"{source.subset}/{source.split}/{candidate.record_id}"
    )
    return {
        "id": f"flashmap-{ordinal:05d}-{candidate.messages_sha256[:12]}",
        "messages": candidate.messages,
        "metadata": {
            "category": source.category,
            "dataset_id": source.dataset_id,
            "dataset_revision": source.dataset_revision,
            "license_spdx": source.license_spdx,
            "normalized_source_sha256": source.input_sha256,
            "record_id": candidate.record_id,
            "selection_algorithm": SELECTION_ALGORITHM,
            "selection_rank_sha256": candidate.rank_sha256,
            "split": source.split,
            "subset": source.subset,
        },
        "source_id": source_id,
    }


def prepare(args: argparse.Namespace) -> Path:
    recipe_path = args.recipe.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise CorpusError(f"refusing to overwrite output directory {output}")
    recipe, recipe_payload, sources = _parse_recipe(recipe_path, args.recipe_sha256)
    selected, duplicate_counts = _select(sources, str(recipe["seed"]))
    if len(selected) != recipe["expected_records"]:
        raise CorpusError("selection did not produce the exact expected record count")
    rows = [
        _output_row(candidate, ordinal)
        for ordinal, candidate in enumerate(selected)
    ]
    prompts_payload = b"".join(_canonical(row) for row in rows)
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        prompts_path = temporary / "prompts.jsonl"
        with prompts_path.open("xb") as stream:
            stream.write(prompts_payload)
        with (temporary / "recipe.json").open("xb") as stream:
            stream.write(recipe_payload)
        manifest: dict[str, object] = {
            "format": FORMAT,
            "prompt_bytes": len(prompts_payload),
            "prompt_file": "prompts.jsonl",
            "prompt_sha256": _sha256_bytes(prompts_payload),
            "recipe_file": "recipe.json",
            "recipe_sha256": _sha256_bytes(recipe_payload),
            "records": len(rows),
            "selection_algorithm": SELECTION_ALGORITHM,
            "sources": [
                {
                    "category": source.category,
                    "dataset_id": source.dataset_id,
                    "dataset_revision": source.dataset_revision,
                    "dataset_url": source.dataset_url,
                    "duplicates_excluded": duplicate_counts[source.category],
                    "input_sha256": source.input_sha256,
                    "license_spdx": source.license_spdx,
                    "license_url": source.license_url,
                    "records": source.quota,
                    "split": source.split,
                    "subset": source.subset,
                }
                for source in sources
            ],
            "tool_sha256": _sha256_file(Path(__file__).resolve()),
        }
        manifest["manifest_content_sha256"] = _manifest_digest(manifest)
        manifest_payload = _canonical(manifest)
        with (temporary / "manifest.json").open("xb") as stream:
            stream.write(manifest_payload)
        os.rename(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validate(output)
    print(
        "FLASHMAP_PROMPT_CORPUS_OK "
        f"records={len(rows)} sha256={_sha256_bytes(prompts_payload)} output={output}"
    )
    return output


def validate(root: Path) -> dict[str, object]:
    root = root.resolve()
    manifest, manifest_payload = _load_strict_json(
        root / "manifest.json", MAX_RECIPE_BYTES
    )
    del manifest_payload
    if (
        set(manifest) != MANIFEST_FIELDS
        or manifest.get("format") != FORMAT
        or manifest.get("prompt_file") != "prompts.jsonl"
        or manifest.get("recipe_file") != "recipe.json"
        or manifest.get("selection_algorithm") != SELECTION_ALGORITHM
    ):
        raise CorpusError("corpus manifest format is not recognized")
    if manifest.get("manifest_content_sha256") != _manifest_digest(manifest):
        raise CorpusError("corpus manifest content digest differs")
    _hex(manifest.get("tool_sha256"), "manifest tool_sha256", 64)
    recipe = root / "recipe.json"
    if _sha256_file(recipe) != manifest.get("recipe_sha256"):
        raise CorpusError("retained recipe digest differs")
    _load_strict_json(recipe, MAX_RECIPE_BYTES)
    prompts = root / "prompts.jsonl"
    if _sha256_file(prompts) != manifest.get("prompt_sha256"):
        raise CorpusError("prompt corpus digest differs")
    if prompts.stat().st_size != manifest.get("prompt_bytes"):
        raise CorpusError("prompt corpus byte count differs")
    expected_files = {"manifest.json", "prompts.jsonl", "recipe.json"}
    if {path.name for path in root.iterdir()} != expected_files:
        raise CorpusError("corpus directory contains missing or unexpected entries")
    ids: set[str] = set()
    message_digests: set[str] = set()
    category_counts: dict[str, int] = {}
    rows = 0
    with prompts.open("rb") as stream:
        for raw_line in stream:
            if not raw_line.endswith(b"\n") or len(raw_line) > MAX_JSONL_LINE_BYTES:
                raise CorpusError("prompt output is not bounded JSONL")
            try:
                row = json.loads(raw_line, object_pairs_hook=_unique_object)
            except (UnicodeDecodeError, json.JSONDecodeError, CorpusError) as error:
                raise CorpusError("prompt output is not strict JSON") from error
            if (
                not isinstance(row, dict)
                or set(row) != OUTPUT_FIELDS
                or _canonical(row) != raw_line
            ):
                raise CorpusError("prompt output is not canonical exporter input")
            row_id = _string(row.get("id"), "output id")
            source_id = _string(row.get("source_id"), "output source_id")
            if row_id in ids or not source_id.startswith("hf:"):
                raise CorpusError("prompt output identity is duplicated or invalid")
            ids.add(row_id)
            _, message_bytes = _messages(row.get("messages"), context=row_id)
            message_sha = _sha256_bytes(message_bytes)
            if message_sha in message_digests:
                raise CorpusError("prompt output repeats exact messages")
            message_digests.add(message_sha)
            metadata = row.get("metadata")
            if not isinstance(metadata, dict) or any(
                not isinstance(value, str) for value in metadata.values()
            ):
                raise CorpusError("prompt output metadata is not portable strings")
            if metadata.get("selection_algorithm") != SELECTION_ALGORITHM:
                raise CorpusError("prompt output selection algorithm differs")
            category = _string(metadata.get("category"), "output category")
            category_counts[category] = category_counts.get(category, 0) + 1
            rows += 1
    if rows != manifest.get("records"):
        raise CorpusError("prompt output record count differs")
    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise CorpusError("corpus manifest sources are invalid")
    source_counts: dict[str, object] = {}
    for source in raw_sources:
        if not isinstance(source, dict) or set(source) != MANIFEST_SOURCE_FIELDS:
            raise CorpusError("corpus manifest source has unexpected fields")
        category = _string(source.get("category"), "manifest source category")
        if category in source_counts:
            raise CorpusError("corpus manifest repeats a category")
        source_counts[category] = source.get("records")
    if category_counts != source_counts:
        raise CorpusError("prompt output category quotas differ")
    return manifest


def self_test() -> None:
    with tempfile.TemporaryDirectory(
        prefix="flashmap-prompt-corpus-", dir=Path.cwd()
    ) as raw:
        root = Path(raw)
        source_specs = (("general", 2), ("code", 1))
        source_rows: dict[str, list[dict[str, object]]] = {
            "general": [
                {
                    "messages": [
                        {"content": "Explain a stable sort.", "role": "user"}
                    ],
                    "record_id": "g1",
                },
                {
                    "messages": [
                        {"content": "Compare queues and stacks.", "role": "user"}
                    ],
                    "record_id": "g2",
                },
                {
                    "messages": [
                        {"content": "Summarize a short memo.", "role": "user"}
                    ],
                    "record_id": "g3",
                },
            ],
            "code": [
                {
                    "messages": [
                        {"content": "Write a bounded ring buffer.", "role": "user"}
                    ],
                    "record_id": "c1",
                },
                {
                    "messages": [
                        {
                            "content": "Repair this parser without changing its API.",
                            "role": "user",
                        }
                    ],
                    "record_id": "c2",
                },
            ],
        }
        sources = []
        for ordinal, (category, quota) in enumerate(source_specs):
            input_name = f"{category}.jsonl"
            payload = b"".join(_canonical(row) for row in source_rows[category])
            (root / input_name).write_bytes(payload)
            sources.append(
                {
                    "category": category,
                    "dataset_id": f"example/{category}",
                    "dataset_revision": f"{ordinal + 1:040x}",
                    "dataset_url": f"https://example.invalid/{category}",
                    "input": input_name,
                    "input_sha256": _sha256_bytes(payload),
                    "license_spdx": "Apache-2.0",
                    "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
                    "max_prompt_chars": 4096,
                    "min_prompt_chars": 8,
                    "quota": quota,
                    "split": "train",
                    "subset": "default",
                }
            )
        recipe = {
            "expected_records": 3,
            "format": RECIPE_FORMAT,
            "seed": "ab" * 32,
            "sources": sources,
        }
        recipe_payload = _canonical(recipe)
        recipe_path = root / "recipe.json"
        recipe_path.write_bytes(recipe_payload)
        common = argparse.Namespace(
            recipe=recipe_path, recipe_sha256=_sha256_bytes(recipe_payload)
        )
        first = prepare(argparse.Namespace(**vars(common), output=root / "first"))
        second = prepare(argparse.Namespace(**vars(common), output=root / "second"))
        if (first / "prompts.jsonl").read_bytes() != (
            second / "prompts.jsonl"
        ).read_bytes():
            raise AssertionError(
                "identical reviewed inputs produced different prompt bytes"
            )
        validate(first)
        with (first / "prompts.jsonl").open("ab") as stream:
            stream.write(b"{}\n")
        try:
            validate(first)
        except CorpusError:
            pass
        else:
            raise AssertionError("mutated prompt corpus was accepted")
    print("FLASHMAP_PROMPT_CORPUS_SELFTEST_OK records=3 sources=2")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("recipe", type=Path)
    prepare_parser.add_argument("output", type=Path)
    prepare_parser.add_argument("--recipe-sha256", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("corpus", type=Path)
    subparsers.add_parser("self-test")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "validate":
        manifest = validate(args.corpus)
        print(
            "FLASHMAP_PROMPT_CORPUS_VALID "
            f"records={manifest['records']} sha256={manifest['prompt_sha256']}"
        )
    else:
        self_test()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CorpusError as error:
        print(f"FLASHMAP_PROMPT_CORPUS_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error

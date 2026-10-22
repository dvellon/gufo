#!/usr/bin/env python3
"""Normalize the six pinned FlashMap prompt sources without teacher answers.

The source inventory below is an approval boundary, not a floating dataset
catalog.  Remote Parquet objects are authenticated by the repository commit,
LFS SHA-256, and byte length exposed by the Hugging Face resolver.  Only a
deterministically ranked set of Parquet row groups and prompt columns is read.
No assistant response is written to disk.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import requests
except ImportError as error:  # pragma: no cover - host dependency diagnostic
    raise SystemExit(
        "flashmap_prompt_sources.py requires pyarrow and requests"
    ) from error


FORMAT = "gufo.flashmap.normalized-prompt-sources.v1"
RECIPE_FORMAT = "gufo.flashmap.prompt-corpus-recipe.v1"
SELECTION = "sha256-ranked-parquet-row-groups-then-records-v1"
SEED = "ddecdaf434055163aea9f7bfad6619e9a973a5e6634753bd799b3e2690d79c91"
ROW_FIELDS = {"messages", "record_id"}
MESSAGE_FIELDS = {"content", "role"}
ALLOWED_PREFIX_ROLES = {"system", "developer"}
MAX_LINE_BYTES = 16 * 1024 * 1024
HTTP_BLOCK_BYTES = 4 * 1024 * 1024
OPENR1_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


class SourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawObject:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class SourceSpec:
    key: str
    adapter: str
    category: str
    dataset_id: str
    revision: str
    subset: str
    split: str
    license_spdx: str
    license_url: str
    output: str
    quota: int
    min_prompt_chars: int
    max_prompt_chars: int
    candidate_records: int
    objects: tuple[RawObject, ...]


SMOLTALK_REVISION = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"
BIGCODE_REVISION = "356bb069eee815daa6e23e9a282eeefe1490ad44"
OPENR1_REVISION = "e4e141ec9dea9f8326f4d347be56105859b2bd68"


def _raw(path: str, sha256: str, size: int) -> RawObject:
    return RawObject(path, sha256, size)


SOURCES = (
    SourceSpec(
        "smol-magpie-ultra",
        "smoltalk-messages-v1",
        "general-instruction",
        "HuggingFaceTB/smoltalk",
        SMOLTALK_REVISION,
        "smol-magpie-ultra",
        "train",
        "Apache-2.0",
        "https://www.apache.org/licenses/LICENSE-2.0",
        "smol-magpie-ultra.normalized.jsonl",
        1750,
        48,
        12000,
        5250,
        (
            _raw("data/smol-magpie-ultra/train-00000-of-00006.parquet", "512cb90bf3c8a6772e97c43977d10177b2726118a107eaf8de4797a604e09df8", 232402904),
            _raw("data/smol-magpie-ultra/train-00001-of-00006.parquet", "abd9a7addee6a33448a887dfa8f2d405f3ad5ad25ed7a87b6bbebdce1232d0ab", 233035425),
            _raw("data/smol-magpie-ultra/train-00002-of-00006.parquet", "f4271506f5ece9c1c1a5bcbaa9c5a39c7334be6830a2a25c1597985578751172", 232992813),
            _raw("data/smol-magpie-ultra/train-00003-of-00006.parquet", "397604789df6e8c874a542a5f71db65838e829bdf0d61eeafeb119fed80bf517", 232333293),
            _raw("data/smol-magpie-ultra/train-00004-of-00006.parquet", "f270bf6178b6f157907ff267cd7fe7f175513eeda5340b5cf7dbb439faaa3bae", 232585260),
            _raw("data/smol-magpie-ultra/train-00005-of-00006.parquet", "da97a3af7cff06a6f29f7f3791cc319b2c31a1795d74cf22d9abf1a074bf31c7", 232747231),
        ),
    ),
    SourceSpec(
        "smol-constraints",
        "smoltalk-messages-v1",
        "constraint-following",
        "HuggingFaceTB/smoltalk",
        SMOLTALK_REVISION,
        "smol-constraints",
        "train",
        "Apache-2.0",
        "https://www.apache.org/licenses/LICENSE-2.0",
        "smol-constraints.normalized.jsonl",
        500,
        48,
        12000,
        1500,
        (_raw("data/smol-constraints/train-00000-of-00001.parquet", "3369eaff911d3511ee21561a3b8607e1acfd773a6ade80f4759490ffdcce7ba4", 16862908),),
    ),
    SourceSpec(
        "smol-rewrite",
        "smoltalk-messages-v1",
        "editing",
        "HuggingFaceTB/smoltalk",
        SMOLTALK_REVISION,
        "smol-rewrite",
        "train",
        "Apache-2.0",
        "https://www.apache.org/licenses/LICENSE-2.0",
        "smol-rewrite.normalized.jsonl",
        250,
        48,
        12000,
        750,
        (_raw("data/smol-rewrite/train-00000-of-00001.parquet", "12c7df2f0ec53fa35c329c1d10243cbb744bb04b6c52823db8e464835747ccba", 38927074),),
    ),
    SourceSpec(
        "smol-summarize",
        "smoltalk-messages-v1",
        "summarization",
        "HuggingFaceTB/smoltalk",
        SMOLTALK_REVISION,
        "smol-summarize",
        "train",
        "Apache-2.0",
        "https://www.apache.org/licenses/LICENSE-2.0",
        "smol-summarize.normalized.jsonl",
        250,
        48,
        12000,
        750,
        (_raw("data/smol-summarize/train-00000-of-00001.parquet", "6fb82c0cb0b6bc4cc484057b035b59d83da8a32cd675379628bc7d295a139229", 119245090),),
    ),
    SourceSpec(
        "self-oss-instruct",
        "bigcode-instruction-v1",
        "code",
        "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        BIGCODE_REVISION,
        "default",
        "train",
        "ODC-By-1.0",
        "https://opendatacommons.org/licenses/by/1-0/",
        "self-oss-instruct.normalized.jsonl",
        1250,
        48,
        12000,
        3750,
        (_raw("data/train-00000-of-00001.parquet", "7bd6092e9940aedebe31f7049c460ba1549a920de5cb4b9bb36b0093cd51fc93", 90128158),),
    ),
    SourceSpec(
        "openr1-math-default",
        "openr1-problem-v1",
        "reasoning-math",
        "open-r1/OpenR1-Math-220k",
        OPENR1_REVISION,
        "default",
        "train",
        "Apache-2.0",
        "https://www.apache.org/licenses/LICENSE-2.0",
        "openr1-math-default.normalized.jsonl",
        1000,
        48,
        12000,
        3000,
        tuple(
            _raw(path, sha256, size)
            for path, sha256, size in (
                ("data/train-00000-of-00010.parquet", "ccc3a95e4875dc4cff9bec2f0fb1c93e4c9cbb24fb24853fd0b96c416e1a263c", 213636225),
                ("data/train-00001-of-00010.parquet", "a7d3eb54d97f7e1237a872947f2246de54a104be224646f62f588abdf6c7db60", 214580713),
                ("data/train-00002-of-00010.parquet", "a44a2354bfa752541ab768b62edf01f3df4b9b2a7660774ab1eb6e7f19f0a86b", 214585972),
                ("data/train-00003-of-00010.parquet", "dd7d2f0f04d80b3ae08484b706c4f7fd9dd29eeff4038c36257a307c77b877e9", 216773329),
                ("data/train-00004-of-00010.parquet", "e358157990d7744643e25f1e692556ac5fd4a2212c008cf67475b14e040861ea", 214835490),
                ("data/train-00005-of-00010.parquet", "30254d14d17b63d1f7963488d53a2d26aa22070f71baa2fe601201b081e38060", 214488314),
                ("data/train-00006-of-00010.parquet", "b3bce93c06d4dc22788c04d0a1da5cbadb5b3db31046809c59a4ec26e96ebb52", 215593265),
                ("data/train-00007-of-00010.parquet", "7def8029cf6d40f7493e9aa4be4b49c251ae7e0cb23b0b117c485e09ff5d5021", 216310260),
                ("data/train-00008-of-00010.parquet", "29cf75a663e91039fc92bffd86a9ad845f992302da52d1fb1563b6c41ae777ed", 213801211),
                ("data/train-00009-of-00010.parquet", "14df2775196a701b97a71cf0285fa75d97980fbda4db9ec9af9c2a07658ca256", 215293135),
            )
        ),
    ),
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank(*parts: str) -> str:
    return _sha256(b"\x00".join(part.encode("utf-8") for part in parts))


def _quoted_header(value: str | None) -> str:
    if not value:
        return ""
    return value[1:-1] if len(value) >= 2 and value[0] == value[-1] == '"' else value


class HttpRangeReader(io.RawIOBase):
    """Small authenticated seekable view over a pinned remote object."""

    def __init__(self, spec: SourceSpec, raw: RawObject):
        super().__init__()
        self._spec = spec
        self._raw = raw
        self._position = 0
        self._blocks: dict[int, bytes] = {}
        self._session = requests.Session()
        self._resolved_url = self._authenticate()
        self.bytes_downloaded = 0

    def _authenticate(self) -> str:
        url = (
            f"https://huggingface.co/datasets/{self._spec.dataset_id}/resolve/"
            f"{self._spec.revision}/{self._raw.path}"
        )
        try:
            response = self._session.head(url, allow_redirects=False, timeout=30)
        except requests.RequestException as error:
            raise SourceError(f"cannot authenticate {self._raw.path}") from error
        if response.status_code not in (302, 303, 307, 308):
            raise SourceError(f"resolver did not redirect pinned object {self._raw.path}")
        if response.headers.get("X-Repo-Commit") != self._spec.revision:
            raise SourceError(f"repository revision differs for {self._raw.path}")
        if _quoted_header(response.headers.get("X-Linked-ETag")) != self._raw.sha256:
            raise SourceError(f"LFS SHA-256 differs for {self._raw.path}")
        try:
            linked_size = int(response.headers.get("X-Linked-Size", ""))
        except ValueError as error:
            raise SourceError(f"object size header is invalid for {self._raw.path}") from error
        if linked_size != self._raw.size:
            raise SourceError(f"object byte length differs for {self._raw.path}")
        location = response.headers.get("Location")
        if not location or not location.startswith("https://"):
            raise SourceError(f"object location is invalid for {self._raw.path}")
        return location

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._raw.size + offset
        else:
            raise ValueError("invalid seek origin")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def _block(self, ordinal: int) -> bytes:
        cached = self._blocks.get(ordinal)
        if cached is not None:
            return cached
        start = ordinal * HTTP_BLOCK_BYTES
        end = min(self._raw.size, start + HTTP_BLOCK_BYTES) - 1
        if start >= self._raw.size:
            return b""
        try:
            response = self._session.get(
                self._resolved_url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=120,
            )
        except requests.RequestException as error:
            raise SourceError(f"range read failed for {self._raw.path}") from error
        expected = end - start + 1
        expected_range = f"bytes {start}-{end}/{self._raw.size}"
        if (
            response.status_code != 206
            or len(response.content) != expected
            or response.headers.get("Content-Range") != expected_range
        ):
            raise SourceError(f"range response is invalid for {self._raw.path}")
        self.bytes_downloaded += len(response.content)
        self._blocks[ordinal] = response.content
        return response.content

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self._raw.size - self._position
        size = min(size, max(0, self._raw.size - self._position))
        output: list[bytes] = []
        while size:
            ordinal = self._position // HTTP_BLOCK_BYTES
            block = self._block(ordinal)
            offset = self._position % HTTP_BLOCK_BYTES
            take = min(size, len(block) - offset)
            if take <= 0:
                break
            output.append(block[offset : offset + take])
            self._position += take
            size -= take
        return b"".join(output)

    def readinto(self, buffer: bytearray) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        self._session.close()
        super().close()


def _validate_schema(spec: SourceSpec, schema: pa.Schema) -> None:
    if spec.adapter == "smoltalk-messages-v1":
        expected_names = (
            [
                "messages",
                "category",
                "difficulty",
                "quality",
                "reward_model_score",
                "conversation_tokens",
            ]
            if spec.key == "smol-magpie-ultra"
            else ["messages"]
        )
        if schema.names != expected_names:
            raise SourceError(f"{spec.key} Parquet schema differs")
        field = schema.field("messages")
        if not pa.types.is_list(field.type):
            raise SourceError(f"{spec.key} messages column is not a list")
        item = field.type.value_type
        if not pa.types.is_struct(item) or [child.name for child in item] != ["content", "role"]:
            raise SourceError(f"{spec.key} message item schema differs")
    elif spec.adapter == "bigcode-instruction-v1":
        expected = ["fingerprint", "sha1", "seed", "response", "concepts", "prompt", "instruction", "id"]
        if schema.names != expected or not pa.types.is_string(schema.field("instruction").type) or not pa.types.is_int64(schema.field("id").type):
            raise SourceError("BigCode Parquet schema differs")
    elif spec.adapter == "openr1-problem-v1":
        expected = ["problem", "solution", "answer", "problem_type", "question_type", "source", "uuid", "is_reasoning_complete", "generations", "correctness_math_verify", "correctness_llama", "finish_reasons", "correctness_count", "messages"]
        if schema.names != expected or not pa.types.is_string(schema.field("problem").type) or not pa.types.is_string(schema.field("uuid").type):
            raise SourceError("OpenR1 Parquet schema differs")
    else:  # pragma: no cover - inventory construction guard
        raise SourceError(f"unknown source adapter {spec.adapter}")


def _prompt_prefix(messages: object) -> list[dict[str, str]] | None:
    if not isinstance(messages, list):
        return None
    result: list[dict[str, str]] = []
    for raw in messages:
        if not isinstance(raw, dict) or set(raw) != MESSAGE_FIELDS:
            return None
        role = raw.get("role")
        content = raw.get("content")
        if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
            return None
        if not _portable_content(content):
            return None
        if role == "user":
            result.append({"content": content, "role": "user"})
            return result
        if role not in ALLOWED_PREFIX_ROLES:
            return None
        result.append({"content": content, "role": role})
    return None


def _portable_content(content: str) -> bool:
    return not any(
        ord(character) < 32 and character not in "\n\r\t" for character in content
    )


def _adapt_row(spec: SourceSpec, row: dict[str, object]) -> tuple[str, list[dict[str, str]]] | None:
    if spec.adapter == "smoltalk-messages-v1":
        messages = _prompt_prefix(row.get("messages"))
        if messages is None:
            return None
        prompt_digest = _sha256(_canonical(messages).rstrip(b"\n"))
        return f"prompt-sha256:{prompt_digest}", messages
    if spec.adapter == "bigcode-instruction-v1":
        identifier = row.get("id")
        instruction = row.get("instruction")
        if isinstance(identifier, bool) or not isinstance(identifier, int):
            return None
        if (
            not isinstance(instruction, str)
            or not instruction.strip()
            or not _portable_content(instruction)
        ):
            return None
        return str(identifier), [{"content": instruction, "role": "user"}]
    if spec.adapter == "openr1-problem-v1":
        identifier = row.get("uuid")
        problem = row.get("problem")
        if not isinstance(identifier, str) or not identifier or "\x00" in identifier:
            return None
        if (
            not isinstance(problem, str)
            or not problem.strip()
            or not _portable_content(problem)
        ):
            return None
        content = f"{OPENR1_INSTRUCTION}\n\n{problem}"
        return identifier, [{"content": content, "role": "user"}]
    raise SourceError(f"unknown adapter {spec.adapter}")


def _columns(spec: SourceSpec) -> list[str]:
    if spec.adapter == "smoltalk-messages-v1":
        return ["messages"]
    if spec.adapter == "bigcode-instruction-v1":
        return ["id", "instruction"]
    return ["uuid", "problem"]


@dataclass
class OpenParquet:
    raw: RawObject
    reader: HttpRangeReader
    parquet: pq.ParquetFile


def _open_parquets(spec: SourceSpec) -> list[OpenParquet]:
    opened: list[OpenParquet] = []
    try:
        for raw in spec.objects:
            reader = HttpRangeReader(spec, raw)
            parquet = pq.ParquetFile(reader)
            _validate_schema(spec, parquet.schema_arrow)
            opened.append(OpenParquet(raw, reader, parquet))
        return opened
    except Exception:
        for item in opened:
            item.reader.close()
        raise


def _normalize_source(spec: SourceSpec) -> tuple[bytes, dict[str, object]]:
    opened = _open_parquets(spec)
    rejection_counts: collections.Counter[str] = collections.Counter()
    candidates: dict[str, tuple[str, dict[str, object]]] = {}
    message_owners: set[str] = set()
    row_groups_read: dict[str, list[int]] = {raw.path: [] for raw in spec.objects}
    try:
        groups: list[tuple[str, int, OpenParquet]] = []
        for item in opened:
            for ordinal in range(item.parquet.num_row_groups):
                groups.append((_rank(SEED, spec.key, item.raw.sha256, str(ordinal)), ordinal, item))
        for _, ordinal, item in sorted(groups, key=lambda value: value[0]):
            table = item.parquet.read_row_group(ordinal, columns=_columns(spec))
            row_groups_read[item.raw.path].append(ordinal)
            for raw_row in table.to_pylist():
                adapted = _adapt_row(spec, raw_row)
                if adapted is None:
                    rejection_counts["schema_or_prompt"] += 1
                    continue
                record_id, messages = adapted
                prompt_chars = sum(len(message["content"]) for message in messages)
                if not spec.min_prompt_chars <= prompt_chars <= spec.max_prompt_chars:
                    rejection_counts["character_envelope"] += 1
                    continue
                row = {"messages": messages, "record_id": record_id}
                message_sha = _sha256(_canonical(messages).rstrip(b"\n"))
                if record_id in candidates:
                    rejection_counts["duplicate_record_id"] += 1
                    continue
                if message_sha in message_owners:
                    rejection_counts["duplicate_messages"] += 1
                    continue
                record_rank = _rank(SEED, spec.key, record_id, message_sha)
                candidates[record_id] = (record_rank, row)
                message_owners.add(message_sha)
            if len(candidates) >= spec.candidate_records:
                break
        if len(candidates) < spec.candidate_records:
            raise SourceError(
                f"{spec.key} yielded {len(candidates)} candidates; "
                f"required {spec.candidate_records}"
            )
        selected = sorted(candidates.values(), key=lambda value: (value[0], str(value[1]["record_id"])))[: spec.candidate_records]
        rows = sorted((value[1] for value in selected), key=lambda row: str(row["record_id"]))
        payload = b"".join(_canonical(row) for row in rows)
        objects = []
        for item in opened:
            objects.append(
                {
                    "bytes_downloaded": item.reader.bytes_downloaded,
                    "lfs_sha256": item.raw.sha256,
                    "path": item.raw.path,
                    "row_groups": row_groups_read[item.raw.path],
                    "size": item.raw.size,
                }
            )
        note = {
            "adapter": spec.adapter,
            "candidate_records": len(rows),
            "category": spec.category,
            "dataset_id": spec.dataset_id,
            "dataset_revision": spec.revision,
            "license_spdx": spec.license_spdx,
            "license_url": spec.license_url,
            "max_prompt_chars": spec.max_prompt_chars,
            "min_prompt_chars": spec.min_prompt_chars,
            "normalized_bytes": len(payload),
            "normalized_file": spec.output,
            "normalized_sha256": _sha256(payload),
            "objects": objects,
            "quota": spec.quota,
            "rejections": dict(sorted(rejection_counts.items())),
            "selection": SELECTION,
            "split": spec.split,
            "subset": spec.subset,
        }
        return payload, note
    finally:
        for item in opened:
            item.reader.close()


def _recipe(notes: list[dict[str, object]]) -> dict[str, object]:
    by_category = {str(note["category"]): note for note in notes}
    return {
        "expected_records": sum(spec.quota for spec in SOURCES),
        "format": RECIPE_FORMAT,
        "seed": SEED,
        "sources": [
            {
                "category": spec.category,
                "dataset_id": spec.dataset_id,
                "dataset_revision": spec.revision,
                "dataset_url": f"https://huggingface.co/datasets/{spec.dataset_id}/tree/{spec.revision}",
                "input": spec.output,
                "input_sha256": by_category[spec.category]["normalized_sha256"],
                "license_spdx": spec.license_spdx,
                "license_url": spec.license_url,
                "max_prompt_chars": spec.max_prompt_chars,
                "min_prompt_chars": spec.min_prompt_chars,
                "quota": spec.quota,
                "split": spec.split,
                "subset": spec.subset,
            }
            for spec in SOURCES
        ],
    }


def prepare(output: Path) -> Path:
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise SourceError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        notes: list[dict[str, object]] = []
        for spec in SOURCES:
            payload, note = _normalize_source(spec)
            (temporary / spec.output).write_bytes(payload)
            notes.append(note)
            print(
                "FLASHMAP_SOURCE_NORMALIZED "
                f"source={spec.key} records={note['candidate_records']} "
                f"sha256={note['normalized_sha256']}"
            )
        recipe_payload = _canonical(_recipe(notes))
        recipe_name = "flashmap_prompt_corpus_recipe.json"
        (temporary / recipe_name).write_bytes(recipe_payload)
        manifest: dict[str, object] = {
            "format": FORMAT,
            "manifest_content_sha256": "",
            "recipe_file": recipe_name,
            "recipe_sha256": _sha256(recipe_payload),
            "selection": SELECTION,
            "sources": notes,
            "tool_sha256": _sha256_file(Path(__file__).resolve()),
        }
        unsigned = dict(manifest)
        unsigned.pop("manifest_content_sha256")
        manifest["manifest_content_sha256"] = _sha256(_canonical(unsigned))
        (temporary / "source-manifest.json").write_bytes(_canonical(manifest))
        os.rename(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validate(output)
    print(
        "FLASHMAP_SOURCES_OK "
        f"sources={len(SOURCES)} recipe_sha256={manifest['recipe_sha256']} output={output}"
    )
    return output


def _load_canonical(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n") or len(payload) > MAX_LINE_BYTES:
        raise SourceError(f"{path.name} is not bounded newline-terminated JSON")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceError(f"{path.name} is not JSON") from error
    if not isinstance(value, dict) or _canonical(value) != payload:
        raise SourceError(f"{path.name} is not canonical JSON")
    return value


def validate(root: Path) -> dict[str, object]:
    root = root.resolve()
    manifest = _load_canonical(root / "source-manifest.json")
    if manifest.get("format") != FORMAT or manifest.get("selection") != SELECTION:
        raise SourceError("source manifest format differs")
    unsigned = dict(manifest)
    content_sha = unsigned.pop("manifest_content_sha256", None)
    if content_sha != _sha256(_canonical(unsigned)):
        raise SourceError("source manifest content digest differs")
    if manifest.get("tool_sha256") != _sha256_file(Path(__file__).resolve()):
        raise SourceError("source normalizer identity differs")
    recipe_name = manifest.get("recipe_file")
    if recipe_name != "flashmap_prompt_corpus_recipe.json":
        raise SourceError("source recipe filename differs")
    recipe_path = root / recipe_name
    if _sha256_file(recipe_path) != manifest.get("recipe_sha256"):
        raise SourceError("source recipe digest differs")
    recipe = _load_canonical(recipe_path)
    notes = manifest.get("sources")
    if not isinstance(notes, list) or len(notes) != len(SOURCES):
        raise SourceError("source manifest inventory differs")
    expected_files = {"source-manifest.json", recipe_name, *(spec.output for spec in SOURCES)}
    if {path.name for path in root.iterdir()} != expected_files:
        raise SourceError("normalized source directory inventory differs")
    recipe_sources = recipe.get("sources")
    if not isinstance(recipe_sources, list) or len(recipe_sources) != len(SOURCES):
        raise SourceError("source recipe inventory differs")
    for spec, note, recipe_source in zip(SOURCES, notes, recipe_sources, strict=True):
        if not isinstance(note, dict) or not isinstance(recipe_source, dict):
            raise SourceError("source manifest entry is invalid")
        path = root / spec.output
        if _sha256_file(path) != note.get("normalized_sha256"):
            raise SourceError(f"{spec.key} normalized digest differs")
        if path.stat().st_size != note.get("normalized_bytes"):
            raise SourceError(f"{spec.key} normalized byte count differs")
        if recipe_source.get("input_sha256") != note.get("normalized_sha256"):
            raise SourceError(f"{spec.key} recipe binding differs")
        rows = 0
        record_ids: set[str] = set()
        message_ids: set[str] = set()
        with path.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n") or len(line) > MAX_LINE_BYTES:
                    raise SourceError(f"{spec.key} normalized JSONL is invalid")
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != ROW_FIELDS or _canonical(row) != line:
                    raise SourceError(f"{spec.key} normalized row differs")
                record_id = row.get("record_id")
                messages = row.get("messages")
                if not isinstance(record_id, str) or not record_id or record_id in record_ids:
                    raise SourceError(f"{spec.key} record identity differs")
                if _prompt_prefix(messages) != messages:
                    raise SourceError(f"{spec.key} prompt messages differ")
                message_id = _sha256(_canonical(messages).rstrip(b"\n"))
                if message_id in message_ids:
                    raise SourceError(f"{spec.key} repeats prompt messages")
                record_ids.add(record_id)
                message_ids.add(message_id)
                rows += 1
        if rows != note.get("candidate_records") or rows != spec.candidate_records:
            raise SourceError(f"{spec.key} candidate count differs")
    return manifest


def self_test() -> None:
    smol = SOURCES[0]
    assert _adapt_row(smol, {"messages": [{"content": "system", "role": "system"}, {"content": "question", "role": "user"}, {"content": "secret teacher answer", "role": "assistant"}]}) == (
        "prompt-sha256:ce00e8d67825259109a2788afe467937e38a8b949983aab88a4430cd5891400f",
        [{"content": "system", "role": "system"}, {"content": "question", "role": "user"}],
    )
    assert _adapt_row(smol, {"messages": [{"content": "answer first", "role": "assistant"}, {"content": "question", "role": "user"}]}) is None
    assert _adapt_row(SOURCES[4], {"id": 42, "instruction": "Implement a queue."}) == (
        "42",
        [{"content": "Implement a queue.", "role": "user"}],
    )
    math = _adapt_row(SOURCES[5], {"uuid": "u-1", "problem": "What is 2+2?"})
    assert math == (
        "u-1",
        [{"content": f"{OPENR1_INSTRUCTION}\n\nWhat is 2+2?", "role": "user"}],
    )
    print("FLASHMAP_PROMPT_SOURCES_SELFTEST_OK adapters=3 sources=6")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("output", type=Path)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("output", type=Path)
    commands.add_parser("self-test")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "validate":
        manifest = validate(args.output)
        print(
            "FLASHMAP_SOURCES_VALID "
            f"sources={len(manifest['sources'])} recipe_sha256={manifest['recipe_sha256']}"
        )
    else:
        self_test()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, SourceError, pa.ArrowException) as error:
        print(f"FLASHMAP_PROMPT_SOURCES_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error

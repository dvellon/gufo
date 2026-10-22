#!/usr/bin/env python3
"""Create and verify bounded, deterministic FlashMap capture-plan shards.

The source is an export from an exact-target generation run.  It contains
token IDs rather than text so chat templates and tokenizer versions cannot
silently change the training sequence between generation and feature capture.
This tool does not generate responses or run the model.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Sequence


FORMAT = "gufo.flashmap.capture-plan.v1"
FEATURE_ABI_SHA256 = (
    "70c629737b587777b4955b454744b8009ffc23901b97c252d8306a9375f66d7c"
)
SOURCE_FIELDS = {
    "id",
    "source_id",
    "prompt_tokens",
    "response_tokens",
    "target_sha256",
    "generator_sha256",
    "generator_git_commit",
    "generation_config_sha256",
    "metadata",
}
FILE_HEADER_BYTES = 320
RECORD_HEADER_BYTES = 192
RECORD_TRAILER_BYTES = 64
ALIGNMENT = 64
FEATURE_BYTES_PER_TOKEN = 6 * 2560 * 2
MAX_JSONL_LINE_BYTES = 16 << 20
WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class PlanError(ValueError):
    """The source export or capture plan is not safe to use."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except UnicodeEncodeError as error:
        raise PlanError("JSON contains a Unicode surrogate") from error


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PlanError(f"JSON repeats key {key!r}")
        result[key] = value
    return result


def _hex(value: object, field: str, digits: int) -> str:
    if not isinstance(value, str) or len(value) != digits:
        raise PlanError(f"{field} must be {digits} hexadecimal digits")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise PlanError(f"{field} is not hexadecimal") from error
    return value.lower()


def _safe_metadata(value: object, path: str = "metadata") -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _safe_metadata(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PlanError(f"{path} has a non-string key")
            _safe_metadata(item, f"{path}.{key}")
        return
    raise PlanError(f"{path} contains a number or unsupported value")


def _tokens(value: object, field: str, vocabulary_size: int) -> list[int]:
    if not isinstance(value, list) or not value:
        raise PlanError(f"{field} must be a non-empty token array")
    result: list[int] = []
    for token in value:
        if (
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < vocabulary_size
        ):
            raise PlanError(f"{field} contains a token outside the vocabulary")
        result.append(token)
    return result


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def _record_bytes(metadata_bytes: int, tokens: int) -> int:
    token_offset = _align(RECORD_HEADER_BYTES + metadata_bytes)
    feature_offset = _align(token_offset + tokens * 4)
    trailer_offset = _align(feature_offset + tokens * FEATURE_BYTES_PER_TOKEN)
    return trailer_offset + RECORD_TRAILER_BYTES


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise PlanError(f"refusing to overwrite {path}")
    with path.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def _publish_no_replace(staging: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise PlanError(f"refusing to overwrite output directory {output}")
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileW
        move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        move_file.restype = ctypes.c_int
        if not move_file(str(staging), str(output)):
            error = ctypes.get_last_error()
            raise OSError(error, os.strerror(error), str(output))
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise PlanError(
                "renameat2 is required for atomic no-overwrite publication"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(staging), -100, os.fsencode(output), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(output))
        return
    raise PlanError("atomic no-overwrite publication is unsupported on this platform")


def _is_reparse_point(status: os.stat_result) -> bool:
    return bool(
        getattr(status, "st_file_attributes", 0)
        & WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT
    )


def _require_plain_directory(path: Path, field: str) -> None:
    try:
        status = path.lstat()
    except OSError as error:
        raise PlanError(f"{field} is not a readable directory") from error
    if (
        not stat.S_ISDIR(status.st_mode)
        or stat.S_ISLNK(status.st_mode)
        or _is_reparse_point(status)
    ):
        raise PlanError(f"{field} must be a non-reparse directory")


def _require_plain_file(path: Path, field: str) -> None:
    try:
        status = path.lstat()
    except OSError as error:
        raise PlanError(f"{field} is not a readable file") from error
    _require_plain_file_status(status, field)


def _require_plain_file_status(status: os.stat_result, field: str) -> None:
    if (
        not stat.S_ISREG(status.st_mode)
        or stat.S_ISLNK(status.st_mode)
        or _is_reparse_point(status)
    ):
        raise PlanError(f"{field} must be a regular non-reparse file")


def _validate_plan_directory_inventory(root: Path, expected: set[str]) -> None:
    _require_plain_directory(root, "plan directory")
    try:
        entries = list(os.scandir(root))
    except OSError as error:
        raise PlanError("plan directory inventory is not readable") from error
    actual = {entry.name for entry in entries}
    if len(actual) != len(entries) or actual != expected:
        raise PlanError("plan directory contains missing or unexpected entries")
    for entry in entries:
        _require_plain_file(Path(entry.path), f"plan entry {entry.name!r}")


def _read_source(
    source: Path,
    *,
    target_sha256: str,
    generator_sha256: str,
    generator_git_commit: str,
    generation_config_sha256: str,
    vocabulary_size: int,
    max_record_tokens: int,
) -> list[dict[str, object]]:
    try:
        raw = source.read_bytes()
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PlanError("source JSONL is not UTF-8") from error
    if not raw or not raw.endswith(b"\n"):
        raise PlanError("source JSONL must be non-empty and newline terminated")
    rows: list[dict[str, object]] = []
    ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        encoded = line.encode("utf-8")
        if not line or len(encoded) > MAX_JSONL_LINE_BYTES:
            raise PlanError(f"source line {line_number} is empty or too large")
        try:
            value = json.loads(line, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, PlanError) as error:
            raise PlanError(f"source line {line_number} is not strict JSON") from error
        if not isinstance(value, dict) or set(value) != SOURCE_FIELDS:
            raise PlanError(
                f"source line {line_number} must contain exactly "
                f"{sorted(SOURCE_FIELDS)}"
            )
        sample_id = value["id"]
        source_id = value["source_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise PlanError("id must be a non-empty string")
        if sample_id in ids:
            raise PlanError(f"duplicate sample id {sample_id!r}")
        ids.add(sample_id)
        if not isinstance(source_id, str) or not source_id:
            raise PlanError("source_id must be a non-empty string")
        if _hex(value["target_sha256"], "target_sha256", 64) != target_sha256:
            raise PlanError("source target identity does not match the plan")
        if _hex(value["generator_sha256"], "generator_sha256", 64) != generator_sha256:
            raise PlanError("source generator identity does not match the plan")
        if (
            _hex(value["generator_git_commit"], "generator_git_commit", 40)
            != generator_git_commit
        ):
            raise PlanError("source generator Git commit does not match the plan")
        if (
            _hex(value["generation_config_sha256"], "generation_config_sha256", 64)
            != generation_config_sha256
        ):
            raise PlanError("source generation configuration does not match the plan")
        prompt_tokens = _tokens(value["prompt_tokens"], "prompt_tokens", vocabulary_size)
        response_tokens = _tokens(
            value["response_tokens"], "response_tokens", vocabulary_size
        )
        tokens = prompt_tokens + response_tokens
        if len(tokens) > max_record_tokens:
            raise PlanError("prompt and response exceed the record token envelope")
        metadata = value["metadata"]
        if not isinstance(metadata, dict):
            raise PlanError("metadata must be an object")
        _safe_metadata(metadata)
        canonical_source = _canonical(value).rstrip(b"\n")
        capture_metadata = {
            "generation_config_sha256": generation_config_sha256,
            "generator_git_commit": generator_git_commit,
            "generator_sha256": generator_sha256,
            "prompt_token_count": str(len(prompt_tokens)),
            "response_token_count": str(len(response_tokens)),
            "source_record_sha256": _sha256_bytes(canonical_source),
            "target_sha256": target_sha256,
            "user_metadata": metadata,
        }
        capture = {
            "id": sample_id,
            "metadata": capture_metadata,
            "source_id": source_id,
            "tokens": tokens,
        }
        capture_bytes = _canonical(capture)
        if len(capture_bytes.rstrip(b"\n")) > MAX_JSONL_LINE_BYTES:
            raise PlanError("canonical capture row exceeds 16 MiB")
        rows.append(
            {
                "capture": capture,
                "capture_bytes": capture_bytes,
                "prompt_token_count": len(prompt_tokens),
                "tokens": tokens,
                "record_bytes": _record_bytes(
                    len(_canonical(capture_metadata).rstrip(b"\n")), len(tokens)
                ),
            }
        )
    return rows


def _manifest_digest(manifest: dict[str, object]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("plan_content_sha256", None)
    return _sha256_bytes(_canonical(unsigned))


def create_plan(args: argparse.Namespace) -> Path:
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise PlanError(f"refusing to overwrite output directory {output}")
    target = _hex(args.target_sha256, "target_sha256", 64)
    generator = _hex(args.generator_sha256, "generator_sha256", 64)
    generator_git = _hex(args.generator_git, "generator_git_commit", 40)
    generation_config = _hex(
        args.generation_config_sha256, "generation_config_sha256", 64
    )
    if min(
        args.expected_records,
        args.vocabulary_size,
        args.max_record_tokens,
        args.max_shard_records,
        args.max_shard_tokens,
    ) <= 0:
        raise PlanError("all numeric limits must be positive")
    rows = _read_source(
        source,
        target_sha256=target,
        generator_sha256=generator,
        generator_git_commit=generator_git,
        generation_config_sha256=generation_config,
        vocabulary_size=args.vocabulary_size,
        max_record_tokens=args.max_record_tokens,
    )
    if len(rows) != args.expected_records:
        raise PlanError(
            f"source has {len(rows)} records; expected exactly {args.expected_records}"
        )
    for row in rows:
        if len(row["tokens"]) > args.max_shard_tokens:
            raise PlanError("one record exceeds the per-shard token limit")

    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        shards: list[dict[str, object]] = []
        dataset_payload = b"".join(
            _canonical(
                {
                    "input_ids": row["tokens"],
                    "loss_mask": [False] * int(row["prompt_token_count"])
                    + [True]
                    * (len(row["tokens"]) - int(row["prompt_token_count"])),
                }
            )
            for row in rows
        )
        _write_new(temporary / "dataset.jsonl", dataset_payload)
        cursor = 0
        global_index = 0
        while cursor < len(rows):
            selected: list[dict[str, object]] = []
            shard_tokens = 0
            while cursor < len(rows) and len(selected) < args.max_shard_records:
                token_count = len(rows[cursor]["tokens"])
                if selected and shard_tokens + token_count > args.max_shard_tokens:
                    break
                selected.append(rows[cursor])
                shard_tokens += token_count
                cursor += 1
            ordinal = len(shards)
            stem = f"shard-{ordinal:05d}"
            input_name = f"{stem}.input.jsonl"
            alignment_name = f"{stem}.alignment.jsonl"
            input_payload = b"".join(row["capture_bytes"] for row in selected)
            alignment_rows = []
            for record_ordinal, row in enumerate(selected):
                capture = row["capture"]
                alignment_rows.append(
                    {
                        "id": capture["id"],
                        "index": global_index,
                        "record_ordinal": record_ordinal,
                        "source_id": capture["source_id"],
                        "token_ids": row["tokens"],
                    }
                )
                global_index += 1
            alignment_payload = b"".join(_canonical(row) for row in alignment_rows)
            _write_new(temporary / input_name, input_payload)
            _write_new(temporary / alignment_name, alignment_payload)
            shards.append(
                {
                    "alignment": alignment_name,
                    "alignment_bytes": len(alignment_payload),
                    "alignment_sha256": _sha256_bytes(alignment_payload),
                    "expected_gfm_bytes": FILE_HEADER_BYTES
                    + sum(int(row["record_bytes"]) for row in selected),
                    "first_index": global_index - len(selected),
                    "input": input_name,
                    "input_bytes": len(input_payload),
                    "input_sha256": _sha256_bytes(input_payload),
                    "last_index": global_index - 1,
                    "records": len(selected),
                    "shard_ordinal": ordinal,
                    "tokens": shard_tokens,
                }
            )
        manifest: dict[str, object] = {
            "format": FORMAT,
            "feature_abi_sha256": FEATURE_ABI_SHA256,
            "dataset": "dataset.jsonl",
            "dataset_bytes": len(dataset_payload),
            "dataset_sha256": _sha256_bytes(dataset_payload),
            "generation_config_sha256": generation_config,
            "generator_git_commit": generator_git,
            "generator_sha256": generator,
            "limits": {
                "max_record_tokens": args.max_record_tokens,
                "max_shard_records": args.max_shard_records,
                "max_shard_tokens": args.max_shard_tokens,
                "vocabulary_size": args.vocabulary_size,
            },
            "planner_sha256": _sha256_file(Path(__file__).resolve()),
            "records": len(rows),
            "shards": shards,
            "source_bytes": source.stat().st_size,
            "source_sha256": _sha256_file(source),
            "target_sha256": target,
            "tokens": sum(len(row["tokens"]) for row in rows),
        }
        manifest["expected_gfm_bytes"] = sum(
            int(shard["expected_gfm_bytes"]) for shard in shards
        )
        manifest["plan_content_sha256"] = _manifest_digest(manifest)
        _write_new(temporary / "plan.json", _canonical(manifest))
        _publish_no_replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validate_plan(output)
    print(
        "FLASHMAP_5K_PLAN_OK "
        f"records={len(rows)} tokens={manifest['tokens']} shards={len(shards)} "
        f"gfm_bytes={manifest['expected_gfm_bytes']} "
        f"sha256={manifest['plan_content_sha256']} output={output}"
    )
    return output


def validate_plan(root: Path) -> dict[str, object]:
    root = root.absolute()
    _require_plain_directory(root, "plan directory")
    manifest_path = root / "plan.json"
    _require_plain_file(manifest_path, "plan manifest")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, PlanError) as error:
        raise PlanError("plan manifest is not strict JSON") from error
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise PlanError("plan manifest format is not recognized")
    if manifest.get("plan_content_sha256") != _manifest_digest(manifest):
        raise PlanError("plan content digest does not match")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise PlanError("plan contains no shards")
    dataset_name = manifest.get("dataset")
    if (
        not isinstance(dataset_name, str)
        or not dataset_name
        or Path(dataset_name).name != dataset_name
        or dataset_name == "plan.json"
    ):
        raise PlanError("plan dataset filename is invalid")
    expected_files = {"plan.json", dataset_name}
    for ordinal, shard in enumerate(shards):
        if not isinstance(shard, dict) or shard.get("shard_ordinal") != ordinal:
            raise PlanError("plan shard order is not contiguous")
        input_name = shard.get("input")
        alignment_name = shard.get("alignment")
        if (
            not isinstance(input_name, str)
            or not input_name
            or not isinstance(alignment_name, str)
            or not alignment_name
        ):
            raise PlanError("plan shard filenames are invalid")
        if (
            Path(input_name).name != input_name
            or Path(alignment_name).name != alignment_name
        ):
            raise PlanError("plan shard filename escapes the plan directory")
        if (
            input_name == alignment_name
            or input_name in expected_files
            or alignment_name in expected_files
        ):
            raise PlanError("plan shard filenames are not unique")
        expected_files.update((input_name, alignment_name))
    _validate_plan_directory_inventory(root, expected_files)
    dataset_path = root / dataset_name
    if _sha256_file(dataset_path) != manifest.get("dataset_sha256"):
        raise PlanError("training dataset digest does not match")
    if dataset_path.stat().st_size != manifest.get("dataset_bytes"):
        raise PlanError("training dataset byte count does not match")
    dataset_lines = dataset_path.read_text(encoding="utf-8").splitlines()
    total_records = 0
    total_tokens = 0
    total_gfm_bytes = 0
    next_index = 0
    all_capture_tokens: list[list[int]] = []
    for ordinal, shard in enumerate(shards):
        if not isinstance(shard, dict) or shard.get("shard_ordinal") != ordinal:
            raise PlanError("plan shard order is not contiguous")
        input_name = shard.get("input")
        alignment_name = shard.get("alignment")
        if not isinstance(input_name, str) or not isinstance(alignment_name, str):
            raise PlanError("plan shard filenames are invalid")
        if Path(input_name).name != input_name or Path(alignment_name).name != alignment_name:
            raise PlanError("plan shard filename escapes the plan directory")
        input_path = root / input_name
        alignment_path = root / alignment_name
        if _sha256_file(input_path) != shard.get("input_sha256"):
            raise PlanError("capture input digest does not match")
        if _sha256_file(alignment_path) != shard.get("alignment_sha256"):
            raise PlanError("alignment digest does not match")
        input_lines = input_path.read_text(encoding="utf-8").splitlines()
        alignment_lines = alignment_path.read_text(encoding="utf-8").splitlines()
        if len(input_lines) != shard.get("records") or len(alignment_lines) != len(input_lines):
            raise PlanError("shard record count does not match")
        shard_tokens = 0
        for record_ordinal, (input_line, alignment_line) in enumerate(
            zip(input_lines, alignment_lines, strict=True)
        ):
            capture = json.loads(input_line, object_pairs_hook=_unique_object)
            alignment = json.loads(alignment_line, object_pairs_hook=_unique_object)
            if alignment.get("record_ordinal") != record_ordinal:
                raise PlanError("alignment record ordinal is not contiguous")
            if alignment.get("index") != next_index:
                raise PlanError("alignment global index is not contiguous")
            if alignment.get("id") != capture.get("id") or alignment.get(
                "source_id"
            ) != capture.get("source_id"):
                raise PlanError("capture and alignment identity differ")
            if alignment.get("token_ids") != capture.get("tokens"):
                raise PlanError("capture and alignment tokens differ")
            all_capture_tokens.append(capture["tokens"])
            shard_tokens += len(capture["tokens"])
            next_index += 1
        if shard_tokens != shard.get("tokens"):
            raise PlanError("shard token count does not match")
        if shard.get("first_index") != total_records or shard.get("last_index") != next_index - 1:
            raise PlanError("shard global index range does not match")
        total_records += len(input_lines)
        total_tokens += shard_tokens
        total_gfm_bytes += int(shard["expected_gfm_bytes"])
    if (
        total_records != manifest.get("records")
        or total_tokens != manifest.get("tokens")
        or total_gfm_bytes != manifest.get("expected_gfm_bytes")
    ):
        raise PlanError("plan totals do not match")
    if len(dataset_lines) != total_records:
        raise PlanError("training dataset record count does not match")
    for index, (line, tokens) in enumerate(
        zip(dataset_lines, all_capture_tokens, strict=True)
    ):
        row = json.loads(line, object_pairs_hook=_unique_object)
        if not isinstance(row, dict) or set(row) != {"input_ids", "loss_mask"}:
            raise PlanError("training dataset row has unexpected fields")
        if row["input_ids"] != tokens:
            raise PlanError(f"training dataset token mismatch at index {index}")
        mask = row["loss_mask"]
        if (
            not isinstance(mask, list)
            or len(mask) != len(tokens)
            or any(not isinstance(value, bool) for value in mask)
            or not any(mask)
        ):
            raise PlanError(f"training dataset loss mask is invalid at index {index}")
        first_true = mask.index(True)
        if any(mask[:first_true]) or not all(mask[first_true:]):
            raise PlanError(f"training dataset loss mask is not contiguous at index {index}")
    return manifest


def self_test() -> None:
    target = "11" * 32
    generator = "22" * 32
    generator_git = "33" * 20
    config = "44" * 32
    with tempfile.TemporaryDirectory(
        prefix="flashmap-5k-plan-", dir=Path.cwd()
    ) as raw:
        root = Path(raw)
        source = root / "source.jsonl"
        records = []
        for index, (prompt, response) in enumerate(
            (([1, 2], [3]), ([4], [5, 6]), ([7, 8], [9, 10]))
        ):
            records.append(
                {
                    "generation_config_sha256": config,
                    "generator_git_commit": generator_git,
                    "generator_sha256": generator,
                    "id": f"sample-{index}",
                    "metadata": {"split": "self-test"},
                    "prompt_tokens": prompt,
                    "response_tokens": response,
                    "source_id": "synthetic",
                    "target_sha256": target,
                }
            )
        source.write_bytes(b"".join(_canonical(record) for record in records))
        common = argparse.Namespace(
            source=source,
            target_sha256=target,
            generator_sha256=generator,
            generator_git=generator_git,
            generation_config_sha256=config,
            expected_records=3,
            vocabulary_size=256,
            max_record_tokens=8,
            max_shard_records=2,
            max_shard_tokens=6,
        )
        first = root / "plan-a"
        second = root / "plan-b"
        create_plan(argparse.Namespace(**vars(common), output=first))
        create_plan(argparse.Namespace(**vars(common), output=second))
        first_manifest = validate_plan(first)
        second_manifest = validate_plan(second)
        if first_manifest != second_manifest:
            raise AssertionError("identical sources produced different plans")
        for path in first.iterdir():
            if path.is_file() and path.read_bytes() != (second / path.name).read_bytes():
                raise AssertionError("identical sources produced different shard bytes")
        unexpected_directory = first / "unexpected-directory"
        unexpected_directory.mkdir()
        try:
            validate_plan(first)
        except PlanError:
            pass
        else:
            raise AssertionError("unexpected plan directory was accepted")
        unexpected_directory.rmdir()
        rejected_statuses = (
            os.stat_result((stat.S_IFLNK | 0o777,) + (0,) * 9),
            SimpleNamespace(
                st_mode=stat.S_IFREG | 0o600,
                st_file_attributes=WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT,
            ),
        )
        for rejected_status in rejected_statuses:
            try:
                _require_plain_file_status(rejected_status, "test link")
            except PlanError:
                pass
            else:
                raise AssertionError("plan symlink or reparse point was accepted")
        bad = json.loads(_canonical(records[0]))
        bad["target_sha256"] = "55" * 32
        source.write_bytes(_canonical(bad))
        try:
            create_plan(
                argparse.Namespace(
                    **{**vars(common), "expected_records": 1},
                    output=root / "rejected",
                )
            )
        except PlanError:
            pass
        else:
            raise AssertionError("mismatched target identity was accepted")
    print("FLASHMAP_5K_PLAN_SELFTEST_OK records=3 shards=2")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("source", type=Path)
    plan.add_argument("output", type=Path)
    plan.add_argument("--target-sha256", required=True)
    plan.add_argument("--generator-sha256", required=True)
    plan.add_argument("--generator-git", required=True)
    plan.add_argument("--generation-config-sha256", required=True)
    plan.add_argument("--expected-records", type=int, default=5000)
    plan.add_argument("--vocabulary-size", type=int, default=248320)
    plan.add_argument("--max-record-tokens", type=int, default=4096)
    plan.add_argument("--max-shard-records", type=int, default=64)
    plan.add_argument("--max-shard-tokens", type=int, default=32768)
    validate = subparsers.add_parser("validate")
    validate.add_argument("plan", type=Path)
    subparsers.add_parser("self-test")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        create_plan(args)
    elif args.command == "validate":
        manifest = validate_plan(args.plan)
        print(
            "FLASHMAP_5K_PLAN_VALID "
            f"records={manifest['records']} tokens={manifest['tokens']} "
            f"shards={len(manifest['shards'])} "
            f"sha256={manifest['plan_content_sha256']}"
        )
    else:
        self_test()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlanError as error:
        print(f"FLASHMAP_5K_PLAN_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error

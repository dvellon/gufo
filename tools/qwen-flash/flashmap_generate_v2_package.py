#!/usr/bin/env python3
"""Prepare, validate and deterministically assemble FlashMap v2 shards.

This tool never runs a model.  It freezes the expensive generator's complete
work allocation before HIP initialisation and validates its durable products
afterwards.  All paths stored in the manifest are simple relative filenames;
the package itself therefore remains relocatable and immutable.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Iterable


FORMAT = "gufo.flashmap.exact-target-shards.v2"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_PACKAGE_FILE_BYTES = 512 * 1024 * 1024
ROW_KEYS = {
    "generation_config_sha256",
    "generator_git_commit",
    "generator_sha256",
    "id",
    "metadata",
    "prompt_tokens",
    "response_tokens",
    "source_id",
    "target_sha256",
}
MANIFEST_KEYS = {
    "decode_width",
    "expected_records",
    "format",
    "generation_config_sha256",
    "generator_git_commit",
    "generator_sha256",
    "max_response_tokens",
    "runtime_identity",
    "shard_records",
    "shards",
    "source_sha256",
    "target_sha256",
}
SHARD_KEYS = {
    "end",
    "index",
    "prompt_file",
    "prompt_sha256",
    "response_file",
    "start",
}


class PackageError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                       sort_keys=True) + "\n").encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: str, digits: int, field: str) -> None:
    if len(value) != digits or any(c not in "0123456789abcdef" for c in value):
        raise PackageError(f"{field} must be {digits} lowercase hexadecimal digits")


def read_lines(path: Path) -> list[bytes]:
    if path.stat().st_size > MAX_PACKAGE_FILE_BYTES:
        raise PackageError(f"{path} exceeds the 512 MiB byte envelope")
    data = path.read_bytes()
    if not data or not data.endswith(b"\n"):
        raise PackageError(f"{path} must be non-empty and newline-terminated")
    lines = data.splitlines(keepends=True)
    if any(line == b"\n" or len(line) > 16 * 1024 * 1024 for line in lines):
        raise PackageError(f"{path} contains an empty or oversized row")
    return lines


def load_json_line(line: bytes, context: str) -> Any:
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackageError(f"{context} is not valid UTF-8 JSON: {error}") from error
    return value


def write_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def publish_no_replace(partial: Path, output: Path) -> None:
    if os.name == "nt":
        move_file = ctypes.windll.kernel32.MoveFileExW
        move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p,
                              ctypes.c_uint32]
        move_file.restype = ctypes.c_int
        # MOVEFILE_WRITE_THROUGH only.  Omitting MOVEFILE_REPLACE_EXISTING is
        # the mandatory no-overwrite guard.
        if not move_file(str(partial), str(output), 0x8):
            raise PackageError("non-overwriting merged publication failed")
        return
    try:
        os.link(partial, output)
    except OSError as error:
        raise PackageError("non-overwriting merged publication failed") from error
    partial.unlink()


def load_manifest(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PackageError("manifest must be an existing regular file")
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise PackageError("manifest exceeds its 4 MiB byte envelope")
    raw = path.read_bytes()
    if expected_sha256 is not None:
        require_sha(expected_sha256, 64, "manifest sha256")
        if digest_bytes(raw) != expected_sha256:
            raise PackageError("manifest identity differs")
    value = load_json_line(raw, "manifest")
    if not isinstance(value, dict) or set(value) != MANIFEST_KEYS:
        raise PackageError("manifest has the wrong fields")
    if canonical(value) != raw:
        raise PackageError("manifest is not canonical newline-terminated JSON")
    if value["format"] != FORMAT:
        raise PackageError("manifest format differs")
    for field in ("generation_config_sha256", "generator_sha256",
                  "runtime_identity", "source_sha256", "target_sha256"):
        require_sha(value[field], 64, field)
    require_sha(value["generator_git_commit"], 40, "generator_git_commit")
    for field in ("decode_width", "expected_records", "max_response_tokens",
                  "shard_records"):
        if isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 1:
            raise PackageError(f"{field} must be a positive integer")
    if value["decode_width"] > 8:
        raise PackageError("decode_width exceeds the runtime maximum of 8")
    if value["shard_records"] % value["decode_width"] != 0:
        raise PackageError("shard_records must be aligned to decode_width")
    shards = value["shards"]
    if not isinstance(shards, list) or not shards:
        raise PackageError("manifest shards must be a non-empty array")
    cursor = 0
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict) or set(shard) != SHARD_KEYS:
            raise PackageError("shard has the wrong fields")
        if shard["index"] != index or shard["start"] != cursor:
            raise PackageError("shard indices or ordinal ranges are not contiguous")
        if (isinstance(shard["end"], bool) or not isinstance(shard["end"], int)
                or shard["end"] <= shard["start"]):
            raise PackageError("shard end is invalid")
        expected_end = min(value["expected_records"],
                           shard["start"] + value["shard_records"])
        if shard["end"] != expected_end:
            raise PackageError("shard range does not match fixed shard_records")
        require_sha(shard["prompt_sha256"], 64, "prompt_sha256")
        expected_prompt = f"prompts-{index:05d}.jsonl"
        expected_response = f"responses-{index:05d}.jsonl"
        if shard["prompt_file"] != expected_prompt or shard["response_file"] != expected_response:
            raise PackageError("shard filename is not the deterministic filename")
        cursor = shard["end"]
    if cursor != value["expected_records"]:
        raise PackageError("shard ranges do not cover expected_records")
    return value


def validate_package(manifest_path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if manifest_path.name != "manifest.json":
        raise PackageError("package manifest filename must be manifest.json")
    manifest = load_manifest(manifest_path, expected_sha256)
    package = manifest_path.parent
    allowed = {"manifest.json"} | {
        shard["prompt_file"] for shard in manifest["shards"]
    }
    for entry in package.iterdir():
        if entry.name not in allowed or entry.is_symlink() or not entry.is_file():
            raise PackageError(f"unexpected package inventory entry: {entry.name}")
    seen_ids: set[str] = set()
    records = 0
    source_digest = hashlib.sha256()
    for shard in manifest["shards"]:
        path = package / shard["prompt_file"]
        if path.is_symlink() or not path.is_file():
            raise PackageError(f"prompt shard is absent: {shard['prompt_file']}")
        if path.stat().st_size > MAX_PACKAGE_FILE_BYTES:
            raise PackageError("prompt shard exceeds the 512 MiB byte envelope")
        raw = path.read_bytes()
        if digest_bytes(raw) != shard["prompt_sha256"]:
            raise PackageError(f"prompt shard identity differs: {shard['prompt_file']}")
        source_digest.update(raw)
        if not raw or not raw.endswith(b"\n"):
            raise PackageError("prompt shard is not newline-terminated")
        lines = raw.splitlines(keepends=True)
        if any(line == b"\n" or len(line) > 16 * 1024 * 1024
               for line in lines):
            raise PackageError("prompt shard contains an empty or oversized row")
        if len(lines) != shard["end"] - shard["start"]:
            raise PackageError("prompt shard row count differs")
        for line in lines:
            value = load_json_line(line, "prompt row")
            if not isinstance(value, dict) or set(value) != {"id", "messages", "metadata", "source_id"}:
                raise PackageError("prompt row has the wrong fields")
            if not isinstance(value["id"], str) or not value["id"] or value["id"] in seen_ids:
                raise PackageError("prompt IDs must be non-empty and unique")
            seen_ids.add(value["id"])
            records += 1
    if records != manifest["expected_records"]:
        raise PackageError("package record count differs")
    if source_digest.hexdigest() != manifest["source_sha256"]:
        raise PackageError("concatenated prompt shards differ from source identity")
    return manifest


def response_rows(manifest: dict[str, Any], manifest_path: Path,
                  responses: Path, shard: dict[str, Any], *,
                  partial: bool) -> list[bytes]:
    suffix = ".partial" if partial else ""
    path = responses / (shard["response_file"] + suffix)
    if path.is_symlink() or not path.is_file():
        raise PackageError(f"response shard is absent: {path.name}")
    lines = read_lines(path)
    expected = shard["end"] - shard["start"]
    if len(lines) > expected or (not partial and len(lines) != expected):
        raise PackageError("response shard row count differs")
    prompt_path = manifest_path.parent / shard["prompt_file"]
    prompt_raw = prompt_path.read_bytes()
    if (prompt_path.is_symlink() or
            digest_bytes(prompt_raw) != shard["prompt_sha256"]):
        raise PackageError("prompt shard changed during response validation")
    prompt_lines = prompt_raw.splitlines(keepends=True)
    for local, line in enumerate(lines):
        value = load_json_line(line, "response row")
        if not isinstance(value, dict) or set(value) != ROW_KEYS or canonical(value) != line:
            raise PackageError("response row fields or canonical encoding differ")
        prompt = load_json_line(prompt_lines[local], "prompt row")
        exact = {
            "id": prompt["id"],
            "source_id": prompt["source_id"],
            "target_sha256": manifest["target_sha256"],
            "generator_sha256": manifest["generator_sha256"],
            "generator_git_commit": manifest["generator_git_commit"],
            "generation_config_sha256": manifest["generation_config_sha256"],
        }
        if any(value[key] != expected_value for key, expected_value in exact.items()):
            raise PackageError("response row identity differs")
        prompt_tokens = value["prompt_tokens"]
        response_tokens = value["response_tokens"]
        if not isinstance(prompt_tokens, list) or not prompt_tokens or not isinstance(response_tokens, list) or not response_tokens:
            raise PackageError("response row token arrays must be non-empty")
        for token in prompt_tokens + response_tokens:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0 or token > 0xFFFFFFFF:
                raise PackageError("response row contains an invalid token")
        if len(response_tokens) > manifest["max_response_tokens"]:
            raise PackageError("response row exceeds the response budget")
        metadata = value["metadata"]
        source_record_sha = digest_bytes(canonical(prompt).rstrip(b"\n"))
        termination = "length" if len(response_tokens) == manifest["max_response_tokens"] else "stop"
        expected_metadata = {
            "prompt_source_file_sha256": manifest["source_sha256"],
            "prompt_source_record_sha256": source_record_sha,
            "termination": termination,
            "user_metadata": prompt["metadata"],
        }
        if metadata != expected_metadata:
            raise PackageError("response row metadata binding differs")
    return lines


def validate_response_inventory(manifest: dict[str, Any], responses: Path,
                                *, allow_partials: bool) -> None:
    if responses.is_symlink() or not responses.is_dir():
        raise PackageError("responses must be an existing real directory")
    allowed = {shard["response_file"] for shard in manifest["shards"]}
    if allow_partials:
        allowed |= {name + ".partial" for name in allowed}
    for entry in responses.iterdir():
        if entry.name not in allowed or entry.is_symlink() or not entry.is_file():
            raise PackageError(f"unexpected response inventory entry: {entry.name}")
    if allow_partials:
        for shard in manifest["shards"]:
            final = responses / shard["response_file"]
            partial = responses / (shard["response_file"] + ".partial")
            if final.exists() and partial.exists():
                raise PackageError("response shard final and partial both exist")


def prepare(args: argparse.Namespace) -> None:
    for field, digits in ((args.source_sha256, 64), (args.target_sha256, 64),
                          (args.generator_sha256, 64), (args.producer_git, 40),
                          (args.generation_config_sha256, 64),
                          (args.runtime_identity, 64)):
        require_sha(field, digits, "reviewed identity")
    if args.decode_width < 1 or args.decode_width > 8:
        raise PackageError("decode width must be in [1, 8]")
    if args.shard_records < args.decode_width or args.shard_records % args.decode_width:
        raise PackageError("shard records must be a positive decode-width multiple")
    if args.package.exists() or args.package.is_symlink():
        raise PackageError("refusing to overwrite package directory")
    if args.source.stat().st_size > MAX_PACKAGE_FILE_BYTES:
        raise PackageError("source exceeds the 512 MiB byte envelope")
    if digest_file(args.source) != args.source_sha256:
        raise PackageError("source identity differs")
    if (args.runtime_manifest.is_symlink() or
            digest_file(args.runtime_manifest) != args.runtime_identity):
        raise PackageError("runtime manifest identity differs")
    if args.generation_config.stat().st_size > 64 * 1024:
        raise PackageError("generation configuration exceeds 64 KiB")
    config_raw = args.generation_config.read_bytes()
    if digest_bytes(config_raw) != args.generation_config_sha256:
        raise PackageError("generation configuration identity differs")
    config = load_json_line(config_raw, "generation configuration")
    max_response = config.get("max_response_tokens") if isinstance(config, dict) else None
    if isinstance(max_response, bool) or not isinstance(max_response, int) or max_response < 1:
        raise PackageError("generation configuration response budget is invalid")
    lines = read_lines(args.source)
    if len(lines) != args.expected_records:
        raise PackageError("source row count differs from expected_records")
    args.package.mkdir(parents=False)
    shards: list[dict[str, Any]] = []
    try:
        for index, start in enumerate(range(0, len(lines), args.shard_records)):
            end = min(len(lines), start + args.shard_records)
            prompt_file = f"prompts-{index:05d}.jsonl"
            response_file = f"responses-{index:05d}.jsonl"
            payload = b"".join(lines[start:end])
            write_exclusive(args.package / prompt_file, payload)
            shards.append({"end": end, "index": index, "prompt_file": prompt_file,
                           "prompt_sha256": digest_bytes(payload),
                           "response_file": response_file, "start": start})
        manifest = {
            "decode_width": args.decode_width,
            "expected_records": args.expected_records,
            "format": FORMAT,
            "generation_config_sha256": args.generation_config_sha256,
            "generator_git_commit": args.producer_git,
            "generator_sha256": args.generator_sha256,
            "max_response_tokens": max_response,
            "runtime_identity": args.runtime_identity,
            "shard_records": args.shard_records,
            "shards": shards,
            "source_sha256": args.source_sha256,
            "target_sha256": args.target_sha256,
        }
        write_exclusive(args.package / "manifest.json", canonical(manifest))
    except BaseException:
        # The directory is new and task-owned.  Failed preparation never
        # masquerades as a complete package because manifest.json is last.
        raise
    manifest_path = args.package / "manifest.json"
    validate_package(manifest_path)
    print(f"FLASHMAP_V2_PACKAGE_OK records={len(lines)} shards={len(shards)} "
          f"decode_width={args.decode_width} sha256={digest_file(manifest_path)}")


def merge(args: argparse.Namespace) -> None:
    manifest = validate_package(args.manifest, args.manifest_sha256)
    validate_response_inventory(manifest, args.responses, allow_partials=False)
    if args.output.exists() or args.output.is_symlink():
        raise PackageError("refusing to overwrite merged output")
    partial = Path(str(args.output) + ".partial")
    if partial.exists() or partial.is_symlink():
        raise PackageError("refusing to overwrite merged output partial")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(partial, flags, 0o600)
    records = 0
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            for shard in manifest["shards"]:
                for line in response_rows(manifest, args.manifest, args.responses,
                                          shard, partial=False):
                    stream.write(line)
                    records += 1
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    publish_no_replace(partial, args.output)
    print(f"FLASHMAP_V2_MERGE_OK records={records} sha256={digest_file(args.output)}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("source", type=Path)
    prep.add_argument("generation_config", type=Path)
    prep.add_argument("runtime_manifest", type=Path)
    prep.add_argument("package", type=Path)
    prep.add_argument("--source-sha256", required=True)
    prep.add_argument("--target-sha256", required=True)
    prep.add_argument("--generator-sha256", required=True)
    prep.add_argument("--producer-git", required=True)
    prep.add_argument("--generation-config-sha256", required=True)
    prep.add_argument("--runtime-identity", required=True)
    prep.add_argument("--expected-records", type=int, required=True)
    prep.add_argument("--decode-width", type=int, required=True)
    prep.add_argument("--shard-records", type=int, required=True)
    prep.set_defaults(run=prepare)
    validate = sub.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--manifest-sha256")
    validate.set_defaults(run=lambda args: print(
        "FLASHMAP_V2_PACKAGE_VALID " + digest_file(args.manifest))
        if validate_package(args.manifest, args.manifest_sha256) else None)
    resume = sub.add_parser("validate-resume")
    resume.add_argument("manifest", type=Path)
    resume.add_argument("responses", type=Path)
    resume.add_argument("--manifest-sha256", required=True)
    resume.add_argument("--shard-index", type=int, required=True)
    def run_resume(args: argparse.Namespace) -> None:
        manifest = validate_package(args.manifest, args.manifest_sha256)
        validate_response_inventory(manifest, args.responses,
                                    allow_partials=True)
        if args.shard_index < 0 or args.shard_index >= len(manifest["shards"]):
            raise PackageError("shard index is outside the manifest")
        rows = response_rows(manifest, args.manifest, args.responses,
                             manifest["shards"][args.shard_index], partial=True)
        print(f"FLASHMAP_V2_RESUME_VALID rows={len(rows)}")
    resume.set_defaults(run=run_resume)
    combine = sub.add_parser("merge")
    combine.add_argument("manifest", type=Path)
    combine.add_argument("responses", type=Path)
    combine.add_argument("output", type=Path)
    combine.add_argument("--manifest-sha256", required=True)
    combine.set_defaults(run=merge)
    return result


def main() -> int:
    try:
        args = parser().parse_args()
        args.run(args)
        return 0
    except (OSError, PackageError, ValueError) as error:
        print(f"FLASHMAP_V2_PACKAGE_FAILED: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

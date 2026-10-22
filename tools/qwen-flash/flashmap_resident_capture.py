#!/usr/bin/env python3
"""Build and validate immutable resident FlashMap capture job manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

if os.name == "nt":
    import ctypes


FORMAT = "gufo.flashmap.resident-capture.v1"


class ResidentCaptureError(ValueError):
    """A resident capture job set is mutable, ambiguous, or unsafe."""


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _hex(value: object, field: str, digits: int = 64) -> str:
    if not isinstance(value, str) or len(value) != digits:
        raise ResidentCaptureError(f"{field} must contain {digits} hex digits")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ResidentCaptureError(f"{field} is not hexadecimal") from error
    if len(decoded) * 2 != digits or value != value.lower():
        raise ResidentCaptureError(f"{field} must be lowercase hexadecimal")
    return value


def _integer(value: object, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResidentCaptureError(f"{field} must be an integer")
    if value < 0 or (positive and value == 0) or value > (1 << 53) - 1:
        raise ResidentCaptureError(f"{field} is outside the portable integer range")
    return value


def _regular_file(path: Path, field: str) -> Path:
    if path.is_symlink():
        raise ResidentCaptureError(f"{field} must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ResidentCaptureError(f"{field} must be a regular non-symlink file")
    return resolved


def _abs_normal(path: Path, field: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_absolute():
        raise ResidentCaptureError(f"{field} must be absolute")
    return resolved


def build_manifest(
    *,
    plan_root: Path,
    capture_root: Path,
    shards: Iterable[dict[str, Any]],
    target_sha256: str,
    producer_sha256: str,
    producer_git_commit: str,
    feature_abi_sha256: str,
    max_context: int,
) -> dict[str, object]:
    """Bind one contiguous, entirely absent shard run to exact inputs/outputs."""

    plan_root = _abs_normal(plan_root, "plan root")
    capture_root = _abs_normal(capture_root, "capture root")
    if capture_root.is_symlink() or not capture_root.is_dir():
        raise ResidentCaptureError("capture root must be an existing non-symlink directory")
    jobs: list[dict[str, object]] = []
    seen_inputs: set[Path] = set()
    seen_outputs: set[Path] = set()
    seen_ordinals: set[int] = set()
    next_index: int | None = None
    for shard in shards:
        ordinal = _integer(shard.get("shard_ordinal"), "shard ordinal")
        first_index = _integer(shard.get("first_index"), "first index")
        last_index = _integer(shard.get("last_index"), "last index")
        records = _integer(shard.get("records"), "record count", positive=True)
        tokens = _integer(shard.get("tokens"), "token count", positive=True)
        input_bytes = _integer(shard.get("input_bytes"), "input bytes", positive=True)
        expected_gfm_bytes = _integer(
            shard.get("expected_gfm_bytes"), "expected GFM bytes", positive=True
        )
        input_sha256 = _hex(shard.get("input_sha256"), "input SHA-256")
        if last_index < first_index or last_index - first_index + 1 != records:
            raise ResidentCaptureError("shard record range differs from count")
        if next_index is not None and first_index != next_index:
            raise ResidentCaptureError("resident shards are not contiguous")
        next_index = last_index + 1
        name = shard.get("input")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ResidentCaptureError("shard input must be a plain filename")
        input_path = _regular_file(plan_root / name, "shard input")
        stat = input_path.stat()
        if stat.st_size != input_bytes or _sha256_file(input_path) != input_sha256:
            raise ResidentCaptureError("shard input identity differs from plan")
        output_path = capture_root / f"shard-{ordinal:05d}"
        if output_path.exists() or output_path.is_symlink():
            raise ResidentCaptureError("resident shard output already exists")
        if input_path in seen_inputs or output_path in seen_outputs or ordinal in seen_ordinals:
            raise ResidentCaptureError("resident shard jobs are duplicated")
        seen_inputs.add(input_path)
        seen_outputs.add(output_path)
        seen_ordinals.add(ordinal)
        jobs.append(
            {
                "first_index": first_index,
                "expected_gfm_bytes": expected_gfm_bytes,
                "input": str(input_path),
                "input_bytes": input_bytes,
                "input_sha256": input_sha256,
                "last_index": last_index,
                "output": str(output_path),
                "records": records,
                "shard_ordinal": ordinal,
                "tokens": tokens,
            }
        )
    if not jobs:
        raise ResidentCaptureError("resident capture requires at least one missing shard")
    return {
        "feature_abi_sha256": _hex(feature_abi_sha256, "feature ABI SHA-256"),
        "format": FORMAT,
        "jobs": jobs,
        "max_context": _integer(max_context, "max context", positive=True),
        "producer_git_commit": _hex(producer_git_commit, "producer Git commit", 40),
        "producer_sha256": _hex(producer_sha256, "producer SHA-256"),
        "target_sha256": _hex(target_sha256, "target SHA-256"),
    }


def validate_manifest(path: Path, expected_sha256: str) -> dict[str, object]:
    """Reparse a manifest and require canonical bytes and complete job bindings."""

    manifest_path = _regular_file(path, "resident manifest")
    expected_sha256 = _hex(expected_sha256, "resident manifest SHA-256")
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ResidentCaptureError("resident manifest SHA-256 differs")
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResidentCaptureError("resident manifest is not strict JSON") from error
    if not isinstance(value, dict) or _canonical(value) != payload:
        raise ResidentCaptureError("resident manifest is not canonical JSON")
    required = {
        "feature_abi_sha256",
        "format",
        "jobs",
        "max_context",
        "producer_git_commit",
        "producer_sha256",
        "target_sha256",
    }
    if set(value) != required or value.get("format") != FORMAT:
        raise ResidentCaptureError("resident manifest schema differs")
    jobs = value.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ResidentCaptureError("resident manifest jobs are empty")
    # Rebuild from the immutable plan-shaped fields and current filesystem.
    plan_root = Path(str(jobs[0].get("input"))).parent
    capture_root = Path(str(jobs[0].get("output"))).parent
    shards = []
    for job in jobs:
        if not isinstance(job, dict) or set(job) != {
            "first_index",
            "expected_gfm_bytes",
            "input",
            "input_bytes",
            "input_sha256",
            "last_index",
            "output",
            "records",
            "shard_ordinal",
            "tokens",
        }:
            raise ResidentCaptureError("resident job schema differs")
        if (
            Path(str(job["input"])).parent != plan_root
            or Path(str(job["output"])).parent != capture_root
        ):
            raise ResidentCaptureError("resident jobs do not share fixed roots")
        shards.append({**job, "input": Path(str(job["input"])).name})
    rebuilt = build_manifest(
        plan_root=plan_root,
        capture_root=capture_root,
        shards=shards,
        target_sha256=str(value["target_sha256"]),
        producer_sha256=str(value["producer_sha256"]),
        producer_git_commit=str(value["producer_git_commit"]),
        feature_abi_sha256=str(value["feature_abi_sha256"]),
        max_context=value["max_context"],
    )
    if rebuilt != value:
        raise ResidentCaptureError("resident manifest differs from filesystem bindings")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ResidentCaptureError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def publish_manifest(root: Path, manifest: dict[str, object]) -> tuple[Path, str]:
    """Publish canonical bytes under their digest without replacing anything."""

    if root.is_symlink():
        raise ResidentCaptureError("resident manifest root must not be a symlink")
    root.mkdir(parents=False, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ResidentCaptureError("resident manifest root is unsafe")
    payload = _canonical(manifest)
    digest = hashlib.sha256(payload).hexdigest()
    destination = root / f"sha256-{digest}.json"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise ResidentCaptureError("resident manifest destination differs")
        return destination.resolve(strict=True), digest
    descriptor, raw = tempfile.mkstemp(
        prefix="resident-", suffix=".json.partial", dir=root
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
            move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
            move.restype = ctypes.c_int
            if not move(str(temporary), str(destination), 0x8):
                error = ctypes.get_last_error()
                if error not in (80, 183):
                    raise OSError(error, "resident manifest publication failed")
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
            else:
                temporary.unlink()
        if destination.is_symlink() or destination.read_bytes() != payload:
            raise ResidentCaptureError("resident manifest destination differs")
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve(strict=True), digest


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="flashmap-resident-", dir=Path.cwd()) as raw:
        root = Path(raw)
        plan = root / "plan"
        output = root / "capture"
        manifests = root / "manifests"
        plan.mkdir()
        output.mkdir()
        shards = []
        for ordinal, first in enumerate((10, 12)):
            payload = f'{{"id":"sample-{ordinal}"}}\n'.encode()
            name = f"shard-{ordinal:05d}.input.jsonl"
            (plan / name).write_bytes(payload)
            shards.append(
                {
                    "first_index": first,
                    "expected_gfm_bytes": 4096,
                    "input": name,
                    "input_bytes": len(payload),
                    "input_sha256": hashlib.sha256(payload).hexdigest(),
                    "last_index": first + 1,
                    "records": 2,
                    "shard_ordinal": ordinal,
                    "tokens": 3,
                }
            )
        manifest = build_manifest(
            plan_root=plan,
            capture_root=output,
            shards=shards,
            target_sha256="11" * 32,
            producer_sha256="22" * 32,
            producer_git_commit="33" * 20,
            feature_abi_sha256="44" * 32,
            max_context=4096,
        )
        path, digest = publish_manifest(manifests, manifest)
        if validate_manifest(path, digest) != manifest:
            raise AssertionError("resident manifest did not round trip")
        try:
            validate_manifest(path, "55" * 32)
        except ResidentCaptureError:
            pass
        else:
            raise AssertionError("wrong resident manifest digest was accepted")
        gapped = [dict(shard) for shard in shards]
        gapped[1]["first_index"] = 13
        gapped[1]["last_index"] = 14
        try:
            build_manifest(
                plan_root=plan,
                capture_root=output,
                shards=gapped,
                target_sha256="11" * 32,
                producer_sha256="22" * 32,
                producer_git_commit="33" * 20,
                feature_abi_sha256="44" * 32,
                max_context=4096,
            )
        except ResidentCaptureError:
            pass
        else:
            raise AssertionError("non-contiguous resident jobs were accepted")
        (output / "shard-00000").mkdir()
        try:
            validate_manifest(path, digest)
        except ResidentCaptureError:
            pass
        else:
            raise AssertionError("existing shard output was accepted")
        (output / "shard-00000").rmdir()
        (plan / "shard-00001.input.jsonl").write_bytes(b"changed\n")
        try:
            validate_manifest(path, digest)
        except ResidentCaptureError:
            pass
        else:
            raise AssertionError("changed shard input was accepted")
    print("FLASHMAP_RESIDENT_CAPTURE_SELFTEST_OK shards=2")


if __name__ == "__main__":
    self_test()

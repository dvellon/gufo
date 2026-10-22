#!/usr/bin/env python3
"""Fail-closed orchestration for a completed FlashMap 5K capture run.

This tool starts after exact-target response generation has finished.  It
validates the completed response export, creates or validates its deterministic
capture plan, and runs at most one capture process at a time.  Published GFM
shards are validated against both the plan and the exact target/capture
identities before they are skipped.  Interrupted ``.partial`` files are never
deleted or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import flashmap_5k_plan as planner
import flashmap_resident_capture as resident
from flashmap_trace import TraceError, TraceExpectations, TraceReader, scan_partial


class OrchestrationError(ValueError):
    """The requested run is incomplete, ambiguous, or identity-mismatched."""


GGUF_IDENTITY_SCHEME = "gguf-sha256-v1"
_SPLIT_GGUF = re.compile(r"^(.*)-([0-9]+)-of-([0-9]+)\.gguf$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _gguf_artifact_paths(path: Path) -> list[Path]:
    """Return the filename-ordered shard set used by Gufo's GGUF loader."""

    first = _regular_file(path, "model")
    match = _SPLIT_GGUF.fullmatch(first.name)
    if match is None:
        return [first]
    index = int(match.group(2))
    count = int(match.group(3))
    if not 1 <= index <= count <= 65535:
        raise OrchestrationError("model split filename has an invalid index or count")
    index_width = len(match.group(2))
    count_width = len(match.group(3))
    return [
        _regular_file(
            first.with_name(
                f"{match.group(1)}-{ordinal:0{index_width}d}-of-"
                f"{count:0{count_width}d}.gguf"
            ),
            f"model shard {ordinal}",
        )
        for ordinal in range(1, count + 1)
    ]


def _gguf_artifact_sha256(path: Path) -> str:
    """Compute Gufo's gguf-sha256-v1 identity over the complete shard set."""

    artifact = hashlib.sha256(GGUF_IDENTITY_SCHEME.encode("ascii"))
    for shard in _gguf_artifact_paths(path):
        before = shard.stat()
        digest = _sha256_file(shard)
        after = shard.stat()
        before_stamp = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_stamp = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_stamp != after_stamp:
            raise OrchestrationError("model shard changed during identity hashing")
        artifact.update(f":{before.st_size}:".encode("ascii"))
        artifact.update(digest.encode("ascii"))
    return artifact.hexdigest()


def _hex(value: str, field: str, digits: int) -> str:
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise OrchestrationError(f"{field} is not hexadecimal") from error
    if len(value) != digits or len(decoded) * 2 != digits:
        raise OrchestrationError(f"{field} must contain {digits} hex digits")
    return value.lower()


def _regular_file(path: Path, field: str) -> Path:
    if path.is_symlink():
        raise OrchestrationError(f"{field} must be a regular non-symlink file")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise OrchestrationError(f"{field} must be a regular non-symlink file")
    return path


def _generation_partial(source: Path) -> Path:
    return source.with_name(source.name + ".partial")


def _validate_completed_source(
    args: argparse.Namespace,
) -> tuple[Path, str, list[dict[str, object]]]:
    source = _regular_file(args.responses, "responses")
    partial = _generation_partial(source)
    if partial.exists() or partial.is_symlink():
        raise OrchestrationError(
            f"response generation is not complete; preserving {partial}"
        )
    rows = planner._read_source(
        source,
        target_sha256=args.target_sha256,
        generator_sha256=args.generator_sha256,
        generator_git_commit=args.generator_git,
        generation_config_sha256=args.generation_config_sha256,
        vocabulary_size=args.vocabulary_size,
        max_record_tokens=args.max_record_tokens,
    )
    if len(rows) != args.expected_records:
        raise OrchestrationError(
            f"responses contain {len(rows)} records; expected {args.expected_records}"
        )
    return source, _sha256_file(source), rows


def _validate_plan_bindings(
    manifest: dict[str, object], args: argparse.Namespace, source_sha256: str
) -> None:
    wanted = {
        "target_sha256": args.target_sha256,
        "generator_sha256": args.generator_sha256,
        "generator_git_commit": args.generator_git,
        "generation_config_sha256": args.generation_config_sha256,
        "feature_abi_sha256": planner.FEATURE_ABI_SHA256,
        "planner_sha256": _sha256_file(Path(planner.__file__).resolve()),
        "source_sha256": source_sha256,
        "records": args.expected_records,
    }
    for field, expected in wanted.items():
        if manifest.get(field) != expected:
            raise OrchestrationError(f"plan {field} differs from this run")
    limits = manifest.get("limits")
    expected_limits = {
        "max_record_tokens": args.max_record_tokens,
        "max_shard_records": args.max_shard_records,
        "max_shard_tokens": args.max_shard_tokens,
        "vocabulary_size": args.vocabulary_size,
    }
    if limits != expected_limits:
        raise OrchestrationError("plan limits differ from this run")


def _validate_plan_derivation(
    root: Path,
    manifest: dict[str, object],
    source_rows: list[dict[str, object]],
) -> None:
    cursor = 0
    for shard in manifest["shards"]:
        records = shard["records"]
        if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
            raise OrchestrationError("plan shard record count is invalid")
        selected = source_rows[cursor : cursor + records]
        if len(selected) != records:
            raise OrchestrationError("plan shard ranges exceed the completed responses")
        expected = b"".join(bytes(row["capture_bytes"]) for row in selected)
        input_path = root / str(shard["input"])
        if input_path.read_bytes() != expected:
            raise OrchestrationError(
                "plan capture inputs are not derived from the completed responses"
            )
        cursor += records
    if cursor != len(source_rows):
        raise OrchestrationError("plan does not cover every completed response")


def _prepare(args: argparse.Namespace) -> dict[str, object]:
    source, source_sha256, source_rows = _validate_completed_source(args)
    if args.plan.is_symlink():
        raise OrchestrationError("plan must not be a symlink")
    output = args.plan.resolve()
    if output.exists() or output.is_symlink():
        manifest = planner.validate_plan(output)
        reused = True
    else:
        namespace = argparse.Namespace(
            source=source,
            output=output,
            target_sha256=args.target_sha256,
            generator_sha256=args.generator_sha256,
            generator_git=args.generator_git,
            generation_config_sha256=args.generation_config_sha256,
            expected_records=args.expected_records,
            vocabulary_size=args.vocabulary_size,
            max_record_tokens=args.max_record_tokens,
            max_shard_records=args.max_shard_records,
            max_shard_tokens=args.max_shard_tokens,
        )
        planner.create_plan(namespace)
        manifest = planner.validate_plan(output)
        reused = False
    _validate_plan_bindings(manifest, args, source_sha256)
    _validate_plan_derivation(output, manifest, source_rows)
    print(
        "FLASHMAP_5K_ORCHESTRATION_PLAN_OK"
        f" reused={int(reused)} records={manifest['records']}"
        f" shards={len(manifest['shards'])}"
        f" source_sha256={source_sha256}"
        f" sha256={manifest['plan_content_sha256']}"
    )
    return manifest


def _load_rows(
    path: Path, *, expected_bytes: object, expected_sha256: object
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        payload = path.read_bytes()
        if len(payload) != expected_bytes:
            raise OrchestrationError("capture input byte count differs from plan")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise OrchestrationError("capture input digest differs from plan")
        text = payload.decode("utf-8")
        if not payload or not payload.endswith(b"\n"):
            raise OrchestrationError(
                "capture input must be non-empty and newline terminated"
            )
        for line in text.splitlines():
            value = json.loads(line, object_pairs_hook=planner._unique_object)
            if not isinstance(value, dict):
                raise OrchestrationError("capture input row is not an object")
            rows.append(value)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        planner.PlanError,
    ) as error:
        raise OrchestrationError("capture input is not strict JSONL") from error
    return rows


def _validate_published_filename(trace: Path) -> str:
    prefix = "sha256-"
    suffix = ".gfm"
    name = trace.name
    digest = name[len(prefix) : -len(suffix)]
    if (
        not name.startswith(prefix)
        or not name.endswith(suffix)
        or len(digest) != 64
        or digest != digest.lower()
    ):
        raise OrchestrationError("published trace filename is not canonical")
    _hex(digest, "published trace filename digest", 64)
    actual = _sha256_file(trace)
    if digest != actual:
        raise OrchestrationError("published trace filename digest differs from bytes")
    return actual


def _validate_trace(
    trace: Path,
    expected: TraceExpectations,
    shard: dict[str, object],
    input_rows: list[dict[str, Any]],
) -> None:
    _validate_published_filename(trace)
    if trace.stat().st_size != shard["expected_gfm_bytes"]:
        raise OrchestrationError("published trace byte count differs from plan")
    try:
        with TraceReader(trace, expected) as reader:
            if reader.header.shard_ordinal != shard["shard_ordinal"]:
                raise OrchestrationError("published trace shard ordinal differs")
            if reader.records != shard["records"] or reader.tokens != shard["tokens"]:
                raise OrchestrationError("published trace totals differ from plan")
            records = list(reader)
            if len(records) != len(input_rows):
                raise OrchestrationError("published trace record count differs")
            pairs = zip(records, input_rows, strict=True)
            for ordinal, (record, row) in enumerate(pairs):
                if record.ordinal != ordinal:
                    raise OrchestrationError("published trace record order differs")
                sample = hashlib.sha256(row["id"].encode("utf-8")).digest()
                source = hashlib.sha256(row["source_id"].encode("utf-8")).digest()
                if record.sample_sha256 != sample or record.source_sha256 != source:
                    raise OrchestrationError("published trace record identity differs")
                if record.tokens() != tuple(row["tokens"]):
                    raise OrchestrationError("published trace tokens differ from plan")
                if record.metadata() != row.get("metadata", {}):
                    raise OrchestrationError(
                        "published trace metadata differs from plan"
                    )
    except (OSError, TraceError, KeyError, TypeError) as error:
        if isinstance(error, OrchestrationError):
            raise
        raise OrchestrationError(f"published trace is invalid: {trace}") from error


def _shard_state(
    directory: Path,
    expected: TraceExpectations,
    shard: dict[str, object],
    input_rows: list[dict[str, Any]],
) -> tuple[str, Path | None]:
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise OrchestrationError(f"capture shard path is not a directory: {directory}")
    if not directory.exists():
        return "missing", None
    published = sorted(directory.glob("sha256-*.gfm"))
    partials = sorted(directory.glob("*.gfm.partial"))
    if any(path.is_symlink() for path in [*published, *partials]):
        raise OrchestrationError(f"capture shard contains a symlink: {directory}")
    unexpected = [
        path
        for path in directory.iterdir()
        if path not in published and path not in partials
    ]
    if unexpected:
        raise OrchestrationError(
            f"capture shard directory has unexpected entries: {directory}"
        )
    if len(published) > 1:
        raise OrchestrationError(
            f"capture shard has multiple published traces: {directory}"
        )
    if published:
        _validate_trace(published[0], expected, shard, input_rows)
        return "complete", published[0]
    for partial in partials:
        try:
            result = scan_partial(partial, expected)
        except (OSError, TraceError) as error:
            raise OrchestrationError(
                f"capture partial is invalid: {partial}"
            ) from error
        if result.header.shard_ordinal != shard["shard_ordinal"]:
            raise OrchestrationError("capture partial shard ordinal differs")
        if result.records > shard["records"] or result.tokens > shard["tokens"]:
            raise OrchestrationError("capture partial exceeds planned shard")
    if partials:
        return "partial", partials[0]
    return "missing", None


class _ExclusiveLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "_ExclusiveLock":
        try:
            self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.write(self.fd, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(self.fd)
        except FileExistsError as error:
            raise OrchestrationError(
                f"capture lock already exists: {self.path}"
            ) from error
        return self

    def __exit__(self, *_: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink()


def _capture(args: argparse.Namespace, manifest: dict[str, object]) -> None:
    capture = _regular_file(args.capture_executable, "capture executable")
    model = _regular_file(args.model, "model")
    if _sha256_file(capture) != args.capture_sha256:
        raise OrchestrationError("capture executable SHA-256 differs")
    if _gguf_artifact_sha256(model) != args.target_sha256:
        raise OrchestrationError("model GGUF artifact identity differs from target")
    if args.runtime_temp.is_symlink():
        raise OrchestrationError(
            "runtime temp must be an existing non-symlink directory"
        )
    runtime_temp = args.runtime_temp.resolve(strict=True)
    if not runtime_temp.is_dir():
        raise OrchestrationError(
            "runtime temp must be an existing non-symlink directory"
        )
    if args.capture_output.is_symlink():
        raise OrchestrationError("capture output must be a non-symlink directory")
    output = args.capture_output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or not output.is_dir():
        raise OrchestrationError("capture output must be a non-symlink directory")
    expected = TraceExpectations(
        target_sha256=args.target_sha256,
        producer_sha256=args.capture_sha256,
        producer_git_commit=args.capture_git,
        vocabulary_size=args.vocabulary_size,
        max_record_tokens=args.max_record_tokens,
    )
    launched = 0
    with _ExclusiveLock(output / "capture.lock"):
        if args.resident_capture:
            selected: list[dict[str, object]] = []
            for shard in manifest["shards"]:
                ordinal = int(shard["shard_ordinal"])
                input_path = args.plan.resolve() / str(shard["input"])
                rows = _load_rows(
                    input_path,
                    expected_bytes=shard["input_bytes"],
                    expected_sha256=shard["input_sha256"],
                )
                directory = output / f"shard-{ordinal:05d}"
                state, artifact = _shard_state(directory, expected, shard, rows)
                if state == "complete":
                    if selected:
                        break
                    print(
                        "FLASHMAP_CAPTURE_SHARD_REUSED"
                        f" ordinal={ordinal} output={artifact}"
                    )
                    continue
                if state == "partial":
                    raise OrchestrationError(
                        f"shard {ordinal} has a preserved partial; recover it "
                        "with the single-shard path before resident capture"
                    )
                selected.append(shard)
                if args.max_new_shards and len(selected) >= args.max_new_shards:
                    break
            if not selected:
                print(
                    "FLASHMAP_5K_CAPTURE_PROGRESS_OK"
                    f" launched=0 total_shards={len(manifest['shards'])}"
                )
                return
            job_manifest = resident.build_manifest(
                plan_root=args.plan.resolve(),
                capture_root=output,
                shards=selected,
                target_sha256=args.target_sha256,
                producer_sha256=args.capture_sha256,
                producer_git_commit=args.capture_git,
                feature_abi_sha256=planner.FEATURE_ABI_SHA256,
                max_context=args.max_record_tokens,
            )
            job_path, job_sha256 = resident.publish_manifest(
                output / "resident-jobs", job_manifest
            )
            resident.validate_manifest(job_path, job_sha256)
            command = [
                str(capture),
                "--model",
                str(model),
                "--resident-manifest",
                str(job_path),
                "--resident-manifest-sha256",
                job_sha256,
            ]
            environment = os.environ.copy()
            environment["TEMP"] = str(runtime_temp)
            environment["TMP"] = str(runtime_temp)
            result = subprocess.run(command, env=environment, check=False)
            if result.returncode != 0:
                raise OrchestrationError(
                    "resident capture failed with exit "
                    f"{result.returncode}; published shards and active partial are preserved"
                )
            for shard in selected:
                ordinal = int(shard["shard_ordinal"])
                input_path = args.plan.resolve() / str(shard["input"])
                rows = _load_rows(
                    input_path,
                    expected_bytes=shard["input_bytes"],
                    expected_sha256=shard["input_sha256"],
                )
                directory = output / f"shard-{ordinal:05d}"
                state, artifact = _shard_state(directory, expected, shard, rows)
                if state != "complete":
                    raise OrchestrationError(
                        f"resident capture returned success without valid shard {ordinal}"
                    )
                launched += 1
                print(
                    "FLASHMAP_CAPTURE_SHARD_OK"
                    f" ordinal={ordinal} records={shard['records']}"
                    f" tokens={shard['tokens']} output={artifact}"
                )
            print(
                "FLASHMAP_5K_CAPTURE_PROGRESS_OK"
                f" launched={launched} total_shards={len(manifest['shards'])}"
                f" resident_manifest_sha256={job_sha256}"
            )
            return
        for shard in manifest["shards"]:
            ordinal = int(shard["shard_ordinal"])
            input_path = args.plan.resolve() / str(shard["input"])
            rows = _load_rows(
                input_path,
                expected_bytes=shard["input_bytes"],
                expected_sha256=shard["input_sha256"],
            )
            directory = output / f"shard-{ordinal:05d}"
            state, artifact = _shard_state(directory, expected, shard, rows)
            if state == "complete":
                print(
                    "FLASHMAP_CAPTURE_SHARD_REUSED"
                    f" ordinal={ordinal} output={artifact}"
                )
                continue
            if state == "partial" and not args.retry_partial_shard:
                raise OrchestrationError(
                    f"shard {ordinal} has a preserved partial; rerun with "
                    "--retry-partial-shard to recapture it without deleting the partial"
                )
            directory.mkdir(parents=False, exist_ok=True)
            command = [
                str(capture),
                "--model",
                str(model),
                "--input",
                str(input_path),
                "--output",
                str(directory),
                "--producer-git",
                args.capture_git,
                "--max-context",
                str(args.max_record_tokens),
                "--shard-ordinal",
                str(ordinal),
            ]
            environment = os.environ.copy()
            environment["TEMP"] = str(runtime_temp)
            environment["TMP"] = str(runtime_temp)
            result = subprocess.run(command, env=environment, check=False)
            if result.returncode != 0:
                raise OrchestrationError(
                    f"capture failed for shard {ordinal} with exit {result.returncode}"
                )
            state, artifact = _shard_state(directory, expected, shard, rows)
            if state != "complete":
                raise OrchestrationError(
                    f"capture returned success without a valid shard {ordinal}"
                )
            launched += 1
            print(
                "FLASHMAP_CAPTURE_SHARD_OK"
                f" ordinal={ordinal} records={shard['records']}"
                f" tokens={shard['tokens']} output={artifact}"
            )
            if args.max_new_shards and launched >= args.max_new_shards:
                break
    print(
        "FLASHMAP_5K_CAPTURE_PROGRESS_OK"
        f" launched={launched} total_shards={len(manifest['shards'])}"
    )


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("responses", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--generator-sha256", required=True)
    parser.add_argument("--generator-git", required=True)
    parser.add_argument("--generation-config-sha256", required=True)
    parser.add_argument("--expected-records", type=int, default=5000)
    parser.add_argument("--vocabulary-size", type=int, default=248320)
    parser.add_argument("--max-record-tokens", type=int, default=4096)
    parser.add_argument("--max-shard-records", type=int, default=64)
    parser.add_argument("--max-shard-tokens", type=int, default=32768)


def _normalize(args: argparse.Namespace) -> None:
    args.target_sha256 = _hex(args.target_sha256, "target_sha256", 64)
    args.generator_sha256 = _hex(args.generator_sha256, "generator_sha256", 64)
    args.generator_git = _hex(args.generator_git, "generator_git", 40)
    args.generation_config_sha256 = _hex(
        args.generation_config_sha256, "generation_config_sha256", 64
    )
    if hasattr(args, "capture_sha256"):
        args.capture_sha256 = _hex(args.capture_sha256, "capture_sha256", 64)
        args.capture_git = _hex(args.capture_git, "capture_git", 40)
        if args.max_new_shards < 0:
            raise OrchestrationError("max_new_shards must be nonnegative")
        if args.resident_capture and args.retry_partial_shard:
            raise OrchestrationError(
                "resident capture does not accept partial-shard retry; recover the "
                "partial with the single-shard path first"
            )
    for name in (
        "expected_records",
        "vocabulary_size",
        "max_record_tokens",
        "max_shard_records",
        "max_shard_tokens",
    ):
        if getattr(args, name) <= 0:
            raise OrchestrationError(f"{name} must be positive")


def _self_test() -> None:
    target = "11" * 32
    generator = "22" * 32
    producer_git = "33" * 20
    config = "44" * 32
    with tempfile.TemporaryDirectory(
        prefix="flashmap-orchestrate-", dir=Path.cwd()
    ) as raw:
        root = Path(raw)
        source = root / "responses.jsonl"
        records = []
        for index in range(3):
            records.append(
                {
                    "generation_config_sha256": config,
                    "generator_git_commit": producer_git,
                    "generator_sha256": generator,
                    "id": f"sample-{index}",
                    "metadata": {"split": "self-test"},
                    "prompt_tokens": [index + 1],
                    "response_tokens": [index + 11],
                    "source_id": "synthetic",
                    "target_sha256": target,
                }
            )
        source.write_bytes(b"".join(planner._canonical(row) for row in records))
        args = argparse.Namespace(
            responses=source,
            plan=root / "plan",
            target_sha256=target,
            generator_sha256=generator,
            generator_git=producer_git,
            generation_config_sha256=config,
            expected_records=3,
            vocabulary_size=256,
            max_record_tokens=8,
            max_shard_records=2,
            max_shard_tokens=4,
        )
        first = _prepare(args)
        second = _prepare(args)
        if first != second:
            raise AssertionError("plan reuse changed the validated manifest")
        changed_source = root / "changed-responses.jsonl"
        changed_records = [dict(row) for row in records]
        changed_records[0]["id"] = "different-sample"
        changed_source.write_bytes(
            b"".join(planner._canonical(row) for row in changed_records)
        )
        changed_rows = planner._read_source(
            changed_source,
            target_sha256=target,
            generator_sha256=generator,
            generator_git_commit=producer_git,
            generation_config_sha256=config,
            vocabulary_size=256,
            max_record_tokens=8,
        )
        try:
            _validate_plan_derivation(args.plan, first, changed_rows)
        except OrchestrationError:
            pass
        else:
            raise AssertionError("plan unrelated to completed responses was accepted")
        first_shard = first["shards"][0]
        first_input = args.plan / first_shard["input"]
        _load_rows(
            first_input,
            expected_bytes=first_shard["input_bytes"],
            expected_sha256=first_shard["input_sha256"],
        )
        original_input = first_input.read_bytes()
        first_input.write_bytes(original_input.replace(b"sample-0", b"sample-X", 1))
        try:
            _load_rows(
                first_input,
                expected_bytes=first_shard["input_bytes"],
                expected_sha256=first_shard["input_sha256"],
            )
        except OrchestrationError:
            pass
        else:
            raise AssertionError("modified capture input was accepted")
        first_input.write_bytes(original_input)
        partial = _generation_partial(source)
        partial.write_bytes(b"in-progress")
        try:
            _prepare(args)
        except OrchestrationError:
            pass
        else:
            raise AssertionError("active response partial was accepted")
        if partial.read_bytes() != b"in-progress":
            raise AssertionError("response partial was modified")
        payload = b"published-trace-self-test"
        digest = hashlib.sha256(payload).hexdigest()
        published = root / f"sha256-{digest}.gfm"
        published.write_bytes(payload)
        if _validate_published_filename(published) != digest:
            raise AssertionError("canonical published trace digest was not returned")
        wrong = root / f"sha256-{'55' * 32}.gfm"
        wrong.write_bytes(payload)
        try:
            _validate_published_filename(wrong)
        except OrchestrationError:
            pass
        else:
            raise AssertionError("mismatched published trace name was accepted")
        single = root / "single.gguf"
        single.write_bytes(b"single synthetic GGUF")
        expected_single = hashlib.sha256()
        expected_single.update(GGUF_IDENTITY_SCHEME.encode("ascii"))
        expected_single.update(f":{single.stat().st_size}:".encode("ascii"))
        expected_single.update(_sha256_file(single).encode("ascii"))
        if _gguf_artifact_sha256(single) != expected_single.hexdigest():
            raise AssertionError("single-file GGUF artifact identity differs")
        shards = []
        expected_split = hashlib.sha256(GGUF_IDENTITY_SCHEME.encode("ascii"))
        for ordinal, contents in enumerate((b"first", b"second"), 1):
            shard = root / f"split-{ordinal:05d}-of-00002.gguf"
            shard.write_bytes(contents)
            shards.append(shard)
            expected_split.update(f":{len(contents)}:".encode("ascii"))
            expected_split.update(hashlib.sha256(contents).hexdigest().encode("ascii"))
        if _gguf_artifact_sha256(shards[1]) != expected_split.hexdigest():
            raise AssertionError("split GGUF artifact identity differs")
    print("FLASHMAP_5K_ORCHESTRATION_SELFTEST_OK records=3 shards=2")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    _add_plan_arguments(prepare)
    capture = commands.add_parser("capture")
    _add_plan_arguments(capture)
    capture.add_argument("--capture-executable", type=Path, required=True)
    capture.add_argument("--capture-sha256", required=True)
    capture.add_argument("--capture-git", required=True)
    capture.add_argument("--model", type=Path, required=True)
    capture.add_argument("--capture-output", type=Path, required=True)
    capture.add_argument("--runtime-temp", type=Path, required=True)
    capture.add_argument("--max-new-shards", type=int, default=0)
    capture.add_argument("--retry-partial-shard", action="store_true")
    capture.add_argument("--resident-capture", action="store_true")
    commands.add_parser("self-test")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "self-test":
        _self_test()
        return 0
    _normalize(args)
    manifest = _prepare(args)
    if args.command == "capture":
        _capture(args, manifest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        planner.PlanError,
        resident.ResidentCaptureError,
        OrchestrationError,
    ) as error:
        print(f"FLASHMAP_5K_ORCHESTRATION_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error

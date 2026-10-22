#!/usr/bin/env python3
"""Manage an append-only FlashMap data-dose campaign.

The campaign starts with one independently published prompt corpus (normally
the existing 5K corpus) and one independently published holdout corpus.  Later
training corpora are added one complete chunk at a time.  Every added chunk is
checked for exact-message overlap with all earlier chunks and with the frozen
holdout before it is atomically published.  Cumulative training milestones are
small immutable prefix manifests; prompt or capture payloads are never copied
to form a milestone.

This tool does not generate responses, capture features, transfer payloads, or
train a model.  Each chunk's copied ``prompts.jsonl`` is an exact input to the
existing generator, plan, and capture tools.  This separation permits direct
generator-to-trainer transfer while both sides retain exact campaign, chunk,
producer, target, plan, and capture identities.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Sequence

import flashmap_prompt_corpus as corpus


FORMAT = "gufo.flashmap.chunk-campaign.v2"
CHUNK_FORMAT = "gufo.flashmap.prompt-chunk.v2"
HOLDOUT_FORMAT = "gufo.flashmap.holdout.v1"
MILESTONE_FORMAT = "gufo.flashmap.cumulative-milestone.v2"
MAX_MANIFEST_BYTES = 4 << 20
GENERATION_FIELDS = {
    "generation_config_sha256",
    "generator_git_commit",
    "generator_sha256",
    "target_sha256",
}
SOURCE_FIELDS = {
    "format",
    "manifest_content_sha256",
    "prompt_bytes",
    "prompt_sha256",
    "records",
    "selection_algorithm",
    "tool_sha256",
}


class CampaignError(RuntimeError):
    """The campaign cannot be safely published or consumed."""


def _canonical(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode("utf-8")
    except UnicodeEncodeError as error:
        raise CampaignError("JSON contains a Unicode surrogate") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _content_digest(value: dict[str, object]) -> str:
    unsigned = dict(value)
    unsigned.pop("manifest_content_sha256", None)
    return _sha256_bytes(_canonical(unsigned))


def _hex(value: object, field: str, digits: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != digits
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CampaignError(f"{field} must be {digits} lowercase hexadecimal digits")
    return value


def _positive(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CampaignError(f"{field} must be a positive integer")
    return value


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise CampaignError(f"refusing to overwrite {path}")
    with path.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def _publish_no_replace(staging: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise CampaignError(f"refusing to overwrite {output}")
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
            raise CampaignError("renameat2 is required for no-replace publication")
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
    raise CampaignError("atomic no-replace publication is unsupported")


def _publish_bytes_no_replace(output: Path, payload: bytes) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    staging = staging_root / "payload"
    try:
        _write_new(staging, payload)
        _publish_no_replace(staging, output)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _strict_object(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise CampaignError(f"manifest must be a non-symlink regular file: {path}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise CampaignError(f"cannot read {path}") from error
    if not payload or len(payload) > MAX_MANIFEST_BYTES or not payload.endswith(b"\n"):
        raise CampaignError(f"{path} is not a bounded newline-terminated manifest")
    try:
        value = json.loads(payload, object_pairs_hook=corpus._unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, corpus.CorpusError) as error:
        raise CampaignError(f"{path} is not strict JSON") from error
    if not isinstance(value, dict) or _canonical(value) != payload:
        raise CampaignError(f"{path} is not a canonical JSON object")
    return value


def _source_record(manifest: dict[str, object]) -> dict[str, object]:
    return {
        "format": manifest["format"],
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "prompt_bytes": manifest["prompt_bytes"],
        "prompt_sha256": manifest["prompt_sha256"],
        "records": manifest["records"],
        "selection_algorithm": manifest["selection_algorithm"],
        "tool_sha256": manifest["tool_sha256"],
    }


def _generation(args: argparse.Namespace) -> dict[str, str]:
    return {
        "generation_config_sha256": _hex(
            args.generation_config_sha256, "generation_config_sha256", 64
        ),
        "generator_git_commit": _hex(args.generator_git, "generator_git", 40),
        "generator_sha256": _hex(args.generator_sha256, "generator_sha256", 64),
        "target_sha256": _hex(args.target_sha256, "target_sha256", 64),
    }


def _parse_milestones(value: str, *, chunk_records: int, maximum: int) -> list[int]:
    try:
        result = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise CampaignError("milestones must be comma-separated integers") from error
    if not result or result != sorted(set(result)):
        raise CampaignError("milestones must be non-empty, unique, and increasing")
    if any(item % chunk_records or item > maximum for item in result):
        raise CampaignError("milestones must be complete in-range chunk prefixes")
    if result[-1] != maximum:
        raise CampaignError("the last planned milestone must equal max_records")
    return result


def _prompt_rows(path: Path) -> Iterable[tuple[bytes, str, str]]:
    with path.open("rb") as stream:
        for raw_line in stream:
            if not raw_line.endswith(b"\n") or len(raw_line) > corpus.MAX_JSONL_LINE_BYTES:
                raise CampaignError("prompt corpus contains non-bounded JSONL")
            try:
                row = json.loads(raw_line, object_pairs_hook=corpus._unique_object)
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                corpus.CorpusError,
            ) as error:
                raise CampaignError("prompt corpus contains non-strict JSON") from error
            if (
                not isinstance(row, dict)
                or set(row) != corpus.OUTPUT_FIELDS
                or _canonical(row) != raw_line
            ):
                raise CampaignError("prompt row is not canonical generator input")
            row_id = corpus._string(row.get("id"), "prompt id")
            _, message_bytes = corpus._messages(row.get("messages"), context=row_id)
            yield raw_line, row_id, _sha256_bytes(message_bytes)


def _corpus_identities(root: Path) -> tuple[dict[str, object], bytes, set[str], set[str]]:
    try:
        manifest = corpus.validate(root)
    except corpus.CorpusError as error:
        raise CampaignError(f"prompt corpus validation failed: {root}") from error
    payload = (root / "prompts.jsonl").read_bytes()
    ids: set[str] = set()
    messages: set[str] = set()
    rows = 0
    for _raw, row_id, message_digest in _prompt_rows(root / "prompts.jsonl"):
        if row_id in ids or message_digest in messages:
            raise CampaignError("prompt corpus repeats an identity or exact messages")
        ids.add(row_id)
        messages.add(message_digest)
        rows += 1
    if rows != manifest["records"]:
        raise CampaignError("prompt row count differs from corpus manifest")
    return manifest, payload, ids, messages


def _make_payload_manifest(
    *,
    format_name: str,
    source: dict[str, object],
    prompt_payload: bytes,
    records: int,
    generation: dict[str, str] | None = None,
    ordinal: int | None = None,
    previous_chunk_manifest_content_sha256: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "format": format_name,
        "prompt_bytes": len(prompt_payload),
        "prompt_file": "prompts.jsonl",
        "prompt_sha256": _sha256_bytes(prompt_payload),
        "records": records,
        "source": source,
    }
    if generation is not None:
        value["generation"] = generation
    if ordinal is not None:
        if previous_chunk_manifest_content_sha256 is None:
            raise CampaignError("chunk requires a previous-manifest chain value")
        value["ordinal"] = ordinal
        value["first_index"] = ordinal * records
        value["last_index"] = (ordinal + 1) * records - 1
        value["previous_chunk_manifest_content_sha256"] = _hex(
            previous_chunk_manifest_content_sha256,
            "previous_chunk_manifest_content_sha256",
            64,
        )
    value["manifest_content_sha256"] = _content_digest(value)
    return value


def _write_payload_directory(
    parent: Path,
    name: str,
    *,
    manifest_name: str,
    manifest: dict[str, object],
    prompt_payload: bytes,
) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=parent))
    try:
        _write_new(staging / "prompts.jsonl", prompt_payload)
        _write_new(staging / manifest_name, _canonical(manifest))
        output = parent / name
        _publish_no_replace(staging, output)
        return output
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def initialize(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise CampaignError(f"refusing to overwrite campaign {output}")
    chunk_records = _positive(args.chunk_records, "chunk_records")
    max_records = _positive(args.max_records, "max_records")
    if max_records % chunk_records:
        raise CampaignError("max_records must be divisible by chunk_records")
    milestones = _parse_milestones(
        args.milestones, chunk_records=chunk_records, maximum=max_records
    )
    initial_manifest, initial_payload, initial_ids, initial_messages = _corpus_identities(
        args.initial_corpus.resolve()
    )
    holdout_manifest, holdout_payload, holdout_ids, holdout_messages = _corpus_identities(
        args.holdout_corpus.resolve()
    )
    if initial_manifest["records"] != chunk_records:
        raise CampaignError("initial corpus must contain exactly one full chunk")
    if initial_ids & holdout_ids or initial_messages & holdout_messages:
        raise CampaignError("initial training chunk overlaps the frozen holdout")
    generation = _generation(args)
    tool_sha256 = _sha256_file(Path(__file__).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (temporary / "chunks").mkdir()
        (temporary / "milestones").mkdir()
        holdout = _make_payload_manifest(
            format_name=HOLDOUT_FORMAT,
            source=_source_record(holdout_manifest),
            prompt_payload=holdout_payload,
            records=int(holdout_manifest["records"]),
        )
        _write_payload_directory(
            temporary,
            "holdout",
            manifest_name="holdout.json",
            manifest=holdout,
            prompt_payload=holdout_payload,
        )
        chunk = _make_payload_manifest(
            format_name=CHUNK_FORMAT,
            source=_source_record(initial_manifest),
            prompt_payload=initial_payload,
            records=chunk_records,
            generation=generation,
            ordinal=0,
            previous_chunk_manifest_content_sha256="0" * 64,
        )
        _write_payload_directory(
            temporary / "chunks",
            "chunk-00000",
            manifest_name="chunk.json",
            manifest=chunk,
            prompt_payload=initial_payload,
        )
        campaign: dict[str, object] = {
            "chunk_records": chunk_records,
            "format": FORMAT,
            "generation": generation,
            "holdout_manifest_content_sha256": holdout["manifest_content_sha256"],
            "initial_chunk_manifest_content_sha256": chunk[
                "manifest_content_sha256"
            ],
            "max_records": max_records,
            "planned_milestones": milestones,
            "tool_sha256": tool_sha256,
        }
        campaign["manifest_content_sha256"] = _content_digest(campaign)
        _write_new(temporary / "campaign.json", _canonical(campaign))
        _publish_no_replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    state = validate(output)
    print(
        "FLASHMAP_CHUNK_CAMPAIGN_INITIALIZED "
        f"chunks={state['chunks']} holdout={state['holdout_records']} "
        f"sha256={state['manifest_content_sha256']} output={output}"
    )
    return output


def _load_campaign(root: Path) -> dict[str, object]:
    campaign = _strict_object(root / "campaign.json")
    if set(campaign) != {
        "chunk_records",
        "format",
        "generation",
        "holdout_manifest_content_sha256",
        "initial_chunk_manifest_content_sha256",
        "manifest_content_sha256",
        "max_records",
        "planned_milestones",
        "tool_sha256",
    } or campaign.get("format") != FORMAT:
        raise CampaignError("campaign manifest format is not recognized")
    if campaign.get("manifest_content_sha256") != _content_digest(campaign):
        raise CampaignError("campaign manifest content digest differs")
    _hex(campaign.get("tool_sha256"), "tool_sha256", 64)
    _hex(
        campaign.get("initial_chunk_manifest_content_sha256"),
        "initial_chunk_manifest_content_sha256",
        64,
    )
    generation = campaign.get("generation")
    if not isinstance(generation, dict) or set(generation) != GENERATION_FIELDS:
        raise CampaignError("campaign generation binding is invalid")
    _hex(generation.get("target_sha256"), "target_sha256", 64)
    _hex(generation.get("generator_sha256"), "generator_sha256", 64)
    _hex(generation.get("generator_git_commit"), "generator_git_commit", 40)
    _hex(generation.get("generation_config_sha256"), "config_sha256", 64)
    chunk_records = _positive(campaign.get("chunk_records"), "chunk_records")
    maximum = _positive(campaign.get("max_records"), "max_records")
    planned = campaign.get("planned_milestones")
    if not isinstance(planned, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in planned
    ):
        raise CampaignError("planned milestones are invalid")
    if _parse_milestones(
        ",".join(str(item) for item in planned),
        chunk_records=chunk_records,
        maximum=maximum,
    ) != planned:
        raise CampaignError("planned milestones are not canonical")
    return campaign


def _validate_source(value: object) -> None:
    if not isinstance(value, dict) or set(value) != SOURCE_FIELDS:
        raise CampaignError("retained corpus source binding is invalid")
    if value.get("format") != corpus.FORMAT:
        raise CampaignError("retained corpus format differs")
    if value.get("selection_algorithm") != corpus.SELECTION_ALGORITHM:
        raise CampaignError("retained corpus selection algorithm differs")
    _positive(value.get("records"), "retained source records")
    _positive(value.get("prompt_bytes"), "retained source bytes")
    _hex(value.get("prompt_sha256"), "retained source prompt digest", 64)
    _hex(value.get("manifest_content_sha256"), "retained source manifest digest", 64)
    _hex(value.get("tool_sha256"), "retained source tool digest", 64)


def _validate_payload(
    directory: Path,
    *,
    manifest_name: str,
    expected_format: str,
) -> tuple[dict[str, object], set[str], set[str]]:
    if directory.is_symlink() or not directory.is_dir():
        raise CampaignError(f"payload directory is invalid: {directory}")
    entries = list(directory.iterdir())
    if {entry.name for entry in entries} != {manifest_name, "prompts.jsonl"} or any(
        entry.is_symlink() for entry in entries
    ):
        raise CampaignError("payload directory inventory is not exact")
    manifest = _strict_object(directory / manifest_name)
    expected = {
        "format",
        "manifest_content_sha256",
        "prompt_bytes",
        "prompt_file",
        "prompt_sha256",
        "records",
        "source",
    }
    if expected_format == CHUNK_FORMAT:
        expected.update(
            {
                "first_index",
                "generation",
                "last_index",
                "ordinal",
                "previous_chunk_manifest_content_sha256",
            }
        )
    if set(manifest) != expected or manifest.get("format") != expected_format:
        raise CampaignError("payload manifest format is not recognized")
    if manifest.get("manifest_content_sha256") != _content_digest(manifest):
        raise CampaignError("payload manifest content digest differs")
    if manifest.get("prompt_file") != "prompts.jsonl":
        raise CampaignError("payload prompt filename differs")
    _validate_source(manifest.get("source"))
    source = manifest["source"]
    if (
        source.get("records") != manifest.get("records")
        or source.get("prompt_bytes") != manifest.get("prompt_bytes")
        or source.get("prompt_sha256") != manifest.get("prompt_sha256")
    ):
        raise CampaignError("retained corpus and copied prompt payload differ")
    prompt = directory / "prompts.jsonl"
    if prompt.stat().st_size != manifest.get("prompt_bytes"):
        raise CampaignError("payload prompt byte count differs")
    if _sha256_file(prompt) != manifest.get("prompt_sha256"):
        raise CampaignError("payload prompt digest differs")
    ids: set[str] = set()
    messages: set[str] = set()
    rows = 0
    for _raw, row_id, message_digest in _prompt_rows(prompt):
        if row_id in ids or message_digest in messages:
            raise CampaignError("payload repeats an identity or exact messages")
        ids.add(row_id)
        messages.add(message_digest)
        rows += 1
    if rows != manifest.get("records"):
        raise CampaignError("payload prompt record count differs")
    return manifest, ids, messages


def _chunk_directories(root: Path) -> list[Path]:
    chunks = root / "chunks"
    if chunks.is_symlink() or not chunks.is_dir():
        raise CampaignError("campaign chunks directory is invalid")
    result = sorted(chunks.iterdir(), key=lambda item: item.name)
    if any(path.name != f"chunk-{ordinal:05d}" for ordinal, path in enumerate(result)):
        raise CampaignError("campaign chunks are not a contiguous prefix")
    return result


def validate(root: Path) -> dict[str, object]:
    root = root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise CampaignError("campaign must be a non-symlink directory")
    campaign = _load_campaign(root)
    if {entry.name for entry in root.iterdir()} != {
        "campaign.json",
        "chunks",
        "holdout",
        "milestones",
    }:
        raise CampaignError("campaign root inventory is not exact")
    holdout, all_ids, all_messages = _validate_payload(
        root / "holdout",
        manifest_name="holdout.json",
        expected_format=HOLDOUT_FORMAT,
    )
    if holdout.get("manifest_content_sha256") != campaign.get(
        "holdout_manifest_content_sha256"
    ):
        raise CampaignError("campaign holdout binding differs")
    chunks: list[dict[str, object]] = []
    chunk_records = int(campaign["chunk_records"])
    maximum_chunks = int(campaign["max_records"]) // chunk_records
    directories = _chunk_directories(root)
    if not directories or len(directories) > maximum_chunks:
        raise CampaignError("campaign chunk count is outside the planned envelope")
    for ordinal, directory in enumerate(directories):
        chunk, ids, messages = _validate_payload(
            directory, manifest_name="chunk.json", expected_format=CHUNK_FORMAT
        )
        if (
            chunk.get("ordinal") != ordinal
            or chunk.get("records") != chunk_records
            or chunk.get("first_index") != ordinal * chunk_records
            or chunk.get("last_index") != (ordinal + 1) * chunk_records - 1
            or chunk.get("generation") != campaign.get("generation")
        ):
            raise CampaignError("chunk order, range, size, or generation binding differs")
        previous = (
            "0" * 64 if ordinal == 0 else chunks[-1]["manifest_content_sha256"]
        )
        if chunk.get("previous_chunk_manifest_content_sha256") != previous:
            raise CampaignError("chunk manifest chain differs")
        if ordinal == 0 and chunk.get("manifest_content_sha256") != campaign.get(
            "initial_chunk_manifest_content_sha256"
        ):
            raise CampaignError("campaign initial chunk binding differs")
        if all_ids & ids or all_messages & messages:
            raise CampaignError("training chunks or holdout overlap")
        all_ids.update(ids)
        all_messages.update(messages)
        chunks.append(chunk)
    milestones = root / "milestones"
    if milestones.is_symlink() or not milestones.is_dir():
        raise CampaignError("campaign milestones directory is invalid")
    milestone_records: list[int] = []
    for path in sorted(milestones.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise CampaignError("milestone inventory contains a non-file")
        manifest = _strict_object(path)
        if set(manifest) != {
            "campaign_manifest_content_sha256",
            "chunks",
            "format",
            "manifest_content_sha256",
            "records",
        } or manifest.get("format") != MILESTONE_FORMAT:
            raise CampaignError("milestone manifest format is not recognized")
        if manifest.get("manifest_content_sha256") != _content_digest(manifest):
            raise CampaignError("milestone content digest differs")
        records = _positive(manifest.get("records"), "milestone records")
        if path.name != f"records-{records:06d}.json":
            raise CampaignError("milestone filename differs from its record count")
        if records not in campaign["planned_milestones"]:
            raise CampaignError("milestone was not planned")
        count = records // chunk_records
        if count > len(chunks):
            raise CampaignError("milestone references unpublished chunks")
        expected = [
            {
                "manifest_content_sha256": chunk["manifest_content_sha256"],
                "ordinal": chunk["ordinal"],
                "prompt_sha256": chunk["prompt_sha256"],
                "records": chunk["records"],
            }
            for chunk in chunks[:count]
        ]
        if (
            manifest.get("campaign_manifest_content_sha256")
            != campaign["manifest_content_sha256"]
            or manifest.get("chunks") != expected
        ):
            raise CampaignError("milestone is not the exact campaign prefix")
        milestone_records.append(records)
    if milestone_records != sorted(set(milestone_records)):
        raise CampaignError("milestone inventory is duplicated or unordered")
    return {
        "chunks": len(chunks),
        "holdout_records": holdout["records"],
        "manifest_content_sha256": campaign["manifest_content_sha256"],
        "milestones": milestone_records,
        "records": len(chunks) * chunk_records,
    }


def add_chunk(args: argparse.Namespace) -> None:
    root = args.campaign.resolve()
    state = validate(root)
    campaign = _load_campaign(root)
    ordinal = int(state["chunks"])
    if int(state["records"]) >= int(campaign["max_records"]):
        raise CampaignError("campaign already contains its maximum records")
    source_manifest, payload, new_ids, new_messages = _corpus_identities(
        args.corpus.resolve()
    )
    if source_manifest["records"] != campaign["chunk_records"]:
        raise CampaignError("new corpus must contain exactly one full chunk")
    existing_ids: set[str] = set()
    existing_messages: set[str] = set()
    for path in [root / "holdout", *_chunk_directories(root)]:
        prompt = path / "prompts.jsonl"
        for _raw, row_id, message_digest in _prompt_rows(prompt):
            existing_ids.add(row_id)
            existing_messages.add(message_digest)
    if existing_ids & new_ids or existing_messages & new_messages:
        raise CampaignError("new training chunk overlaps prior chunks or holdout")
    manifest = _make_payload_manifest(
        format_name=CHUNK_FORMAT,
        source=_source_record(source_manifest),
        prompt_payload=payload,
        records=int(campaign["chunk_records"]),
        generation=campaign["generation"],
        ordinal=ordinal,
        previous_chunk_manifest_content_sha256=_strict_object(
            _chunk_directories(root)[-1] / "chunk.json"
        )["manifest_content_sha256"],
    )
    output = _write_payload_directory(
        root / "chunks",
        f"chunk-{ordinal:05d}",
        manifest_name="chunk.json",
        manifest=manifest,
        prompt_payload=payload,
    )
    validate(root)
    print(
        "FLASHMAP_CHUNK_ADDED "
        f"ordinal={ordinal} records={manifest['records']} "
        f"sha256={manifest['manifest_content_sha256']} output={output}"
    )


def publish_milestone(args: argparse.Namespace) -> None:
    root = args.campaign.resolve()
    state = validate(root)
    campaign = _load_campaign(root)
    records = _positive(args.records, "records")
    if records not in campaign["planned_milestones"]:
        raise CampaignError("requested records are not a planned milestone")
    if records > state["records"]:
        raise CampaignError("requested milestone includes unpublished chunks")
    chunk_records = int(campaign["chunk_records"])
    chunks = [
        _strict_object(path / "chunk.json")
        for path in _chunk_directories(root)[: records // chunk_records]
    ]
    manifest: dict[str, object] = {
        "campaign_manifest_content_sha256": campaign["manifest_content_sha256"],
        "chunks": [
            {
                "manifest_content_sha256": chunk["manifest_content_sha256"],
                "ordinal": chunk["ordinal"],
                "prompt_sha256": chunk["prompt_sha256"],
                "records": chunk["records"],
            }
            for chunk in chunks
        ],
        "format": MILESTONE_FORMAT,
        "records": records,
    }
    manifest["manifest_content_sha256"] = _content_digest(manifest)
    output = root / "milestones" / f"records-{records:06d}.json"
    _publish_bytes_no_replace(output, _canonical(manifest))
    validate(root)
    print(
        "FLASHMAP_MILESTONE_PUBLISHED "
        f"records={records} chunks={len(chunks)} "
        f"sha256={manifest['manifest_content_sha256']} output={output}"
    )


def _make_corpus(root: Path, name: str, messages: list[str]) -> Path:
    source_payload = b"".join(
        corpus._canonical(
            {
                "messages": [{"content": message, "role": "user"}],
                "record_id": f"{name}-{index}",
            }
        )
        for index, message in enumerate(messages)
    )
    source_path = root / f"{name}.jsonl"
    source_path.write_bytes(source_payload)
    recipe = {
        "expected_records": len(messages),
        "format": corpus.RECIPE_FORMAT,
        "seed": _sha256_bytes(name.encode("utf-8")),
        "sources": [
            {
                "category": name,
                "dataset_id": f"example/{name}",
                "dataset_revision": _sha256_bytes(name.encode("utf-8"))[:40],
                "dataset_url": f"https://example.invalid/{name}",
                "input": source_path.name,
                "input_sha256": _sha256_bytes(source_payload),
                "license_spdx": "Apache-2.0",
                "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
                "max_prompt_chars": 4096,
                "min_prompt_chars": 8,
                "quota": len(messages),
                "split": "train",
                "subset": "default",
            }
        ],
    }
    recipe_payload = corpus._canonical(recipe)
    recipe_path = root / f"{name}-recipe.json"
    recipe_path.write_bytes(recipe_payload)
    return corpus.prepare(
        argparse.Namespace(
            recipe=recipe_path,
            recipe_sha256=_sha256_bytes(recipe_payload),
            output=root / f"{name}-corpus",
        )
    )


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="flashmap-chunk-campaign-", dir=Path.cwd()) as raw:
        root = Path(raw)
        initial = _make_corpus(root, "initial", ["Initial prompt alpha.", "Initial prompt beta."])
        holdout = _make_corpus(root, "holdout", ["Holdout prompt alpha."])
        second = _make_corpus(root, "second", ["Second prompt alpha.", "Second prompt beta."])
        overlap = _make_corpus(root, "overlap", ["Holdout prompt alpha.", "Unique prompt gamma."])
        campaign = root / "campaign"
        initialize(
            argparse.Namespace(
                chunk_records=2,
                generation_config_sha256="1" * 64,
                generator_git="2" * 40,
                generator_sha256="3" * 64,
                holdout_corpus=holdout,
                initial_corpus=initial,
                max_records=4,
                milestones="4",
                output=campaign,
                target_sha256="4" * 64,
            )
        )
        try:
            add_chunk(argparse.Namespace(campaign=campaign, corpus=overlap))
        except CampaignError:
            pass
        else:
            raise AssertionError("holdout-overlapping chunk was accepted")
        if len(_chunk_directories(campaign)) != 1:
            raise AssertionError("rejected chunk changed campaign")
        add_chunk(argparse.Namespace(campaign=campaign, corpus=second))
        publish_milestone(argparse.Namespace(campaign=campaign, records=4))
        state = validate(campaign)
        if state["chunks"] != 2 or state["milestones"] != [4]:
            raise AssertionError("campaign state differs")
        try:
            publish_milestone(argparse.Namespace(campaign=campaign, records=4))
        except CampaignError:
            pass
        else:
            raise AssertionError("milestone overwrite was accepted")
    print("FLASHMAP_CHUNK_CAMPAIGN_SELFTEST_OK chunks=2 milestones=1")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("initial_corpus", type=Path)
    init.add_argument("holdout_corpus", type=Path)
    init.add_argument("output", type=Path)
    init.add_argument("--chunk-records", type=int, default=5000)
    init.add_argument("--max-records", type=int, default=50000)
    init.add_argument(
        "--milestones",
        default="10000,15000,20000,25000,30000,35000,40000,45000,50000",
    )
    init.add_argument("--target-sha256", required=True)
    init.add_argument("--generator-sha256", required=True)
    init.add_argument("--generator-git", required=True)
    init.add_argument("--generation-config-sha256", required=True)
    add = commands.add_parser("add-chunk")
    add.add_argument("campaign", type=Path)
    add.add_argument("corpus", type=Path)
    milestone = commands.add_parser("publish-milestone")
    milestone.add_argument("campaign", type=Path)
    milestone.add_argument("records", type=int)
    check = commands.add_parser("validate")
    check.add_argument("campaign", type=Path)
    commands.add_parser("self-test")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            initialize(args)
        elif args.command == "add-chunk":
            add_chunk(args)
        elif args.command == "publish-milestone":
            publish_milestone(args)
        elif args.command == "validate":
            state = validate(args.campaign)
            print(
                "FLASHMAP_CHUNK_CAMPAIGN_VALID "
                f"records={state['records']} chunks={state['chunks']} "
                f"holdout={state['holdout_records']} "
                f"milestones={','.join(str(value) for value in state['milestones'])} "
                f"sha256={state['manifest_content_sha256']}"
            )
        else:
            self_test()
        return 0
    except (CampaignError, OSError, corpus.CorpusError) as error:
        print(f"FLASHMAP_CHUNK_CAMPAIGN_FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

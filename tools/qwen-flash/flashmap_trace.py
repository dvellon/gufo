#!/usr/bin/env python3
"""Fail-closed reader and recovery scanner for Gufo FlashMap trace v2.

The module deliberately has no third-party dependencies.  It validates the
complete shard before exposing records and keeps recovery of an incomplete
``.partial`` tail separate from validation of a published shard.
"""

from __future__ import annotations

import argparse
import array
import dataclasses
import hashlib
import hmac
import json
import mmap
import os
from pathlib import Path
import struct
import sys
from typing import Iterator, Mapping, Sequence


FILE_MAGIC = b"GFMTRC2\0"
RECORD_MAGIC = b"GFMREC2\0"
COMMIT_MAGIC = b"GFMEND2\0"
FORMAT_MAJOR = 2
FORMAT_MINOR = 0
FILE_HEADER_BYTES = 320
FILE_HEADER_HASH_OFFSET = 288
RECORD_HEADER_BYTES = 192
RECORD_TRAILER_BYTES = 64
ALIGNMENT = 64
TOKEN_DTYPE_U32LE = 1
FEATURE_DTYPE_BF16_RNE_LE = 1
KNOWN_FILE_FLAGS = 1
KNOWN_RECORD_FLAGS = 0
IDENTITY_SCHEME = "gguf-sha256-v1"
FLASHMAP_TAPS = (1, 12, 23, 34, 45, 47)
FLASHMAP_HIDDEN = 2560
FEATURE_ABI = (
    b"gufo.flashmap.feature.v1\n"
    b"architecture=qwen4exp\n"
    b"alignment=post-input-token\n"
    b"tap-index=zero-based-completed-layer\n"
    b"taps=1,12,23,34,45,47\n"
    b"aux-count=5\n"
    b"final-tap=47\n"
    b"readout=target-final-hyperconnection-mixer\n"
    b"hidden=2560\n"
    b"capture=f32\n"
    b"storage=bf16-rne-le\n"
    b"layout=token,tap,hidden\n"
    b"target-only=true\n"
)
FEATURE_ABI_SHA256 = hashlib.sha256(FEATURE_ABI).digest()

_RECORD_HEADER = struct.Struct("<8sHHIQQIIQQQQQQ32s32s32sII")
_RECORD_TRAILER = struct.Struct("<32sQQII8s")


class TraceError(ValueError):
    """The trace is malformed, corrupt, or bound to unexpected artifacts."""


def _digest(value: bytes | str, field: str, size: int = 32) -> bytes:
    if isinstance(value, str):
        try:
            value = bytes.fromhex(value)
        except ValueError as error:
            raise TraceError(f"{field} is not hexadecimal") from error
    if len(value) != size:
        raise TraceError(f"{field} must contain exactly {size} bytes")
    return value


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def _u16(data: mmap.mmap, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: mmap.mmap, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _u64(data: mmap.mmap, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def _all_zero(data: mmap.mmap, start: int, end: int, field: str) -> None:
    if end < start or any(data[start:end]):
        raise TraceError(f"{field} is not canonical zero padding")


def _sha256_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.digest()


@dataclasses.dataclass(frozen=True)
class TraceExpectations:
    """External identities that a trace is required to match exactly."""

    target_sha256: bytes | str
    producer_sha256: bytes | str
    producer_git_commit: bytes | str
    vocabulary_size: int
    max_record_tokens: int
    feature_abi_sha256: bytes | str = FEATURE_ABI_SHA256
    taps: tuple[int, ...] = FLASHMAP_TAPS
    hidden: int = FLASHMAP_HIDDEN
    identity_scheme: str = IDENTITY_SCHEME

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_sha256", _digest(self.target_sha256, "target_sha256")
        )
        object.__setattr__(
            self,
            "producer_sha256",
            _digest(self.producer_sha256, "producer_sha256"),
        )
        object.__setattr__(
            self,
            "producer_git_commit",
            _digest(self.producer_git_commit, "producer_git_commit", 20),
        )
        object.__setattr__(
            self,
            "feature_abi_sha256",
            _digest(self.feature_abi_sha256, "feature_abi_sha256"),
        )
        if not 0 < self.vocabulary_size <= 0xFFFFFFFF:
            raise TraceError("vocabulary_size is outside U32")
        if not 0 < self.max_record_tokens <= 0xFFFFFFFF:
            raise TraceError("max_record_tokens is outside U32")
        if not self.taps or len(self.taps) > 8:
            raise TraceError("taps must contain between one and eight entries")
        if tuple(sorted(set(self.taps))) != self.taps:
            raise TraceError("taps must be strictly increasing and unique")
        if not 0 < self.hidden <= 0xFFFFFFFF:
            raise TraceError("hidden is outside U32")


@dataclasses.dataclass(frozen=True)
class TraceHeader:
    flags: int
    hidden: int
    taps: tuple[int, ...]
    max_record_tokens: int
    vocabulary_size: int
    identity_scheme: str
    target_sha256: bytes
    feature_abi_sha256: bytes
    producer_sha256: bytes
    producer_git_commit: bytes
    shard_ordinal: int
    creation_unix_ns: int
    shard_uuid: bytes


@dataclasses.dataclass(frozen=True)
class RecoveryResult:
    header: TraceHeader
    committed_end: int
    records: int
    tokens: int
    incomplete_tail: bool


@dataclasses.dataclass(frozen=True)
class _RecordDescriptor:
    offset: int
    total_bytes: int
    ordinal: int
    token_count: int
    metadata_offset: int
    metadata_bytes: int
    token_offset: int
    feature_offset: int
    feature_bytes: int
    trailer_offset: int
    sample_sha256: bytes
    source_sha256: bytes


class TraceRecord:
    """One validated record backed by its reader's memory map."""

    def __init__(self, data: mmap.mmap, descriptor: _RecordDescriptor):
        self._data = data
        self._descriptor = descriptor

    @property
    def ordinal(self) -> int:
        return self._descriptor.ordinal

    @property
    def token_count(self) -> int:
        return self._descriptor.token_count

    @property
    def sample_sha256(self) -> bytes:
        return self._descriptor.sample_sha256

    @property
    def source_sha256(self) -> bytes:
        return self._descriptor.source_sha256

    def metadata(self) -> Mapping[str, object]:
        descriptor = self._descriptor
        if descriptor.metadata_bytes == 0:
            return {}
        raw = self._data[
            descriptor.offset + descriptor.metadata_offset :
            descriptor.offset
            + descriptor.metadata_offset
            + descriptor.metadata_bytes
        ]
        return json.loads(raw.decode("utf-8"))

    def tokens(self) -> tuple[int, ...]:
        descriptor = self._descriptor
        start = descriptor.offset + descriptor.token_offset
        return struct.unpack_from(
            f"<{descriptor.token_count}I", self._data, start
        )

    def feature_bytes(self) -> memoryview:
        descriptor = self._descriptor
        start = descriptor.offset + descriptor.feature_offset
        end = start + descriptor.feature_bytes
        return memoryview(self._data)[start:end]


def _parse_header(data: mmap.mmap, expected: TraceExpectations) -> TraceHeader:
    if len(data) < FILE_HEADER_BYTES:
        raise TraceError("trace is shorter than its file header")
    if data[0:8] != FILE_MAGIC:
        raise TraceError("file magic does not identify FlashMap trace v2")
    if _u16(data, 8) != FORMAT_MAJOR or _u16(data, 10) != FORMAT_MINOR:
        raise TraceError("unsupported trace format version")
    if _u32(data, 12) != FILE_HEADER_BYTES:
        raise TraceError("file header size is not canonical")
    if _u32(data, 16) != KNOWN_FILE_FLAGS:
        raise TraceError("file header flags do not require BF16 RNE")
    if _u32(data, 20) != TOKEN_DTYPE_U32LE:
        raise TraceError("token dtype is not U32LE")
    if _u32(data, 24) != FEATURE_DTYPE_BF16_RNE_LE:
        raise TraceError("feature dtype is not BF16-RNE-LE")
    hidden = _u32(data, 28)
    tap_count = _u32(data, 32)
    max_record_tokens = _u32(data, 36)
    vocabulary_size = _u32(data, 40)
    if _u32(data, 44) != 0:
        raise TraceError("file header reserved scalar is nonzero")
    if not 0 < tap_count <= 8:
        raise TraceError("tap count is outside the header envelope")
    tap_slots = struct.unpack_from("<8I", data, 48)
    taps = tuple(tap_slots[:tap_count])
    if any(tap_slots[tap_count:]):
        raise TraceError("unused tap slots are nonzero")
    scheme_raw = data[80:96]
    scheme = scheme_raw.rstrip(b"\0")
    if b"\0" in scheme or scheme_raw != scheme.ljust(16, b"\0"):
        raise TraceError("identity scheme is not canonically NUL padded")
    try:
        identity_scheme = scheme.decode("ascii")
    except UnicodeDecodeError as error:
        raise TraceError("identity scheme is not ASCII") from error
    target_sha256 = bytes(data[96:128])
    feature_abi_sha256 = bytes(data[128:160])
    producer_sha256 = bytes(data[160:192])
    producer_git_commit = bytes(data[192:212])
    if _u32(data, 212) != 0:
        raise TraceError("file header source-alignment field is nonzero")
    shard_ordinal = _u64(data, 216)
    creation_unix_ns = _u64(data, 224)
    shard_uuid = bytes(data[232:248])
    _all_zero(data, 248, 288, "file header reserved region")
    stored_digest = bytes(data[288:320])
    if not hmac.compare_digest(
        stored_digest, hashlib.sha256(data[0:288]).digest()
    ):
        raise TraceError("file header digest differs")

    actual = {
        "hidden": hidden,
        "taps": taps,
        "max_record_tokens": max_record_tokens,
        "vocabulary_size": vocabulary_size,
        "identity_scheme": identity_scheme,
        "target_sha256": target_sha256,
        "feature_abi_sha256": feature_abi_sha256,
        "producer_sha256": producer_sha256,
        "producer_git_commit": producer_git_commit,
    }
    for field, wanted in (
        ("hidden", expected.hidden),
        ("taps", expected.taps),
        ("max_record_tokens", expected.max_record_tokens),
        ("vocabulary_size", expected.vocabulary_size),
        ("identity_scheme", expected.identity_scheme),
        ("target_sha256", expected.target_sha256),
        ("feature_abi_sha256", expected.feature_abi_sha256),
        ("producer_sha256", expected.producer_sha256),
        ("producer_git_commit", expected.producer_git_commit),
    ):
        if actual[field] != wanted:
            raise TraceError(f"file header {field} differs from expectation")
    if not any(target_sha256) or not any(feature_abi_sha256) or not any(
        producer_sha256
    ):
        raise TraceError("authoritative SHA-256 binding is zero")
    if creation_unix_ns == 0 or not any(shard_uuid):
        raise TraceError("shard creation identity is empty")
    return TraceHeader(
        flags=_u32(data, 16),
        hidden=hidden,
        taps=taps,
        max_record_tokens=max_record_tokens,
        vocabulary_size=vocabulary_size,
        identity_scheme=identity_scheme,
        target_sha256=target_sha256,
        feature_abi_sha256=feature_abi_sha256,
        producer_sha256=producer_sha256,
        producer_git_commit=producer_git_commit,
        shard_ordinal=shard_ordinal,
        creation_unix_ns=creation_unix_ns,
        shard_uuid=shard_uuid,
    )


def _check_bf16_finite(data: mmap.mmap, start: int, end: int) -> None:
    values_per_chunk = 1 << 20
    for offset in range(start, end, values_per_chunk * 2):
        words = array.array("H")
        words.frombytes(data[offset : min(end, offset + values_per_chunk * 2)])
        if sys.byteorder != "little":
            words.byteswap()
        if any((word & 0x7F80) == 0x7F80 for word in words):
            raise TraceError("feature payload contains non-finite BF16")


def _canonical_metadata(raw: bytes) -> Mapping[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TraceError("record metadata is not UTF-8 canonical JSON") from error
    if not isinstance(value, dict):
        raise TraceError("record metadata must be a JSON object")

    def reject_numbers(item: object) -> None:
        # The C++ producer's minimal JSON representation cannot preserve the
        # source spelling of integers, decimals, and negative zero. Keep the
        # wire contract to the shared lossless subset instead of accepting a
        # shard one conforming producer cannot reproduce.
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            raise TraceError("numeric record metadata is not portable")
        if isinstance(item, list):
            for child in item:
                reject_numbers(child)
        elif isinstance(item, dict):
            for child in item.values():
                reject_numbers(child)

    reject_numbers(value)
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if raw != canonical:
        raise TraceError("record metadata is not in canonical JSON form")
    return value


def _parse_record(
    data: mmap.mmap,
    offset: int,
    header: TraceHeader,
    expected_ordinal: int,
    allow_incomplete_tail: bool,
) -> tuple[_RecordDescriptor | None, int, bool]:
    remaining = len(data) - offset
    if remaining < RECORD_HEADER_BYTES:
        if allow_incomplete_tail:
            tail = bytes(data[offset:])
            if (
                remaining < len(RECORD_MAGIC)
                and tail == RECORD_MAGIC[:remaining]
            ) or (
                remaining >= len(RECORD_MAGIC)
                and tail[: len(RECORD_MAGIC)] == RECORD_MAGIC
            ):
                return None, offset, True
        raise TraceError("truncated or corrupt record header")
    fields = _RECORD_HEADER.unpack_from(data, offset)
    (
        magic,
        major,
        minor,
        header_bytes,
        total_bytes,
        ordinal,
        token_count,
        metadata_bytes,
        metadata_offset,
        token_offset,
        feature_offset,
        trailer_offset,
        token_bytes,
        feature_bytes,
        sample_sha256,
        token_sha256,
        source_sha256,
        flags,
        reserved,
    ) = fields
    if magic != RECORD_MAGIC or major != FORMAT_MAJOR or minor != FORMAT_MINOR:
        raise TraceError("record identity or version differs")
    if header_bytes != RECORD_HEADER_BYTES or metadata_offset != RECORD_HEADER_BYTES:
        raise TraceError("record header or metadata offset is not canonical")
    if flags & ~KNOWN_RECORD_FLAGS or reserved != 0:
        raise TraceError("record contains unknown flags or nonzero reserved data")
    if ordinal != expected_ordinal:
        raise TraceError("record ordinal is not contiguous")
    if not 0 < token_count <= header.max_record_tokens:
        raise TraceError("record token count is outside the shard envelope")
    if token_bytes != token_count * 4:
        raise TraceError("record token byte count differs")
    expected_feature_bytes = token_count * len(header.taps) * header.hidden * 2
    if feature_bytes != expected_feature_bytes:
        raise TraceError("record feature byte count differs")
    expected_token_offset = _align(RECORD_HEADER_BYTES + metadata_bytes)
    expected_feature_offset = _align(expected_token_offset + token_bytes)
    expected_trailer_offset = _align(expected_feature_offset + feature_bytes)
    expected_total_bytes = expected_trailer_offset + RECORD_TRAILER_BYTES
    if (
        token_offset != expected_token_offset
        or feature_offset != expected_feature_offset
        or trailer_offset != expected_trailer_offset
        or total_bytes != expected_total_bytes
    ):
        raise TraceError("record offsets or total size are not canonical")
    record_end = offset + total_bytes
    if record_end > len(data):
        if allow_incomplete_tail:
            return None, offset, True
        raise TraceError("record is truncated before its committed trailer")

    metadata_start = offset + metadata_offset
    metadata_end = metadata_start + metadata_bytes
    if metadata_bytes:
        _canonical_metadata(bytes(data[metadata_start:metadata_end]))
    _all_zero(data, metadata_end, offset + token_offset, "metadata padding")
    token_start = offset + token_offset
    token_end = token_start + token_bytes
    if not hmac.compare_digest(
        token_sha256, hashlib.sha256(data[token_start:token_end]).digest()
    ):
        raise TraceError("record token digest differs")
    for (token,) in struct.iter_unpack("<I", data[token_start:token_end]):
        if token >= header.vocabulary_size:
            raise TraceError("record contains an out-of-vocabulary token")
    _all_zero(data, token_end, offset + feature_offset, "token padding")
    feature_start = offset + feature_offset
    feature_end = feature_start + feature_bytes
    _check_bf16_finite(data, feature_start, feature_end)
    _all_zero(data, feature_end, offset + trailer_offset, "feature padding")

    trailer = _RECORD_TRAILER.unpack_from(data, offset + trailer_offset)
    (
        record_sha256,
        duplicate_total,
        duplicate_ordinal,
        duplicate_tokens,
        duplicate_flags,
        commit_magic,
    ) = trailer
    if commit_magic != COMMIT_MAGIC:
        raise TraceError("record commit marker is absent or corrupt")
    if (
        duplicate_total != total_bytes
        or duplicate_ordinal != ordinal
        or duplicate_tokens != token_count
        or duplicate_flags != flags
    ):
        raise TraceError("record trailer duplicates differ")
    if not hmac.compare_digest(
        record_sha256, hashlib.sha256(data[offset : offset + trailer_offset]).digest()
    ):
        raise TraceError("record digest differs")
    if not any(sample_sha256) or not any(source_sha256):
        raise TraceError("record sample or source identity is zero")
    descriptor = _RecordDescriptor(
        offset=offset,
        total_bytes=total_bytes,
        ordinal=ordinal,
        token_count=token_count,
        metadata_offset=metadata_offset,
        metadata_bytes=metadata_bytes,
        token_offset=token_offset,
        feature_offset=feature_offset,
        feature_bytes=feature_bytes,
        trailer_offset=trailer_offset,
        sample_sha256=sample_sha256,
        source_sha256=source_sha256,
    )
    return descriptor, record_end, False


def _scan(
    data: mmap.mmap,
    expected: TraceExpectations,
    allow_incomplete_tail: bool,
) -> tuple[RecoveryResult, list[_RecordDescriptor]]:
    header = _parse_header(data, expected)
    offset = FILE_HEADER_BYTES
    descriptors: list[_RecordDescriptor] = []
    samples: set[bytes] = set()
    tokens = 0
    incomplete = False
    while offset < len(data):
        descriptor, next_offset, incomplete = _parse_record(
            data, offset, header, len(descriptors), allow_incomplete_tail
        )
        if incomplete:
            break
        assert descriptor is not None
        if descriptor.sample_sha256 in samples:
            raise TraceError("duplicate sample identity in shard")
        samples.add(descriptor.sample_sha256)
        descriptors.append(descriptor)
        tokens += descriptor.token_count
        offset = next_offset
    return (
        RecoveryResult(
            header=header,
            committed_end=offset,
            records=len(descriptors),
            tokens=tokens,
            incomplete_tail=incomplete,
        ),
        descriptors,
    )


class TraceReader:
    """Validate and memory-map one complete, published trace shard."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        expected: TraceExpectations,
        *,
        require_content_addressed_name: bool = True,
    ) -> None:
        self.path = Path(path)
        self._file = self.path.open("rb")
        try:
            if os.fstat(self._file.fileno()).st_size < FILE_HEADER_BYTES:
                raise TraceError("trace is shorter than its file header")
            self._data = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            result, self._descriptors = _scan(self._data, expected, False)
            if result.incomplete_tail or result.committed_end != len(self._data):
                raise TraceError("published shard has an incomplete tail")
            if not self._descriptors:
                raise TraceError("published shard contains no records")
            self.header = result.header
            self.records = result.records
            self.tokens = result.tokens
            self.sha256 = _sha256_file(self.path)
            if require_content_addressed_name:
                expected_name = f"sha256-{self.sha256.hex()}.gfm"
                if self.path.name != expected_name:
                    raise TraceError("published shard filename is not content addressed")
        except Exception:
            if hasattr(self, "_data"):
                self._data.close()
            self._file.close()
            raise

    def __enter__(self) -> "TraceReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._data.close()
        self._file.close()

    def __iter__(self) -> Iterator[TraceRecord]:
        for descriptor in self._descriptors:
            yield TraceRecord(self._data, descriptor)


def scan_partial(
    path: str | os.PathLike[str], expected: TraceExpectations
) -> RecoveryResult:
    """Validate committed records and locate a recoverable incomplete EOF tail."""

    partial = Path(path)
    with partial.open("rb") as source:
        if os.fstat(source.fileno()).st_size < FILE_HEADER_BYTES:
            raise TraceError("trace is shorter than its file header")
        data = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            result, _ = _scan(data, expected, True)
            return result
        finally:
            data.close()


def truncate_incomplete_tail(
    path: str | os.PathLike[str], expected: TraceExpectations
) -> RecoveryResult:
    """Truncate only a validated incomplete EOF tail, never corrupt records."""

    partial = Path(path)
    if partial.suffix != ".partial":
        raise TraceError("recovery truncation is restricted to .partial files")
    result = scan_partial(partial, expected)
    if result.incomplete_tail:
        with partial.open("r+b") as output:
            output.truncate(result.committed_end)
            output.flush()
            os.fsync(output.fileno())
    return result


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--producer-git", default="00" * 20)
    parser.add_argument("--vocabulary-size", type=int, required=True)
    parser.add_argument("--max-record-tokens", type=int, required=True)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--truncate-incomplete-tail", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_cli()
    expected = TraceExpectations(
        target_sha256=args.target_sha256,
        producer_sha256=args.producer_sha256,
        producer_git_commit=args.producer_git,
        vocabulary_size=args.vocabulary_size,
        max_record_tokens=args.max_record_tokens,
    )
    if args.partial:
        result = (
            truncate_incomplete_tail(args.trace, expected)
            if args.truncate_incomplete_tail
            else scan_partial(args.trace, expected)
        )
        print(
            "FLASHMAP_PARTIAL_OK"
            f" records={result.records} tokens={result.tokens}"
            f" committed_end={result.committed_end}"
            f" incomplete_tail={int(result.incomplete_tail)}"
        )
        return 0
    if args.truncate_incomplete_tail:
        raise TraceError("--truncate-incomplete-tail requires --partial")
    with TraceReader(args.trace, expected) as reader:
        print(
            "FLASHMAP_TRACE_VALID"
            f" records={reader.records} tokens={reader.tokens}"
            f" sha256={reader.sha256.hex()}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TraceError) as error:
        print(f"FLASHMAP_TRACE_INVALID: {error}", file=sys.stderr)
        raise SystemExit(1) from error

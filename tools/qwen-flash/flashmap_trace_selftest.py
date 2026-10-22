#!/usr/bin/env python3
"""Narrow, CPU-only synthetic validation for ``flashmap_trace.py``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile

import flashmap_trace as trace


TARGET = hashlib.sha256(b"synthetic exact target GGUF").digest()
PRODUCER = hashlib.sha256(b"synthetic capture executable").digest()
GIT_COMMIT = hashlib.sha1(b"synthetic source revision").digest()
VOCABULARY = 256
MAX_TOKENS = 8


def align(value: int) -> int:
    return (value + trace.ALIGNMENT - 1) & ~(trace.ALIGNMENT - 1)


def make_header() -> bytes:
    output = bytearray(trace.FILE_HEADER_BYTES)
    output[0:8] = trace.FILE_MAGIC
    struct.pack_into("<HHI", output, 8, 2, 0, trace.FILE_HEADER_BYTES)
    struct.pack_into(
        "<8I",
        output,
        16,
        1,
        trace.TOKEN_DTYPE_U32LE,
        trace.FEATURE_DTYPE_BF16_RNE_LE,
        trace.FLASHMAP_HIDDEN,
        len(trace.FLASHMAP_TAPS),
        MAX_TOKENS,
        VOCABULARY,
        0,
    )
    struct.pack_into(
        "<8I",
        output,
        48,
        *trace.FLASHMAP_TAPS,
        *(0 for _ in range(8 - len(trace.FLASHMAP_TAPS))),
    )
    scheme = trace.IDENTITY_SCHEME.encode("ascii")
    output[80:96] = scheme.ljust(16, b"\0")
    output[96:128] = TARGET
    output[128:160] = trace.FEATURE_ABI_SHA256
    output[160:192] = PRODUCER
    output[192:212] = GIT_COMMIT
    struct.pack_into("<IQQ", output, 212, 0, 7, 1_800_000_000_000_000_000)
    output[232:248] = bytes(range(1, 17))
    output[288:320] = hashlib.sha256(output[0:288]).digest()
    return bytes(output)


def make_record(
    ordinal: int,
    tokens: tuple[int, ...],
    *,
    sample: str | None = None,
    source: str | None = None,
    nonfinite: bool = False,
) -> bytes:
    metadata = json.dumps(
        {"domain": "synthetic", "split": "test"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    token_payload = struct.pack(f"<{len(tokens)}I", *tokens)
    value_count = (
        len(tokens) * len(trace.FLASHMAP_TAPS) * trace.FLASHMAP_HIDDEN
    )
    feature_payload = bytearray(struct.pack("<H", 0x3F80) * value_count)
    if nonfinite:
        struct.pack_into("<H", feature_payload, 0, 0x7F80)
    metadata_offset = trace.RECORD_HEADER_BYTES
    token_offset = align(metadata_offset + len(metadata))
    feature_offset = align(token_offset + len(token_payload))
    trailer_offset = align(feature_offset + len(feature_payload))
    total_bytes = trailer_offset + trace.RECORD_TRAILER_BYTES
    output = bytearray(total_bytes)
    sample_digest = hashlib.sha256(
        (sample or f"sample-{ordinal}").encode("utf-8")
    ).digest()
    source_digest = hashlib.sha256(
        (source or f"source-{ordinal}").encode("utf-8")
    ).digest()
    trace._RECORD_HEADER.pack_into(
        output,
        0,
        trace.RECORD_MAGIC,
        2,
        0,
        trace.RECORD_HEADER_BYTES,
        total_bytes,
        ordinal,
        len(tokens),
        len(metadata),
        metadata_offset,
        token_offset,
        feature_offset,
        trailer_offset,
        len(token_payload),
        len(feature_payload),
        sample_digest,
        hashlib.sha256(token_payload).digest(),
        source_digest,
        0,
        0,
    )
    output[metadata_offset : metadata_offset + len(metadata)] = metadata
    output[token_offset : token_offset + len(token_payload)] = token_payload
    output[feature_offset : feature_offset + len(feature_payload)] = feature_payload
    record_digest = hashlib.sha256(output[0:trailer_offset]).digest()
    trace._RECORD_TRAILER.pack_into(
        output,
        trailer_offset,
        record_digest,
        total_bytes,
        ordinal,
        len(tokens),
        0,
        trace.COMMIT_MAGIC,
    )
    return bytes(output)


def expect_invalid(function, message: str) -> None:
    try:
        function()
    except trace.TraceError:
        return
    raise AssertionError(message)


def expectations(**changes: object) -> trace.TraceExpectations:
    values: dict[str, object] = {
        "target_sha256": TARGET,
        "producer_sha256": PRODUCER,
        "producer_git_commit": GIT_COMMIT,
        "vocabulary_size": VOCABULARY,
        "max_record_tokens": MAX_TOKENS,
    }
    values.update(changes)
    return trace.TraceExpectations(**values)


def run() -> None:
    header = make_header()
    first = make_record(0, (1, 2, 3))
    second = make_record(1, (5, 8))
    with tempfile.TemporaryDirectory(prefix="gufo-flashmap-trace-") as directory:
        root = Path(directory)
        partial = root / "capture.gfm.partial"
        partial.write_bytes(header + first + second + trace.RECORD_MAGIC[:4])

        recovery = trace.scan_partial(partial, expectations())
        assert recovery.records == 2
        assert recovery.tokens == 5
        assert recovery.incomplete_tail
        assert recovery.committed_end == len(header) + len(first) + len(second)
        expect_invalid(
            lambda: trace.TraceReader(
                partial, expectations(), require_content_addressed_name=False
            ),
            "published validation accepted an incomplete tail",
        )
        trace.truncate_incomplete_tail(partial, expectations())
        recovered = trace.scan_partial(partial, expectations())
        assert not recovered.incomplete_tail
        assert partial.stat().st_size == recovery.committed_end

        digest = hashlib.sha256(partial.read_bytes()).hexdigest()
        published = root / f"sha256-{digest}.gfm"
        partial.rename(published)
        with trace.TraceReader(published, expectations()) as reader:
            assert reader.records == 2 and reader.tokens == 5
            records = list(reader)
            assert records[0].tokens() == (1, 2, 3)
            assert records[1].tokens() == (5, 8)
            assert records[0].metadata() == {
                "domain": "synthetic",
                "split": "test",
            }
            assert len(records[0].feature_bytes()) == (
                3 * len(trace.FLASHMAP_TAPS) * trace.FLASHMAP_HIDDEN * 2
            )
            del records

        wrong_bindings = (
            {"target_sha256": hashlib.sha256(b"wrong target").digest()},
            {"producer_sha256": hashlib.sha256(b"wrong producer").digest()},
            {"producer_git_commit": bytes.fromhex("11" * 20)},
            {"feature_abi_sha256": hashlib.sha256(b"wrong ABI").digest()},
            {"vocabulary_size": VOCABULARY + 1},
            {"max_record_tokens": MAX_TOKENS + 1},
            {"hidden": trace.FLASHMAP_HIDDEN + 1},
            {"taps": (1, 12, 23, 34, 46, 47)},
        )
        for wrong in wrong_bindings:
            expect_invalid(
                lambda wrong=wrong: trace.TraceReader(
                    published, expectations(**wrong)
                ),
                f"wrong binding was accepted: {next(iter(wrong))}",
            )

        corrupt_header = root / "corrupt-header.gfm"
        shutil.copyfile(published, corrupt_header)
        mutated = bytearray(corrupt_header.read_bytes())
        mutated[96] ^= 1
        corrupt_header.write_bytes(mutated)
        expect_invalid(
            lambda: trace.TraceReader(
                corrupt_header,
                expectations(),
                require_content_addressed_name=False,
            ),
            "mutated header was accepted",
        )

        corrupt_payload = root / "corrupt-payload.gfm"
        shutil.copyfile(published, corrupt_payload)
        mutated = bytearray(corrupt_payload.read_bytes())
        mutated[trace.FILE_HEADER_BYTES + align(trace.RECORD_HEADER_BYTES + 37)] ^= 1
        corrupt_payload.write_bytes(mutated)
        expect_invalid(
            lambda: trace.TraceReader(
                corrupt_payload,
                expectations(),
                require_content_addressed_name=False,
            ),
            "mutated record payload was accepted",
        )

        nonfinite = root / "nonfinite.gfm"
        nonfinite.write_bytes(header + make_record(0, (1,), nonfinite=True))
        expect_invalid(
            lambda: trace.TraceReader(
                nonfinite,
                expectations(),
                require_content_addressed_name=False,
            ),
            "non-finite BF16 payload was accepted",
        )

        out_of_vocabulary = root / "out-of-vocabulary.gfm"
        out_of_vocabulary.write_bytes(
            header + make_record(0, (VOCABULARY,))
        )
        expect_invalid(
            lambda: trace.TraceReader(
                out_of_vocabulary,
                expectations(),
                require_content_addressed_name=False,
            ),
            "out-of-vocabulary token was accepted",
        )

        duplicate = root / "duplicate.gfm"
        duplicate.write_bytes(
            header
            + make_record(0, (1,), sample="same", source="source-a")
            + make_record(1, (2,), sample="same", source="source-b")
        )
        expect_invalid(
            lambda: trace.TraceReader(
                duplicate,
                expectations(),
                require_content_addressed_name=False,
            ),
            "duplicate sample identity was accepted",
        )

        repeated_source = root / "repeated-source.gfm"
        repeated_source.write_bytes(
            header
            + make_record(0, (1,), sample="unique-a", source="shared-source")
            + make_record(1, (2,), sample="unique-b", source="shared-source")
        )
        with trace.TraceReader(
            repeated_source,
            expectations(),
            require_content_addressed_name=False,
        ) as reader:
            assert reader.records == 2

        corrupt_tail = root / "corrupt.gfm.partial"
        damaged = bytearray(first)
        damaged[-1] ^= 1
        corrupt_tail.write_bytes(header + damaged)
        expect_invalid(
            lambda: trace.scan_partial(corrupt_tail, expectations()),
            "a full corrupt record was treated as recoverable truncation",
        )

        third = make_record(2, (13,))
        for cut in (1, 7, 8, 64, 191, 192, len(third) - 1):
            truncated = root / f"truncated-{cut}.gfm.partial"
            truncated.write_bytes(header + first + second + third[:cut])
            result = trace.scan_partial(truncated, expectations())
            assert result.records == 2 and result.incomplete_tail
            assert result.committed_end == len(header) + len(first) + len(second)

    print(
        "FLASHMAP_TRACE_SELFTEST_OK"
        f" taps={len(trace.FLASHMAP_TAPS)} hidden={trace.FLASHMAP_HIDDEN}"
        f" abi={trace.FEATURE_ABI_SHA256.hex()}"
    )


if __name__ == "__main__":
    run()

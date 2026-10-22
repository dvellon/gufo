#!/usr/bin/env python3
"""Focused synthetic Q8_0 test for flashmap_verifier_bundle.py."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np


TOOL = Path(__file__).with_name("flashmap_verifier_bundle.py")


def _module():
    spec = importlib.util.spec_from_file_location("flashmap_verifier_bundle", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _metadata(key: str, tag: int, value: bytes) -> bytes:
    return _string(key) + struct.pack("<I", tag) + value


def _q8(scale: float, seed: int) -> tuple[bytes, np.ndarray]:
    quants = np.array([((index + seed) % 31) - 15 for index in range(32)], dtype=np.int8)
    raw = struct.pack("<e", scale) + quants.tobytes()
    values = quants.astype(np.float32) * np.float32(scale)
    return raw, values


def _fixture(path: Path) -> tuple[np.ndarray, np.ndarray]:
    metadata = b"".join(
        [
            _metadata("general.architecture", 8, _string("qwen4exp")),
            _metadata("qwen4exp.block_count", 4, struct.pack("<I", 3)),
            _metadata("qwen4exp.nextn_predict_layers", 4, struct.pack("<I", 1)),
            _metadata("qwen4exp.embedding_length", 4, struct.pack("<I", 32)),
            _metadata("qwen4exp.context_length", 4, struct.pack("<I", 4096)),
        ]
    )
    rows = []
    payloads = []
    for seed, scale in ((0, 0.5), (3, -0.25), (7, 0.125), (11, 0.75)):
        payload, values = _q8(scale, seed)
        payloads.append(payload)
        rows.append(values)
    embedding = np.stack(rows[:2])
    head = np.stack(rows[2:])
    descriptor = b""
    offset = 0
    for name, payload in (
        ("token_embd.weight", b"".join(payloads[:2])),
        ("output.weight", b"".join(payloads[2:])),
    ):
        descriptor += _string(name)
        descriptor += struct.pack("<IQQIQ", 2, 32, 2, 8, offset)
        offset += len(payload)
    header = b"GGUF" + struct.pack("<IQQ", 3, 2, 5) + metadata + descriptor
    header += b"\0" * ((-len(header)) % 32)
    path.write_bytes(header + b"".join(payloads))
    return embedding, head


def _read_safetensors(path: Path) -> tuple[dict, int]:
    data = path.read_bytes()
    length = struct.unpack_from("<Q", data)[0]
    return json.loads(data[8 : 8 + length]), 8 + length


def _expected_bf16(values: np.ndarray) -> bytes:
    bits = values.astype(np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype("<u2").tobytes()


def main() -> None:
    tool = _module()
    with tempfile.TemporaryDirectory(prefix="flashmap-verifier-selftest-") as temp:
        root = Path(temp)
        source, output = root / "tiny.gguf", root / "bundle"
        embedding, head = _fixture(source)
        producer_git = "0" * 40
        paths, _states = tool._discover_shards(source)
        identity, _records = tool._source_identity(paths)
        rejected = root / "wrong-identity"
        try:
            tool.export(
                source,
                rejected,
                rows_per_chunk=1,
                producer_git=producer_git,
                expected_source_identity="f" * 64,
            )
        except tool.BundleError as error:
            assert "GGUF identity differs" in str(error) and not rejected.exists()
        else:
            raise AssertionError("export accepted the wrong GGUF identity")
        result = tool.export(
            source,
            output,
            rows_per_chunk=1,
            producer_git=producer_git,
            expected_source_identity=identity,
        )
        header, data_offset = _read_safetensors(output / "model.safetensors")
        payload = (output / "model.safetensors").read_bytes()[data_offset:]
        expected = _expected_bf16(embedding) + _expected_bf16(head)
        assert payload == expected
        assert header["model.embed_tokens.weight"]["shape"] == [2, 32]
        assert header["lm_head.weight"]["shape"] == [2, 32]
        config = json.loads((output / "config.json").read_text())
        assert config["hidden_size"] == 32 and config["num_hidden_layers"] == 2
        assert config["verifier_last_hidden_state_is_post_norm"] is True
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["source"]["identity"] == result["source_identity"]
        assert manifest["producer"]["git_commit"] == producer_git
        assert manifest["flashmap"]["verifier_last_hidden_state_is_post_norm"] is True
        assert manifest["files"]["model.safetensors"]["sha256"] == hashlib.sha256(
            (output / "model.safetensors").read_bytes()
        ).hexdigest()
        try:
            tool.export(
                source,
                output,
                rows_per_chunk=1,
                producer_git=producer_git,
                expected_source_identity=identity,
            )
        except tool.BundleError as error:
            assert "already exists" in str(error)
        else:
            raise AssertionError("export overwrote an existing bundle")
    print("FLASHMAP_VERIFIER_BUNDLE_SELFTEST_OK shape=2x32 dtype=BF16")


if __name__ == "__main__":
    main()

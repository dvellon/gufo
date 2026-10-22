#!/usr/bin/env python3
"""Export the exact GGUF verifier embedding and head for offline FlashMap.

This is deliberately not a GGUF-to-Hugging-Face model converter.  It publishes
only the two verifier-owned matrices that Speculators asks for while training a
FlashMap drafter.  The source Q8_0 blocks are decoded in bounded row batches and
rounded once, round-to-nearest-even, to BF16.  The resulting bundle therefore
tracks the quantized verifier Gufo actually runs, not an unavailable upstream
full-precision checkpoint.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import BinaryIO

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gufo import gguf  # noqa: E402

SCHEMA = "gufo-flashmap-verifier-bundle-v1"
IDENTITY_SCHEME = "gguf-sha256-v1"
FEATURE_ABI = "70c629737b587777b4955b454744b8009ffc23901b97c252d8306a9375f66d7c"
SOURCE_TENSORS = (
    ("token_embd.weight", "model.embed_tokens.weight"),
    ("output.weight", "lm_head.weight"),
)
_SPLIT = re.compile(r"^(.*)-([0-9]+)-of-([0-9]+)\.gguf$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BundleError(RuntimeError):
    """An input or publication contract was violated."""


def _sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _stamp(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    if not path.is_file():
        raise BundleError(f"source is not a regular file: {path}")
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _discover_shards(source: Path) -> tuple[list[Path], list[dict]]:
    source = source.resolve(strict=True)
    first = gguf.parse_gguf(source)
    count = first["user_kv"].get("split.count", 1)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 65535:
        raise BundleError("GGUF split.count is invalid")
    if count == 1:
        paths = [source]
    else:
        match = _SPLIT.fullmatch(source.name)
        if match is None:
            raise BundleError("split GGUF name must be <name>-00001-of-00002.gguf")
        if int(match.group(3)) != count:
            raise BundleError("filename split count differs from GGUF metadata")
        index_width, count_width = len(match.group(2)), len(match.group(3))
        paths = [
            source.with_name(
                f"{match.group(1)}-{index + 1:0{index_width}d}-of-"
                f"{count:0{count_width}d}.gguf"
            ).resolve(strict=True)
            for index in range(count)
        ]
    states = [gguf.parse_gguf(path) for path in paths]
    architecture = states[0]["metadata"].get("general.architecture")
    total = states[0]["user_kv"].get("split.tensors.count")
    for index, state in enumerate(states):
        if state["version"] != states[0]["version"]:
            raise BundleError("GGUF shard versions differ")
        if state["metadata"].get("general.architecture") not in (None, architecture):
            raise BundleError("GGUF shard architectures differ")
        if count > 1 and (
            state["user_kv"].get("split.no") != index
            or state["user_kv"].get("split.count") != count
        ):
            raise BundleError(f"GGUF shard {index} has inconsistent split metadata")
    actual = sum(state["n_tensors"] for state in states)
    if count > 1 and total != actual:
        raise BundleError(f"GGUF split tensor count differs: metadata={total}, actual={actual}")
    return paths, states


def _source_identity(paths: list[Path]) -> tuple[str, list[dict]]:
    artifact = hashlib.sha256()
    artifact.update(IDENTITY_SCHEME.encode("ascii"))
    records = []
    for path in paths:
        size = path.stat().st_size
        digest = _sha256(path)
        artifact.update(f":{size}:".encode("ascii"))
        artifact.update(digest.encode("ascii"))
        records.append({"name": path.name, "bytes": size, "sha256": digest})
    return artifact.hexdigest(), records


def _tensor_index(paths: list[Path], states: list[dict]) -> dict[str, tuple[Path, dict, dict]]:
    result: dict[str, tuple[Path, dict, dict]] = {}
    for path, state in zip(paths, states, strict=True):
        for tensor in state["tensors"]:
            name = tensor["name"]
            if name in result:
                raise BundleError(f"duplicate tensor across GGUF shards: {name}")
            result[name] = (path, state, tensor)
    return result


def _q8_descriptor(entry: tuple[Path, dict, dict]) -> tuple[int, int, int, int]:
    path, state, tensor = entry
    shape = tensor["shape"]
    if tensor["type"] != 8 or len(shape) != 2:
        raise BundleError(
            f"{tensor['name']} must be rank-2 Q8_0, got {tensor['type_name']} {shape}"
        )
    rows, width = shape
    if rows <= 0 or width <= 0 or width % 32:
        raise BundleError(f"{tensor['name']} has invalid Q8_0 shape {shape}")
    row_bytes = width // 32 * 34
    size = gguf.tensor_bytes(state, tensor)
    if size != rows * row_bytes:
        raise BundleError(f"{tensor['name']} byte size does not match its shape")
    offset = state["data_offset"] + tensor["offset"]
    if offset < 0 or offset + size > path.stat().st_size:
        raise BundleError(f"{tensor['name']} payload exceeds its GGUF shard")
    return rows, width, row_bytes, offset


def _bf16_rne(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise BundleError("Q8_0 decode produced a non-finite FP32 value")
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype("<u2", copy=False)


def _stream_tensor(
    output: BinaryIO,
    entry: tuple[Path, dict, dict],
    rows_per_chunk: int,
) -> tuple[int, str]:
    path, _state, _tensor = entry
    rows, width, row_bytes, offset = _q8_descriptor(entry)
    payload_hash = hashlib.sha256()
    written = 0
    source = np.memmap(path, mode="r", dtype=np.uint8)
    try:
        for row in range(0, rows, rows_per_chunk):
            count = min(rows_per_chunk, rows - row)
            start = offset + row * row_bytes
            raw = source[start : start + count * row_bytes]
            values = gguf.DEQUANT[8](raw).reshape(count, width)
            encoded = _bf16_rne(values).tobytes(order="C")
            output.write(encoded)
            payload_hash.update(encoded)
            written += len(encoded)
    finally:
        del source
    expected = rows * width * 2
    if written != expected:
        raise BundleError(f"short BF16 export: wrote {written}, expected {expected}")
    return written, payload_hash.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _write_safetensors(
    path: Path,
    entries: dict[str, tuple[Path, dict, dict]],
    identity: str,
    producer_git: str,
    exporter_sha256: str,
    rows_per_chunk: int,
) -> tuple[str, dict[str, dict]]:
    cursor = 0
    header: dict[str, object] = {
        "__metadata__": {
            "format": "pt",
            "gufo.schema": SCHEMA,
            "gufo.source_identity_scheme": IDENTITY_SCHEME,
            "gufo.source_identity": identity,
            "gufo.producer_git": producer_git,
            "gufo.exporter_sha256": exporter_sha256,
            "gufo.precision": "Q8_0->FP32->BF16_RNE",
            "gufo.verifier_last_hidden_state_is_post_norm": "true",
        }
    }
    descriptors: dict[str, tuple[Path, dict, dict]] = {}
    for source_name, hf_name in SOURCE_TENSORS:
        entry = entries[source_name]
        rows, width, _row_bytes, _offset = _q8_descriptor(entry)
        length = rows * width * 2
        header[hf_name] = {
            "dtype": "BF16",
            "shape": [rows, width],
            "data_offsets": [cursor, cursor + length],
        }
        descriptors[hf_name] = entry
        cursor += length
    encoded = _canonical_json(header)
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    tensor_records: dict[str, dict] = {}
    whole_hash = hashlib.sha256()
    with path.open("xb") as output:
        prefix = len(encoded).to_bytes(8, "little") + encoded
        output.write(prefix)
        whole_hash.update(prefix)
        for hf_name, entry in descriptors.items():
            # Hashing through a small adapter avoids rereading multi-GB output.
            class _HashedWriter:
                def write(self, data: bytes) -> int:
                    whole_hash.update(data)
                    return output.write(data)

            written, payload_sha = _stream_tensor(_HashedWriter(), entry, rows_per_chunk)
            source_name = entry[2]["name"]
            tensor_records[hf_name] = {
                "source_tensor": source_name,
                "source_type": "Q8_0",
                "shape": entry[2]["shape"],
                "output_dtype": "BF16",
                "output_bytes": written,
                "output_payload_sha256": payload_sha,
            }
        output.flush()
        os.fsync(output.fileno())
    return whole_hash.hexdigest(), tensor_records


def _minimal_config(state: dict, rows: int, width: int, identity: str) -> dict:
    architecture = state["metadata"].get("general.architecture")
    if architecture != "qwen4exp":
        raise BundleError(f"expected qwen4exp GGUF, got {architecture!r}")
    kv = state["user_kv"]
    blocks = kv.get("qwen4exp.block_count")
    nextn = kv.get("qwen4exp.nextn_predict_layers", 0)
    context = kv.get("qwen4exp.context_length")
    embedded = kv.get("qwen4exp.embedding_length")
    for label, value in (("block_count", blocks), ("context_length", context)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise BundleError(f"qwen4exp.{label} is missing or invalid")
    if embedded != width or not isinstance(nextn, int) or nextn < 0 or nextn >= blocks:
        raise BundleError("GGUF architecture dimensions differ from verifier matrices")
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": rows,
        "hidden_size": width,
        "num_hidden_layers": blocks - nextn,
        "max_position_embeddings": context,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "gufo_minimal_verifier_weights_only": True,
        "gufo_source_identity_scheme": IDENTITY_SCHEME,
        "gufo_source_identity": identity,
        "verifier_last_hidden_state_is_post_norm": True,
    }


def _write_durable(path: Path, data: bytes) -> str:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def _publish_no_replace(staging: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise BundleError(f"output already exists: {output}")
    if sys.platform == "win32":
        if not ctypes.windll.kernel32.MoveFileW(str(staging), str(output)):
            raise ctypes.WinError()
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise BundleError("renameat2 is required for atomic no-overwrite publication")
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
    raise BundleError("atomic no-overwrite publication is unsupported on this platform")


def export(
    source: Path,
    output: Path,
    rows_per_chunk: int,
    producer_git: str,
    expected_source_identity: str,
) -> dict:
    if rows_per_chunk <= 0 or rows_per_chunk > 4096:
        raise BundleError("--rows-per-chunk must be in [1,4096]")
    if _GIT_COMMIT.fullmatch(producer_git) is None:
        raise BundleError("--producer-git must be one lowercase 40-hex commit")
    if _SHA256.fullmatch(expected_source_identity) is None:
        raise BundleError("--expected-source-identity must be one lowercase SHA-256")
    exporter_sha256 = _sha256(Path(__file__).resolve())
    output = output.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise BundleError(f"output already exists: {output}")
    paths, states = _discover_shards(source)
    stamps = [_stamp(path) for path in paths]
    identity, shard_records = _source_identity(paths)
    if identity != expected_source_identity:
        raise BundleError(
            f"GGUF identity differs: expected {expected_source_identity}, found {identity}"
        )
    entries = _tensor_index(paths, states)
    missing = [name for name, _ in SOURCE_TENSORS if name not in entries]
    if missing:
        raise BundleError(f"GGUF lacks required verifier tensors: {missing}")
    first_shape = entries[SOURCE_TENSORS[0][0]][2]["shape"]
    second_shape = entries[SOURCE_TENSORS[1][0]][2]["shape"]
    if first_shape != second_shape:
        raise BundleError(f"embedding/head shapes differ: {first_shape} vs {second_shape}")
    config = _minimal_config(states[0], first_shape[0], first_shape[1], identity)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        weights_sha, tensor_records = _write_safetensors(
            staging / "model.safetensors",
            entries,
            identity,
            producer_git,
            exporter_sha256,
            rows_per_chunk,
        )
        config_bytes = _canonical_json(config) + b"\n"
        config_sha = _write_durable(staging / "config.json", config_bytes)
        notice = (
            "Minimal offline FlashMap verifier bundle. It is not a runnable target model.\n"
            "The captured sixth verifier row is already post target hyper-connection "
            "mixer/norm; set verifier_last_hidden_state_is_post_norm=true.\n"
            "Weights are exact BF16-RNE derivations of the served Q8_0 GGUF weights, "
            "not the original unquantized checkpoint.\n"
        ).encode("utf-8")
        notice_sha = _write_durable(staging / "README.txt", notice)
        if [_stamp(path) for path in paths] != stamps:
            raise BundleError("GGUF artifact changed during export")
        manifest = {
            "schema": SCHEMA,
            "feature_abi_sha256": FEATURE_ABI,
            "source": {
                "identity_scheme": IDENTITY_SCHEME,
                "identity": identity,
                "shards": shard_records,
            },
            "producer": {
                "git_commit": producer_git,
                "exporter_sha256": exporter_sha256,
            },
            "precision": {
                "source": "Q8_0",
                "decode": "FP32",
                "output": "BF16",
                "rounding": "round-to-nearest-even",
                "caveat": "Derived from quantized served weights; not original full precision.",
            },
            "flashmap": {"verifier_last_hidden_state_is_post_norm": True},
            "files": {
                "model.safetensors": {"sha256": weights_sha},
                "config.json": {"sha256": config_sha},
                "README.txt": {"sha256": notice_sha},
            },
            "tensors": tensor_records,
        }
        manifest_bytes = _canonical_json(manifest) + b"\n"
        manifest_sha = _write_durable(staging / "manifest.json", manifest_bytes)
        try:
            directory = os.open(staging, os.O_RDONLY)
        except OSError:
            directory = -1
        if directory >= 0:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        _publish_no_replace(staging, output)
        published = True
        try:
            parent = os.open(output.parent, os.O_RDONLY)
        except OSError:
            parent = -1
        if parent >= 0:
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        return {
            "output": str(output),
            "source_identity": identity,
            "manifest_sha256": manifest_sha,
            "weights_sha256": weights_sha,
            "shape": first_shape,
        }
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="one shard of the target GGUF")
    parser.add_argument("--output", type=Path, required=True, help="new bundle directory")
    parser.add_argument("--rows-per-chunk", type=int, default=256)
    parser.add_argument(
        "--producer-git",
        required=True,
        help="40-hex Gufo source commit (exporter bytes are bound separately)",
    )
    parser.add_argument(
        "--expected-source-identity",
        required=True,
        help=f"expected {IDENTITY_SCHEME} digest of the complete GGUF set",
    )
    args = parser.parse_args()
    try:
        result = export(
            args.model,
            args.output,
            args.rows_per_chunk,
            args.producer_git,
            args.expected_source_identity,
        )
    except (BundleError, OSError, ValueError, KeyError) as error:
        raise SystemExit(f"FLASHMAP_VERIFIER_BUNDLE_ERROR {error}") from error
    print(
        "FLASHMAP_VERIFIER_BUNDLE_OK "
        f"shape={result['shape'][0]}x{result['shape'][1]} "
        f"source={result['source_identity']} manifest={result['manifest_sha256']}"
    )


if __name__ == "__main__":
    main()

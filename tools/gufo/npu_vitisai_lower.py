#!/usr/bin/env python3
"""Fail-closed, compile-only FlashMap BF16 lowering for Windows Vitis AI.

The tool consumes the authenticated static-export package produced by
``flashmap_xdna2_export.py`` and emits a self-contained embedded EPContext
model plus the strict lowering manifest accepted by that exporter.  Session
construction is used only as a compilation/load operation: this module never
calls ``InferenceSession.run``.

This is an optional developer tool.  ONNX and ONNX Runtime are imported only
after the repository-only manifest/report helpers have accepted their inputs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence


LOWERING_FORMAT = "speculators.flashmap-xdna2-lowering.v1"
SOURCE_FORMAT = "speculators.flashmap-xdna2-static-export.v1"
EVIDENCE_FORMAT = "gufo.npu-vitisai-lowering-evidence.v1"
EXPECTED_PROVIDER = "VitisAIExecutionProvider"
TOOL_NAME = "gufo-npu-vitisai-lower"
TOOL_VERSION = "1"
DRAFT_VOCABULARY_MANIFEST = "draft-vocabulary-manifest.json"
NON_EXTERNAL_ARTIFACTS = {"model.onnx", "d2t.npy", DRAFT_VOCABULARY_MANIFEST}
# AMD's Ryzen AI 1.8 VAIML option contract explicitly supports levels 1, 2,
# and 3.  This lane deliberately pins the highest documented level rather than
# inheriting the compiler default.  Transformer storage is likewise explicit.
OPTIMIZE_LEVEL = 3
PREFERRED_DATA_STORAGE = "unvectorized"
HEX_DIGITS = frozenset("0123456789abcdef")

TARGET_HIDDEN = 2560
TARGET_VOCAB = 248320
DRAFT_VOCAB = 65536
DIT_HIDDEN = 2048
DIT_LAYERS = 6
DIT_HEADS = 16
HEAD_DIM = 128
BLOCK_SIZE = 7
CONTEXT_SIZE = 128
INPUT_NAMES = [
    "noise",
    "rope_cos",
    "rope_sin",
    *[
        name
        for layer in range(DIT_LAYERS)
        for name in (f"ctx_k_{layer}", f"ctx_v_{layer}")
    ],
    "ctx_mask",
]
INPUT_SHAPES = {
    "noise": [1, BLOCK_SIZE, TARGET_HIDDEN],
    "rope_cos": [1, BLOCK_SIZE, 1, HEAD_DIM // 2],
    "rope_sin": [1, BLOCK_SIZE, 1, HEAD_DIM // 2],
    **{
        name: [1, DIT_HEADS, CONTEXT_SIZE, HEAD_DIM]
        for layer in range(DIT_LAYERS)
        for name in (f"ctx_k_{layer}", f"ctx_v_{layer}")
    },
    "ctx_mask": [1, 1, BLOCK_SIZE, CONTEXT_SIZE],
}


class LoweringError(RuntimeError):
    """A condition that makes the compiled artifact ineligible for use."""


def canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise LoweringError("value is not canonical finite JSON") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX_DIGITS for character in value)
    ):
        raise LoweringError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LoweringError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise LoweringError(f"JSON contains non-standard constant {value!r}")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise LoweringError(f"JSON number {value!r} is not finite")
    return result


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise LoweringError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise LoweringError(f"{label} is not a JSON object")
    return value


def _portable_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise LoweringError(f"{label} is not a portable relative path")
    path = Path(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise LoweringError(f"{label} is not a portable relative path")
    return value


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise LoweringError(f"{label} is absent, a symlink, or not a regular file")
    return path


def _descriptor_list(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise LoweringError(f"{label} is empty or not a list")
    result: list[dict[str, Any]] = []
    previous = ""
    folded: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise LoweringError(f"{label} descriptor schema differs")
        name = _portable_relative(item["path"], f"{label} path")
        if name <= previous or name.casefold() in folded:
            raise LoweringError(f"{label} is not uniquely sorted")
        previous = name
        folded.add(name.casefold())
        size = item["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise LoweringError(f"{label} byte count is invalid")
        _digest(item["sha256"], f"{label} digest")
        result.append(item)
    return result


def _expected_geometry() -> dict[str, Any]:
    return {
        "batch": 1,
        "block_size": BLOCK_SIZE,
        "context_size": CONTEXT_SIZE,
        "target_hidden_size": TARGET_HIDDEN,
        "draft_vocabulary_size": DRAFT_VOCAB,
        "target_vocabulary_size": TARGET_VOCAB,
        "dit_hidden_size": DIT_HIDDEN,
        "dit_layers": DIT_LAYERS,
        "dit_heads": DIT_HEADS,
        "head_dim": HEAD_DIM,
        "num_jumps": 1,
        "jump_grid": [0.0, 1.0],
        "sample_from_anchor": True,
        "time_schedule": "decoding_error",
        "noise_gate": 1.0,
        "ctx_mask_allowed_values": [0.0, -1.0e9],
    }


def load_source_export(path: Path) -> tuple[dict[str, Any], str]:  # noqa: C901
    """Authenticate the exact FlashMap static-export package."""

    path = _regular_file(path, "source export manifest").resolve()
    try:
        payload = path.read_bytes()
        manifest = strict_json_bytes(payload, "source export manifest")
    except OSError as error:
        raise LoweringError("source export manifest could not be read") from error
    fields = {
        "format",
        "geometry",
        "identities",
        "checkpoint_files",
        "verifier_files",
        "artifact_files",
        "input_names",
        "inputs",
        "output",
        "external_data_bound",
    }
    if set(manifest) != fields or manifest["format"] != SOURCE_FORMAT:
        raise LoweringError("source export manifest schema or format differs")
    if manifest["geometry"] != _expected_geometry():
        raise LoweringError("source static geometry differs")
    identity_names = {
        "target_artifact_sha256",
        "feature_abi_sha256",
        "response_corpus_sha256",
        "checkpoint_content_sha256",
        "verifier_bundle_content_sha256",
        "d2t_sha256",
        "draft_vocabulary_manifest_sha256",
        "artifact_content_sha256",
    }
    identities = manifest["identities"]
    if not isinstance(identities, dict) or set(identities) != identity_names:
        raise LoweringError("source export identities differ")
    for name in identity_names:
        _digest(identities[name], f"source {name}")
    checkpoint = _descriptor_list(manifest["checkpoint_files"], "checkpoint files")
    verifier = _descriptor_list(manifest["verifier_files"], "verifier files")
    artifacts = _descriptor_list(manifest["artifact_files"], "artifact files")
    bindings = (
        (checkpoint, "checkpoint_content_sha256"),
        (verifier, "verifier_bundle_content_sha256"),
        (artifacts, "artifact_content_sha256"),
    )
    for records, identity in bindings:
        if hashlib.sha256(canonical_json(records)).hexdigest() != identities[identity]:
            raise LoweringError(f"source {identity} differs")
    expected_inputs = {
        name: {
            "shape": INPUT_SHAPES[name],
            "export_dtype": "float32",
            "logical_dtype": (
                "float32" if name in {"rope_cos", "rope_sin", "ctx_mask"} else "bfloat16"
            ),
        }
        for name in INPUT_NAMES
    }
    if manifest["input_names"] != INPUT_NAMES or manifest["inputs"] != expected_inputs:
        raise LoweringError("source input ABI differs")
    if manifest["output"] != {
        "name": "draft_logits",
        "shape": [1, BLOCK_SIZE, DRAFT_VOCAB],
        "dtype": "float32",
    }:
        raise LoweringError("source output ABI differs")
    expected_external = any(
        item["path"] not in NON_EXTERNAL_ARTIFACTS for item in artifacts
    )
    if manifest["external_data_bound"] is not expected_external:
        raise LoweringError("source external-data declaration differs")
    root = path.parent
    expected_names = {item["path"] for item in artifacts}
    if any("/" in name for name in expected_names):
        raise LoweringError("source artifact paths must be direct children")
    entries = list(root.iterdir())
    if any(item.is_symlink() or item.is_dir() for item in entries):
        raise LoweringError("source package contains a directory or symlink")
    actual_names = {item.name for item in entries if item.is_file()}
    if actual_names != {*expected_names, path.name}:
        raise LoweringError("source package file inventory differs")
    for item in artifacts:
        artifact = _regular_file(root / item["path"], "source artifact")
        if artifact.stat().st_size != item["bytes"] or sha256_file(artifact) != item["sha256"]:
            raise LoweringError(f"source artifact {item['path']!r} differs")
    by_name = {item["path"]: item for item in artifacts}
    required_artifacts = {"model.onnx", "d2t.npy", DRAFT_VOCABULARY_MANIFEST}
    if not required_artifacts <= set(by_name):
        raise LoweringError("source package lacks a required core artifact")
    if by_name["d2t.npy"]["sha256"] != identities["d2t_sha256"]:
        raise LoweringError("source d2t identity differs")
    if (
        by_name[DRAFT_VOCABULARY_MANIFEST]["sha256"]
        != identities["draft_vocabulary_manifest_sha256"]
    ):
        raise LoweringError("source draft-vocabulary manifest identity differs")
    if path.read_bytes() != payload:
        raise LoweringError("source export manifest changed while being authenticated")
    return manifest, hashlib.sha256(payload).hexdigest()


def _tensor_shape(value: Any, label: str) -> list[int]:
    tensor = value.type.tensor_type
    if not tensor.HasField("shape"):
        raise LoweringError(f"{label} has no tensor shape")
    result: list[int] = []
    for dimension in tensor.shape.dim:
        if dimension.WhichOneof("value") != "dim_value" or dimension.dim_value <= 0:
            raise LoweringError(f"{label} is not fully static")
        result.append(int(dimension.dim_value))
    return result


def _external_tensor_locations(message: Any, onnx: Any) -> list[str]:
    """Return validated external TensorProto locations, including attributes."""

    descriptor = getattr(message, "DESCRIPTOR", None)
    if descriptor is None:
        return []
    if descriptor.full_name == "onnx.TensorProto":
        entries = list(getattr(message, "external_data", ()))
        data_location = int(getattr(message, "data_location", 0))
        if not entries and data_location == int(onnx.TensorProto.DEFAULT):
            return []
        if data_location != int(onnx.TensorProto.EXTERNAL) or not entries:
            raise LoweringError("source ONNX tensor has inconsistent external-data metadata")
        metadata: dict[str, str] = {}
        for entry in entries:
            if (
                not entry.key
                or entry.key in metadata
                or not isinstance(entry.value, str)
            ):
                raise LoweringError("source ONNX tensor has invalid external-data metadata")
            metadata[entry.key] = entry.value
        if set(metadata) - {"location", "offset", "length", "checksum"}:
            raise LoweringError("source ONNX uses an unknown external-data field")
        location = _portable_relative(
            metadata.get("location"), "external tensor location"
        )
        for name in ("offset", "length"):
            if name in metadata and (
                not metadata[name].isascii() or not metadata[name].isdigit()
            ):
                raise LoweringError(f"source ONNX external-data {name} is invalid")
        return [location]
    result: list[str] = []
    for field, value in message.ListFields():
        if field.type != field.TYPE_MESSAGE:
            continue
        if field.is_repeated:
            for item in value:
                result.extend(_external_tensor_locations(item, onnx))
        else:
            result.extend(_external_tensor_locations(value, onnx))
    return result


def preflight_source_onnx(path: Path, manifest: dict[str, Any], onnx: Any) -> None:
    """Reject symbolic/non-tensor ABI and unbound external data before compilation."""

    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as error:
        raise LoweringError("source ONNX could not be parsed") from error
    if not model.graph.node:
        raise LoweringError("source ONNX graph is empty")
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    if [value.name for value in inputs] != INPUT_NAMES or len(outputs) != 1:
        raise LoweringError("source ONNX input/output names differ")
    for value in [*inputs, *outputs, *model.graph.value_info]:
        if not value.type.HasField("tensor_type"):
            raise LoweringError(f"source value {value.name!r} is not a tensor")
        _tensor_shape(value, f"source value {value.name!r}")
    for value in inputs:
        if (
            value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
            or _tensor_shape(value, f"source input {value.name!r}") != INPUT_SHAPES[value.name]
        ):
            raise LoweringError(f"source input {value.name!r} ABI differs")
    output = outputs[0]
    if (
        output.name != "draft_logits"
        or output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        or _tensor_shape(output, "source output") != [1, BLOCK_SIZE, DRAFT_VOCAB]
    ):
        raise LoweringError("source output ABI differs")
    artifact_names = {item["path"] for item in manifest["artifact_files"]}
    for tensor in model.graph.initializer:
        if any(dimension <= 0 for dimension in tensor.dims):
            raise LoweringError(f"initializer {tensor.name!r} has a non-positive dimension")
    external = set(_external_tensor_locations(model, onnx))
    if any("/" in location or location not in artifact_names for location in external):
        raise LoweringError("source ONNX external data is not manifest-bound")
    expected_external = artifact_names - NON_EXTERNAL_ARTIFACTS
    if external != expected_external or bool(external) != manifest["external_data_bound"]:
        raise LoweringError("source ONNX external-data inventory differs from its manifest")
    for node in model.graph.node:
        if any(
            attribute.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
            for attribute in node.attribute
        ):
            raise LoweringError("source ONNX contains an unvalidated control-flow subgraph")
    try:
        onnx.checker.check_model(str(path), full_check=False)
    except Exception as error:
        raise LoweringError("source ONNX failed structural validation") from error


def vaiml_config() -> dict[str, Any]:
    """Return the one accepted transformer BF16 compiler configuration."""

    return {
        "passes": [
            {"name": "init", "plugin": "vaip-pass_init"},
            {
                "name": "vaiml_partition",
                "plugin": "vaip-pass_vaiml_partition",
                "vaiml_config": {
                    "optimize_level": OPTIMIZE_LEVEL,
                    "preferred_data_storage": PREFERRED_DATA_STORAGE,
                },
            },
        ],
        "target": "VAIML",
        "targets": [{"name": "VAIML", "pass": ["init", "vaiml_partition"]}],
    }


def parse_assignment_report_payload(payload: bytes) -> dict[str, int]:
    """Parse AMD's documented ``deviceStat`` report and require full NPU placement.

    AMD's public schema documents additional report sections but only one
    assignment schema: a top-level ``deviceStat`` list containing ``all``,
    ``CPU``, and ``NPU`` records with integer ``nodeNum`` values.  Extra report
    sections and extra per-device diagnostic fields are retained as evidence;
    alternate assignment layouts are deliberately not guessed.
    """

    report = strict_json_bytes(payload, "Vitis AI assignment report")
    stats = report.get("deviceStat")
    if not isinstance(stats, list) or len(stats) != 3:
        raise LoweringError("assignment report does not use documented deviceStat schema")
    result: dict[str, int] = {}
    operator_sets: dict[str, set[str]] = {}
    for record in stats:
        if not isinstance(record, dict) or not {"name", "nodeNum", "supportedOpType"} <= set(record):
            raise LoweringError("assignment report deviceStat record differs")
        name = record["name"]
        count = record["nodeNum"]
        operators = record["supportedOpType"]
        if name not in {"all", "CPU", "NPU"} or name in result:
            raise LoweringError("assignment report contains an unknown or repeated device")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise LoweringError("assignment report nodeNum is not a non-negative integer")
        if not isinstance(operators, list) or any(
            not isinstance(operator, str) or not operator for operator in operators
        ):
            raise LoweringError("assignment report supportedOpType differs")
        if len(operators) != len(set(operators)):
            raise LoweringError("assignment report repeats a supported operator type")
        result[name] = count
        operator_sets[name] = set(operators)
    if set(result) != {"all", "CPU", "NPU"}:
        raise LoweringError("assignment report device inventory differs")
    if result["all"] <= 0 or result["CPU"] != 0 or result["NPU"] != result["all"]:
        raise LoweringError(
            "source graph was not placed wholly on the NPU: "
            f"all={result['all']} CPU={result['CPU']} NPU={result['NPU']}"
        )
    if operator_sets["CPU"] or operator_sets["NPU"] != operator_sets["all"]:
        raise LoweringError("assignment report operator inventory contradicts node placement")
    return result


def inspect_epcontext(path: Path, source_manifest: dict[str, Any], onnx: Any) -> dict[str, Any]:
    try:
        model = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model, full_check=False)
    except Exception as error:
        raise LoweringError("candidate EPContext ONNX is invalid") from error
    if _external_tensor_locations(model, onnx):
        raise LoweringError("candidate EPContext ONNX contains external tensor data")
    if model.graph.initializer or model.graph.sparse_initializer or model.functions:
        raise LoweringError("candidate EPContext ONNX contains executable data outside EPContext")
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    if [value.name for value in inputs] != INPUT_NAMES or len(outputs) != 1:
        raise LoweringError("candidate EPContext ABI names differ")
    for value in [*inputs, *outputs, *model.graph.value_info]:
        if not value.type.HasField("tensor_type"):
            raise LoweringError(f"candidate value {value.name!r} is not a tensor")
        _tensor_shape(value, f"candidate value {value.name!r}")
    for value in inputs:
        if (
            value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
            or _tensor_shape(value, f"candidate input {value.name!r}") != INPUT_SHAPES[value.name]
        ):
            raise LoweringError(f"candidate input {value.name!r} ABI differs")
    output = outputs[0]
    if (
        output.name != source_manifest["output"]["name"]
        or output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        or _tensor_shape(output, "candidate output") != source_manifest["output"]["shape"]
    ):
        raise LoweringError("candidate output ABI differs")
    nodes = list(model.graph.node)
    if not nodes or any(
        node.op_type != "EPContext" or node.domain != "com.microsoft"
        for node in nodes
    ):
        raise LoweringError("candidate graph contains a non-EPContext node")
    consumed_inputs = {name for node in nodes for name in node.input if name}
    produced_outputs = [name for node in nodes for name in node.output if name]
    if not set(INPUT_NAMES) <= consumed_inputs or output.name not in produced_outputs:
        raise LoweringError("candidate EPContext graph does not bind its declared ABI")
    if len(produced_outputs) != len(set(produced_outputs)):
        raise LoweringError("candidate EPContext graph repeats a produced value")
    context_bytes = 0
    partitions: list[dict[str, Any]] = []
    for node in nodes:
        attribute_names = [attribute.name for attribute in node.attribute]
        if len(attribute_names) != len(set(attribute_names)):
            raise LoweringError("candidate EPContext repeats an attribute")
        attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        if attributes.get("embed_mode") != 1:
            raise LoweringError("candidate EPContext is not embed_mode=1")
        source = attributes.get("source")
        if isinstance(source, bytes):
            source = source.decode("utf-8", errors="strict")
        if source != EXPECTED_PROVIDER:
            raise LoweringError("candidate EPContext source is not Vitis AI")
        context = attributes.get("ep_cache_context")
        if not isinstance(context, bytes) or not context:
            raise LoweringError("candidate EPContext has no embedded compiler context")
        context_bytes += len(context)
        partitions.append({"node_name": node.name, "context_bytes": len(context)})
    return {
        "nodes": len(nodes),
        "embedded_context_bytes": context_bytes,
        "partitions": partitions,
    }


def make_lowering_manifest(
    source_manifest_sha256: str,
    source_artifact_sha256: str,
    draft_vocabulary_manifest_sha256: str,
    candidate_sha256: str,
    d2t_sha256: str,
    tool_sha256: str,
    evidence_sha256: str,
    toolchain_sha256: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Construct the exact Speculators v1 top-level lowering schema."""

    for value, label in (
        (source_manifest_sha256, "source manifest"),
        (source_artifact_sha256, "source artifact"),
        (draft_vocabulary_manifest_sha256, "draft-vocabulary manifest"),
        (candidate_sha256, "candidate model"),
        (d2t_sha256, "draft-to-target"),
        (tool_sha256, "tool executable"),
        (evidence_sha256, "lowering evidence"),
        (toolchain_sha256, "toolchain evidence"),
    ):
        _digest(value, label)
    if not isinstance(options, dict):
        raise LoweringError("lowering options are not an object")
    return {
        "format": LOWERING_FORMAT,
        "source_export_manifest": {
            "path": "source/export-manifest.json",
            "sha256": source_manifest_sha256,
        },
        "source_artifact_content_sha256": source_artifact_sha256,
        "draft_vocabulary_manifest_sha256": draft_vocabulary_manifest_sha256,
        "candidate_model": {"path": "candidate.onnx", "sha256": candidate_sha256},
        "kind": "bf16",
        "tool": {
            "name": TOOL_NAME,
            "version": TOOL_VERSION,
            "executable": {"path": "npu_vitisai_lower.py", "sha256": tool_sha256},
        },
        "evidence": {
            "path": "lowering-evidence.json",
            "sha256": evidence_sha256,
            "toolchain_sha256": toolchain_sha256,
        },
        "options": options,
        "calibration": None,
        "draft_to_target": {"path": "d2t.npy", "sha256": d2t_sha256},
    }


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _environment(name: str, value: str | None):
    present = name in os.environ
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if present:
            assert previous is not None
            os.environ[name] = previous
        else:
            os.environ.pop(name, None)


@contextmanager
def _captured_file_descriptors(path: Path):
    """Contain Python/CRT stdout and stderr produced during provider calls."""

    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with path.open("ab", buffering=0) as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def require_ep_library(explicit: Path, expected_sha256: str) -> Path:
    """Authenticate the caller-selected managed EP DLL without discovery."""

    expected = _digest(expected_sha256, "expected Vitis AI provider DLL")
    library = _regular_file(explicit, "Vitis AI provider DLL").resolve()
    if library.name.casefold() != "onnxruntime_vitisai_ep.dll":
        raise LoweringError("Vitis AI provider DLL basename differs")
    if sha256_file(library) != expected:
        raise LoweringError("Vitis AI provider DLL digest differs")
    return library


def _tree_inventory(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise LoweringError(f"inventory root {root} is absent or unsafe")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise LoweringError(f"isolated tree contains symlink {path}")
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _device_record(device: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("ep_name", "device_type", "vendor", "vendor_id", "device_id", "ep_metadata"):
        try:
            value = getattr(device, name)
        except (AttributeError, RuntimeError):
            continue
        result[name] = _json_safe(value)
    if result.get("ep_name") != EXPECTED_PROVIDER:
        raise LoweringError("selected execution-provider device identity differs")
    return result


def _authenticated_user() -> dict[str, str]:
    """Bind the Windows logon name and SID without retaining credentials."""

    try:
        completed = subprocess.run(
            ["whoami.exe", "/user", "/fo", "csv", "/nh"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        line = completed.stdout.strip()
        # whoami's CSV fields cannot contain an unescaped quote.  Parsing this
        # exact two-field shape avoids accepting localized prose or warnings.
        import csv  # noqa: PLC0415

        rows = list(csv.reader([line]))
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise LoweringError("authenticated Windows user identity is unavailable") from error
    if len(rows) != 1 or len(rows[0]) != 2:
        raise LoweringError("authenticated Windows user identity schema differs")
    name, sid = rows[0]
    if not name or not sid.startswith("S-1-"):
        raise LoweringError("authenticated Windows user identity is invalid")
    return {"name": name, "sid": sid}


def _runtime_inventory(module: Any) -> list[dict[str, Any]]:
    root = Path(module.__file__).resolve().parent
    candidates = sorted(
        path
        for pattern in ("*.pyd", "*.dll")
        for path in root.rglob(pattern)
        if path.is_file() and not path.is_symlink()
    )
    if not candidates:
        candidates = [_regular_file(Path(module.__file__).resolve(), "runtime module")]
    return [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in candidates
    ]


def _provider_inventory(library: Path) -> list[dict[str, Any]]:
    entries = list(library.parent.glob("*.dll"))
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise LoweringError("provider package contains an unsafe DLL entry")
    files = sorted(entries)
    if library not in files:
        raise LoweringError("provider library is absent from its package inventory")
    return [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    ]


def _copy_source_package(
    source_manifest: Path,
    target: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
) -> None:
    target.mkdir()
    shutil.copyfile(source_manifest, target / "export-manifest.json")
    if sha256_file(target / "export-manifest.json") != manifest_sha256:
        raise LoweringError("source manifest changed while copying")
    for item in manifest["artifact_files"]:
        destination = target / item["path"]
        shutil.copyfile(source_manifest.parent / item["path"], destination)
        if destination.stat().st_size != item["bytes"] or sha256_file(destination) != item["sha256"]:
            raise LoweringError(f"source artifact {item['path']!r} changed while copying")


def _session_options(ort: Any, devices: list[Any], provider_options: dict[str, str], compile_path: str | None) -> Any:
    options = ort.SessionOptions()
    options.log_severity_level = 0
    options.logid = TOOL_NAME
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if compile_path is not None:
        options.add_session_config_entry("ep.context_enable", "1")
        options.add_session_config_entry("ep.context_file_path", compile_path)
        options.add_session_config_entry("ep.context_embed_mode", "1")
    options.add_provider_for_devices(devices, provider_options)
    return options


def lower(
    source_manifest_path: Path,
    output: Path,
    ep_library: Path,
    ep_library_sha256: str,
) -> dict[str, Any]:  # noqa: C901
    source_manifest_path = _regular_file(
        source_manifest_path, "source export manifest"
    ).resolve()
    if output.exists() or output.is_symlink():
        raise LoweringError("output already exists")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest, authenticated_source_manifest_digest = load_source_export(
        source_manifest_path
    )
    try:
        import onnx  # noqa: PLC0415
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError as error:
        raise LoweringError("onnx and onnxruntime are required for lowering") from error
    library = require_ep_library(ep_library, ep_library_sha256)
    provider_package_before = _provider_inventory(library)
    try:
        ort.register_execution_provider_library(EXPECTED_PROVIDER, str(library))
        devices = [device for device in ort.get_ep_devices() if device.ep_name == EXPECTED_PROVIDER]
    except Exception as error:
        raise LoweringError("Vitis AI dynamic provider registration failed") from error
    if len(devices) != 1:
        raise LoweringError(f"expected exactly one Vitis AI device, found {len(devices)}")

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.vitisai-", dir=output.parent))
    complete = False
    try:
        (stage / "cache").mkdir()
        (stage / "logs").mkdir()
        _copy_source_package(
            source_manifest_path,
            stage / "source",
            manifest,
            authenticated_source_manifest_digest,
        )
        preflight_source_onnx(stage / "source" / "model.onnx", manifest, onnx)
        shutil.copyfile(stage / "source" / "d2t.npy", stage / "d2t.npy")
        tool_source = _regular_file(Path(__file__), "lowering tool executable").resolve()
        tool_copy = stage / "npu_vitisai_lower.py"
        shutil.copyfile(tool_source, tool_copy)
        tool_digest = sha256_file(tool_source)
        if sha256_file(tool_copy) != tool_digest:
            raise LoweringError("lowering tool changed while copying")
        config_path = stage / "vaiml-config.json"
        config_path.write_bytes(canonical_json(vaiml_config()))
        provider_options = {
            "config_file": "vaiml-config.json",
            "cache_dir": "cache",
            "cache_key": f"flashmap-bf16-{manifest['identities']['artifact_content_sha256']}",
            "enable_cache_file_io_in_mem": "0",
            "ai_analyzer_visualization": "true",
            "ai_analyzer_profiling": "false",
        }
        session_entries = {
            "session.disable_cpu_ep_fallback": "1",
            "ep.context_enable": "1",
            "ep.context_file_path": "candidate.onnx",
            "ep.context_embed_mode": "1",
        }
        with _working_directory(stage), _environment(
            "XLNX_ONNX_EP_REPORT_FILE", "assignment-report.json"
        ):
            try:
                with _captured_file_descriptors(stage / "logs" / "compile.log"):
                    session = ort.InferenceSession(
                        "source/model.onnx",
                        sess_options=_session_options(
                            ort, devices, provider_options, "candidate.onnx"
                        ),
                    )
            except Exception as error:
                raise LoweringError("Vitis AI compile-only session creation failed") from error
            if session.get_providers() != [EXPECTED_PROVIDER]:
                raise LoweringError(f"compile session provider inventory differs: {session.get_providers()!r}")
            del session
        candidate = _regular_file(stage / "candidate.onnx", "candidate EPContext model")
        inspection = inspect_epcontext(candidate, manifest, onnx)
        candidate_digest = sha256_file(candidate)
        reports = list((stage / "cache").rglob("assignment-report.json"))
        if len(reports) != 1:
            raise LoweringError(f"expected one isolated assignment report, found {len(reports)}")
        report_payload = _regular_file(
            reports[0], "isolated assignment report"
        ).read_bytes()
        assignment = parse_assignment_report_payload(report_payload)
        report_copy = stage / "assignment-report.json"
        report_copy.write_bytes(report_payload)
        (stage / "reload-cache").mkdir()
        reload_provider_options = dict(provider_options)
        reload_provider_options["cache_dir"] = "reload-cache"
        reload_provider_options["cache_key"] = (
            f"flashmap-embedded-reload-{candidate_digest}"
        )
        before_reload = _tree_inventory(stage / "reload-cache")
        if before_reload:
            raise LoweringError("fresh EPContext reload cache is not initially empty")
        with _working_directory(stage), _environment("XLNX_ONNX_EP_REPORT_FILE", None):
            try:
                with _captured_file_descriptors(stage / "logs" / "reload.log"):
                    reload_session = ort.InferenceSession(
                        "candidate.onnx",
                        sess_options=_session_options(
                            ort, devices, reload_provider_options, None
                        ),
                    )
            except Exception as error:
                raise LoweringError("fresh embedded EPContext reload failed") from error
            if reload_session.get_providers() != [EXPECTED_PROVIDER]:
                raise LoweringError(f"reload session provider inventory differs: {reload_session.get_providers()!r}")
            del reload_session
        after_reload = _tree_inventory(stage / "reload-cache")
        if after_reload:
            raise LoweringError("fresh EPContext reload populated its isolated empty cache")

        source_manifest_digest = sha256_file(stage / "source" / "export-manifest.json")
        config_digest = sha256_file(config_path)
        report_digest = sha256_file(report_copy)
        d2t_digest = sha256_file(stage / "d2t.npy")
        provider_package = _provider_inventory(library)
        if provider_package != provider_package_before:
            raise LoweringError("Vitis AI provider package changed during lowering")
        provider_package_digest = hashlib.sha256(
            canonical_json(provider_package)
        ).hexdigest()
        log_inventory = _tree_inventory(stage / "logs")
        toolchain = {
            "python": {
                "version": sys.version,
                "executable": str(Path(sys.executable).resolve()),
                "executable_sha256": sha256_file(Path(sys.executable).resolve()),
            },
            "onnx": {"version": onnx.__version__, "files": _runtime_inventory(onnx)},
            "onnxruntime": {"version": ort.__version__, "files": _runtime_inventory(ort)},
        }
        toolchain_digest = hashlib.sha256(canonical_json(toolchain)).hexdigest()
        evidence = {
            "format": EVIDENCE_FORMAT,
            "source": {
                "export_manifest_sha256": source_manifest_digest,
                "artifact_content_sha256": manifest["identities"]["artifact_content_sha256"],
                "d2t_sha256": manifest["identities"]["d2t_sha256"],
                "draft_vocabulary_manifest_sha256": manifest["identities"][
                    "draft_vocabulary_manifest_sha256"
                ],
                "model_sha256": next(
                    item["sha256"] for item in manifest["artifact_files"] if item["path"] == "model.onnx"
                ),
            },
            "execution": {
                "authenticated_user": _authenticated_user(),
                "python_user": getpass.getuser(),
                "host": platform.node(),
                "os": {
                    "platform": platform.platform(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                },
                "zero_inference": True,
            },
            "toolchain": toolchain,
            "provider": {
                "name": EXPECTED_PROVIDER,
                "library": {
                    "path": str(library),
                    "bytes": library.stat().st_size,
                    "sha256": sha256_file(library),
                    "caller_expected_sha256": ep_library_sha256,
                },
                "package_directory": str(library.parent),
                "package_dlls": provider_package,
                "package_dll_inventory_sha256": provider_package_digest,
                "device_and_driver_metadata": _device_record(devices[0]),
            },
            "compiler": {
                "config": {"path": "vaiml-config.json", "sha256": config_digest, "value": vaiml_config()},
                "provider_options": provider_options,
                "reload_provider_options": reload_provider_options,
                "session_config_entries": session_entries,
            },
            "assignment_report": {
                "path": "assignment-report.json",
                "sha256": report_digest,
                "schema_adapter": "amd-deviceStat-v1",
                "node_counts": assignment,
                "original_cache_path": reports[0].relative_to(stage).as_posix(),
            },
            "candidate": {
                "path": "candidate.onnx",
                "sha256": candidate_digest,
                "inspection": inspection,
            },
            "reload": {
                "fresh_session": True,
                "cpu_fallback_disabled": True,
                "cache_content_unchanged": True,
                "cache_directory": "reload-cache",
                "cache_inventory_sha256": hashlib.sha256(canonical_json(after_reload)).hexdigest(),
            },
            "logs": {
                "directory": "logs",
                "inventory": log_inventory,
                "inventory_sha256": hashlib.sha256(
                    canonical_json(log_inventory)
                ).hexdigest(),
            },
        }
        evidence_path = stage / "lowering-evidence.json"
        evidence_path.write_bytes(canonical_json(evidence))
        manifest_options = {
            "precision": "bf16",
            "compile_only": True,
            "inference_calls": 0,
            "ep_context_embed_mode": 1,
            "cpu_ep_fallback": False,
            "provider_registration": "register_execution_provider_library",
            "provider_attachment": "SessionOptions.add_provider_for_devices",
            "provider_library_sha256": ep_library_sha256,
            "provider_package_dll_inventory_sha256": provider_package_digest,
            "provider_options": provider_options,
            "reload_provider_options": reload_provider_options,
            "session_config_entries": session_entries,
            "vaiml_config_sha256": config_digest,
        }
        lowering_manifest = make_lowering_manifest(
            source_manifest_digest,
            manifest["identities"]["artifact_content_sha256"],
            manifest["identities"]["draft_vocabulary_manifest_sha256"],
            candidate_digest,
            d2t_digest,
            tool_digest,
            sha256_file(evidence_path),
            toolchain_digest,
            manifest_options,
        )
        (stage / "lowering-manifest.json").write_bytes(canonical_json(lowering_manifest))
        allowed_root = {
            "source",
            "cache",
            "reload-cache",
            "logs",
            "candidate.onnx",
            "d2t.npy",
            "vaiml-config.json",
            "assignment-report.json",
            "lowering-evidence.json",
            "npu_vitisai_lower.py",
            "lowering-manifest.json",
        }
        unexpected = {item.name for item in stage.iterdir()} - allowed_root
        if unexpected:
            raise LoweringError(f"provider emitted unbound root artifacts: {sorted(unexpected)!r}")
        stage.rename(output)
        complete = True
        return lowering_manifest
    finally:
        if not complete and stage.exists():
            shutil.rmtree(stage)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-export-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ep-library", type=Path, required=True)
    parser.add_argument("--ep-library-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = lower(
            args.source_export_manifest,
            args.output,
            args.ep_library,
            args.ep_library_sha256,
        )
    except (LoweringError, OSError) as error:
        print(f"FLASHMAP_VITISAI_LOWER_INVALID {error}")
        return 1
    print(
        "FLASHMAP_VITISAI_LOWER_OK "
        f"candidate={manifest['candidate_model']['sha256']} output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

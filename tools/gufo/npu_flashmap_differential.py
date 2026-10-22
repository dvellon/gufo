#!/usr/bin/env python3
"""Offline, fail-closed FlashMap NPU differential qualification.

The harness replays captured real drafter inputs through a CPU reference and
the dynamically registered Windows Vitis AI execution provider.  A run is
qualified only when every captured greedy draft agrees and ONNX Runtime's
profile attributes every executed graph node to Vitis AI.  CPU execution is a
diagnostic fallback only: its drafts are never reported as serving-eligible.

This is a developer tool.  It does not add ONNX Runtime to Gufo's production
dependencies and it does not call the target verifier itself.  Qualified token
IDs are intended for Session::DecodeDraftStep, which remains authoritative.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any, Sequence

import numpy as np


FORMAT = "gufo.flashmap-npu-differential.v1"
EXPECTED_PROVIDER = "VitisAIExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
HEX_DIGITS = frozenset("0123456789abcdef")


class QualificationError(RuntimeError):
    """A condition that makes NPU drafts ineligible for serving."""


@contextmanager
def working_directory(path: Path):
    """Contain provider-created relative compiler artifacts."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX_DIGITS for character in value)
    ):
        raise QualificationError(f"{label} is not lowercase SHA-256")
    return value


def _relative_file(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise QualificationError(f"{label} is not a portable relative path")
    relative = Path(value)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise QualificationError(f"{label} is not a contained relative path")
    path = root.joinpath(relative)
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"{label} is absent, a symlink, or not a file")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise QualificationError(f"{label} escapes the capture directory") from error
    return path


def _exact_object(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise QualificationError(f"{label} fields differ from the v1 ABI")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationError(f"JSON repeats object key {key!r}")
        result[key] = value
    return result


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QualificationError(f"{label} is not a positive integer")
    return value


def _load_array(root: Path, descriptor: object, label: str) -> np.ndarray:
    item = _exact_object(
        descriptor, {"path", "sha256", "dtype", "shape"}, label
    )
    path = _relative_file(root, item["path"], f"{label} path")
    expected_digest = _hex_digest(item["sha256"], f"{label} digest")
    if _sha256(path) != expected_digest:
        raise QualificationError(f"{label} content digest differs")
    dtype = item["dtype"]
    shape = item["shape"]
    if not isinstance(dtype, str) or not isinstance(shape, list) or any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
        for dimension in shape
    ):
        raise QualificationError(f"{label} dtype or shape is invalid")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise QualificationError(f"{label} is not a safe NPY array") from error
    if not isinstance(array, np.ndarray):
        raise QualificationError(f"{label} did not contain one NPY array")
    if str(array.dtype) != dtype or list(array.shape) != shape:
        raise QualificationError(f"{label} array ABI differs")
    if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
        raise QualificationError(f"{label} contains unsupported values")
    return array


def load_capture(
    manifest_path: Path,
) -> tuple[
    Path,
    int,
    int,
    dict[str, Any],
    np.ndarray | None,
    dict[str, str],
    list[dict[str, Any]],
    dict[str, str],
]:
    """Load a strict real-input replay manifest and all bound NPY arrays."""

    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise QualificationError("capture manifest is absent, a symlink, or not a file")
    try:
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(
            manifest_payload.decode("utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationError("capture manifest is not strict UTF-8 JSON") from error
    manifest = _exact_object(
        manifest,
        {
            "format",
            "target_artifact_sha256",
            "feature_abi_sha256",
            "model",
            "block_size",
            "target_vocabulary_size",
            "draft_to_target",
            "provenance",
            "output",
            "cases",
        },
        "capture manifest",
    )
    if manifest["format"] != FORMAT:
        raise QualificationError("capture format differs")
    identities = {
        "manifest": hashlib.sha256(manifest_payload).hexdigest(),
        "target": _hex_digest(
            manifest["target_artifact_sha256"], "target artifact identity"
        ),
        "feature_abi": _hex_digest(
            manifest["feature_abi_sha256"], "feature ABI identity"
        ),
    }
    root = manifest_path.parent
    model = _exact_object(manifest["model"], {"path", "sha256"}, "model")
    model_path = _relative_file(root, model["path"], "model path")
    model_digest = _hex_digest(model["sha256"], "model digest")
    if _sha256(model_path) != model_digest:
        raise QualificationError("ONNX model content digest differs")
    identities["model"] = model_digest
    provenance = _exact_object(
        manifest["provenance"],
        {
            "source_export_manifest_sha256",
            "source_artifact_content_sha256",
            "lowering_manifest_sha256",
            "independent_reference_manifest_sha256",
            "candidate_model_sha256",
        },
        "provenance",
    )
    provenance = {
        key: _hex_digest(value, f"provenance {key}")
        for key, value in provenance.items()
    }
    if provenance["candidate_model_sha256"] != model_digest:
        raise QualificationError(
            "provenance candidate model identity differs from model digest"
        )
    block_size = _positive_integer(manifest["block_size"], "block size")
    if block_size != 7:
        raise QualificationError("block size differs from the seven-proposal ABI")
    target_vocabulary_size = _positive_integer(
        manifest["target_vocabulary_size"], "target vocabulary size"
    )
    output = _exact_object(
        manifest["output"], {"name", "kind", "argmax_axis"}, "output"
    )
    if (
        not isinstance(output["name"], str)
        or not output["name"]
        or output["kind"] not in ("token_ids", "logits")
        or isinstance(output["argmax_axis"], bool)
        or not isinstance(output["argmax_axis"], int)
    ):
        raise QualificationError("output ABI is invalid")
    raw_draft_to_target = manifest["draft_to_target"]
    draft_to_target: np.ndarray | None = None
    if raw_draft_to_target is not None:
        raw_mapping = _exact_object(
            raw_draft_to_target,
            {"path", "sha256", "dtype", "shape"},
            "draft-to-target mapping",
        )
        identities["draft_to_target"] = _hex_digest(
            raw_mapping["sha256"], "draft-to-target mapping digest"
        )
        draft_to_target = _load_array(
            root, raw_mapping, "draft-to-target mapping"
        )
        if draft_to_target.ndim != 1 or draft_to_target.dtype.kind not in "iu":
            raise QualificationError(
                "draft-to-target mapping is not one-dimensional integral data"
            )
        mapped = [int(token) for token in draft_to_target]
        if len(mapped) != len(set(mapped)):
            raise QualificationError("draft-to-target mapping repeats a target token")
        if any(not 0 <= token < target_vocabulary_size for token in mapped):
            raise QualificationError(
                "draft-to-target mapping contains an out-of-range target token"
            )
    if output["kind"] == "token_ids" and draft_to_target is not None:
        raise QualificationError("token-id output must not have a draft vocabulary map")
    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        raise QualificationError("capture contains no replay cases")
    identifiers: set[str] = set()
    loaded: list[dict[str, Any]] = []
    for ordinal, raw_case in enumerate(cases):
        case = _exact_object(
            raw_case, {"id", "inputs", "reference_tokens"}, f"case {ordinal}"
        )
        identifier = case["id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise QualificationError(f"case {ordinal} identity is empty or repeated")
        identifiers.add(identifier)
        raw_inputs = case["inputs"]
        if not isinstance(raw_inputs, dict) or not raw_inputs:
            raise QualificationError(f"case {identifier} has no inputs")
        inputs: dict[str, np.ndarray] = {}
        for name, descriptor in raw_inputs.items():
            if not isinstance(name, str) or not name:
                raise QualificationError(f"case {identifier} has an invalid input name")
            inputs[name] = _load_array(root, descriptor, f"case {identifier} input {name}")
        tokens = case["reference_tokens"]
        if (
            not isinstance(tokens, list)
            or len(tokens) != block_size
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or not 0 <= token < target_vocabulary_size
                for token in tokens
            )
        ):
            raise QualificationError(f"case {identifier} reference tokens are invalid")
        loaded.append({"id": identifier, "inputs": inputs, "tokens": tokens})
    return (
        model_path,
        block_size,
        target_vocabulary_size,
        output,
        draft_to_target,
        provenance,
        loaded,
        identities,
    )


def greedy_tokens(
    value: np.ndarray,
    output: dict[str, Any],
    block_size: int,
    target_vocabulary_size: int,
    draft_to_target: np.ndarray | None,
) -> list[int]:
    """Convert one drafter output into its deterministic proposal block."""

    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise QualificationError("drafter output contains non-finite values")
    if output["kind"] == "token_ids":
        if draft_to_target is not None:
            raise QualificationError(
                "token-id output must not have a draft vocabulary map"
            )
        if array.dtype.kind not in "iu":
            raise QualificationError("token-id output is not integral")
        tokens = array.reshape(-1)
    else:
        if array.ndim < 2:
            raise QualificationError("logit output has fewer than two dimensions")
        axis = output["argmax_axis"]
        if not -array.ndim <= axis < array.ndim:
            raise QualificationError("logit argmax axis is out of range")
        logit_width = array.shape[axis]
        local_tokens = np.argmax(array, axis=axis).reshape(-1)
        if draft_to_target is None:
            if logit_width != target_vocabulary_size:
                raise QualificationError(
                    "restricted logit output has no draft-to-target mapping"
                )
            tokens = local_tokens
        else:
            if draft_to_target.size != logit_width:
                raise QualificationError(
                    "draft-to-target mapping length differs from logit width"
                )
            tokens = draft_to_target[local_tokens]
    if tokens.size != block_size:
        raise QualificationError(
            f"drafter produced {tokens.size} proposals instead of {block_size}"
        )
    result = [int(token) for token in tokens]
    if any(not 0 <= token < target_vocabulary_size for token in result):
        raise QualificationError("drafter produced a token outside the vocabulary")
    return result


def profile_providers(profile_path: Path) -> tuple[list[str], int]:
    """Return every provider assigned to a profiled ONNX node event."""

    try:
        events = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationError("ONNX provider profile is unreadable") from error
    if not isinstance(events, list):
        raise QualificationError("ONNX provider profile is not an event list")
    providers: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        arguments = event.get("args")
        if not isinstance(arguments, dict):
            continue
        provider = arguments.get("provider")
        if isinstance(provider, str) and provider:
            providers.append(provider)
    return sorted(set(providers)), len(providers)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(np.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def discover_ep_library(explicit: Path | None) -> Path:
    if explicit is not None:
        candidates = [explicit]
    else:
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        candidates = sorted(
            (program_files / "WindowsApps").glob(
                "WindowsWorkload.EP.AMD.VitisAI.Framework.*_x64__8wekyb3d8bbwe/"
                "ExecutionProvider/onnxruntime_vitisai_ep.dll"
            ),
            reverse=True,
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise QualificationError("Vitis AI execution-provider library was not found")


def _output_index(session: Any, output_name: str) -> int:
    names = [item.name for item in session.get_outputs()]
    if names.count(output_name) != 1:
        raise QualificationError("bound drafter output is absent or repeated")
    return names.index(output_name)


def _validate_inputs(session: Any, cases: list[dict[str, Any]]) -> None:
    names = [item.name for item in session.get_inputs()]
    if len(names) != len(set(names)):
        raise QualificationError("ONNX model repeats an input name")
    expected = set(names)
    for case in cases:
        if set(case["inputs"]) != expected:
            raise QualificationError(f"case {case['id']} input names differ from ONNX")


def _reference_replay(
    ort: Any,
    model_path: Path,
    output: dict[str, Any],
    block_size: int,
    target_vocabulary_size: int,
    draft_to_target: np.ndarray | None,
    cases: list[dict[str, Any]],
) -> tuple[list[list[int]], list[float]]:
    session = ort.InferenceSession(str(model_path), providers=[CPU_PROVIDER])
    _validate_inputs(session, cases)
    output_index = _output_index(session, output["name"])
    drafts: list[list[int]] = []
    timings: list[float] = []
    for case in cases:
        started = time.perf_counter_ns()
        values = session.run(None, case["inputs"])
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
        tokens = greedy_tokens(
            values[output_index],
            output,
            block_size,
            target_vocabulary_size,
            draft_to_target,
        )
        if tokens != case["tokens"]:
            raise QualificationError(
                f"CPU reference differs from captured tokens for case {case['id']}"
            )
        drafts.append(tokens)
    return drafts, timings


def run_differential(
    manifest_path: Path,
    ep_library: Path | None,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if warmup < 1 or iterations < 2:
        raise QualificationError("warmup must be positive and iterations at least two")
    (
        model_path,
        block_size,
        target_vocabulary_size,
        output,
        draft_to_target,
        provenance,
        cases,
        identities,
    ) = load_capture(manifest_path)
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise QualificationError("onnxruntime is not installed") from error

    cpu_drafts, cpu_timings = _reference_replay(
        ort,
        model_path,
        output,
        block_size,
        target_vocabulary_size,
        draft_to_target,
        cases,
    )
    library = discover_ep_library(ep_library)
    try:
        ort.register_execution_provider_library(EXPECTED_PROVIDER, str(library))
        devices = [
            device for device in ort.get_ep_devices() if device.ep_name == EXPECTED_PROVIDER
        ]
    except Exception as error:
        raise QualificationError("Vitis AI provider registration failed") from error
    if not devices:
        raise QualificationError("Vitis AI NPU device was not found")

    with tempfile.TemporaryDirectory(prefix="gufo-flashmap-npu-diff-") as temp_name:
        temp = Path(temp_name)
        with working_directory(temp):
            options = ort.SessionOptions()
            options.enable_profiling = True
            options.profile_file_prefix = str(temp / "vitisai-profile")
            options.add_provider_for_devices(devices, {})
            started = time.perf_counter_ns()
            try:
                session = ort.InferenceSession(str(model_path), sess_options=options)
            except Exception as error:
                raise QualificationError("Vitis AI session or compilation failed") from error
            cold_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            if EXPECTED_PROVIDER not in session.get_providers():
                raise QualificationError("Vitis AI provider was not attached")
            _validate_inputs(session, cases)
            output_index = _output_index(session, output["name"])
            first = cases[0]
            for _ in range(warmup):
                session.run(None, first["inputs"])

            timings: list[float] = []
            npu_drafts: list[list[int]] = []
            for ordinal, case in enumerate(cases):
                last_values = None
                case_timings: list[float] = []
                for _ in range(iterations):
                    started = time.perf_counter_ns()
                    last_values = session.run(None, case["inputs"])
                    elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
                    case_timings.append(elapsed)
                    timings.append(elapsed)
                assert last_values is not None
                tokens = greedy_tokens(
                    last_values[output_index],
                    output,
                    block_size,
                    target_vocabulary_size,
                    draft_to_target,
                )
                if tokens != cpu_drafts[ordinal] or tokens != case["tokens"]:
                    raise QualificationError(
                        f"NPU greedy draft differs for case {case['id']}"
                    )
                npu_drafts.append(tokens)
                case["latency"] = {
                    "medianMilliseconds": statistics.median(case_timings),
                    "p95Milliseconds": percentile(case_timings, 0.95),
                }
            profile_path = Path(session.end_profiling())
            providers, node_events = profile_providers(profile_path)
            if node_events == 0 or providers != [EXPECTED_PROVIDER]:
                raise QualificationError(
                    "CPU fallback or incomplete NPU placement detected: "
                    f"node_events={node_events} providers={providers}"
                )

    if _sha256(manifest_path) != identities["manifest"]:
        raise QualificationError("capture manifest changed during qualification")
    if _sha256(model_path) != identities["model"]:
        raise QualificationError("ONNX model changed during qualification")

    return {
        "format": "gufo.flashmap-npu-differential-result.v1",
        "qualified": True,
        "draftsEligibleForTargetVerification": True,
        "manifest": str(manifest_path.resolve()),
        "manifestSha256": identities["manifest"],
        "model": str(model_path.resolve()),
        "modelSha256": identities["model"],
        "targetArtifactSha256": identities["target"],
        "featureAbiSha256": identities["feature_abi"],
        "provider": EXPECTED_PROVIDER,
        "epLibrary": str(library),
        "ortVersion": ort.__version__,
        "cases": len(cases),
        "blockSize": block_size,
        "targetVocabularySize": target_vocabulary_size,
        "draftVocabularySize": (
            int(draft_to_target.size)
            if draft_to_target is not None
            else target_vocabulary_size
        ),
        "draftToTargetSha256": (
            identities["draft_to_target"]
            if draft_to_target is not None
            else None
        ),
        "provenance": provenance,
        "agreement": {
            "capturedReference": len(cases),
            "cpuReference": len(cpu_drafts),
            "npu": len(npu_drafts),
        },
        "profiledNodeEvents": node_events,
        "profiledProviders": providers,
        "coldSessionAndCompileMilliseconds": cold_ms,
        "cpuReferenceMedianMilliseconds": statistics.median(cpu_timings),
        "npuWarmMedianMilliseconds": statistics.median(timings),
        "npuWarmP95Milliseconds": percentile(timings, 0.95),
        "caseLatency": {case["id"]: case["latency"] for case in cases},
        "cpuFallback": {
            "ranAsReference": True,
            "draftsEligibleForTargetVerification": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--ep-library", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_differential(
            args.manifest, args.ep_library, args.warmup, args.iterations
        )
    except QualificationError as error:
        failure = {
            "format": "gufo.flashmap-npu-differential-result.v1",
            "qualified": False,
            "draftsEligibleForTargetVerification": False,
            "cpuFallback": {
                "draftsEligibleForTargetVerification": False,
                "reason": str(error),
            },
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

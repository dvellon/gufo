#!/usr/bin/env python3
"""Synthetic parser/manifest checks for npu_vitisai_lower.py."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile


TOOL = Path(__file__).with_name("npu_vitisai_lower.py")
SPEC = importlib.util.spec_from_file_location("npu_vitisai_lower", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def expect_failure(action, text: str) -> None:
    try:
        action()
    except MODULE.LoweringError as error:
        if text not in str(error):
            raise AssertionError(f"expected {text!r}, got {error!r}") from error
    else:
        raise AssertionError(f"expected failure containing {text!r}")


def report(all_nodes: int = 19, cpu: int = 0, npu: int = 19) -> bytes:
    return json.dumps(
        {
            "deviceStat": [
                {"name": "all", "nodeNum": all_nodes, "supportedOpType": ["::MatMul"]},
                {"name": "CPU", "nodeNum": cpu, "supportedOpType": []},
                {"name": "NPU", "nodeNum": npu, "supportedOpType": ["::MatMul"]},
            ],
            "nodeStat": [{"name": "bound-but-not-interpreted"}],
        },
        separators=(",", ":"),
    ).encode()


def main() -> int:
    assert MODULE.parse_assignment_report_payload(report()) == {
        "all": 19,
        "CPU": 0,
        "NPU": 19,
    }
    expect_failure(
        lambda: MODULE.parse_assignment_report_payload(report(cpu=1, npu=18)),
        "not placed wholly",
    )
    expect_failure(
        lambda: MODULE.parse_assignment_report_payload(
            json.dumps({"partitionSummary": {"CPU": 0, "NPU": 19}}).encode()
        ),
        "documented deviceStat",
    )
    expect_failure(
        lambda: MODULE.parse_assignment_report_payload(
            b'{"deviceStat":[],"deviceStat":[]}'
        ),
        "repeats key",
    )
    manifest = MODULE.make_lowering_manifest(
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
        "6" * 64,
        "7" * 64,
        "8" * 64,
        {
            "precision": "bf16",
            "compile_only": True,
            "inference_calls": 0,
            "ep_context_embed_mode": 1,
            "cpu_ep_fallback": False,
        },
    )
    assert set(manifest) == {
        "format",
        "source_export_manifest",
        "source_artifact_content_sha256",
        "draft_vocabulary_manifest_sha256",
        "candidate_model",
        "kind",
        "tool",
        "evidence",
        "options",
        "calibration",
        "draft_to_target",
    }
    assert manifest["format"] == MODULE.LOWERING_FORMAT
    assert manifest["kind"] == "bf16"
    assert manifest["calibration"] is None
    assert manifest["draft_vocabulary_manifest_sha256"] == "3" * 64
    assert manifest["tool"] == {
        "name": MODULE.TOOL_NAME,
        "version": MODULE.TOOL_VERSION,
        "executable": {
            "path": "npu_vitisai_lower.py",
            "sha256": "6" * 64,
        },
    }
    assert manifest["evidence"] == {
        "path": "lowering-evidence.json",
        "sha256": "7" * 64,
        "toolchain_sha256": "8" * 64,
    }
    assert manifest["draft_to_target"] == {
        "path": "d2t.npy",
        "sha256": "5" * 64,
    }
    assert MODULE.vaiml_config()["passes"][1]["vaiml_config"] == {
        "optimize_level": 3,
        "preferred_data_storage": "unvectorized",
    }
    with tempfile.TemporaryDirectory(dir=TOOL.parent) as temporary:
        root = Path(temporary)
        for name, payload in (
            ("d2t.npy", b"d2t"),
            (MODULE.DRAFT_VOCABULARY_MANIFEST, b"vocabulary"),
            ("model.onnx", b"model"),
        ):
            (root / name).write_bytes(payload)
        artifacts = [
            {
                "path": name,
                "bytes": (root / name).stat().st_size,
                "sha256": MODULE.sha256_file(root / name),
            }
            for name in ("d2t.npy", MODULE.DRAFT_VOCABULARY_MANIFEST, "model.onnx")
        ]
        checkpoint = [{"path": "config.json", "bytes": 1, "sha256": "a" * 64}]
        verifier = [{"path": "manifest.json", "bytes": 1, "sha256": "b" * 64}]
        source = {
            "format": MODULE.SOURCE_FORMAT,
            "geometry": MODULE._expected_geometry(),
            "identities": {
                "target_artifact_sha256": "c" * 64,
                "feature_abi_sha256": "d" * 64,
                "response_corpus_sha256": "e" * 64,
                "checkpoint_content_sha256": hashlib.sha256(
                    MODULE.canonical_json(checkpoint)
                ).hexdigest(),
                "verifier_bundle_content_sha256": hashlib.sha256(
                    MODULE.canonical_json(verifier)
                ).hexdigest(),
                "d2t_sha256": MODULE.sha256_file(root / "d2t.npy"),
                "draft_vocabulary_manifest_sha256": MODULE.sha256_file(
                    root / MODULE.DRAFT_VOCABULARY_MANIFEST
                ),
                "artifact_content_sha256": hashlib.sha256(
                    MODULE.canonical_json(artifacts)
                ).hexdigest(),
            },
            "checkpoint_files": checkpoint,
            "verifier_files": verifier,
            "artifact_files": artifacts,
            "input_names": MODULE.INPUT_NAMES,
            "inputs": {
                name: {
                    "shape": MODULE.INPUT_SHAPES[name],
                    "export_dtype": "float32",
                    "logical_dtype": (
                        "float32"
                        if name in {"rope_cos", "rope_sin", "ctx_mask"}
                        else "bfloat16"
                    ),
                }
                for name in MODULE.INPUT_NAMES
            },
            "output": {
                "name": "draft_logits",
                "shape": [1, MODULE.BLOCK_SIZE, MODULE.DRAFT_VOCAB],
                "dtype": "float32",
            },
            "external_data_bound": False,
        }
        source_path = root / "export-manifest.json"
        source_path.write_bytes(MODULE.canonical_json(source))
        loaded, source_digest = MODULE.load_source_export(source_path)
        assert loaded == source
        assert source_digest == MODULE.sha256_file(source_path)
    expect_failure(
        lambda: MODULE.make_lowering_manifest(
            "not-a-digest",
            "2" * 64,
            "3" * 64,
            "4" * 64,
            "5" * 64,
            "6" * 64,
            "7" * 64,
            "8" * 64,
            {},
        ),
        "source manifest",
    )
    print("NPU_VITISAI_LOWER_SELFTEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

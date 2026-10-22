#!/usr/bin/env python3
"""Focused repository-only checks for npu_flashmap_differential.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np


TOOL = Path(__file__).with_name("npu_flashmap_differential.py")
SPEC = importlib.util.spec_from_file_location("npu_flashmap_differential", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expect_failure(action, text: str) -> None:
    try:
        action()
    except MODULE.QualificationError as error:
        if text not in str(error):
            raise AssertionError(f"expected {text!r}, got {error!r}") from error
    else:
        raise AssertionError(f"expected failure containing {text!r}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="gufo-npu-diff-selftest-") as name:
        root = Path(name)
        model = root / "drafter.onnx"
        model.write_bytes(b"selftest-model")
        value = np.arange(8, dtype=np.float32).reshape(1, 8)
        array = root / "context.npy"
        np.save(array, value, allow_pickle=False)
        mapping = root / "draft_to_target.npy"
        draft_to_target = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int32)
        np.save(mapping, draft_to_target, allow_pickle=False)
        manifest = {
            "format": MODULE.FORMAT,
            "target_artifact_sha256": "1" * 64,
            "feature_abi_sha256": "2" * 64,
            "model": {"path": model.name, "sha256": digest(model)},
            "block_size": 7,
            "target_vocabulary_size": 16,
            "draft_to_target": {
                "path": mapping.name,
                "sha256": digest(mapping),
                "dtype": "int32",
                "shape": [8],
            },
            "provenance": {
                "source_export_manifest_sha256": "3" * 64,
                "source_artifact_content_sha256": "4" * 64,
                "lowering_manifest_sha256": "5" * 64,
                "independent_reference_manifest_sha256": "6" * 64,
                "candidate_model_sha256": digest(model),
            },
            "output": {"name": "draft_logits", "kind": "logits", "argmax_axis": -1},
            "cases": [
                {
                    "id": "real-capture-0001",
                    "inputs": {
                        "context": {
                            "path": array.name,
                            "sha256": digest(array),
                            "dtype": "float32",
                            "shape": [1, 8],
                        }
                    },
                    "reference_tokens": [7, 6, 5, 4, 3, 2, 1],
                }
            ],
        }
        manifest_path = root / "capture.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (
            loaded_model,
            block,
            vocabulary,
            output,
            loaded_mapping,
            provenance,
            cases,
            identities,
        ) = MODULE.load_capture(manifest_path)
        assert loaded_model == model and block == 7 and vocabulary == 16
        assert np.array_equal(loaded_mapping, draft_to_target)
        assert len(cases) == 1
        assert identities["manifest"] == digest(manifest_path)
        assert identities["model"] == digest(model)
        assert identities["target"] == "1" * 64
        assert identities["feature_abi"] == "2" * 64
        assert identities["draft_to_target"] == digest(mapping)
        assert provenance == manifest["provenance"]
        logits = np.eye(8, dtype=np.float32)[[0, 1, 2, 3, 4, 5, 6]][None, ...]
        assert MODULE.greedy_tokens(
            logits, output, block, vocabulary, loaded_mapping
        ) == [
            7,
            6,
            5,
            4,
            3,
            2,
            1,
        ]
        expect_failure(
            lambda: MODULE.greedy_tokens(logits[..., :7], output, block, 16, loaded_mapping),
            "length differs from logit width",
        )
        expect_failure(
            lambda: MODULE.greedy_tokens(logits, output, block, 16, None),
            "restricted logit output has no draft-to-target mapping",
        )

        profile = root / "profile.json"
        profile.write_text(
            json.dumps(
                [
                    {"args": {"provider": MODULE.EXPECTED_PROVIDER}},
                    {"args": {"provider": MODULE.EXPECTED_PROVIDER}},
                    {"args": {}},
                ]
            ),
            encoding="utf-8",
        )
        assert MODULE.profile_providers(profile) == ([MODULE.EXPECTED_PROVIDER], 2)
        profile.write_text(
            json.dumps(
                [
                    {"args": {"provider": MODULE.EXPECTED_PROVIDER}},
                    {"args": {"provider": MODULE.CPU_PROVIDER}},
                ]
            ),
            encoding="utf-8",
        )
        assert MODULE.profile_providers(profile) == (
            [MODULE.CPU_PROVIDER, MODULE.EXPECTED_PROVIDER],
            2,
        )

        damaged = json.loads(manifest_path.read_text(encoding="utf-8"))
        damaged["cases"][0]["inputs"]["context"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(damaged), encoding="utf-8")
        expect_failure(lambda: MODULE.load_capture(manifest_path), "content digest")

        duplicate = manifest_path.read_text(encoding="utf-8").replace(
            '"format":', '"format": "duplicate", "format":', 1
        )
        manifest_path.write_text(duplicate, encoding="utf-8")
        expect_failure(lambda: MODULE.load_capture(manifest_path), "repeats object key")

        wrong_block = dict(manifest)
        wrong_block["block_size"] = 6
        manifest_path.write_text(json.dumps(wrong_block), encoding="utf-8")
        expect_failure(lambda: MODULE.load_capture(manifest_path), "seven-proposal ABI")

        wrong_vocabulary = json.loads(json.dumps(manifest))
        wrong_vocabulary["cases"][0]["reference_tokens"][0] = 16
        manifest_path.write_text(json.dumps(wrong_vocabulary), encoding="utf-8")
        expect_failure(lambda: MODULE.load_capture(manifest_path), "reference tokens")

        duplicate_mapping = np.array([7, 7, 5, 4, 3, 2, 1, 0], dtype=np.int32)
        np.save(mapping, duplicate_mapping, allow_pickle=False)
        duplicate_map_manifest = json.loads(json.dumps(manifest))
        duplicate_map_manifest["draft_to_target"]["sha256"] = digest(mapping)
        manifest_path.write_text(
            json.dumps(duplicate_map_manifest), encoding="utf-8"
        )
        expect_failure(lambda: MODULE.load_capture(manifest_path), "repeats a target token")

        out_of_range_mapping = np.array(
            [7, 6, 5, 4, 3, 2, 1, 16], dtype=np.int32
        )
        np.save(mapping, out_of_range_mapping, allow_pickle=False)
        out_of_range_manifest = json.loads(json.dumps(manifest))
        out_of_range_manifest["draft_to_target"]["sha256"] = digest(mapping)
        manifest_path.write_text(json.dumps(out_of_range_manifest), encoding="utf-8")
        expect_failure(lambda: MODULE.load_capture(manifest_path), "out-of-range")

        bad_provenance = json.loads(json.dumps(manifest))
        bad_provenance["provenance"]["candidate_model_sha256"] = "9" * 64
        manifest_path.write_text(json.dumps(bad_provenance), encoding="utf-8")
        expect_failure(
            lambda: MODULE.load_capture(manifest_path),
            "candidate model identity differs",
        )

        missing_provenance = json.loads(json.dumps(manifest))
        del missing_provenance["provenance"]["lowering_manifest_sha256"]
        manifest_path.write_text(json.dumps(missing_provenance), encoding="utf-8")
        expect_failure(lambda: MODULE.load_capture(manifest_path), "fields differ")

    print("NPU_FLASHMAP_DIFFERENTIAL_SELFTEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

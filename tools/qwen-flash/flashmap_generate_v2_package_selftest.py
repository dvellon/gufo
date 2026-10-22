#!/usr/bin/env python3
"""Focused CPU-only tests for the FlashMap v2 package contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import flashmap_generate_v2_package as package


SHA_SOURCE = "1" * 64
SHA_TARGET = "2" * 64
SHA_GENERATOR = "3" * 64
SHA_CONFIG = "4" * 64
SHA_RUNTIME = "5" * 64
GIT = "6" * 40


def expect_failure(action, contains: str) -> None:
    try:
        action()
    except package.PackageError as error:
        if contains not in str(error):
            raise AssertionError(f"wrong failure: {error}") from error
    else:
        raise AssertionError(f"expected failure containing {contains!r}")


def prompt(ordinal: int) -> dict:
    return {
        "id": f"sample-{ordinal:03d}",
        "messages": [{"content": f"Return {ordinal}.", "role": "user"}],
        "metadata": {"ordinal": str(ordinal)},
        "source_id": "synthetic",
    }


def response(source: dict, ordinal: int, max_tokens: int = 3) -> dict:
    record_sha = package.digest_bytes(package.canonical(source).rstrip(b"\n"))
    tokens = [100 + ordinal, 200 + ordinal]
    return {
        "generation_config_sha256": SHA_CONFIG,
        "generator_git_commit": GIT,
        "generator_sha256": SHA_GENERATOR,
        "id": source["id"],
        "metadata": {
            "prompt_source_file_sha256": SHA_SOURCE,
            "prompt_source_record_sha256": record_sha,
            "termination": "stop" if len(tokens) < max_tokens else "length",
            "user_metadata": source["metadata"],
        },
        "prompt_tokens": [ordinal + 1],
        "response_tokens": tokens,
        "source_id": source["source_id"],
        "target_sha256": SHA_TARGET,
    }


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="flashmap-v2-package-", dir=Path.cwd()))
    try:
        prompts = [prompt(index) for index in range(5)]
        source = root / "source.jsonl"
        source.write_bytes(b"".join(package.canonical(row) for row in prompts))
        global SHA_SOURCE
        SHA_SOURCE = package.digest_file(source)
        config = root / "config.json"
        config.write_bytes(package.canonical({"max_response_tokens": 3}))
        global SHA_CONFIG
        SHA_CONFIG = package.digest_file(config)
        runtime = root / "runtime.json"
        runtime.write_bytes(package.canonical({"runtime": "synthetic"}))
        global SHA_RUNTIME
        SHA_RUNTIME = package.digest_file(runtime)
        destination = root / "package"
        package.prepare(argparse.Namespace(
            source=source,
            generation_config=config,
            runtime_manifest=runtime,
            package=destination,
            source_sha256=SHA_SOURCE,
            target_sha256=SHA_TARGET,
            generator_sha256=SHA_GENERATOR,
            producer_git=GIT,
            generation_config_sha256=SHA_CONFIG,
            runtime_identity=SHA_RUNTIME,
            expected_records=5,
            decode_width=2,
            shard_records=4,
        ))
        manifest_path = destination / "manifest.json"
        manifest_sha = package.digest_file(manifest_path)
        manifest = package.validate_package(manifest_path, manifest_sha)
        assert manifest["decode_width"] == 2
        assert [(row["start"], row["end"]) for row in manifest["shards"]] == [(0, 4), (4, 5)]

        responses = root / "responses"
        responses.mkdir()
        for shard in manifest["shards"]:
            rows = [package.canonical(response(prompts[index], index))
                    for index in range(shard["start"], shard["end"])]
            response_path = responses / shard["response_file"]
            response_path.write_bytes(b"".join(rows))
            response_path.chmod(0o444)
            package.write_exclusive(
                responses / (shard["response_file"] + ".complete.json"),
                package.completion_payload(
                    manifest_sha,
                    manifest,
                    shard,
                    package.digest_file(response_path),
                ),
            )
            assert len(package.response_rows(manifest, manifest_path, responses,
                                             shard, partial=False)) == len(rows)

        merged = root / "merged.jsonl"
        package.merge(argparse.Namespace(
            manifest=manifest_path,
            manifest_sha256=manifest_sha,
            responses=responses,
            output=merged,
        ))
        assert merged.read_bytes() == b"".join(
            package.canonical(response(row, index))
            for index, row in enumerate(prompts))

        first = manifest["shards"][0]
        complete = responses / first["response_file"]
        partial = Path(str(complete) + ".partial")
        partial.write_bytes(b"")
        assert package.response_rows(
            manifest, manifest_path, responses, first, partial=True
        ) == []
        partial.write_bytes(complete.read_bytes().splitlines(keepends=True)[0])
        assert len(package.response_rows(manifest, manifest_path, responses,
                                         first, partial=True)) == 1
        partial.write_bytes(b"".join(reversed(
            complete.read_bytes().splitlines(keepends=True)[:2])))
        expect_failure(lambda: package.response_rows(
            manifest, manifest_path, responses, first, partial=True),
            "identity differs")
        partial.unlink()

        receipt = responses / (first["response_file"] + ".complete.json")
        receipt_partial = Path(str(receipt) + ".partial")
        receipt.chmod(0o600)
        receipt.rename(receipt_partial)
        assert len(package.response_rows(
            manifest,
            manifest_path,
            responses,
            first,
            partial=False,
            completion_partial=True,
        )) == 4
        receipt_partial.rename(receipt)
        receipt.chmod(0o444)

        complete.chmod(0o600)
        expect_failure(lambda: package.response_rows(
            manifest, manifest_path, responses, first, partial=False),
            "read-only")
        complete.chmod(0o444)
        receipt.chmod(0o600)
        receipt.write_bytes(receipt.read_bytes().replace(
            package.digest_file(complete).encode("ascii"), b"f" * 64
        ))
        receipt.chmod(0o444)
        expect_failure(lambda: package.response_rows(
            manifest, manifest_path, responses, first, partial=False),
            "binding differs")

        unexpected = responses / "not-in-manifest.txt"
        unexpected.write_text("unexpected\n", encoding="utf-8")
        expect_failure(lambda: package.validate_response_inventory(
            manifest, responses, allow_partials=True),
            "unexpected response inventory entry")
        unexpected.unlink()

        bad_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bad_manifest["decode_width"] = 3
        bad = root / "bad-manifest.json"
        bad.write_bytes(package.canonical(bad_manifest))
        expect_failure(lambda: package.load_manifest(bad),
                       "aligned to decode_width")
        expect_failure(lambda: package.merge(argparse.Namespace(
            manifest=manifest_path,
            manifest_sha256=manifest_sha,
            responses=responses,
            output=merged,
        )), "overwrite merged output")
        print("FLASHMAP_V2_PACKAGE_SELFTEST_OK records=5 shards=2 width=2")
        return 0
    finally:
        for path in root.rglob("*"):
            try:
                path.chmod(0o700 if path.is_dir() else 0o600)
            except OSError:
                pass
        shutil.rmtree(root, ignore_errors=False)


if __name__ == "__main__":
    raise SystemExit(main())

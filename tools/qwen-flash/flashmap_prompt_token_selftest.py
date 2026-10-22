#!/usr/bin/env python3
"""Focused contract/parity check for Gufo's prompt-token derivation mode.

This script never tokenizes in Python. It invokes the exact Gufo generator's
CPU-only ``--derive-prompt-tokens`` mode, compares that result through the
generator's native parity mode, and proves no-replace publication on rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path


TARGET_ARTIFACT_SHA256 = (
    "bdf8425b3453aacfa4f247e94c13abeb98c4356120c39da2fa60b90b5a7116a9"
)
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class SelfTestError(RuntimeError):
    """Raised when the focused derivation contract differs."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if expect_success != (completed.returncode == 0):
        raise SelfTestError(
            f"unexpected command status {completed.returncode}: "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--response-output", type=Path, required=True)
    parser.add_argument("--response-output-sha256", required=True)
    parser.add_argument("--producer-git", required=True)
    parser.add_argument("--producer-tree", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    if not HEX64.fullmatch(args.response_output_sha256):
        raise SelfTestError("response output SHA-256 is malformed")
    if not HEX40.fullmatch(args.producer_git) or not HEX40.fullmatch(
        args.producer_tree
    ):
        raise SelfTestError("producer commit or tree is malformed")
    for path in (args.generator, args.model, args.response_output):
        if not path.is_file():
            raise SelfTestError(f"required file is absent: {path}")
    if sha256(args.response_output) != args.response_output_sha256:
        raise SelfTestError("reviewed response output identity differs")
    if args.output_directory.exists() or args.output_directory.is_symlink():
        raise SelfTestError("refusing to replace focused self-test output")
    args.output_directory.mkdir(parents=True)

    root = Path(__file__).resolve().parents[2]
    fixture = root / "tools/qwen-flash/flashmap_generation_fixture.jsonl"
    config = root / "tools/qwen-flash/flashmap_generation_config_128.json"
    token_output = args.output_directory / "fixture.tokens.jsonl"
    derive = [
        str(args.generator.resolve()),
        "--derive-prompt-tokens",
        "--model",
        str(args.model.resolve()),
        "--input",
        str(fixture),
        "--output",
        str(token_output),
        "--generation-config",
        str(config),
        "--generation-config-sha256",
        sha256(config),
        "--input-sha256",
        sha256(fixture),
        "--executable-sha256",
        sha256(args.generator),
        "--producer-git",
        args.producer_git,
        "--producer-tree",
        args.producer_tree,
        "--tokenizer-model-file-sha256",
        sha256(args.model),
        "--tokenizer-artifact-sha256",
        TARGET_ARTIFACT_SHA256,
        "--partition",
        "fixture",
        "--expected-records",
        "1",
    ]
    derived = run(derive)
    if "FLASHMAP_PROMPT_TOKENS_OK records=1 tokens=58" not in derived.stdout:
        raise SelfTestError("focused fixture token count differs")

    token_sha256 = sha256(token_output)
    parity = run(
        [
            str(args.generator.resolve()),
            "--verify-prompt-token-parity",
            "--token-input",
            str(token_output),
            "--token-input-sha256",
            token_sha256,
            "--response-output",
            str(args.response_output.resolve()),
            "--response-output-sha256",
            args.response_output_sha256,
            "--expected-records",
            "1",
        ]
    )
    if "FLASHMAP_PROMPT_TOKEN_PARITY_OK records=1" not in parity.stdout:
        raise SelfTestError("native prompt-token parity receipt is absent")

    refused = run(derive, expect_success=False)
    if "refusing to overwrite prompt-token export" not in refused.stderr:
        raise SelfTestError("repeat derivation did not fail for no-replace")
    if sha256(token_output) != token_sha256:
        raise SelfTestError("published token fixture changed after refusal")

    print(
        "FLASHMAP_PROMPT_TOKEN_SELFTEST_OK "
        f"tokens_sha256={token_sha256} response_sha256={args.response_output_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

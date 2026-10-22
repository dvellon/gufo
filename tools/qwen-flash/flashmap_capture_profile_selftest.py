#!/usr/bin/env python3
"""Focused self-test for flashmap_capture_profile.py."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("flashmap_capture_profile.py")
SPEC = importlib.util.spec_from_file_location("flashmap_capture_profile", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEMP_ROOT = REPOSITORY_ROOT / "build" / "gufo-runtime-temp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
    root = Path(temporary)
    # Completion interval = 5 fixed seconds + 2 seconds per decimal MB.
    completed_ns = 1_000_000_000
    sizes = [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000]
    for ordinal, size in enumerate(sizes):
        directory = root / f"shard-{ordinal:05d}"
        directory.mkdir()
        artifact = directory / f"sha256-{'a' * 64}.gfm"
        artifact.write_bytes(b"x")
        os.truncate(artifact, size)
        if ordinal:
            completed_ns += int((5.0 + 2.0 * (size / 1_000_000.0)) * 1e9)
        os.utime(artifact, ns=(completed_ns, completed_ns))
    preserved = root / "shard-00003" / "failed.gfm.partial"
    preserved.write_bytes(b"do-not-read")
    shards, partials = MODULE.scan(root)
    report = MODULE.estimate(shards, 30.0)
    assert len(shards) == 5
    assert partials == [str(preserved)]
    assert abs(report["fixed_seconds_per_process"] - 5.0) < 1e-9
    assert abs(report["seconds_per_megabyte"] - 2.0) < 1e-9
    assert abs(report["correlation"] - 1.0) < 1e-9
    assert report["estimated_resident_speedup"] > 1.0

print("FLASHMAP_CAPTURE_PROFILE_SELFTEST_OK")

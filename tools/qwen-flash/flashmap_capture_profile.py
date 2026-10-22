#!/usr/bin/env python3
"""Estimate fixed and byte-proportional FlashMap shard capture costs.

The probe deliberately reads directory metadata only.  It never opens a GFM
payload or a live ``.partial`` and is therefore safe to use while reviewing a
completed capture campaign.  Adjacent completion times are regressed against
the newly published shard size; long recovery gaps are excluded explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Shard:
    ordinal: int
    bytes: int
    completed_ns: int


def _ordinal(directory: Path) -> int:
    prefix = "shard-"
    if not directory.name.startswith(prefix):
        raise ValueError(f"unexpected shard directory: {directory.name}")
    text = directory.name[len(prefix) :]
    if not text.isdigit():
        raise ValueError(f"invalid shard ordinal: {directory.name}")
    return int(text)


def scan(root: Path) -> tuple[list[Shard], list[str]]:
    if not root.is_dir():
        raise ValueError(f"capture root is not a directory: {root}")
    shards: list[Shard] = []
    partials: list[str] = []
    for directory in sorted(root.glob("shard-*"), key=_ordinal):
        if not directory.is_dir():
            raise ValueError(f"shard path is not a directory: {directory}")
        published = list(directory.glob("sha256-*.gfm"))
        live = list(directory.glob("*.gfm.partial"))
        if len(published) > 1:
            raise ValueError(f"multiple published GFM files: {directory}")
        partials.extend(str(path) for path in live)
        if not published:
            continue
        artifact = published[0]
        stat = artifact.stat()
        shards.append(
            Shard(
                ordinal=_ordinal(directory),
                bytes=stat.st_size,
                completed_ns=stat.st_mtime_ns,
            )
        )
    if not shards:
        raise ValueError("capture root has no published GFM shards")
    ordinals = [item.ordinal for item in shards]
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("duplicate shard ordinal")
    return shards, partials


def estimate(shards: list[Shard], max_gap_seconds: float) -> dict[str, object]:
    if max_gap_seconds <= 0 or not math.isfinite(max_gap_seconds):
        raise ValueError("max gap must be finite and positive")
    rows: list[tuple[float, float]] = []
    for previous, current in zip(shards, shards[1:]):
        seconds = (current.completed_ns - previous.completed_ns) / 1e9
        if (
            current.ordinal == previous.ordinal + 1
            and 0 < seconds <= max_gap_seconds
        ):
            rows.append((current.bytes / 1_000_000.0, seconds))
    if len(rows) < 3:
        raise ValueError("need at least three adjacent healthy shard intervals")
    x_mean = statistics.fmean(row[0] for row in rows)
    y_mean = statistics.fmean(row[1] for row in rows)
    x_variance = sum((row[0] - x_mean) ** 2 for row in rows)
    if x_variance == 0:
        raise ValueError("published shard sizes have no variation")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in rows) / x_variance
    intercept = y_mean - slope * x_mean
    y_variance = sum((row[1] - y_mean) ** 2 for row in rows)
    correlation = (
        sum((x - x_mean) * (y - y_mean) for x, y in rows)
        / math.sqrt(x_variance * y_variance)
        if y_variance
        else 0.0
    )
    total_megabytes = sum(item.bytes for item in shards) / 1_000_000.0
    serial_estimate = len(shards) * intercept + slope * total_megabytes
    resident_estimate = max(intercept, 0.0) + slope * total_megabytes
    return {
        "schema": "flashmap-capture-process-profile-v1",
        "published_shards": len(shards),
        "healthy_adjacent_intervals": len(rows),
        "median_interval_seconds": statistics.median(row[1] for row in rows),
        "median_shard_megabytes": statistics.median(row[0] for row in rows),
        "fixed_seconds_per_process": intercept,
        "seconds_per_megabyte": slope,
        "correlation": correlation,
        "published_megabytes": total_megabytes,
        "estimated_serial_seconds": serial_estimate,
        "estimated_resident_seconds": resident_estimate,
        "estimated_resident_speedup": (
            serial_estimate / resident_estimate
            if resident_estimate > 0
            else None
        ),
        "estimated_fixed_fraction": (
            len(shards) * max(intercept, 0.0) / serial_estimate
            if serial_estimate > 0
            else None
        ),
        "max_gap_seconds": max_gap_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--max-gap-seconds", type=float, default=300.0)
    args = parser.parse_args()
    shards, partials = scan(args.capture_root)
    if args.expected_shards is not None and len(shards) != args.expected_shards:
        raise SystemExit(
            f"published shard count {len(shards)} != {args.expected_shards}"
        )
    report = estimate(shards, args.max_gap_seconds)
    report["preserved_partial_count"] = len(partials)
    report["preserved_partials"] = partials
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

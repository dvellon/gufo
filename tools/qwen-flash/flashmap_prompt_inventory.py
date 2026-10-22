#!/usr/bin/env python3
"""Materialize the pinned 50K FlashMap prompt inventory and fixed evaluation sets.

The network-facing ``normalize`` phase reuses the already-reviewed source
adapters and immutable raw-object inventory.  It only raises their bounded
candidate limits.  The offline ``build`` phase publishes one 52K parent corpus,
proves the reviewed original 5K is contained in it, excludes those exact
identities, and deterministically publishes nine new 5K chunks plus disjoint
1K holdout and 1K final-test corpora through ``flashmap_prompt_corpus``.

Nothing here generates answers.  Every output is no-replace and binds the
source revisions, object hashes, licenses, original corpus, tool identities,
canonical messages and category allocation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence

import flashmap_chunk_campaign as campaign
import flashmap_prompt_corpus as corpus


FORMAT = "gufo.flashmap.prompt-inventory.v1"
SOURCE_FORMAT = "gufo.flashmap.campaign-source-inventory.v1"
PLAN_FORMAT = "gufo.flashmap.prompt-inventory-plan.v1"
PLAN_FILE = Path(__file__).with_name("flashmap_prompt_inventory_plan.v1.json")
PROFILE = "dose-50k-fixed-eval-v1"
PARTITION_ALGORITHM = "sha256-rank-per-category-after-exact-5k-exclusion-v1"
PARTITION_SEED = "b7ff1cc03e4a74a266e02264cbe86aef799922b227379501997ba91e0cbd3d77"
SOURCE_SELECTION = "sha256-ranked-parquet-row-groups-then-records-v1"
SOURCE_RECIPE = "flashmap_prompt_corpus_recipe.json"
MAX_MANIFEST_BYTES = 8 << 20

# Quotas include the already-reviewed 5K corpus.  Removing it leaves exactly
# 45K new training records and two disjoint 1K evaluation partitions.
EXPANDED_QUOTAS = {
    "general-instruction": 18_200,
    "constraint-following": 5_200,
    "editing": 2_600,
    "summarization": 2_600,
    "code": 13_000,
    "reasoning-math": 10_400,
}
CANDIDATE_RECORDS = {
    "general-instruction": 20_020,
    "constraint-following": 5_720,
    "editing": 2_860,
    "summarization": 2_860,
    "code": 14_300,
    "reasoning-math": 11_440,
}
CHUNK_QUOTAS = {
    "general-instruction": 1_750,
    "constraint-following": 500,
    "editing": 250,
    "summarization": 250,
    "code": 1_250,
    "reasoning-math": 1_000,
}
EVAL_QUOTAS = {
    "general-instruction": 350,
    "constraint-following": 100,
    "editing": 50,
    "summarization": 50,
    "code": 250,
    "reasoning-math": 200,
}
CHUNKS = 9
EXPECTED_SOURCE_PLAN = [
    {
        "category": category,
        "dataset_id": dataset_id,
        "dataset_revision": revision,
        "license_spdx": license_spdx,
        "license_url": license_url,
        "subsets": [subset],
    }
    for category, dataset_id, revision, license_spdx, license_url, subset in (
        (
            "general-instruction",
            "HuggingFaceTB/smoltalk",
            "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa",
            "Apache-2.0",
            "https://www.apache.org/licenses/LICENSE-2.0",
            "smol-magpie-ultra",
        ),
        (
            "constraint-following",
            "HuggingFaceTB/smoltalk",
            "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa",
            "Apache-2.0",
            "https://www.apache.org/licenses/LICENSE-2.0",
            "smol-constraints",
        ),
        (
            "editing",
            "HuggingFaceTB/smoltalk",
            "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa",
            "Apache-2.0",
            "https://www.apache.org/licenses/LICENSE-2.0",
            "smol-rewrite",
        ),
        (
            "summarization",
            "HuggingFaceTB/smoltalk",
            "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa",
            "Apache-2.0",
            "https://www.apache.org/licenses/LICENSE-2.0",
            "smol-summarize",
        ),
        (
            "code",
            "bigcode/self-oss-instruct-sc2-exec-filter-50k",
            "356bb069eee815daa6e23e9a282eeefe1490ad44",
            "ODC-By-1.0",
            "https://opendatacommons.org/licenses/by/1-0/",
            "default",
        ),
        (
            "reasoning-math",
            "open-r1/OpenR1-Math-220k",
            "e4e141ec9dea9f8326f4d347be56105859b2bd68",
            "Apache-2.0",
            "https://www.apache.org/licenses/LICENSE-2.0",
            "default",
        ),
    )
]


class InventoryError(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_digest(value: dict[str, object]) -> str:
    unsigned = dict(value)
    unsigned.pop("manifest_content_sha256", None)
    return _sha256_bytes(_canonical(unsigned))


def _hex(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise InventoryError(f"{field} must be 64 lowercase hexadecimal digits")
    return value


def _strict_object(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise InventoryError(f"manifest is not a regular file: {path}")
    payload = path.read_bytes()
    if not payload or len(payload) > MAX_MANIFEST_BYTES or not payload.endswith(b"\n"):
        raise InventoryError(f"manifest is not bounded newline-terminated JSON: {path}")
    try:
        value = json.loads(payload, object_pairs_hook=corpus._unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, corpus.CorpusError) as error:
        raise InventoryError(f"manifest is not strict JSON: {path}") from error
    if not isinstance(value, dict) or _canonical(value) != payload:
        raise InventoryError(f"manifest is not canonical JSON: {path}")
    return value


def _plan() -> tuple[dict[str, object], str]:
    plan = _strict_object(PLAN_FILE)
    expected = {
        "candidate_records": CANDIDATE_RECORDS,
        "chunk_quotas": CHUNK_QUOTAS,
        "chunks": CHUNKS,
        "corpus_tool_sha256": _sha256_file(
            Path(__file__).with_name("flashmap_prompt_corpus.py")
        ),
        "eval_quotas": EVAL_QUOTAS,
        "exclusion": {
            "manifest_content_sha256": (
                "72040f19e4fd2bc8ab8eae68ae6beb244da0d61291b57c30655f710a6999af32"
            ),
            "prompt_sha256": (
                "81c962011ca55e1214495c41e23c0c69f1ca6819157264ce732e4fef3509a5f5"
            ),
            "records": 5_000,
        },
        "expanded_quotas": EXPANDED_QUOTAS,
        "format": PLAN_FORMAT,
        "partitions": {
            "holdout_records": 1_000,
            "new_training_records": 45_000,
            "test_records": 1_000,
            "total_training_with_exclusion": 50_000,
        },
        "profile": PROFILE,
        "source_tool_sha256": _sha256_file(
            Path(__file__).with_name("flashmap_prompt_sources.py")
        ),
        "sources": EXPECTED_SOURCE_PLAN,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise InventoryError(f"reviewed inventory plan differs: {key}")
    return plan, _sha256_file(PLAN_FILE)


def _expanded_specs() -> tuple[Any, ...]:
    # Keep pyarrow/requests optional for offline validation and self-test.
    import flashmap_prompt_sources as sources

    return tuple(
        replace(
            spec,
            quota=EXPANDED_QUOTAS[spec.category],
            candidate_records=CANDIDATE_RECORDS[spec.category],
        )
        for spec in sources.SOURCES
    )


def normalize(output: Path) -> Path:
    import flashmap_prompt_sources as sources

    _, plan_sha256 = _plan()
    specs = _expanded_specs()
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise InventoryError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        notes: list[dict[str, object]] = []
        for spec in specs:
            payload, note = sources._normalize_source(spec)
            (temporary / spec.output).write_bytes(payload)
            notes.append(note)
            print(
                "FLASHMAP_CAMPAIGN_SOURCE_NORMALIZED "
                f"source={spec.key} records={note['candidate_records']} "
                f"sha256={note['normalized_sha256']}"
            )
        by_category = {str(note["category"]): note for note in notes}
        recipe = {
            "expected_records": sum(EXPANDED_QUOTAS.values()),
            "format": corpus.RECIPE_FORMAT,
            "seed": sources.SEED,
            "sources": [
                {
                    "category": spec.category,
                    "dataset_id": spec.dataset_id,
                    "dataset_revision": spec.revision,
                    "dataset_url": (
                        f"https://huggingface.co/datasets/{spec.dataset_id}/tree/"
                        f"{spec.revision}"
                    ),
                    "input": spec.output,
                    "input_sha256": by_category[spec.category]["normalized_sha256"],
                    "license_spdx": spec.license_spdx,
                    "license_url": spec.license_url,
                    "max_prompt_chars": spec.max_prompt_chars,
                    "min_prompt_chars": spec.min_prompt_chars,
                    "quota": spec.quota,
                    "split": spec.split,
                    "subset": spec.subset,
                }
                for spec in specs
            ],
        }
        recipe_payload = _canonical(recipe)
        (temporary / SOURCE_RECIPE).write_bytes(recipe_payload)
        manifest: dict[str, object] = {
            "base_source_tool_sha256": _sha256_file(Path(sources.__file__).resolve()),
            "format": SOURCE_FORMAT,
            "manifest_content_sha256": "",
            "plan_sha256": plan_sha256,
            "profile": PROFILE,
            "recipe_file": SOURCE_RECIPE,
            "recipe_sha256": _sha256_bytes(recipe_payload),
            "selection": SOURCE_SELECTION,
            "sources": notes,
            "tool_sha256": _sha256_file(Path(__file__).resolve()),
        }
        manifest["manifest_content_sha256"] = _content_digest(manifest)
        (temporary / "source-manifest.json").write_bytes(_canonical(manifest))
        campaign._publish_no_replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validate_source(output)
    print(
        "FLASHMAP_CAMPAIGN_SOURCES_OK "
        f"records={sum(EXPANDED_QUOTAS.values())} "
        f"sha256={manifest['manifest_content_sha256']} output={output}"
    )
    return output


def validate_source(root: Path) -> dict[str, object]:
    _, plan_sha256 = _plan()
    root = root.resolve()
    manifest = _strict_object(root / "source-manifest.json")
    if (
        manifest.get("format") != SOURCE_FORMAT
        or manifest.get("profile") != PROFILE
        or manifest.get("plan_sha256") != plan_sha256
        or manifest.get("selection") != SOURCE_SELECTION
        or manifest.get("manifest_content_sha256") != _content_digest(manifest)
    ):
        raise InventoryError("expanded source manifest identity differs")
    if manifest.get("tool_sha256") != _sha256_file(Path(__file__).resolve()):
        raise InventoryError("expanded source inventory tool identity differs")
    if manifest.get("base_source_tool_sha256") != _sha256_file(
        Path(__file__).with_name("flashmap_prompt_sources.py")
    ):
        raise InventoryError("reviewed source adapter tool identity differs")
    recipe = _strict_object(root / SOURCE_RECIPE)
    if (
        _sha256_file(root / SOURCE_RECIPE) != manifest.get("recipe_sha256")
        or recipe.get("format") != corpus.RECIPE_FORMAT
        or recipe.get("expected_records") != sum(EXPANDED_QUOTAS.values())
    ):
        raise InventoryError("expanded source recipe differs")
    raw_sources = recipe.get("sources")
    notes = manifest.get("sources")
    if not isinstance(raw_sources, list) or not isinstance(notes, list):
        raise InventoryError("expanded source inventory is absent")
    if len(raw_sources) != len(EXPANDED_QUOTAS) or len(notes) != len(raw_sources):
        raise InventoryError("expanded source inventory length differs")
    expected_files = {"source-manifest.json", SOURCE_RECIPE}
    for recipe_source, note in zip(raw_sources, notes, strict=True):
        if not isinstance(recipe_source, dict) or not isinstance(note, dict):
            raise InventoryError("expanded source entry is invalid")
        category = recipe_source.get("category")
        if category not in EXPANDED_QUOTAS:
            raise InventoryError("expanded source category differs")
        name = recipe_source.get("input")
        if not isinstance(name, str) or Path(name).name != name:
            raise InventoryError("expanded source filename differs")
        expected_files.add(name)
        path = root / name
        if (
            recipe_source.get("quota") != EXPANDED_QUOTAS[category]
            or note.get("candidate_records") != CANDIDATE_RECORDS[category]
            or recipe_source.get("input_sha256") != _sha256_file(path)
            or note.get("normalized_sha256") != recipe_source.get("input_sha256")
            or path.stat().st_size != note.get("normalized_bytes")
        ):
            raise InventoryError(f"expanded source binding differs for {category}")
        for field in (
            "category",
            "dataset_id",
            "dataset_revision",
            "license_spdx",
            "license_url",
            "max_prompt_chars",
            "min_prompt_chars",
            "split",
            "subset",
        ):
            if recipe_source.get(field) != note.get(field):
                raise InventoryError(
                    f"expanded source attribution differs for {category}: {field}"
                )
    if {path.name for path in root.iterdir()} != expected_files:
        raise InventoryError("expanded source directory inventory differs")
    return manifest


def _rows(root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = corpus.validate(root)
    result: list[dict[str, object]] = []
    with (root / "prompts.jsonl").open("rb") as stream:
        for raw in stream:
            value = json.loads(raw, object_pairs_hook=corpus._unique_object)
            if not isinstance(value, dict) or corpus._canonical(value) != raw:
                raise InventoryError("prompt row is not canonical")
            result.append(value)
    if len(result) != manifest["records"]:
        raise InventoryError("prompt row count differs")
    return manifest, result


def _message_digest(row: dict[str, object]) -> str:
    _, payload = corpus._messages(row.get("messages"), context=str(row.get("id")))
    return _sha256_bytes(payload)


def _rank(row: dict[str, object], category: str) -> str:
    return _sha256_bytes(
        b"\x00".join(
            (
                PARTITION_SEED.encode("ascii"),
                category.encode("utf-8"),
                str(row["source_id"]).encode("utf-8"),
                _message_digest(row).encode("ascii"),
            )
        )
    )


def _partition_rows(
    expanded: list[dict[str, object]], excluded: list[dict[str, object]]
) -> dict[str, list[dict[str, object]]]:
    expanded_by_identity = {
        (str(row["source_id"]), _message_digest(row)): row for row in expanded
    }
    if len(expanded_by_identity) != len(expanded):
        raise InventoryError("expanded corpus repeats an identity or exact messages")
    excluded_keys = {
        (str(row["source_id"]), _message_digest(row)) for row in excluded
    }
    if len(excluded_keys) != len(excluded) or not excluded_keys <= set(expanded_by_identity):
        raise InventoryError("reviewed 5K is not an exact subset of expanded corpus")
    remaining = [row for key, row in expanded_by_identity.items() if key not in excluded_keys]
    by_category: dict[str, list[dict[str, object]]] = {
        category: [] for category in CHUNK_QUOTAS
    }
    for row in remaining:
        metadata = row.get("metadata")
        category = metadata.get("category") if isinstance(metadata, dict) else None
        if category not in by_category:
            raise InventoryError("expanded corpus category differs")
        by_category[str(category)].append(row)
    partitions = {"holdout": [], "test": []}
    partitions.update({f"chunk-{ordinal:05d}": [] for ordinal in range(1, CHUNKS + 1)})
    for category, rows in by_category.items():
        ordered = sorted(rows, key=lambda row: (_rank(row, category), str(row["source_id"])))
        needed = EVAL_QUOTAS[category] * 2 + CHUNK_QUOTAS[category] * CHUNKS
        if len(ordered) != needed:
            raise InventoryError(
                f"{category} has {len(ordered)} post-exclusion prompts; required {needed}"
            )
        cursor = 0
        for name, quota in (
            ("holdout", EVAL_QUOTAS[category]),
            ("test", EVAL_QUOTAS[category]),
        ):
            partitions[name].extend(ordered[cursor : cursor + quota])
            cursor += quota
        for ordinal in range(1, CHUNKS + 1):
            quota = CHUNK_QUOTAS[category]
            partitions[f"chunk-{ordinal:05d}"].extend(ordered[cursor : cursor + quota])
            cursor += quota
    return partitions


def _publish_partition(
    rows: list[dict[str, object]], parent_recipe: dict[str, object], output: Path
) -> dict[str, object]:
    sources = parent_recipe.get("sources")
    if not isinstance(sources, list):
        raise InventoryError("parent corpus recipe sources are invalid")
    by_category: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        metadata = row["metadata"]
        by_category.setdefault(str(metadata["category"]), []).append(row)
    normalized = output.parent / f".{output.name}-normalized"
    normalized.mkdir()
    child_sources = []
    try:
        for ordinal, raw_source in enumerate(sources):
            if not isinstance(raw_source, dict):
                raise InventoryError("parent recipe source is invalid")
            category = str(raw_source["category"])
            selected = by_category.get(category, [])
            name = f"source-{ordinal:02d}.jsonl"
            payload = b"".join(
                _canonical(
                    {
                        "messages": row["messages"],
                        "record_id": row["metadata"]["record_id"],
                    }
                )
                for row in selected
            )
            (normalized / name).write_bytes(payload)
            child = dict(raw_source)
            child.update(
                {"input": name, "input_sha256": _sha256_bytes(payload), "quota": len(selected)}
            )
            child_sources.append(child)
        recipe = {
            "expected_records": len(rows),
            "format": corpus.RECIPE_FORMAT,
            "seed": PARTITION_SEED,
            "sources": child_sources,
        }
        recipe_payload = _canonical(recipe)
        recipe_path = normalized / "recipe.json"
        recipe_path.write_bytes(recipe_payload)
        result = corpus.prepare(
            argparse.Namespace(
                recipe=recipe_path,
                recipe_sha256=_sha256_bytes(recipe_payload),
                output=output,
            )
        )
        return corpus.validate(result)
    finally:
        shutil.rmtree(normalized, ignore_errors=True)


def build(
    source_root: Path,
    exclusion_root: Path,
    output: Path,
    source_manifest_sha256: str,
    exclusion_manifest_sha256: str,
) -> Path:
    _, plan_sha256 = _plan()
    source_manifest = validate_source(source_root)
    if source_manifest["manifest_content_sha256"] != _hex(
        source_manifest_sha256, "source manifest SHA-256"
    ):
        raise InventoryError("reviewed expanded-source manifest differs")
    exclusion_manifest, excluded = _rows(exclusion_root.resolve())
    if exclusion_manifest["manifest_content_sha256"] != _hex(
        exclusion_manifest_sha256, "exclusion manifest SHA-256"
    ):
        raise InventoryError("reviewed exclusion manifest differs")
    if exclusion_manifest.get("records") != 5_000:
        raise InventoryError("exclusion corpus is not the reviewed 5K boundary")
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise InventoryError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        parent_recipe = source_root.resolve() / SOURCE_RECIPE
        expanded_root = corpus.prepare(
            argparse.Namespace(
                recipe=parent_recipe,
                recipe_sha256=_sha256_file(parent_recipe),
                output=temporary / "expanded-corpus",
            )
        )
        expanded_manifest, expanded = _rows(expanded_root)
        if expanded_manifest.get("records") != 52_000:
            raise InventoryError("expanded corpus does not contain exactly 52K prompts")
        partitions = _partition_rows(expanded, excluded)
        shutil.copytree(exclusion_root.resolve(), temporary / "excluded-5k")
        recipe = _strict_object(parent_recipe)
        (temporary / "chunks").mkdir()
        published: list[dict[str, object]] = []
        for name, rows in partitions.items():
            destination = (
                temporary / name
                if name in {"holdout", "test"}
                else temporary / "chunks" / name
            )
            manifest = _publish_partition(rows, recipe, destination)
            expected = 1_000 if name in {"holdout", "test"} else 5_000
            if manifest.get("records") != expected:
                raise InventoryError(f"{name} record count differs")
            published.append(
                {
                    "manifest_content_sha256": manifest["manifest_content_sha256"],
                    "name": name,
                    "prompt_sha256": manifest["prompt_sha256"],
                    "records": manifest["records"],
                }
            )
        manifest = {
            "expanded_corpus_manifest_content_sha256": expanded_manifest[
                "manifest_content_sha256"
            ],
            "expanded_corpus_prompt_sha256": expanded_manifest["prompt_sha256"],
            "exclusion_manifest_content_sha256": exclusion_manifest[
                "manifest_content_sha256"
            ],
            "exclusion_prompt_sha256": exclusion_manifest["prompt_sha256"],
            "format": FORMAT,
            "manifest_content_sha256": "",
            "partition_algorithm": PARTITION_ALGORITHM,
            "partition_seed": PARTITION_SEED,
            "partitions": sorted(published, key=lambda item: str(item["name"])),
            "plan_sha256": plan_sha256,
            "profile": PROFILE,
            "source_manifest_content_sha256": source_manifest[
                "manifest_content_sha256"
            ],
            "tool_sha256": _sha256_file(Path(__file__).resolve()),
        }
        manifest["manifest_content_sha256"] = _content_digest(manifest)
        (temporary / "manifest.json").write_bytes(_canonical(manifest))
        campaign._publish_no_replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validate(output)
    print(
        "FLASHMAP_PROMPT_INVENTORY_OK train_records=45000 holdout=1000 test=1000 "
        f"sha256={manifest['manifest_content_sha256']} output={output}"
    )
    return output


def validate(root: Path) -> dict[str, object]:
    _, plan_sha256 = _plan()
    root = root.resolve()
    manifest = _strict_object(root / "manifest.json")
    if (
        manifest.get("format") != FORMAT
        or manifest.get("profile") != PROFILE
        or manifest.get("partition_algorithm") != PARTITION_ALGORITHM
        or manifest.get("partition_seed") != PARTITION_SEED
        or manifest.get("plan_sha256") != plan_sha256
        or manifest.get("manifest_content_sha256") != _content_digest(manifest)
        or manifest.get("tool_sha256") != _sha256_file(Path(__file__).resolve())
    ):
        raise InventoryError("prompt inventory manifest differs")
    expected_names = {"holdout", "test", *(f"chunk-{i:05d}" for i in range(1, CHUNKS + 1))}
    raw_partitions = manifest.get("partitions")
    if not isinstance(raw_partitions, list) or {
        item.get("name") for item in raw_partitions if isinstance(item, dict)
    } != expected_names:
        raise InventoryError("prompt inventory partition set differs")
    expanded_manifest, expanded = _rows(root / "expanded-corpus")
    if (
        expanded_manifest.get("manifest_content_sha256")
        != manifest.get("expanded_corpus_manifest_content_sha256")
        or expanded_manifest.get("prompt_sha256")
        != manifest.get("expanded_corpus_prompt_sha256")
        or expanded_manifest.get("records") != 52_000
    ):
        raise InventoryError("expanded corpus binding differs")
    exclusion_manifest, excluded = _rows(root / "excluded-5k")
    if (
        exclusion_manifest.get("manifest_content_sha256")
        != manifest.get("exclusion_manifest_content_sha256")
        or exclusion_manifest.get("prompt_sha256")
        != manifest.get("exclusion_prompt_sha256")
        or exclusion_manifest.get("records") != 5_000
    ):
        raise InventoryError("excluded 5K corpus binding differs")
    expanded_identities = {
        (str(row["source_id"]), _message_digest(row)) for row in expanded
    }
    excluded_identities = {
        (str(row["source_id"]), _message_digest(row)) for row in excluded
    }
    if len(expanded_identities) != 52_000 or len(excluded_identities) != 5_000:
        raise InventoryError("expanded or excluded corpus repeats identities")
    identities: set[tuple[str, str]] = set()
    for item in raw_partitions:
        name = str(item["name"])
        path = root / name if name in {"holdout", "test"} else root / "chunks" / name
        child_manifest, rows = _rows(path)
        if (
            child_manifest.get("manifest_content_sha256")
            != item.get("manifest_content_sha256")
            or child_manifest.get("prompt_sha256") != item.get("prompt_sha256")
            or child_manifest.get("records") != item.get("records")
        ):
            raise InventoryError(f"{name} manifest binding differs")
        for row in rows:
            identity = (str(row["source_id"]), _message_digest(row))
            if identity in identities:
                raise InventoryError("prompt inventory partitions overlap")
            identities.add(identity)
    if len(identities) != 47_000:
        raise InventoryError("prompt inventory total differs")
    if identities != expanded_identities - excluded_identities:
        raise InventoryError("prompt inventory is not the exact post-exclusion set")
    if {path.name for path in root.iterdir()} != {
        "chunks",
        "excluded-5k",
        "expanded-corpus",
        "holdout",
        "manifest.json",
        "test",
    }:
        raise InventoryError("prompt inventory directory contents differ")
    return manifest


def self_test() -> None:
    _plan()

    def row(category: str, ordinal: int) -> dict[str, object]:
        messages = [{"content": f"{category} prompt {ordinal}", "role": "user"}]
        return {
            "id": f"{category}-{ordinal}",
            "messages": messages,
            "metadata": {"category": category, "record_id": str(ordinal)},
            "source_id": f"hf:example/{category}/{ordinal}",
        }

    original_chunk = dict(CHUNK_QUOTAS)
    original_eval = dict(EVAL_QUOTAS)
    try:
        CHUNK_QUOTAS.clear()
        CHUNK_QUOTAS.update({"general": 1, "code": 1})
        EVAL_QUOTAS.clear()
        EVAL_QUOTAS.update({"general": 1, "code": 1})
        expanded = [row(category, i) for category in CHUNK_QUOTAS for i in range(12)]
        excluded = [row(category, 0) for category in CHUNK_QUOTAS]
        # 1 excluded + 9 train + 2 eval = 12 per category.
        first = _partition_rows(expanded, excluded)
        second = _partition_rows(list(reversed(expanded)), excluded)
        if {
            name: [str(item["source_id"]) for item in rows]
            for name, rows in first.items()
        } != {
            name: [str(item["source_id"]) for item in rows]
            for name, rows in second.items()
        }:
            raise AssertionError("partitioning depends on source row order")
        keys = [
            (str(item["source_id"]), _message_digest(item))
            for rows in first.values()
            for item in rows
        ]
        if len(keys) != len(set(keys)) or len(keys) != 22:
            raise AssertionError("partitions are not complete and disjoint")
    finally:
        CHUNK_QUOTAS.clear()
        CHUNK_QUOTAS.update(original_chunk)
        EVAL_QUOTAS.clear()
        EVAL_QUOTAS.update(original_eval)
    print("FLASHMAP_PROMPT_INVENTORY_SELFTEST_OK partitions=11 records=22")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("normalize")
    source.add_argument("output", type=Path)
    check_source = commands.add_parser("validate-source")
    check_source.add_argument("source", type=Path)
    prepare = commands.add_parser("build")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("exclusion", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--source-manifest-sha256", required=True)
    prepare.add_argument("--exclusion-manifest-sha256", required=True)
    check = commands.add_parser("validate")
    check.add_argument("inventory", type=Path)
    commands.add_parser("self-test")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "normalize":
        normalize(args.output)
    elif args.command == "validate-source":
        manifest = validate_source(args.source)
        print(
            "FLASHMAP_CAMPAIGN_SOURCES_VALID "
            f"sha256={manifest['manifest_content_sha256']}"
        )
    elif args.command == "build":
        build(
            args.source,
            args.exclusion,
            args.output,
            args.source_manifest_sha256,
            args.exclusion_manifest_sha256,
        )
    elif args.command == "validate":
        manifest = validate(args.inventory)
        print(
            "FLASHMAP_PROMPT_INVENTORY_VALID "
            f"sha256={manifest['manifest_content_sha256']}"
        )
    else:
        self_test()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (InventoryError, corpus.CorpusError, campaign.CampaignError) as error:
        print(f"FLASHMAP_PROMPT_INVENTORY_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error

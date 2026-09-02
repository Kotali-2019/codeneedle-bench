#!/usr/bin/env python3
"""Generate deterministic, fresh positional-recall corpora.

The checked-in fixtures are reproducible from this script.  They intentionally
contain high-entropy identifiers, strings, and constants so an exact answer
must come from the supplied context rather than familiarity with public code.
Rotate the seed/version for future model generations after these fixtures have
been public long enough to risk entering training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "fixtures" / "novel"
GENERATOR_VERSION = "novel-recall-v3"
DEFAULT_SEED = 20260901
TARGET_SEEDS = (20260901, 20260917, 20261003)
TARGETS_PER_SEED = 6
ANCHOR_COUNT = len(TARGET_SEEDS) * TARGETS_PER_SEED
PRIMARY_LINES = 48

# Token counts vary by model tokenizer.  The character targets use the same
# empirical ~3.5 chars/token ratio as the repository's Python/jQuery fixtures.
CORPORA = {
    "novel_16k": {"approx_tokens": 16_000, "target_chars": 56_000},
    "novel_64k": {"approx_tokens": 64_000, "target_chars": 224_000},
    "novel_128k": {"approx_tokens": 128_000, "target_chars": 448_000},
}

_ALPHABET = string.ascii_lowercase + string.digits


def _token(rng: random.Random, length: int = 12) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(length))


def _identifier(rng: random.Random, length: int = 8) -> str:
    return rng.choice(string.ascii_lowercase) + _token(rng, length - 1)


def _function(index: int, rng: random.Random, prefix: str) -> str:
    """Return one valid Python function with a high-entropy 48-line window."""
    fn_tag = _token(rng, 10)
    names = [_identifier(rng, 8) for _ in range(24)]
    words = [_token(rng, 14) for _ in range(16)]
    nums = [rng.randrange(10_000, 9_999_999) for _ in range(48)]
    function_name = f"{prefix}_{index:04d}_{fn_tag}"

    lines = [
        f"def {function_name}(arg_a, arg_b, arg_c):",
        f'    """Novel recall marker {words[0]}::{words[1]}::{words[2]}."""',
        f"    {names[0]} = (arg_a + {nums[0]}) ^ (arg_b * {nums[1]})",
        f"    {names[1]} = ({names[0]} << {nums[2] % 7 + 1}) & {nums[3]}",
        f"    {names[2]} = [{nums[4]}, {nums[5]}, {nums[6]}, {nums[7]}]",
        f'    {names[3]} = "{words[3]}-{words[4]}-{words[5]}"',
        f"    if {names[0]} % {nums[8] % 89 + 11} == {nums[9] % 11}:",
        f"        {names[1]} ^= {nums[10]}",
        "    else:",
        f"        {names[1]} += {nums[11]}",
        f"    {names[0]} = ({names[0]} + {names[1]}) % {nums[12]}",
        f"    {names[2]}.append(({names[0]} ^ {nums[13]}) & 0xffff)",
        f"    {names[4]} = sum({names[2]}) + len({names[3]})",
        f"    for {names[5]}, {names[6]} in enumerate({names[2]}):",
        f"        {names[4]} ^= ({names[6]} + {names[5]} * {nums[14]})",
        f"    if {names[4]} & {nums[15] % 255 + 1}:",
        f"        {names[2]}.reverse()",
        f'    {names[7]} = {{"{words[6]}": {names[0]}, "{words[7]}": {names[4]}}}',
        f'    {names[7]}["{words[8]}"] = {names[3]}[{nums[16] % 5}:]',
        f"    # opaque checkpoint {words[9]} / {nums[17]}",
        f"    {names[8]} = ({names[4]} ^ {names[0]}) & 0xffffffff",
        f"    {names[9]} = tuple(value ^ {names[8]} for value in {names[2]})",
        f"    if len({names[9]}) == {len(nums[4:8]) + 1}:",
        f"        {names[8]} = ({names[8]} << 1) ^ {nums[0]}",
        "    else:",
        f"        {names[8]} = ({names[8]} >> 1) + {nums[1]}",
        f'    {names[10]} = "{fn_tag}:{words[0]}:{index:04d}"',
        f"    {names[7]}[{names[10]}] = {names[9]}",
        f"    {names[11]} = ({names[8]} + {nums[18]}) ^ {nums[19]}",
        f"    {names[12]} = [{names[11]}, {names[4]}, {nums[20]}]",
        f"    for {names[13]} in range({nums[21] % 4 + 2}):",
        f"        {names[11]} = ({names[11]} * {nums[22] % 997 + 17} + {names[13]}) & 0xffffffff",
        f"        {names[12]}.append({names[11]} ^ {nums[23]})",
        f'    {names[14]} = "{words[10]}::{words[11]}::{nums[24]}"',
        f"    {names[15]} = sum({names[12]}) % {nums[25]}",
        f"    if {names[15]} > {nums[26] % 5000}:",
        f"        {names[16]} = {names[14]}[::-1]",
        "    else:",
        f'        {names[16]} = {names[14]} + "-{words[12]}"',
        f"    {names[17]} = {{value: pos for pos, value in enumerate({names[12]})}}",
        f"    {names[18]} = ({names[15]} << {nums[27] % 5 + 1}) ^ len({names[16]})",
        f"    for {names[19]}, {names[20]} in {names[17]}.items():",
        f"        {names[18]} ^= ({names[19]} & 0xffff) + {names[20]} * {nums[28]}",
        f"    {names[21]} = ({names[18]} ^ {nums[29]}) & {nums[30]}",
        f'    {names[22]} = ("{words[13]}", "{words[14]}", "{words[15]}")',
        f"    {names[23]} = {names[22]}[{names[21]} % len({names[22]})]",
        f"    {names[7]}[{names[23]}] = {names[21]}",
        f"    {names[7]}[{names[16]}] = tuple({names[12]})",
        f"    {names[8]} = ({names[8]} + {names[18]} + {nums[31]}) & 0xffffffff",
        f"    {names[2]}.extend(({names[21]}, {nums[32]}, {nums[33]}))",
        f"    # terminal checkpoint {words[15]} / {nums[34]} / {nums[35]}",
        f"    {names[10]} = f\"{{{names[10]}}}:{words[14]}:{{{names[8]}}}\"",
        "    return {",
        f'        "state": {names[8]},',
        f'        "route": {names[2]},',
        f'        "payload": {names[7]},',
        f'        "marker": {names[10]},',
        "    }",
    ]
    return "\n".join(lines)


def _shadow_function(target_name: str, rng: random.Random) -> str:
    """A short, structurally similar neighbor that can distract retrieval.

    Shadows deliberately remain below the extractor's 20-line minimum: they
    are context interference, never benchmark targets themselves.
    """
    names = [_identifier(rng, 8) for _ in range(7)]
    words = [_token(rng, 14) for _ in range(4)]
    nums = [rng.randrange(10_000, 9_999_999) for _ in range(10)]
    lines = [
        f"def {target_name}_shadow(arg_a, arg_b, arg_c):",
        f'    """Novel recall marker {words[0]}::{words[1]}::{words[2]}."""',
        f"    {names[0]} = (arg_a + {nums[0]}) ^ (arg_b * {nums[1]})",
        f"    {names[1]} = ({names[0]} << {nums[2] % 7 + 1}) & {nums[3]}",
        f"    {names[2]} = [{nums[4]}, {nums[5]}, {nums[6]}, {nums[7]}]",
        f'    {names[3]} = "{words[1]}-{words[2]}-{words[3]}"',
        f"    if {names[0]} % {nums[8] % 89 + 11} == {nums[9] % 11}:",
        f"        {names[1]} ^= {nums[4]}",
        "    else:",
        f"        {names[1]} += {nums[5]}",
        f"    {names[4]} = sum({names[2]}) + len({names[3]})",
        f"    {names[5]} = ({names[4]} ^ {names[0]}) & 0xffffffff",
        f"    {names[6]} = tuple(value ^ {names[5]} for value in {names[2]})",
        f"    return {names[6]}",
    ]
    return "\n".join(lines)


def _function_name(block: str) -> str:
    return block.split("(", 1)[0].removeprefix("def ")


def generate_source(
    name: str, target_chars: int, seed: int = DEFAULT_SEED
) -> tuple[str, int, list[str], dict[str, int]]:
    header = (
        '"""Deterministic novel-code recall fixture.\n\n'
        "Generated by tools/generate_novel_corpora.py; do not hand-edit.\n"
        "The identifiers and constants are intentionally opaque.\n"
        '"""\n\n'
        f'GENERATOR_VERSION = "{GENERATOR_VERSION}"\n'
        f"CORPUS_SEED = {seed}\n"
        f'CORPUS_NAME = "{name}"\n\n'
    )
    # The exact same anchors appear in every size.  Their positions are spread
    # across each file, while a size-dependent number of distractors expands
    # the distance between them.  This keeps target difficulty paired when
    # measuring context-length decay.
    anchors: list[str] = []
    target_seed_by_name: dict[str, int] = {}
    for seed_index, target_seed in enumerate(TARGET_SEEDS):
        anchor_rng = random.Random(target_seed ^ 0xA5A5_5A5A)
        for local_index in range(TARGETS_PER_SEED):
            index = seed_index * TARGETS_PER_SEED + local_index
            block = _function(index, anchor_rng, "needle") + "\n\n"
            anchors.append(block)
            target_seed_by_name[_function_name(block)] = target_seed
    target_names = [_function_name(block) for block in anchors]

    shadow_rng = random.Random(seed ^ 0xC3C3_3C3C)
    bundles = [
        _shadow_function(target_name, shadow_rng) + "\n\n" + anchor
        for target_name, anchor in zip(target_names, anchors)
    ]

    filler_rng = random.Random(seed)
    fillers: list[str] = []
    current_chars = len(header) + sum(len(block) for block in bundles)
    while current_chars < target_chars:
        block = _function(len(fillers), filler_rng, "distractor") + "\n\n"
        fillers.append(block)
        current_chars += len(block)

    total_blocks = len(bundles) + len(fillers)
    anchor_positions = {
        round(index * (total_blocks - 1) / (ANCHOR_COUNT - 1))
        for index in range(ANCHOR_COUNT)
    }
    parts = [header]
    anchor_index = 0
    filler_index = 0
    for position in range(total_blocks):
        if position in anchor_positions:
            parts.append(bundles[anchor_index])
            anchor_index += 1
        else:
            parts.append(fillers[filler_index])
            filler_index += 1

    assert anchor_index == len(bundles) and filler_index == len(fillers)
    # `functions` in the manifest means extractor-eligible functions. Shadows
    # are intentionally shorter than the universal 20-line extraction floor.
    extractable_functions = len(anchors) + len(fillers)
    source = "".join(parts).rstrip() + "\n"
    return source, extractable_functions, target_names, target_seed_by_name


def generated_files(seed: int = DEFAULT_SEED) -> tuple[dict[str, str], str]:
    files: dict[str, str] = {}
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "target_seeds": list(TARGET_SEEDS),
        "primary_lines": PRIMARY_LINES,
        "decoys_per_target": 1,
        "paired_targets": [],
        "corpora": {},
    }
    for name, spec in CORPORA.items():
        source, functions, target_names, target_seed_by_name = generate_source(
            name, spec["target_chars"], seed
        )
        if not manifest["paired_targets"]:
            manifest["paired_targets"] = target_names
        else:
            assert manifest["paired_targets"] == target_names
        manifest["target_seed_by_name"] = target_seed_by_name
        filename = f"{name}.py"
        files[filename] = source
        manifest["corpora"][name] = {
            "approx_tokens": spec["approx_tokens"],
            "target_chars": spec["target_chars"],
            "actual_chars": len(source),
            "lines": source.count("\n"),
            "functions": functions,
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        }
    return files, json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def write_generated(output_dir: Path, seed: int = DEFAULT_SEED) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files, manifest = generated_files(seed)
    for filename, contents in files.items():
        (output_dir / filename).write_text(contents, encoding="utf-8", newline="\n")
    (output_dir / "manifest.json").write_text(manifest, encoding="utf-8", newline="\n")


def check_generated(output_dir: Path, seed: int = DEFAULT_SEED) -> bool:
    files, manifest = generated_files(seed)
    expected = {**files, "manifest.json": manifest}
    stale = []
    for filename, contents in expected.items():
        path = output_dir / filename
        if not path.is_file() or path.read_text(encoding="utf-8") != contents:
            stale.append(filename)
    if stale:
        print("stale or missing generated fixture(s): " + ", ".join(stale))
        return False
    print(f"generated fixtures are current ({len(files)} corpora, seed {seed})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--check", action="store_true", help="verify checked-in outputs")
    args = parser.parse_args()

    if args.check:
        return 0 if check_generated(args.output_dir, args.seed) else 1
    write_generated(args.output_dir, args.seed)
    print(f"wrote {len(CORPORA)} corpora to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

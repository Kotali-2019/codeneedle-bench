"""Parse a source file and enumerate named functions with enough body lines to test."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .textio import read_text


MIN_BODY_LINES = 20
BONUS_CAP = 40  # extra lines past the primary 20 that count toward the "blue" bonus

# Per-line classification, used by the scorer to decide what earns credit.
# Blank lines are structural filler — a model reproducing them demonstrates no
# recall, so they never count (see bench/scorer.py). Comments and docstrings
# ARE genuine recall (prose can't be inferred from surrounding code), so they
# count by default but are reported separately from code.
KIND_CODE = "code"
KIND_BLANK = "blank"
KIND_COMMENT = "comment"
KIND_DOCSTRING = "docstring"
PROSE_KINDS = (KIND_COMMENT, KIND_DOCSTRING)


@dataclass
class FunctionTarget:
    name: str
    start_line: int           # 1-indexed line of first body line (after the opening brace)
    body_lines: list[str]     # body lines starting at start_line, excluding the closing brace line
    language: str = "js"      # "js" or "py" — controls the prompt wording
    source_path: Path | None = None  # which file this came from (for multi-file corpora)
    body_kinds: list[str] | None = None  # parallel to body_lines; one KIND_* per line
    # The definition's own source text, from where it starts through the line
    # that opens the body. The prompt quotes this instead of guessing at
    # `function <name>(` — 5 of 16 sampled jQuery targets are property or
    # assignment style, for which that guess names text the file never contains.
    signature_text: str | None = None
    # True when another definition shares BOTH this name and this exact
    # signature, so no prompt could distinguish them. Such targets are
    # unanswerable and get excluded from sampling.
    ambiguous: bool = False
    # Corpus configs may widen the recall window for harder suites. Extraction
    # still uses MIN_BODY_LINES as its universal floor; configure_primary_window
    # removes functions that are too short before sampling.
    primary_line_count: int = MIN_BODY_LINES

    @property
    def primary_lines(self) -> list[str]:
        return self.body_lines[:self.primary_line_count]

    @property
    def bonus_lines(self) -> list[str]:
        start = self.primary_line_count
        return self.body_lines[start:start + BONUS_CAP]

    @property
    def primary_kinds(self) -> list[str]:
        return self._kinds()[:self.primary_line_count]

    @property
    def bonus_kinds(self) -> list[str]:
        start = self.primary_line_count
        return self._kinds()[start:start + BONUS_CAP]

    def _kinds(self) -> list[str]:
        """Kinds for every body line, falling back to a content-only guess."""
        if self.body_kinds is not None and len(self.body_kinds) == len(self.body_lines):
            return self.body_kinds
        return [KIND_BLANK if l.strip() == "" else KIND_CODE for l in self.body_lines]

    @property
    def code_line_count(self) -> int:
        """Code lines in the primary window.

        A window that's nearly all docstring tests prose recall almost
        exclusively — `http_server.log_message` has exactly one code line in
        its 20. Use this to filter or flag such targets.
        """
        return sum(1 for k in self.primary_kinds if k == KIND_CODE)


@dataclass
class Source:
    """A combined corpus: one or more files concatenated for a single benchmark run."""
    files: list[Path]
    text: str                       # full text fed to the model
    targets: list[FunctionTarget]
    language: str

    @property
    def display_name(self) -> str:
        if len(self.files) == 1:
            return self.files[0].name
        return f"{len(self.files)} files from {self.files[0].parent}"


def configure_primary_window(source: Source, primary_lines: int) -> Source:
    """Apply a corpus's primary-window size and drop targets too short for it."""
    if primary_lines < MIN_BODY_LINES:
        raise ValueError(
            f"primary_lines must be at least {MIN_BODY_LINES}, got {primary_lines}"
        )
    source.targets = [
        target for target in source.targets if len(target.body_lines) >= primary_lines
    ]
    for target in source.targets:
        target.primary_line_count = primary_lines
    return source


def language_of(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".js", ".mjs", ".cjs"):
        return "js"
    if suffix == ".py":
        return "py"
    raise ValueError(f"Unsupported file type: {suffix!r}. Supported: .js, .mjs, .cjs, .py")


def extract(path: Path) -> list[FunctionTarget]:
    source = read_text(path)
    lang = language_of(path)
    if lang == "js":
        targets = _extract_js(source)
    else:
        targets = _extract_py(source)
    # Classify every line once per file, then slice per target. Safe to index
    # by `start_line` here because targets still carry per-file line numbers —
    # load_source_glob() applies the multi-file offset only afterwards.
    kinds = line_kinds(source, lang)
    for t in targets:
        t.language = lang
        t.source_path = path
        begin = t.start_line - 1
        t.body_kinds = kinds[begin:begin + len(t.body_lines)]
    return targets


# --- line classification ------------------------------------------------------


def line_kinds(source: str, lang: str) -> list[str]:
    """Classify each 1-indexed source line as code / blank / comment / docstring."""
    lines = source.splitlines()
    kinds = [KIND_BLANK if l.strip() == "" else KIND_CODE for l in lines]
    marks = _prose_lines_py(source) if lang == "py" else _prose_lines_js(source)
    for lineno, kind in marks.items():
        idx = lineno - 1
        if 0 <= idx < len(kinds) and kinds[idx] != KIND_BLANK:
            kinds[idx] = kind
    return kinds


def _prose_lines_py(source: str) -> dict[int, str]:
    """1-indexed line → KIND_DOCSTRING / KIND_COMMENT for Python."""
    import ast

    marks: dict[int, str] = {}
    for i, l in enumerate(source.splitlines(), 1):
        if l.lstrip().startswith("#"):
            marks[i] = KIND_COMMENT

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return marks

    scopes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, scopes):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        # A docstring is a bare string expression in the first statement slot.
        if (
            isinstance(first, ast.Expr)
            and isinstance(getattr(first, "value", None), ast.Constant)
            and isinstance(first.value.value, str)
        ):
            for ln in range(first.lineno, getattr(first, "end_lineno", first.lineno) + 1):
                marks[ln] = KIND_DOCSTRING
    return marks


def _prose_lines_js(source: str) -> dict[int, str]:
    """1-indexed line → KIND_COMMENT for JavaScript.

    Only whole-line comments are marked. A line of code with a trailing `//`
    comment stays code — the model still has to reproduce the code on it.
    """
    import esprima

    opts = {"loc": True, "comment": True, "tolerant": True}
    try:
        tree = esprima.parseModule(source, options=opts)
    except Exception:
        try:
            tree = esprima.parseScript(source, options=opts)
        except Exception:
            return {}

    lines = source.splitlines()
    marks: dict[int, str] = {}

    def line_at(lineno: int) -> str | None:
        idx = lineno - 1
        return lines[idx] if 0 <= idx < len(lines) else None

    def blank_before(lineno: int, col: int) -> bool:
        l = line_at(lineno)
        return l is not None and l[:col].strip() == ""

    def blank_after(lineno: int, col: int) -> bool:
        l = line_at(lineno)
        return l is not None and l[col:].strip() == ""

    for c in getattr(tree, "comments", None) or []:
        start, scol = c.loc.start.line, c.loc.start.column
        end, ecol = c.loc.end.line, c.loc.end.column
        if start == end:
            # Single-line comment: only a whole-line one counts. A line of code
            # with a trailing `//` stays code — the code still must be recalled.
            if blank_before(start, scol) and blank_after(end, ecol):
                marks[start] = KIND_COMMENT
            continue
        # Multi-line block comment. Interior lines are unambiguously comment.
        for ln in range(start + 1, end):
            marks[ln] = KIND_COMMENT
        # The opening line is all comment from `/*` on; it counts if nothing
        # precedes it. Likewise the closing line, if nothing follows `*/`.
        if blank_before(start, scol):
            marks[start] = KIND_COMMENT
        if blank_after(end, ecol):
            marks[end] = KIND_COMMENT
    return marks


def load_source_glob(
    directory: Path,
    glob: str,
    limit: int | None = None,
) -> Source:
    """Glob a directory for source files, concatenate them, extract all targets.

    Files are concatenated with comment-marker headers so the model can see file
    boundaries. All files must be the same language. Across files, duplicate
    function names are deduplicated (first occurrence wins) so the prompt is
    unambiguous when looked up by name.
    """
    paths = sorted(p for p in directory.glob(glob) if p.is_file())
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no files match {directory}/{glob}")

    lang = language_of(paths[0])
    for p in paths[1:]:
        if language_of(p) != lang:
            raise ValueError(
                f"mixed languages in glob: {paths[0]} is {lang}, {p} is {language_of(p)}"
            )

    parts: list[str] = []
    targets: list[FunctionTarget] = []
    seen_names: set[str] = set()
    line_offset = 0

    for p in paths:
        text = read_text(p)
        header = _file_header(lang, p)
        parts.append(header)
        parts.append(text)
        if not text.endswith("\n"):
            parts.append("\n")
        parts.append("\n")  # blank line between files

        header_line_count = header.count("\n")
        for t in extract(p):
            if t.name in seen_names:
                # Skip cross-file collisions — prompt would be ambiguous by name.
                continue
            seen_names.add(t.name)
            t.start_line += line_offset + header_line_count
            targets.append(t)

        line_offset += header.count("\n") + text.count("\n") + (0 if text.endswith("\n") else 1) + 1

    combined = "".join(parts)
    return Source(files=paths, text=combined, targets=targets, language=lang)


def _file_header(lang: str, path: Path) -> str:
    marker = "//" if lang == "js" else "#"
    return f"{marker} ====== {path} ======\n"


# --- JavaScript ---------------------------------------------------------------


def _block_occurrences(lines: list[str], block: str) -> int:
    """How many times `block` appears as consecutive whole lines in `lines`.

    Ambiguity has to be judged against the WHOLE file, not just the functions
    long enough to be targets: the model sees every definition. `send_head` in
    http_server.py is declared twice with an identical signature, but only one
    body is long enough to be extracted — the prompt is still ambiguous.
    """
    needle = block.splitlines()
    if not needle:
        return 0
    n = len(needle)
    return sum(1 for i in range(len(lines) - n + 1) if lines[i:i + n] == needle)


def _extract_js(source: str) -> list[FunctionTarget]:
    import esprima

    try:
        tree = esprima.parseModule(
            source, options={"loc": True, "tolerant": True}
        )
    except Exception:
        tree = esprima.parseScript(
            source, options={"loc": True, "tolerant": True}
        )

    lines = source.splitlines()
    targets: list[FunctionTarget] = []
    seen: set[str] = set()
    # Every eligible definition's signature, keyed by name — including ones
    # skipped as duplicates. Used afterwards to tell a name that merely repeats
    # (disambiguated by its signature) from one that is genuinely ambiguous.
    signatures: dict[str, list[str]] = {}

    def emit(name: str, node, block) -> None:
        brace_line = block.loc.start.line  # line of '{'
        close_line = block.loc.end.line    # line of '}'
        if close_line - brace_line < MIN_BODY_LINES + 1:
            return
        # lines strictly between { and }
        body = lines[brace_line:close_line - 1]
        if len(body) < MIN_BODY_LINES:
            return

        # The signature spans from where the definition starts through the
        # line carrying the opening brace, so the body begins on the very next
        # line. Quoting the whole span keeps multi-line signatures correct.
        signature = "\n".join(lines[node.loc.start.line - 1:brace_line])
        signatures.setdefault(name, []).append(signature)

        if name in seen:
            return
        seen.add(name)
        targets.append(
            FunctionTarget(
                name=name,
                start_line=brace_line + 1,
                body_lines=body,
                signature_text=signature,
            )
        )

    def hint_for(parent_type: str | None, key: str, parent) -> str | None:
        if parent_type == "VariableDeclarator" and key == "init":
            pid = getattr(parent, "id", None)
            if pid is not None and getattr(pid, "type", None) == "Identifier":
                return pid.name
        elif parent_type == "AssignmentExpression" and key == "right":
            left = getattr(parent, "left", None)
            if left is None:
                return None
            if getattr(left, "type", None) == "Identifier":
                return left.name
            if getattr(left, "type", None) == "MemberExpression":
                prop = getattr(left, "property", None)
                if prop is not None and getattr(prop, "type", None) == "Identifier":
                    return prop.name
        elif parent_type == "Property" and key == "value":
            k = getattr(parent, "key", None)
            if k is not None and getattr(k, "type", None) == "Identifier":
                return k.name
            if k is not None and getattr(k, "type", None) == "Literal":
                return str(k.value)
        elif parent_type == "MethodDefinition" and key == "value":
            k = getattr(parent, "key", None)
            if k is not None and getattr(k, "type", None) == "Identifier":
                return k.name
        return None

    def walk(node, name_hint: str | None = None) -> None:
        if node is None or not hasattr(node, "type"):
            return
        t = node.type

        if t == "FunctionDeclaration":
            nm = (node.id.name if getattr(node, "id", None) else None) or name_hint
            body = getattr(node, "body", None)
            if nm and body is not None and body.type == "BlockStatement":
                emit(nm, node, body)
        elif t == "FunctionExpression":
            nm = (
                (node.id.name if getattr(node, "id", None) else None)
                or name_hint
            )
            body = getattr(node, "body", None)
            if nm and body is not None and body.type == "BlockStatement":
                emit(nm, node, body)
        elif t == "ArrowFunctionExpression":
            body = getattr(node, "body", None)
            if name_hint and body is not None and body.type == "BlockStatement":
                emit(name_hint, node, body)

        # recurse
        for key, val in vars(node).items():
            if key == "loc":
                continue
            if isinstance(val, list):
                for item in val:
                    if hasattr(item, "type"):
                        walk(item, hint_for(t, key, node))
            elif hasattr(val, "type"):
                walk(val, hint_for(t, key, node))

    walk(tree)
    for t in targets:
        # Ambiguous when the quoted signature is not unique in the file, so no
        # prompt built from it could single this definition out.
        t.ambiguous = _block_occurrences(lines, t.signature_text or "") > 1
    return targets


# --- Python -------------------------------------------------------------------


def _extract_py(source: str) -> list[FunctionTarget]:
    import ast

    tree = ast.parse(source)
    lines = source.splitlines()
    targets: list[FunctionTarget] = []
    seen: set[str] = set()

    signatures: dict[str, list[str]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        start = node.body[0].lineno
        end = max(getattr(n, "end_lineno", n.lineno) for n in node.body)
        body = lines[start - 1:end]
        if len(body) < MIN_BODY_LINES:
            continue

        # `node.lineno` is the `def` line (decorators sit above it and are not
        # part of the signature). The span runs to the line before the body,
        # so multi-line signatures are quoted whole.
        signature = "\n".join(lines[node.lineno - 1:start - 1])
        signatures.setdefault(node.name, []).append(signature)

        if node.name in seen:
            continue
        seen.add(node.name)
        targets.append(
            FunctionTarget(name=node.name, start_line=start, body_lines=body,
                           signature_text=signature)
        )

    for t in targets:
        t.ambiguous = _block_occurrences(lines, t.signature_text or "") > 1
    return targets


# --- Sampling -----------------------------------------------------------------


def stratified_sample(
    targets: list[FunctionTarget],
    total_lines: int,
    k: int = 16,
    seed: int = 42,
) -> list[FunctionTarget]:
    """Sample k targets spread across file position — tests recall at all depths, not just the tail."""
    if len(targets) <= k:
        return list(targets)
    targets = sorted(targets, key=lambda t: t.start_line)
    rng = random.Random(seed)
    buckets: list[list[FunctionTarget]] = [[] for _ in range(k)]
    for t in targets:
        idx = min(k - 1, (t.start_line * k) // max(1, total_lines))
        buckets[idx].append(t)
    chosen: list[FunctionTarget] = []
    for b in buckets:
        if b:
            chosen.append(rng.choice(b))
    chosen_names = {t.name for t in chosen}
    pool = [t for t in targets if t.name not in chosen_names]
    rng.shuffle(pool)
    while len(chosen) < k and pool:
        chosen.append(pool.pop())
    return chosen[:k]

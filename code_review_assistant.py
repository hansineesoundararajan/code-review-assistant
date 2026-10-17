#!/usr/bin/env python3
"""
Code Review Assistant.

Analyses Python source code and produces review comments covering bugs,
security vulnerabilities, style deviations and optimisation opportunities.

The pipeline has three stages:

    preprocess  ->  static analysis  ->  LLM inference

The static layer is deterministic and runs on the standard library alone
(ast, tokenize, py_compile), so it works offline and never ships code
anywhere. Its structured output is then handed to a local LLM, which
explains each finding, ranks it and proposes a fix. The LLM may add
findings of its own, but only above a confidence threshold -- static
findings stay authoritative because they are the ones we can prove.

Everything lives in this one file on purpose: it is meant to be dropped
into a repo, a CI job or a notebook cell without an install step.

Usage:
    python code_review_assistant.py review path/to/file.py
    python code_review_assistant.py review src/ --format markdown
    python code_review_assistant.py review app.py --llm ollama --model qwen2.5-coder:7b
    python code_review_assistant.py eval --llm none
    python code_review_assistant.py ui
"""

from __future__ import annotations

import argparse
import ast
import builtins
import functools
import io
import json
import keyword
import multiprocessing
import os
import queue
import re
import shutil
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import time
import tokenize
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

__version__ = "1.2.0"

# Severity ordering, worst first. Used for sorting and for --min-severity.
SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]
SEVERITY_RANK = {name: i for i, name in enumerate(SEVERITY_ORDER)}

CATEGORIES = ("bug", "security", "style", "optimization")

# The brief asks for snippets up to ~500 lines. Anything longer gets chunked
# before it reaches the model, because small code models lose the plot on
# long inputs and we would rather review 500 good lines than 3000 bad ones.
DEFAULT_MAX_LINES = 500
DEFAULT_LINE_LENGTH = 100

BUILTIN_NAMES = set(dir(builtins))


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------

@dataclass
class Finding:
    """One review comment about one place in one file."""

    rule: str
    category: str
    severity: str
    line: int
    message: str
    column: int = 0
    suggestion: str = ""
    rationale: str = ""
    confidence: float = 1.0
    source: str = "static"          # "static" or "llm"
    snippet: str = ""

    def key(self) -> Tuple[str, int, str]:
        return (self.category, self.line, self.rule)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FileReport:
    path: str
    findings: List[Finding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    static_seconds: float = 0.0
    llm_seconds: float = 0.0
    # every finding the model proposed, before the confidence cut; the
    # evaluation uses these to show where the threshold should sit
    llm_candidates: List[Finding] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return self.static_seconds + self.llm_seconds

    def counts_by_category(self) -> Dict[str, int]:
        out = {c: 0 for c in CATEGORIES}
        for f in self.findings:
            out[f.category] = out.get(f.category, 0) + 1
        return out

    def counts_by_severity(self) -> Dict[str, int]:
        out = {s: 0 for s in SEVERITY_ORDER}
        for f in self.findings:
            out[f.severity] = out.get(f.severity, 0) + 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "findings": [f.to_dict() for f in self.findings],
            "metrics": self.metrics,
            "errors": self.errors,
            "timing": {
                "static_seconds": round(self.static_seconds, 4),
                "llm_seconds": round(self.llm_seconds, 4),
                "total_seconds": round(self.total_seconds, 4),
            },
        }


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

class SourceUnit:
    """A single file (or snippet) prepared for analysis.

    Parsing is allowed to fail. Half the value of a review tool is what it
    tells you about code that does not compile yet, so when ast.parse blows
    up we keep the text, record the error, and let the line-based rules run
    anyway.
    """

    def __init__(self, text: str, path: str = "<snippet>", max_lines: int = DEFAULT_MAX_LINES):
        self.path = path
        self.raw_text = text
        self.text = self._normalise(text)
        self.lines = self.text.split("\n")
        self.max_lines = max_lines
        self.truncated = False
        self.parse_error: Optional[str] = None
        self.parse_error_line: Optional[int] = None
        self.tree: Optional[ast.AST] = None
        self.tokens: List[tokenize.TokenInfo] = []
        self.token_error: Optional[str] = None

        if len(self.lines) > max_lines:
            self.truncated = True

        self._parse()
        self._tokenize()

    @staticmethod
    def _normalise(text: str) -> str:
        if text.startswith("﻿"):
            text = text[1:]
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @classmethod
    def from_path(cls, path: str | Path, max_lines: int = DEFAULT_MAX_LINES) -> "SourceUnit":
        p = Path(path)
        # errors="replace" so a stray byte in a file does not kill a batch run
        text = p.read_text(encoding="utf-8", errors="replace")
        return cls(text, str(p), max_lines=max_lines)

    def _parse(self) -> None:
        try:
            self.tree = ast.parse(self.text, filename=self.path)
        except SyntaxError as exc:
            self.parse_error = exc.msg or "syntax error"
            self.parse_error_line = exc.lineno or 1
        except ValueError as exc:
            # null bytes and similar
            self.parse_error = str(exc)
            self.parse_error_line = 1

    def _tokenize(self) -> None:
        try:
            stream = io.StringIO(self.text)
            self.tokens = list(tokenize.generate_tokens(stream.readline))
        except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
            self.token_error = str(exc)
            self.tokens = []

    @property
    def is_parseable(self) -> bool:
        return self.tree is not None

    def line(self, n: int) -> str:
        if 1 <= n <= len(self.lines):
            return self.lines[n - 1]
        return ""

    def numbered(self, start: int = 1, end: Optional[int] = None) -> str:
        end = end if end is not None else len(self.lines)
        width = len(str(end))
        out = []
        for i in range(start, min(end, len(self.lines)) + 1):
            out.append(f"{str(i).rjust(width)} | {self.lines[i - 1]}")
        return "\n".join(out)

    def chunks(self, size: Optional[int] = None) -> List[Tuple[int, int]]:
        """Split into (start_line, end_line) windows for the model.

        Splits on top-level definitions where possible so a function is not
        cut in half; falls back to a fixed window otherwise.
        """
        size = size or self.max_lines
        if len(self.lines) <= size:
            return [(1, len(self.lines))]

        boundaries = [1]
        if self.tree is not None:
            for node in self.tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    boundaries.append(node.lineno)
        boundaries.append(len(self.lines) + 1)
        boundaries = sorted(set(boundaries))

        windows: List[Tuple[int, int]] = []
        start = boundaries[0]
        for b in boundaries[1:]:
            if b - start >= size:
                windows.append((start, b - 1))
                start = b
        if start <= len(self.lines):
            windows.append((start, len(self.lines)))

        # A file with no top-level defs (or one enormous function) still has to
        # be cut up, so fall back to fixed windows for anything left oversized.
        final: List[Tuple[int, int]] = []
        for lo, hi in windows:
            while hi - lo + 1 > size:
                final.append((lo, lo + size - 1))
                lo += size
            if lo <= hi:
                final.append((lo, hi))
        return final


# --------------------------------------------------------------------------
# quality metrics
# --------------------------------------------------------------------------

DECISION_NODES = (
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler,
    ast.With, ast.AsyncWith, ast.Assert, ast.IfExp, ast.comprehension,
)

# match statements only exist from 3.10 onwards
match_type = getattr(ast, "match_case", None)


def cyclomatic_complexity(node: ast.AST) -> int:
    """McCabe complexity. One, plus one per branch point."""
    score = 1
    for child in ast.walk(node):
        if isinstance(child, DECISION_NODES):
            score += 1
        elif isinstance(child, ast.BoolOp):
            score += len(child.values) - 1
        elif match_type is not None and isinstance(child, match_type):
            score += 1
    return score


def max_nesting_depth(node: ast.AST) -> int:
    nesting_types = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With,
                     ast.AsyncWith, ast.Try, ast.FunctionDef, ast.AsyncFunctionDef)

    def walk(n: ast.AST, depth: int) -> int:
        best = depth
        for child in ast.iter_child_nodes(n):
            d = depth + 1 if isinstance(child, nesting_types) else depth
            best = max(best, walk(child, d))
        return best

    return walk(node, 0)


def collect_metrics(unit: SourceUnit) -> Dict[str, Any]:
    total = len(unit.lines)
    blank = sum(1 for ln in unit.lines if not ln.strip())
    comment = sum(1 for ln in unit.lines if ln.strip().startswith("#"))
    metrics: Dict[str, Any] = {
        "total_lines": total,
        "code_lines": total - blank - comment,
        "blank_lines": blank,
        "comment_lines": comment,
        "comment_ratio": round(comment / total, 3) if total else 0.0,
        "parseable": unit.is_parseable,
    }

    if unit.tree is None:
        return metrics

    functions = [n for n in ast.walk(unit.tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n for n in ast.walk(unit.tree) if isinstance(n, ast.ClassDef)]
    complexities = [cyclomatic_complexity(f) for f in functions]

    metrics.update({
        "functions": len(functions),
        "classes": len(classes),
        "max_complexity": max(complexities) if complexities else 0,
        "mean_complexity": round(statistics.fmean(complexities), 2) if complexities else 0.0,
        "max_nesting": max_nesting_depth(unit.tree),
        "module_complexity": cyclomatic_complexity(unit.tree),
    })

    if complexities:
        worst = max(zip(complexities, functions), key=lambda pair: pair[0])
        metrics["most_complex_function"] = {
            "name": worst[1].name,
            "line": worst[1].lineno,
            "complexity": worst[0],
        }
    return metrics


# --------------------------------------------------------------------------
# static rules
# --------------------------------------------------------------------------

class RuleVisitor(ast.NodeVisitor):
    """Shared plumbing for the AST rule sets."""

    def __init__(self, unit: SourceUnit, config: Dict[str, Any]):
        self.unit = unit
        self.config = config
        self.findings: List[Finding] = []

    def report(self, node: ast.AST, rule: str, category: str, severity: str,
               message: str, suggestion: str = "", rationale: str = "") -> None:
        line = getattr(node, "lineno", 1)
        self.findings.append(Finding(
            rule=rule,
            category=category,
            severity=severity,
            line=line,
            column=getattr(node, "col_offset", 0),
            message=message,
            suggestion=suggestion,
            rationale=rationale,
            snippet=self.unit.line(line).strip()[:160],
        ))

    def run(self) -> List[Finding]:
        if self.unit.tree is not None:
            self.visit(self.unit.tree)
        return self.findings


def call_name(node: ast.Call) -> str:
    """Dotted name of a call target, e.g. 'os.system' or 'eval'."""
    func = node.func
    parts: List[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def is_constant_str(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def literal_text(node: ast.AST) -> str:
    """Best-effort recovery of the static parts of a built string.

    Enough to tell whether an f-string or a concatenation was building SQL,
    without pretending to evaluate anything.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(literal_text(v) for v in node.values)
    if isinstance(node, ast.FormattedValue):
        return " ? "
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        return literal_text(node.left) + " " + literal_text(node.right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in {"format", "join"}:
            return literal_text(node.func.value)
    return ""


class BugRules(RuleVisitor):
    """Patterns that are usually a defect rather than a matter of taste."""

    MUTABLE_DEFAULTS = (ast.List, ast.Dict, ast.Set)

    def _check_mutable_defaults(self, node) -> None:
        defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d]
        for default in defaults:
            if isinstance(default, self.MUTABLE_DEFAULTS):
                self.report(
                    default, "B001", "bug", "high",
                    f"Mutable default argument in '{node.name}'",
                    "Use None as the default and build the container inside the function.",
                    "Default arguments are evaluated once at definition time, so the same "
                    "object is reused on every call and keeps state between them.",
                )
            elif isinstance(default, ast.Call) and call_name(default) in {"list", "dict", "set"}:
                self.report(
                    default, "B001", "bug", "high",
                    f"Mutable default argument in '{node.name}'",
                    "Use None as the default and build the container inside the function.",
                )

    def _check_returns(self, node) -> None:
        returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
        # Only look at returns belonging to this function, not nested ones.
        nested = set()
        for child in ast.walk(node):
            if child is not node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nested.update(id(n) for n in ast.walk(child))
        returns = [r for r in returns if id(r) not in nested]
        if len(returns) < 2:
            return
        with_value = [r for r in returns if r.value is not None]
        without = [r for r in returns if r.value is None]
        if with_value and without:
            self.report(
                without[0], "B002", "bug", "medium",
                f"'{node.name}' sometimes returns a value and sometimes returns None",
                "Return an explicit value on every path, or restructure so the function "
                "always returns the same kind of thing.",
                "Callers that unpack or use the result will break on the None path.",
            )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_mutable_defaults(node)
        self._check_returns(node)
        self._check_unreachable(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def _check_unreachable(self, node) -> None:
        for body_holder in ast.walk(node):
            body = getattr(body_holder, "body", None)
            if not isinstance(body, list):
                continue
            for i, stmt in enumerate(body[:-1]):
                if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                    nxt = body[i + 1]
                    self.report(
                        nxt, "B003", "bug", "medium",
                        "Unreachable code after a return/raise/break/continue",
                        "Delete the dead statements or move them above the exit.",
                    )
                    break

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        bare = node.type is None
        broad = isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}
        body_is_pass = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)

        if bare and body_is_pass:
            self.report(
                node, "B004", "bug", "high",
                "Bare except that silently swallows every error",
                "Catch the specific exception you expect and log or re-raise the rest.",
                "A bare except also catches KeyboardInterrupt and SystemExit, so the "
                "program becomes hard to stop and real failures disappear.",
            )
        elif bare:
            self.report(
                node, "B005", "bug", "medium",
                "Bare except catches everything, including KeyboardInterrupt",
                "Use 'except Exception:' at minimum, ideally a specific exception type.",
            )
        elif broad and body_is_pass:
            self.report(
                node, "B004", "bug", "medium",
                "Exception caught and discarded without handling",
                "Log the exception or re-raise it; silent failure hides bugs.",
            )
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Eq, ast.NotEq)):
                if isinstance(comparator, ast.Constant) and comparator.value is None:
                    self.report(
                        node, "B006", "bug", "low",
                        "Comparing to None with == instead of 'is'",
                        "Use 'is None' / 'is not None'.",
                        "== goes through __eq__, which a class can override, so the "
                        "comparison can give a surprising answer.",
                    )
                elif isinstance(comparator, ast.Constant) and isinstance(comparator.value, bool):
                    self.report(
                        node, "B007", "bug", "low",
                        f"Comparing to {comparator.value} explicitly",
                        "Use the value directly: 'if flag:' or 'if not flag:'.",
                    )
                elif isinstance(comparator, ast.Constant) and isinstance(comparator.value, float):
                    self.report(
                        node, "B008", "bug", "medium",
                        "Exact equality comparison against a float",
                        "Compare with a tolerance, e.g. math.isclose(a, b).",
                        "Binary floating point cannot represent most decimals exactly, so "
                        "values that should match often differ in the last bits.",
                    )
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        seen = {}
        for k in node.keys:
            if isinstance(k, ast.Constant):
                if k.value in seen:
                    self.report(
                        k, "B009", "bug", "high",
                        f"Duplicate key {k.value!r} in dict literal",
                        "Remove or rename the duplicate; the later value silently wins.",
                    )
                seen[k.value] = True
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        if isinstance(node.test, ast.Tuple) and node.test.elts:
            self.report(
                node, "B010", "bug", "critical",
                "assert on a tuple literal is always true",
                "Drop the parentheses: 'assert cond, message'.",
            )
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._check_mutation_during_iteration(node)
        self.generic_visit(node)

    def _check_mutation_during_iteration(self, node: ast.For) -> None:
        iter_name = None
        if isinstance(node.iter, ast.Name):
            iter_name = node.iter.id
        elif isinstance(node.iter, ast.Call):
            target = node.iter.func
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                if target.attr in {"keys", "values", "items"}:
                    iter_name = target.value.id
        if iter_name is None:
            return

        mutators = {"append", "remove", "pop", "insert", "clear", "extend", "update"}
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                owner = child.func.value
                if isinstance(owner, ast.Name) and owner.id == iter_name:
                    if child.func.attr in mutators:
                        self.report(
                            child, "B011", "bug", "high",
                            f"'{iter_name}' is modified while being iterated over",
                            "Iterate over a copy (list(x)) or build a new collection.",
                            "Mutating a container during iteration skips elements or raises "
                            "RuntimeError, depending on the type.",
                        )

    def visit_Try(self, node: ast.Try) -> None:
        for stmt in node.finalbody:
            if isinstance(stmt, ast.Return):
                self.report(
                    stmt, "B012", "bug", "high",
                    "return inside finally discards exceptions",
                    "Move the return out of the finally block.",
                    "A return in finally swallows any exception that was propagating.",
                )
        self.generic_visit(node)


class SecurityRules(RuleVisitor):
    """Vulnerability patterns. Deliberately conservative."""

    DANGEROUS_CALLS = {
        "eval": ("S001", "critical", "eval() executes arbitrary code",
                 "Use ast.literal_eval for data, or an explicit parser."),
        "exec": ("S002", "critical", "exec() executes arbitrary code",
                 "Restructure so the behaviour is selected from a dict of known functions."),
        "os.system": ("S003", "high", "os.system passes the string to a shell",
                      "Use subprocess.run([...]) with an argument list."),
        "os.popen": ("S003", "high", "os.popen passes the string to a shell",
                     "Use subprocess.run([...]) with an argument list."),
        "pickle.load": ("S004", "high", "pickle.load executes code from the payload",
                        "Use json for untrusted data, or verify the source first."),
        "pickle.loads": ("S004", "high", "pickle.loads executes code from the payload",
                         "Use json for untrusted data, or verify the source first."),
        "marshal.loads": ("S004", "high", "marshal.loads is unsafe on untrusted data",
                          "Use json instead."),
        "tempfile.mktemp": ("S005", "medium", "tempfile.mktemp has a race condition",
                            "Use tempfile.NamedTemporaryFile or mkstemp."),
        "ssl._create_unverified_context": ("S006", "high", "TLS verification disabled",
                                           "Use the default context so certificates are checked."),
    }

    WEAK_HASHES = {"md5", "sha1"}
    SECRET_HINTS = re.compile(r"(password|passwd|pwd|secret|token|api_?key|access_?key|"
                              r"private_?key|credential)", re.I)
    SQL_HINTS = re.compile(r"\b(select\s|insert\s+into|update\s|delete\s+from|drop\s+table)",
                           re.I)

    def __init__(self, unit: SourceUnit, config: Dict[str, Any]):
        super().__init__(unit, config)
        self.tainted_sql: set[str] = set()
        self.param_stack: List[set] = []    # argument names of the enclosing functions

    def visit_Module(self, node: ast.Module) -> None:
        # Queries are usually assembled a line or two above the execute() call,
        # so before walking we note which variables were built from a dynamic
        # string that looks like SQL.
        for child in ast.walk(node):
            if not isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            value = child.value
            if value is None:
                continue
            dynamic = isinstance(value, (ast.JoinedStr, ast.BinOp)) or (
                isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and value.func.attr == "format"
            )
            if not dynamic or not self.SQL_HINTS.search(literal_text(value)):
                continue
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    self.tainted_sql.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = call_name(node)

        if name in self.DANGEROUS_CALLS:
            rule, severity, message, suggestion = self.DANGEROUS_CALLS[name]
            # eval on a plain literal is harmless; do not cry wolf.
            if name in {"eval", "exec"} and node.args and isinstance(node.args[0], ast.Constant):
                pass
            else:
                self.report(node, rule, "security", severity, message, suggestion)

        if name.startswith("subprocess.") or name in {"run", "call", "Popen", "check_output"}:
            for kw in node.keywords:
                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    self.report(
                        node, "S007", "security", "high",
                        "subprocess called with shell=True",
                        "Pass a list of arguments and leave shell at its default.",
                        "With shell=True any interpolated value can inject extra commands.",
                    )

        if name in {"yaml.load"}:
            has_safe_loader = any(
                kw.arg == "Loader" and "Safe" in ast.dump(kw.value) for kw in node.keywords
            )
            if not has_safe_loader:
                self.report(
                    node, "S008", "security", "high",
                    "yaml.load without a safe loader can construct arbitrary objects",
                    "Use yaml.safe_load(...).",
                )

        if name.startswith("hashlib.") and name.split(".")[-1] in self.WEAK_HASHES:
            self.report(
                node, "S009", "security", "medium",
                f"{name.split('.')[-1]} is not safe for passwords or signatures",
                "Use hashlib.sha256 for integrity, or bcrypt/argon2 for passwords.",
            )

        if name.startswith("random.") and self._in_security_context(node):
            self.report(
                node, "S010", "security", "medium",
                "random is predictable and should not generate secrets",
                "Use the secrets module for tokens, passwords and nonces.",
            )

        for kw in node.keywords:
            if kw.arg == "verify" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                self.report(
                    node, "S006", "security", "high",
                    "Certificate verification disabled (verify=False)",
                    "Leave verification on; fix the certificate chain instead.",
                )

        self._check_sql(node, name)
        self.generic_visit(node)

    def _in_security_context(self, node: ast.AST) -> bool:
        """Only flag random() when it is feeding something that looks secret."""
        line = self.unit.line(getattr(node, "lineno", 0))
        return bool(self.SECRET_HINTS.search(line))

    def _check_sql(self, node: ast.Call, name: str) -> None:
        if not name.endswith("execute") and not name.endswith("executemany"):
            return
        if not node.args:
            return
        arg = node.args[0]
        built_dynamically = (
            isinstance(arg, ast.JoinedStr)
            or (isinstance(arg, ast.BinOp) and isinstance(arg.op, (ast.Add, ast.Mod)))
            or (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute)
                and arg.func.attr == "format")
            or (isinstance(arg, ast.Name) and arg.id in self.tainted_sql)
        )
        if built_dynamically:
            self.report(
                node, "S011", "security", "critical",
                "SQL query built by string interpolation",
                "Use a parameterised query: cursor.execute(sql, (value,)).",
                "Interpolating user input into SQL is the classic injection vector; "
                "the driver can escape values safely if you let it.",
            )

    # 'token_type', 'password_field' and friends hold metadata, not secrets.
    METADATA_SUFFIX = re.compile(
        r"_(type|name|field|header|pattern|regex|re|url|uri|id|prefix|suffix|"
        r"param|env|var|column|label|kind|format|error|class|attr)s?$", re.I
    )
    PLACEHOLDERS = {"", "none", "null", "changeme", "change_me", "xxx", "todo",
                    "your_key_here", "replace_me", "example", "test", "dummy"}

    @classmethod
    def _looks_like_secret(cls, value: str) -> bool:
        """Distinguish a real credential from a word that happens to sit in
        a variable called 'token'. Short, lowercase, alphabetic values are
        almost always identifiers; real keys mix character classes."""
        value = value.strip()
        if len(value) < 8 or value.lower() in cls.PLACEHOLDERS:
            return False
        has_digit = any(c.isdigit() for c in value)
        has_alpha = any(c.isalpha() for c in value)
        mixed_case = value != value.lower() and value != value.upper()
        has_punct = any(c in "-_/+=." for c in value)
        if has_alpha and (has_digit or mixed_case):
            return True
        return len(value) >= 20 and has_punct

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if not self.SECRET_HINTS.search(target.id):
                continue
            if self.METADATA_SUFFIX.search(target.id):
                continue
            if is_constant_str(node.value) and self._looks_like_secret(node.value.value):
                self.report(
                    node, "S012", "security", "critical",
                    f"Hardcoded credential in '{target.id}'",
                    "Read it from an environment variable or a secret manager.",
                    "Anything committed to a repository should be assumed public.",
                )
        self.generic_visit(node)

    def _enter_function(self, node) -> None:
        args = node.args
        names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
        names.update(a.arg for a in (args.vararg, args.kwarg) if a is not None)
        self.param_stack.append(names - {"self", "cls"})
        self.generic_visit(node)
        self.param_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_function(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Assert(self, node: ast.Assert) -> None:
        # asserts vanish under python -O, so they must not be the only check
        # on a caller's input. An assert about internal state is what assert
        # is for: flagging those made this rule mostly noise on real code
        # (8 of 10 wrong in the stdlib audit), so only arguments count.
        params = self.param_stack[-1] if self.param_stack else set()
        checks_argument = any(isinstance(n, ast.Name) and n.id in params
                              for n in ast.walk(node.test))
        if checks_argument and isinstance(node.test, (ast.Compare, ast.Call)):
            self.report(
                node, "S013", "security", "low",
                "assert checks an argument but is removed when Python runs with -O",
                "Raise an explicit exception for checks that must always run.",
            )
        self.generic_visit(node)


class StyleRules(RuleVisitor):
    """PEP 8 and readability rules that need the syntax tree."""

    def __init__(self, unit: SourceUnit, config: Dict[str, Any]):
        super().__init__(unit, config)
        self.scope_stack: List[str] = ["module"]

    def visit_Module(self, node: ast.Module) -> None:
        if not ast.get_docstring(node) and len(self.unit.lines) > 20:
            self.findings.append(Finding(
                rule="Y001", category="style", severity="info", line=1,
                message="Module has no docstring",
                suggestion="Add a short docstring saying what this module is for.",
            ))
        self._check_import_placement(node)
        self.generic_visit(node)

    @staticmethod
    def _is_dunder_assignment(stmt: ast.stmt) -> bool:
        """__all__, __version__ and friends conventionally sit above the imports."""
        targets: List[ast.expr] = []
        if isinstance(stmt, ast.Assign):
            targets = list(stmt.targets)
        elif isinstance(stmt, ast.AnnAssign):
            targets = [stmt.target]
        return bool(targets) and all(
            isinstance(t, ast.Name) and t.id.startswith("__") and t.id.endswith("__")
            for t in targets
        )

    @staticmethod
    def _is_conditional_import(stmt: ast.stmt) -> bool:
        """try/except ImportError and if TYPE_CHECKING blocks are still imports."""
        if not isinstance(stmt, (ast.If, ast.Try)):
            return False
        inner = [n for n in ast.walk(stmt)
                 if isinstance(n, ast.stmt) and not isinstance(n, (ast.If, ast.Try))]
        return bool(inner) and all(
            isinstance(n, (ast.Import, ast.ImportFrom, ast.Pass, ast.Raise,
                           ast.Assign, ast.ExceptHandler))
            for n in inner
        )

    def _check_import_placement(self, node: ast.Module) -> None:
        seen_code = False
        for stmt in node.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                if seen_code:
                    self.report(
                        stmt, "Y002", "style", "low",
                        "Import is not at the top of the file",
                        "Move imports above the module-level code.",
                    )
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue  # docstring
            elif self._is_dunder_assignment(stmt) or self._is_conditional_import(stmt):
                continue
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                   ast.Assign, ast.AnnAssign, ast.If, ast.Try)):
                seen_code = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        name = node.name
        if not re.fullmatch(r"_{0,2}[a-z][a-z0-9_]*_{0,2}", name):
            self.report(
                node, "Y003", "style", "low",
                f"Function name '{name}' is not snake_case",
                "Rename to lower_case_with_underscores (PEP 8).",
            )
        if not ast.get_docstring(node) and not name.startswith("_"):
            body_size = len(node.body)
            if body_size > 3:
                self.report(
                    node, "Y004", "style", "info",
                    f"Public function '{name}' has no docstring",
                    "Describe what it does, its arguments and what it returns.",
                )
        # pylint's default max-args is 5, so match that rather than invent one.
        arg_limit = self.config.get("max_args", 5)
        arg_count = len(node.args.args) + len(node.args.kwonlyargs)
        if node.args.args and node.args.args[0].arg in {"self", "cls"}:
            arg_count -= 1
        if arg_count > arg_limit:
            self.report(
                node, "Y005", "style", "medium",
                f"'{name}' takes {arg_count} parameters (limit {arg_limit})",
                "Group related parameters into a dataclass or pass a config object.",
                "Long parameter lists are easy to call in the wrong order and usually "
                "mean the function is doing more than one job.",
            )
        statements = sum(1 for n in ast.walk(node) if isinstance(n, ast.stmt))
        if statements > 60:
            self.report(
                node, "Y006", "style", "medium",
                f"'{name}' is very long ({statements} statements)",
                "Split out the independent steps into helper functions.",
            )
        complexity = cyclomatic_complexity(node)
        limit = self.config.get("max_complexity", 10)
        if complexity > limit:
            self.report(
                node, "Y007", "style", "medium",
                f"'{name}' has cyclomatic complexity {complexity} (limit {limit})",
                "Flatten the branches with early returns or extract helpers.",
                "Highly branched functions are the ones that get tested least and "
                "break most often.",
            )
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if not re.fullmatch(r"_?[A-Z][A-Za-z0-9]*", node.name):
            self.report(
                node, "Y008", "style", "low",
                f"Class name '{node.name}' is not PascalCase",
                "Rename to CapitalisedWords (PEP 8).",
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                self.report(
                    node, "Y009", "style", "medium",
                    f"Wildcard import from '{node.module}'",
                    "Import the names you actually use.",
                    "Wildcard imports hide where a name came from and can shadow builtins.",
                )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Lambda) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                self.report(
                    node, "Y010", "style", "low",
                    f"Lambda assigned to '{node.targets[0].id}'",
                    "Use a def; it gives the function a real name in tracebacks.",
                )
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in BUILTIN_NAMES:
                self.report(
                    node, "Y011", "style", "medium",
                    f"'{target.id}' shadows a builtin",
                    f"Rename the variable; '{target.id}' is no longer usable as the builtin here.",
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if call_name(node) == "type" and isinstance(getattr(node, "parent", None), ast.Compare):
            pass  # handled below via Compare
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if isinstance(node.left, ast.Call) and call_name(node.left) == "type":
            for op in node.ops:
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    self.report(
                        node, "Y012", "style", "low",
                        "type(x) == ... instead of isinstance",
                        "Use isinstance(x, SomeType), which also accepts subclasses.",
                    )
        self.generic_visit(node)


class OptimizationRules(RuleVisitor):
    """Things that are correct but wasteful."""

    def visit_For(self, node: ast.For) -> None:
        before = len(self.findings)
        self._check_range_len(node)
        self._check_dict_keys(node)
        self._check_string_concat(node)
        self._check_list_concat(node)

        # If a more specific rule already has something to say about this
        # loop header, stay quiet. Two comments on one line read as noise and
        # the developer only acts on one of them anyway.
        header_flagged = any(f.line == node.lineno for f in self.findings[before:])
        if not header_flagged:
            self._check_append_only_loop(node)
        self.generic_visit(node)

    def _check_range_len(self, node: ast.For) -> None:
        it = node.iter
        if isinstance(it, ast.Call) and call_name(it) == "range" and len(it.args) == 1:
            inner = it.args[0]
            if isinstance(inner, ast.Call) and call_name(inner) == "len":
                self.report(
                    node, "O001", "optimization", "low",
                    "for i in range(len(x)) can be enumerate(x)",
                    "Use 'for i, item in enumerate(x):' and drop the indexing.",
                    "enumerate avoids repeated indexing and reads better.",
                )

    def _check_dict_keys(self, node: ast.For) -> None:
        it = node.iter
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute):
            if it.func.attr == "keys" and not it.args:
                self.report(
                    node, "O002", "optimization", "info",
                    "Iterating over .keys() is redundant",
                    "Iterate over the dict directly.",
                )

    def _check_string_concat(self, node: ast.For) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                if isinstance(child.target, ast.Name):
                    if is_constant_str(child.value) or isinstance(child.value, ast.JoinedStr):
                        self.report(
                            child, "O003", "optimization", "medium",
                            f"String '{child.target.id}' is built by repeated concatenation",
                            "Append the pieces to a list and ''.join(...) once at the end.",
                            "Strings are immutable, so each += copies the whole string; "
                            "the loop becomes quadratic in the output size.",
                        )

    def _check_append_only_loop(self, node: ast.For) -> None:
        if len(node.body) != 1 or node.orelse:
            return
        stmt = node.body[0]
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "append":
                self.report(
                    node, "O004", "optimization", "low",
                    "Loop only appends; a list comprehension is faster and clearer",
                    "Rewrite as [expr for item in iterable].",
                )

    def _check_list_concat(self, node: ast.For) -> None:
        for child in ast.walk(node):
            if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                if isinstance(child.value, ast.List):
                    self.report(
                        child, "O005", "optimization", "medium",
                        "List built with += [...] inside a loop",
                        "Use list.append(...) or extend once outside the loop.",
                    )

    def visit_Call(self, node: ast.Call) -> None:
        name = call_name(node)
        if name in {"sum", "any", "all", "min", "max", "sorted"} and node.args:
            if isinstance(node.args[0], ast.ListComp):
                self.report(
                    node, "O006", "optimization", "low",
                    f"{name}() given a list comprehension",
                    f"Drop the brackets: {name}(x for x in ...) avoids building the list.",
                )
        if name == "len" and isinstance(getattr(node, "parent", None), ast.Compare):
            pass
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, ast.In) and isinstance(comparator, (ast.List, ast.Tuple)):
                if len(comparator.elts) >= 5 and all(isinstance(e, ast.Constant)
                                                     for e in comparator.elts):
                    self.report(
                        node, "O007", "optimization", "low",
                        "Membership test against a list/tuple literal",
                        "Use a set literal; membership is O(1) instead of O(n).",
                    )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        # d[k] fetched again right after 'k in d' -> two lookups. Opt-in: on real
        # code it was 0 for 8 in the stdlib audit, because 'k in d' then d[k] is
        # the clearest way to write it and .get() merges "missing" with None.
        if "O008" not in self.config.get("enable", ()):
            self.generic_visit(node)
            return
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.In):
            container = test.comparators[0]
            key = test.left
            if isinstance(container, ast.Name) and isinstance(key, (ast.Name, ast.Constant)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name):
                        if child.value.id == container.id and ast.dump(child.slice) == ast.dump(key):
                            self.report(
                                node, "O008", "optimization", "info",
                                f"'{container.id}' is looked up twice (membership then index)",
                                "Use dict.get(key) once, or try/except KeyError.",
                            )
                            break
        self.generic_visit(node)


def unused_name_checks(unit: SourceUnit) -> List[Finding]:
    """Unused imports and unused local variables.

    Kept separate from the visitors because it needs two passes: collect
    every Load first, then decide what was never read.
    """
    findings: List[Finding] = []
    tree = unit.tree
    if tree is None:
        return findings

    loaded: set[str] = set()
    attribute_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded.add(node.id)
        elif isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                attribute_roots.add(root.id)

    used = loaded | attribute_roots

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound not in used:
                    findings.append(Finding(
                        rule="Y013", category="style", severity="low", line=node.lineno,
                        message=f"'{bound}' is imported but never used",
                        suggestion="Remove the import.",
                        snippet=unit.line(node.lineno).strip()[:160],
                    ))
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue  # these are compiler directives, not names to use
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name
                if bound not in used:
                    findings.append(Finding(
                        rule="Y013", category="style", severity="low", line=node.lineno,
                        message=f"'{bound}' is imported but never used",
                        suggestion="Remove the import.",
                        snippet=unit.line(node.lineno).strip()[:160],
                    ))

    # Unused locals, one function at a time.
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assigned: Dict[str, int] = {}
        read: set[str] = set()
        skip: set[str] = set()
        # Names that the syntax forces you to bind: one slot of a tuple unpack
        # (data, consumed = encode(...)), a for-loop or comprehension variable.
        # In the stdlib audit these were 8 of the 9 unused-variable findings,
        # and none was a bug.
        for node in ast.walk(func):
            targets = []
            if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                targets.append(node.target)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets.extend(t for t in getattr(node, "targets", [getattr(node, "target", None)])
                               if isinstance(t, (ast.Tuple, ast.List)))
            elif isinstance(node, ast.withitem) and isinstance(node.optional_vars, (ast.Tuple, ast.List)):
                targets.append(node.optional_vars)
            for target in targets:
                if isinstance(target, (ast.Tuple, ast.List)) or isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                    skip.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
        for node in ast.walk(func):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                skip.update(node.names)
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    assigned.setdefault(node.id, node.lineno)
                else:
                    read.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not func:
                # names used by a nested function count as read
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Name):
                        read.add(inner.id)
        for name, line in assigned.items():
            if name.startswith("_") or name in skip or name in read:
                continue
            findings.append(Finding(
                rule="B013", category="bug", severity="low", line=line,
                message=f"'{name}' is assigned but never used in '{func.name}'",
                suggestion="Remove it, or prefix with _ if it is deliberately discarded.",
                rationale="An unused assignment is often a typo in the name that is read later.",
                snippet=unit.line(line).strip()[:160],
            ))
    return findings


def line_level_checks(unit: SourceUnit, config: Dict[str, Any]) -> List[Finding]:
    """Rules that work on raw text, so they survive a syntax error."""
    findings: List[Finding] = []
    limit = config.get("max_line_length", DEFAULT_LINE_LENGTH)

    uses_tabs = False
    uses_spaces = False
    blank_run = 0

    for i, raw in enumerate(unit.lines, start=1):
        if len(raw) > limit:
            findings.append(Finding(
                rule="Y014", category="style", severity="low", line=i,
                message=f"Line is {len(raw)} characters (limit {limit})",
                suggestion="Wrap it, or pull part of the expression into a named variable.",
                snippet=raw.strip()[:160],
            ))
        if raw != raw.rstrip():
            findings.append(Finding(
                rule="Y015", category="style", severity="info", line=i,
                message="Trailing whitespace",
                suggestion="Strip the end of the line.",
            ))
        indent = raw[:len(raw) - len(raw.lstrip())]
        if "\t" in indent:
            uses_tabs = True
        if indent.startswith(" ") and indent.strip(" ") == "":
            uses_spaces = True

        stripped = raw.strip()
        if not stripped:
            blank_run += 1
            if blank_run == 3:
                findings.append(Finding(
                    rule="Y017", category="style", severity="info", line=i,
                    message="More than two consecutive blank lines",
                    suggestion="Two blank lines between top-level definitions, one inside them.",
                ))
        else:
            blank_run = 0

    # Semicolons are found through the tokeniser rather than the raw text,
    # because prose in a docstring is full of semicolons and a regex over the
    # line cannot tell the difference.
    for tok in unit.tokens:
        if tok.type == tokenize.OP and tok.string == ";":
            findings.append(Finding(
                rule="Y016", category="style", severity="low", line=tok.start[0],
                message="Multiple statements on one line",
                suggestion="Put each statement on its own line.",
                snippet=unit.line(tok.start[0]).strip()[:160],
            ))

    if uses_tabs and uses_spaces:
        findings.append(Finding(
            rule="Y018", category="style", severity="medium", line=1,
            message="File mixes tabs and spaces for indentation",
            suggestion="Convert to 4 spaces throughout (PEP 8).",
        ))

    if unit.text and not unit.text.endswith("\n"):
        findings.append(Finding(
            rule="Y019", category="style", severity="info", line=len(unit.lines),
            message="File does not end with a newline",
            suggestion="Add a trailing newline.",
        ))

    # TODO/FIXME markers are worth surfacing in a review, at low priority
    for i, raw in enumerate(unit.lines, start=1):
        m = re.search(r"#\s*(TODO|FIXME|XXX|HACK)\b", raw)
        if m:
            findings.append(Finding(
                rule="Y020", category="style", severity="info", line=i,
                message=f"{m.group(1)} comment left in the code",
                suggestion="Resolve it or link it to a tracked issue.",
                snippet=raw.strip()[:160],
            ))
    return findings


# A developer who has looked at a finding and decided it does not apply needs
# a way to say so, otherwise the tool gets ignored wholesale after a week.
SUPPRESSION = re.compile(
    r"#\s*(?:noqa|review:\s*ignore)"
    r"(?:\s*[:\[]\s*(?P<rules>[A-Za-z0-9_,\s]+?)\s*\]?)?\s*(?:#.*)?$"
)


def collect_suppressions(unit: SourceUnit) -> Dict[int, Optional[set]]:
    """Map line number -> set of suppressed rules, or None meaning 'all'."""
    out: Dict[int, Optional[set]] = {}
    for i, raw in enumerate(unit.lines, start=1):
        m = SUPPRESSION.search(raw)
        if not m:
            continue
        rules = m.group("rules")
        if rules:
            out[i] = {r.strip() for r in rules.split(",") if r.strip()}
        else:
            out[i] = None
    return out


def apply_suppressions(findings: List[Finding],
                       suppressions: Dict[int, Optional[set]]) -> List[Finding]:
    if not suppressions:
        return findings
    kept = []
    for f in findings:
        entry = suppressions.get(f.line, False)
        if entry is False:
            kept.append(f)
        elif entry is None:
            continue           # bare noqa suppresses everything on the line
        elif f.rule not in entry:
            kept.append(f)
    return kept


# External rules that say the same thing as one of ours. When both fire on the
# same line only ours is kept, because ours carries the explanation and the
# fix. Anything missing from this table is new information and stays.
EQUIVALENT_RULES: Dict[str, Tuple[str, ...]] = {
    "ruff:F401": ("Y013",), "ruff:F841": ("B013",), "ruff:F601": ("B009",),
    "ruff:F403": ("Y009",), "ruff:E501": ("Y014",), "ruff:W291": ("Y015",),
    "ruff:W293": ("Y015",), "ruff:E711": ("B006",), "ruff:E712": ("B007",),
    "ruff:E721": ("Y012",), "ruff:E731": ("Y010",), "ruff:E402": ("Y002",),
    "ruff:E722": ("B004", "B005"), "ruff:B006": ("B001",), "ruff:B012": ("B012",),
    "ruff:C901": ("Y007",), "ruff:SIM118": ("O002",), "ruff:F631": ("B010",),
    "ruff:S110": ("B004",), "ruff:SIM107": ("B012",),
    # ruff reports a loop that only appends; we report the same loop under
    # whichever rule claimed its header, so any of ours counts as the twin
    "ruff:PERF401": ("O001", "O002", "O003", "O004", "O005"),
    "ruff:PERF402": ("O001", "O002", "O003", "O004", "O005"),
    "ruff:S101": ("S013",), "ruff:S102": ("S002",), "ruff:S105": ("S012",),
    "ruff:S106": ("S012",), "ruff:S107": ("S012",), "ruff:S301": ("S004",),
    "ruff:S302": ("S004",), "ruff:S306": ("S005",), "ruff:S307": ("S001",),
    "ruff:S311": ("S010",), "ruff:S324": ("S009",), "ruff:S501": ("S006",),
    "ruff:S506": ("S008",), "ruff:S602": ("S007",), "ruff:S604": ("S007",),
    "ruff:S605": ("S003",), "ruff:S608": ("S011",),
    "bandit:B101": ("S013",), "bandit:B102": ("S002",), "bandit:B105": ("S012",),
    "bandit:B106": ("S012",), "bandit:B107": ("S012",), "bandit:B301": ("S004",),
    "bandit:B302": ("S004",), "bandit:B306": ("S005",), "bandit:B307": ("S001",),
    "bandit:B311": ("S010",), "bandit:B324": ("S009",), "bandit:B501": ("S006",),
    "bandit:B506": ("S008",), "bandit:B602": ("S007",), "bandit:B605": ("S003",),
    "bandit:B608": ("S011",),
}

# One ruff call covers what flake8, pycodestyle, pyflakes, bugbear, bandit's
# security checks and radon's complexity score would each do separately.
RUFF_SELECT = "E,W,F,B,S,C90,PERF,SIM"


def ruff_category(code: str) -> str:
    # SIM starts with S, so the longer prefixes have to be tried first
    if code.startswith(("PERF", "SIM")):
        return "optimization"
    if code.startswith("S") and code != "syntax":
        return "security"
    if code in {"F401", "F811"}:
        return "style"          # unused or redefined imports
    if code.startswith(("B", "F")) or code == "syntax":
        return "bug"
    return "style"


# The ruff security rules that describe a clear hole rather than a habit worth
# a second look. Everything else in the S family is advisory (importing
# subprocess, calling it without shell=True, a string called "token_type")
# and is reported at medium so it cannot drown out the real findings.
RUFF_HIGH_RISK = {"S102", "S301", "S302", "S307", "S501", "S506", "S602", "S604", "S605"}


def ruff_severity(code: str) -> str:
    if code == "syntax":
        return "critical"
    # SIM and PERF have to be handled before the S family: they start with S/P
    if code.startswith(("PERF", "SIM")):
        return "low"
    if code.startswith("S"):
        if code in RUFF_HIGH_RISK:
            return "high"
        return "low" if code == "S101" else "medium"
    if code in {"F821", "F823"}:
        return "high"            # names that do not exist
    if code in {"F401", "F841", "F811"}:
        return "low"
    if code.startswith(("F", "B", "C90")):
        return "medium"
    return "low"


def ruff_code(item: Dict[str, Any]) -> str:
    code = item.get("code")
    # Syntax errors carry no code in some ruff versions and the string
    # "invalid-syntax" in others.
    if not code or code in {"invalid-syntax", "syntax-error"}:
        return "syntax"
    return code


def parse_ruff_json(raw: str) -> List[Finding]:
    """Turn `ruff check --output-format json` into Findings."""
    try:
        items = json.loads(raw) if raw.strip() else []
    except json.JSONDecodeError:
        return []
    out: List[Finding] = []
    for item in items:
        code = ruff_code(item)
        where = item.get("location") or {}
        out.append(Finding(
            rule=f"ruff:{code}",
            category=ruff_category(code),
            severity=ruff_severity(code),
            line=int(where.get("row", 1)),
            column=max(int(where.get("column", 1)) - 1, 0),
            message=item.get("message", ""),
            source="static",
        ))
    return out


def drop_equivalents(external: Iterable[Finding], builtin: Sequence[Finding],
                     window: int = 3) -> List[Finding]:
    """Drop external findings that repeat one of ours.

    The window is a few lines wide because the two sides do not always pick
    the same line for one problem: ruff flags the f-string that builds a SQL
    query, we flag the execute() call that runs it.
    """
    have = [(f.line, f.rule) for f in builtin]
    parse_failed = any(rule == "P001" for _, rule in have)
    kept = []
    for f in external:
        if f.rule == "ruff:syntax" and parse_failed:
            continue        # we already reported that the file does not parse
        twins = EQUIVALENT_RULES.get(f.rule, ())
        if any(rule in twins and abs(line - f.line) <= window for line, rule in have):
            continue
        kept.append(f)
    return kept


class ExternalTools:
    """Optional adapters for ruff, bandit and pylint.

    None of these are required. When they are installed the extra findings
    are merged in; when they are not, the built-in rules carry the load.
    Ruff is the one worth having: it is written in Rust, so it covers style,
    bug-pattern, security and complexity checks in a single call that costs
    about ten milliseconds a file. Pylint does a much heavier analysis, so it
    is opt-in rather than part of the default set.

    Each tool runs on a temporary copy of the text under review, so snippets
    that only exist in memory (the UI, the evaluation corpus) get checked too.
    """

    DEFAULT_TOOLS = ("ruff", "bandit")

    def __init__(self, enabled: bool = False, timeout: int = 60,
                 tools: Sequence[str] = DEFAULT_TOOLS, line_length: int = DEFAULT_LINE_LENGTH):
        self.enabled = enabled
        self.timeout = timeout
        self.tools = tuple(tools)
        self.line_length = line_length

    @staticmethod
    def available() -> Dict[str, bool]:
        return {name: shutil.which(name) is not None for name in ("ruff", "bandit", "pylint")}

    def installed(self) -> List[str]:
        return [t for t in self.tools if shutil.which(t)]

    def run(self, unit: "SourceUnit") -> List[Finding]:
        wanted = self.installed()
        if not self.enabled or not wanted:
            return []

        runners = {"ruff": self._ruff, "bandit": self._bandit, "pylint": self._pylint}
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "unit.py")
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(unit.text)
            if len(wanted) == 1:
                results = [runners[wanted[0]](target)]
            else:
                # the tools are separate processes, so threads are enough
                with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
                    results = list(pool.map(lambda name: runners[name](target), wanted))

        found = [f for group in results for f in group]
        for f in found:
            f.snippet = unit.line(f.line).strip()[:160]
        return found

    def _ruff(self, path: str) -> List[Finding]:
        # S101 flags every assert; ours only flags asserts on arguments
        raw = self._exec(["ruff", "check", "--select", RUFF_SELECT, "--ignore", "S101",
                          "--output-format", "json", "--exit-zero", "--no-cache",
                          "--isolated", "--line-length", str(self.line_length), path])
        return parse_ruff_json(raw)

    def _exec(self, cmd: List[str]) -> str:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            return proc.stdout
        except (subprocess.SubprocessError, OSError):
            return ""

    def _pylint(self, path: str) -> List[Finding]:
        raw = self._exec(["pylint", "--output-format=json", "--score=n", path])
        try:
            items = json.loads(raw) if raw.strip() else []
        except json.JSONDecodeError:
            return []
        severity_map = {"error": "high", "fatal": "critical", "warning": "medium",
                        "convention": "low", "refactor": "low", "info": "info"}
        out = []
        for item in items:
            out.append(Finding(
                rule=f"pylint:{item.get('symbol', item.get('message-id', '?'))}",
                category="style" if item.get("type") in {"convention", "refactor"} else "bug",
                severity=severity_map.get(item.get("type", "info"), "info"),
                line=item.get("line", 1),
                column=item.get("column", 0),
                message=item.get("message", ""),
                source="static",
            ))
        return out

    def _bandit(self, path: str) -> List[Finding]:
        raw = self._exec(["bandit", "-f", "json", "-q", "-s", "B101", path])
        try:
            data = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return []
        severity_map = {"HIGH": "critical", "MEDIUM": "high", "LOW": "medium"}
        out = []
        for item in data.get("results", []):
            out.append(Finding(
                rule=f"bandit:{item.get('test_id', '?')}",
                category="security",
                severity=severity_map.get(item.get("issue_severity", "LOW"), "medium"),
                line=item.get("line_number", 1),
                message=item.get("issue_text", ""),
                source="static",
            ))
        return out


class StaticEngine:
    """Runs every rule set over one SourceUnit."""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 external: Optional[ExternalTools] = None):
        self.config = config or {}
        self.external = external or ExternalTools(enabled=False)

    def analyse(self, unit: SourceUnit) -> Tuple[List[Finding], Dict[str, Any], List[str]]:
        findings: List[Finding] = []
        errors: List[str] = []

        if unit.parse_error:
            errors.append(f"line {unit.parse_error_line}: {unit.parse_error}")
            findings.append(Finding(
                rule="P001", category="bug", severity="critical",
                line=unit.parse_error_line or 1,
                message=f"File does not parse: {unit.parse_error}",
                suggestion="Fix the syntax error; the deeper checks cannot run until then.",
                snippet=unit.line(unit.parse_error_line or 1).strip()[:160],
            ))

        if unit.truncated:
            errors.append(
                f"file has {len(unit.lines)} lines, above the {unit.max_lines}-line limit; "
                "the LLM stage will see it in chunks"
            )

        for rules in (BugRules, SecurityRules, StyleRules, OptimizationRules):
            try:
                findings.extend(rules(unit, self.config).run())
            except RecursionError:
                errors.append(f"{rules.__name__} hit the recursion limit and was skipped")
            except Exception as exc:  # a broken rule must not kill the review
                errors.append(f"{rules.__name__} failed: {exc}")

        try:
            findings.extend(unused_name_checks(unit))
        except Exception as exc:
            errors.append(f"unused-name analysis failed: {exc}")

        findings.extend(line_level_checks(unit, self.config))

        if self.external.enabled:
            if self.external.installed():
                findings.extend(drop_equivalents(self.external.run(unit), findings))
            else:
                errors.append("external tools requested but none are installed "
                              f"(looked for {', '.join(self.external.tools)})")

        findings = apply_suppressions(findings, collect_suppressions(unit))
        metrics = collect_metrics(unit)
        return dedupe(findings), metrics, errors


def dedupe(findings: Iterable[Finding]) -> List[Finding]:
    seen: Dict[Tuple[str, int, str], Finding] = {}
    for f in findings:
        k = f.key()
        if k not in seen:
            seen[k] = f
    return sorted(seen.values(), key=lambda f: (f.line, SEVERITY_RANK.get(f.severity, 9), f.rule))


# --------------------------------------------------------------------------
# LLM inference
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a careful senior engineer reviewing Python code.

You are given source code and a list of findings from a static analyser. The
static findings are already verified -- your job is to make them useful to a
developer, not to doubt them.

For each static finding, write a one or two sentence explanation of why it
matters in this specific code. Name the variable, function or value involved;
an explanation that would fit any program is not useful. The fix should be a
concrete change, ideally the corrected line of code.

You may add findings of your own, but only for real defects a static analyser
would miss: wrong logic, off-by-one errors, wrong comparison operators,
missing cases, resources that are never closed, bad error handling, misuse of
an API. Read each docstring as the specification: a function that does not
do what its docstring says has a bug. Do not report anything you are unsure
about, and do not repeat the static findings as new ones.
Precision matters much more than recall here. If you have nothing to add,
return an empty list for new_findings.

Reply with JSON only, no prose, no markdown fence, in exactly this shape:

{
  "explanations": [
    {"line": 12, "rule": "B001", "explanation": "...", "fix": "..."}
  ],
  "new_findings": [
    {"line": 20, "category": "bug", "severity": "high", "message": "...",
     "explanation": "...", "fix": "...", "confidence": 0.8}
  ]
}

category must be one of: bug, security, style, optimization.
severity must be one of: critical, high, medium, low, info.
confidence is your own estimate between 0 and 1."""


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def require_local(url: str, allow_remote: bool = False) -> str:
    """Refuse to send code anywhere but this machine unless told otherwise.

    The brief rules out sending source to an external service without
    permission, so the rule lives in code instead of in the documentation.
    """
    host = urllib.parse.urlparse(url).hostname or ""
    if host not in LOCAL_HOSTS and not allow_remote:
        raise ValueError(
            f"{url} is not a local address, so source code would leave this machine. "
            "Pass --allow-remote if that is really intended."
        )
    return url.rstrip("/")


class LLMBackend:
    name = "base"

    def generate(self, prompt: str, system: str = "") -> str:
        raise NotImplementedError

    def warm_up(self) -> None:
        """Load weights ahead of the first request. Most backends have none."""

    def describe(self) -> str:
        return self.name


class OllamaBackend(LLMBackend):
    """Talks to a local Ollama server. Nothing leaves the machine."""

    name = "ollama"

    def __init__(self, model: str = "qwen2.5-coder:7b",
                 host: str = "http://localhost:11434", timeout: int = 180,
                 temperature: float = 0.0, allow_remote: bool = False):
        self.model = model
        self.host = require_local(host, allow_remote)
        self.timeout = timeout
        self.temperature = temperature

    def generate(self, prompt: str, system: str = "") -> str:
        import urllib.request

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": 8192},
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "")

    def describe(self) -> str:
        return f"ollama:{self.model}"


class OpenAICompatBackend(LLMBackend):
    """Any server that speaks the OpenAI chat-completions protocol.

    vLLM, llama.cpp's server, LM Studio and Ollama's /v1 endpoint all do, so
    this one class is the way to plug in a faster inference engine without
    touching the rest of the pipeline. vLLM in particular batches concurrent
    requests, which is what a multi-worker review run produces.
    """

    name = "openai"

    def __init__(self, model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
                 base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY",
                 timeout: int = 180, temperature: float = 0.0, max_tokens: int = 1024,
                 allow_remote: bool = False):
        self.model = model
        self.base_url = require_local(base_url, allow_remote)
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected response from {self.base_url}: {str(body)[:200]}") from exc

    def describe(self) -> str:
        return f"openai-compatible:{self.model}"


class TransformersBackend(LLMBackend):
    """Runs an open-weights code model locally through transformers.

    Default is a small Qwen2.5-Coder because it fits a free GPU and still
    follows a JSON instruction reliably enough. quantize="4bit" squeezes the
    7B model into roughly 5 GB so it fits a single T4.
    """

    name = "hf"

    def __init__(self, model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                 max_new_tokens: int = 1536, temperature: float = 0.0,
                 device: Optional[str] = None, quantize: Optional[str] = None):
        if quantize not in (None, "4bit"):
            raise ValueError(f"unsupported quantize option: {quantize!r} (use '4bit')")
        self.model_id = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.quantize = quantize
        self._pipe = None

    def warm_up(self) -> None:
        self._load()

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are needed for the hf backend: "
                "pip install transformers torch accelerate"
            ) from exc

        tok = AutoTokenizer.from_pretrained(self.model_id)

        if self.quantize == "4bit":
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit loading needs a CUDA GPU")
            from transformers import BitsAndBytesConfig
            config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            # device 0 is whichever GPU this process was pinned to
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, quantization_config=config, device_map={"": 0})
            device = "cuda"
        else:
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype)
            model.to(device)

        model.eval()
        self._pipe = (tok, model, device)
        return self._pipe

    def generate(self, prompt: str, system: str = "") -> str:
        import torch

        tok, model, device = self._load()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tok.eos_token_id,
                **self._sampling(),
            )
        generated = out[0][inputs["input_ids"].shape[-1]:]
        return tok.decode(generated, skip_special_tokens=True)

    def _sampling(self) -> dict:
        # Greedy decoding by default, so the same input gives the same review on
        # every run. The neutral values override the model's own sampling defaults,
        # which would otherwise trigger a warning when sampling is off.
        if self.temperature > 0:
            return {"do_sample": True, "temperature": self.temperature}
        return {"do_sample": False, "temperature": 1.0, "top_p": 1.0, "top_k": 50}

    def describe(self) -> str:
        suffix = f" ({self.quantize})" if self.quantize else ""
        return f"hf:{self.model_id}{suffix}"


class OfflineBackend(LLMBackend):
    """Deterministic stand-in used when no model is available.

    It does not invent findings. It only turns the static rule's own
    rationale into the same JSON shape a model would return, so the rest of
    the pipeline, the CLI and the evaluation all still run end to end with
    no network and no GPU. Anything it produces is labelled 'offline' in the
    output so results are never passed off as model output.
    """

    name = "offline"

    def generate(self, prompt: str, system: str = "") -> str:
        lines = []
        for match in re.finditer(r'"line":\s*(\d+),\s*"rule":\s*"([^"]+)",'
                                 r'\s*"message":\s*"((?:[^"\\]|\\.)*)"', prompt):
            line, rule, message = match.group(1), match.group(2), match.group(3)
            lines.append({
                "line": int(line),
                "rule": rule,
                "explanation": f"{message}. Flagged by rule {rule}.",
                "fix": "",
            })
        return json.dumps({"explanations": lines, "new_findings": []})

    def describe(self) -> str:
        return "offline (no model)"


def make_backend(kind: str, model: Optional[str] = None, base_url: Optional[str] = None,
                 allow_remote: bool = False, quantize: Optional[str] = None,
                 **kwargs) -> Optional[LLMBackend]:
    """Build a backend from plain values.

    Everything here is a string, a bool or None on purpose: worker processes
    receive these values and build their own backend, because a loaded model
    cannot be pickled and sent across.
    """
    kind = (kind or "none").lower()
    if kind in {"none", "off", ""}:
        return None
    if kind == "ollama":
        extra = {"host": base_url} if base_url else {}
        return OllamaBackend(model=model or "qwen2.5-coder:7b",
                             allow_remote=allow_remote, **extra, **kwargs)
    if kind == "openai":
        extra = {"base_url": base_url} if base_url else {}
        return OpenAICompatBackend(model=model or "Qwen/Qwen2.5-Coder-7B-Instruct",
                                   allow_remote=allow_remote, **extra, **kwargs)
    if kind in {"hf", "transformers"}:
        return TransformersBackend(model=model or "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                                   quantize=quantize, **kwargs)
    if kind == "offline":
        return OfflineBackend()
    raise ValueError(f"unknown LLM backend: {kind}")


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of a model response.

    Small models wrap JSON in prose, in a markdown fence, or produce a
    trailing comma. We try progressively more forgiving strategies rather
    than failing the whole review over formatting.
    """
    if not text:
        return None

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        return None
    return None


class LLMReviewer:
    """Second stage: explain the static findings and, carefully, extend them."""

    def __init__(self, backend: LLMBackend, min_confidence: float = 0.6,
                 max_findings_in_prompt: int = 25, window: int = 200):
        self.backend = backend
        self.min_confidence = min_confidence
        self.max_findings_in_prompt = max_findings_in_prompt
        # A small model reads 200 lines far more reliably than 500, and a
        # shorter window keeps its answer short enough not to be cut off.
        self.window = window
        self.last_candidates: List[Finding] = []

    def build_prompt(self, unit: SourceUnit, findings: Sequence[Finding],
                     start: int, end: int) -> str:
        in_window = [f for f in findings if start <= f.line <= end]
        payload = [
            {"line": f.line, "rule": f.rule, "message": f.message, "category": f.category}
            for f in in_window[:self.max_findings_in_prompt]
        ]
        return (
            f"File: {unit.path}\n"
            f"Lines {start} to {end}.\n\n"
            "Source:\n"
            f"{unit.numbered(start, end)}\n\n"
            "Static analyser findings (already verified):\n"
            f"{json.dumps(payload, indent=1)}\n\n"
            "Return the JSON described in the instructions."
        )

    def review(self, unit: SourceUnit, findings: List[Finding]) -> Tuple[List[Finding], List[str]]:
        notes: List[str] = []
        by_position = {(f.line, f.rule): f for f in findings}
        added: List[Finding] = []
        self.last_candidates = []

        for start, end in unit.chunks(self.window):
            prompt = self.build_prompt(unit, findings, start, end)
            try:
                raw = self.backend.generate(prompt, system=SYSTEM_PROMPT)
            except Exception as exc:
                notes.append(f"LLM call failed for lines {start}-{end}: {exc}")
                continue

            data = extract_json(raw)
            if data is None:
                notes.append(f"Could not parse model output for lines {start}-{end}; "
                             "static findings kept as-is")
                continue

            for item in data.get("explanations", []) or []:
                try:
                    line = int(item.get("line", 0))
                except (TypeError, ValueError):
                    continue
                target = by_position.get((line, item.get("rule", "")))
                if target is None:
                    # models drift by a line; accept the nearest match on that rule
                    candidates = [f for f in findings
                                  if f.rule == item.get("rule") and abs(f.line - line) <= 2]
                    target = candidates[0] if candidates else None
                if target is None:
                    continue
                explanation = str(item.get("explanation", "")).strip()
                fix = str(item.get("fix", "")).strip()
                if explanation:
                    target.rationale = explanation
                if fix:
                    target.suggestion = fix
                target.source = "static+llm"

            for item in data.get("new_findings", []) or []:
                finding = self._coerce_new_finding(item, unit, start, end)
                if finding is None:
                    continue
                if any(f.line == finding.line and f.category == finding.category
                       for f in findings):
                    continue  # the static layer already said this
                self.last_candidates.append(finding)
                if finding.confidence >= self.min_confidence:
                    added.append(finding)

        return added, notes

    def _coerce_new_finding(self, item: Dict[str, Any], unit: SourceUnit,
                            start: int, end: int) -> Optional[Finding]:
        try:
            line = int(item.get("line", 0))
        except (TypeError, ValueError):
            return None
        if not (start <= line <= end):
            return None

        category = str(item.get("category", "")).lower().strip()
        if category not in CATEGORIES:
            return None

        severity = str(item.get("severity", "medium")).lower().strip()
        if severity not in SEVERITY_RANK:
            severity = "medium"

        try:
            confidence = min(max(float(item.get("confidence", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            confidence = 0.5

        message = str(item.get("message", "")).strip()
        if not message:
            return None

        return Finding(
            rule="LLM001",
            category=category,
            severity=severity,
            line=line,
            message=message,
            suggestion=str(item.get("fix", "")).strip(),
            rationale=str(item.get("explanation", "")).strip(),
            confidence=round(confidence, 2),
            source="llm",
            snippet=unit.line(line).strip()[:160],
        )


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

class CodeReviewAssistant:
    def __init__(self, backend: Optional[LLMBackend] = None,
                 config: Optional[Dict[str, Any]] = None,
                 use_external_tools: bool = False,
                 min_confidence: float = 0.6,
                 external_tools: Sequence[str] = ExternalTools.DEFAULT_TOOLS):
        self.config = config or {}
        self.engine = StaticEngine(
            self.config,
            ExternalTools(
                enabled=use_external_tools,
                tools=external_tools,
                line_length=self.config.get("max_line_length", DEFAULT_LINE_LENGTH),
            ),
        )
        self.reviewer = LLMReviewer(backend, min_confidence) if backend else None
        self.backend = backend

    def review_text(self, text: str, path: str = "<snippet>") -> FileReport:
        unit = SourceUnit(text, path, max_lines=self.config.get("max_lines", DEFAULT_MAX_LINES))
        report = FileReport(path=path)

        t0 = time.perf_counter()
        findings, metrics, errors = self.engine.analyse(unit)
        report.static_seconds = time.perf_counter() - t0
        report.metrics = metrics
        report.errors = errors

        if self.reviewer is not None and unit.is_parseable:
            t1 = time.perf_counter()
            added, notes = self.reviewer.review(unit, findings)
            report.llm_seconds = time.perf_counter() - t1
            report.llm_candidates = list(self.reviewer.last_candidates)
            findings.extend(added)
            report.errors.extend(notes)
        elif self.reviewer is not None:
            report.errors.append("LLM stage skipped: the file does not parse")

        report.findings = self._postprocess(findings)
        return report

    def review_file(self, path: str | Path) -> FileReport:
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            report = FileReport(path=str(p))
            report.errors.append(f"could not read file: {exc}")
            return report
        return self.review_text(text, str(p))

    def review_paths(self, paths: Sequence[str | Path],
                     exclude: Sequence[str] = ()) -> List[FileReport]:
        reports = []
        for file_path in iter_python_files(paths, exclude):
            reports.append(self.review_file(file_path))
        return reports

    def _postprocess(self, findings: List[Finding]) -> List[Finding]:
        findings = dedupe(findings)

        min_sev = self.config.get("min_severity")
        if min_sev:
            cutoff = SEVERITY_RANK.get(min_sev, len(SEVERITY_ORDER))
            findings = [f for f in findings if SEVERITY_RANK.get(f.severity, 9) <= cutoff]

        wanted = self.config.get("categories")
        if wanted:
            findings = [f for f in findings if f.category in wanted]

        limit = self.config.get("max_findings")
        if limit:
            findings = sorted(findings, key=lambda f: (SEVERITY_RANK.get(f.severity, 9), f.line))
            findings = findings[:limit]

        return sorted(findings, key=lambda f: (f.line, SEVERITY_RANK.get(f.severity, 9)))


def iter_python_files(paths: Sequence[str | Path],
                      exclude: Sequence[str] = ()) -> Iterator[Path]:
    skip_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules",
                 ".mypy_cache", ".pytest_cache", "build", "dist", ".ipynb_checkpoints"}
    patterns = [re.compile(p) for p in exclude]

    for raw in paths:
        p = Path(raw)
        if p.is_file():
            if not any(pat.search(str(p)) for pat in patterns):
                yield p
        elif p.is_dir():
            for found in sorted(p.rglob("*.py")):
                if any(part in skip_dirs for part in found.parts):
                    continue
                if any(pat.search(str(found)) for pat in patterns):
                    continue
                yield found


# --------------------------------------------------------------------------
# parallel execution
# --------------------------------------------------------------------------
#
# Two kinds of work run through here and they want different things. Static
# analysis is CPU-bound Python, so more processes help up to the core count.
# Model inference is GPU-bound, so the useful number of processes is the
# number of GPUs, with one copy of the model pinned to each. The same pool
# handles both: every worker builds its own assistant (and, for a local
# model, loads its own weights) when it starts.
#
# Workers are started with "spawn", never "fork". A forked child inherits the
# parent's CUDA state, which breaks as soon as it touches the GPU.

_WORKER: Dict[str, Any] = {}


def detect_gpus() -> List[int]:
    """GPU indices, found through nvidia-smi.

    Asking torch would initialise CUDA in this process, and a parent that has
    touched CUDA can leave its workers unable to use the same device.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [int(tok) for tok in proc.stdout.split() if tok.isdigit()]


def parse_gpus(text: Optional[str]) -> List[int]:
    """'auto' -> whatever nvidia-smi reports, '0,1' -> [0, 1], '' or 'none' -> []."""
    if not text or text.lower() == "none":
        return []
    if text.lower() == "auto":
        return detect_gpus()
    return [int(part) for part in text.split(",") if part.strip()]


def resolve_workers(requested: int, backend_kind: str, gpus: Sequence[int],
                    cpu_bound: bool = False) -> int:
    """Pick a worker count when the user did not.

    A local model gets one worker per GPU, and a hosted or absent one stays at
    a single process. A purely CPU-bound job (the benchmark) gets every core.
    """
    if requested and requested > 0:
        return requested
    if (backend_kind or "").lower() in {"hf", "transformers"}:
        return max(len(gpus), 1)
    return (os.cpu_count() or 1) if cpu_bound else 1


def _init_worker(spec: Optional[Dict[str, Any]], config: Dict[str, Any], gpu_slots,
                 use_external: bool, tools: Sequence[str], min_confidence: float) -> None:
    gpu = None
    if gpu_slots is not None:
        try:
            # blocking get: the parent's queue feeder may not have flushed yet
            gpu = gpu_slots.get(timeout=30)
        except queue.Empty:
            gpu = None
    if gpu is not None:
        # has to be set before torch is imported anywhere in this process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    _WORKER["gpu"] = gpu
    _WORKER["error"] = None
    try:
        backend = make_backend(**spec) if spec else None
        if backend is not None:
            backend.warm_up()
        _WORKER["assistant"] = CodeReviewAssistant(
            backend=backend, config=config, use_external_tools=use_external,
            external_tools=tools, min_confidence=min_confidence,
        )
    except Exception as exc:
        # An exception escaping an initializer only produces a bare
        # BrokenProcessPool in the parent, so keep the reason and raise it
        # from the first task instead.
        _WORKER["error"] = f"{type(exc).__name__}: {exc}"


def _worker_assistant() -> "CodeReviewAssistant":
    if _WORKER.get("error"):
        raise RuntimeError(f"worker could not start: {_WORKER['error']}")
    return _WORKER["assistant"]


def _task_review_file(path: str) -> FileReport:
    return _worker_assistant().review_file(path)


def _task_score_case(index: int, tolerance: int) -> Dict[str, Any]:
    return score_case(_worker_assistant(), TEST_CORPUS[index], tolerance)


def _task_bench_file(path: str) -> Dict[str, Any]:
    return bench_file(_worker_assistant(), path)


def _task_worker_info(_: int) -> Dict[str, Any]:
    time.sleep(0.3)      # long enough that every worker gets a turn
    return {"pid": os.getpid(), "gpu": _WORKER.get("gpu"),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}


def run_parallel(task, items: Sequence[Any], workers: int,
                 spec: Optional[Dict[str, Any]] = None,
                 config: Optional[Dict[str, Any]] = None,
                 gpus: Sequence[int] = (), use_external: bool = False,
                 tools: Sequence[str] = ExternalTools.DEFAULT_TOOLS,
                 min_confidence: float = 0.6, chunksize: int = 1) -> List[Any]:
    """Map `task` over `items` in `workers` processes; results keep input order."""
    ctx = multiprocessing.get_context("spawn")
    slots = None
    if gpus:
        slots = ctx.Queue()
        for i in range(workers):
            slots.put(gpus[i % len(gpus)])      # round-robin if workers > GPUs

    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_init_worker,
        initargs=(spec, config or {}, slots, use_external, tuple(tools), min_confidence),
    ) as pool:
        return list(pool.map(task, items, chunksize=chunksize))


def review_paths_parallel(paths: Sequence[str], exclude: Sequence[str], workers: int,
                          spec: Optional[Dict[str, Any]], config: Dict[str, Any],
                          gpus: Sequence[int] = (), use_external: bool = False,
                          tools: Sequence[str] = ExternalTools.DEFAULT_TOOLS,
                          min_confidence: float = 0.6) -> List[FileReport]:
    files = [str(p) for p in iter_python_files(paths, exclude)]
    return run_parallel(_task_review_file, files, workers, spec, config, gpus,
                        use_external, tools, min_confidence)


# --------------------------------------------------------------------------
# output formats
# --------------------------------------------------------------------------

SEVERITY_LABEL = {
    "critical": "CRIT",
    "high": "HIGH",
    "medium": "MED ",
    "low": "LOW ",
    "info": "INFO",
}

ANSI = {
    "critical": "\033[1;31m",
    "high": "\033[31m",
    "medium": "\033[33m",
    "low": "\033[36m",
    "info": "\033[90m",
    "reset": "\033[0m",
    "bold": "\033[1m",
}


def _colour(text: str, key: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{ANSI.get(key, '')}{text}{ANSI['reset']}"


def format_text(reports: Sequence[FileReport], colour: bool = True,
                show_metrics: bool = True) -> str:
    out: List[str] = []
    total = 0

    for report in reports:
        out.append(_colour(f"\n{report.path}", "bold", colour))
        out.append("-" * max(len(report.path), 12))

        if report.errors:
            for err in report.errors:
                out.append(f"  ! {err}")

        if not report.findings:
            out.append("  no issues found")
        for f in report.findings:
            total += 1
            tag = _colour(SEVERITY_LABEL.get(f.severity, "????"), f.severity, colour)
            marker = "*" if f.source != "static" else " "
            out.append(f"  {tag}{marker} {f.line:>4}  [{f.category}/{f.rule}] {f.message}")
            if f.snippet:
                out.append(f"              | {f.snippet}")
            if f.rationale:
                out.append(f"          why: {f.rationale}")
            if f.suggestion:
                out.append(f"          fix: {f.suggestion}")

        if show_metrics and report.metrics:
            m = report.metrics
            bits = [f"{m.get('total_lines', 0)} lines"]
            if m.get("functions") is not None:
                bits.append(f"{m['functions']} functions")
            if m.get("max_complexity"):
                bits.append(f"max complexity {m['max_complexity']}")
            if m.get("max_nesting"):
                bits.append(f"max nesting {m['max_nesting']}")
            out.append(f"  metrics: {', '.join(bits)}")
        out.append(f"  time: {report.total_seconds:.3f}s "
                   f"(static {report.static_seconds:.3f}s, llm {report.llm_seconds:.3f}s)")

    by_sev: Dict[str, int] = {}
    for report in reports:
        for f in report.findings:
            by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
    summary = ", ".join(f"{by_sev[s]} {s}" for s in SEVERITY_ORDER if by_sev.get(s))
    out.append("")
    out.append(_colour(f"{total} findings across {len(reports)} file(s)"
                       f"{': ' + summary if summary else ''}", "bold", colour))
    out.append("(* = enriched or produced by the LLM stage)")
    return "\n".join(out)


def format_markdown(reports: Sequence[FileReport]) -> str:
    out = ["# Code review", ""]
    total = sum(len(r.findings) for r in reports)
    out.append(f"{total} findings across {len(reports)} file(s).")
    out.append("")

    for report in reports:
        out.append(f"## `{report.path}`")
        out.append("")
        if report.errors:
            for err in report.errors:
                out.append(f"> {err}")
            out.append("")
        if not report.findings:
            out.append("No issues found.")
            out.append("")
            continue

        out.append("| Line | Severity | Category | Rule | Issue |")
        out.append("| ---: | --- | --- | --- | --- |")
        for f in report.findings:
            msg = f.message.replace("|", "\\|")
            out.append(f"| {f.line} | {f.severity} | {f.category} | `{f.rule}` | {msg} |")
        out.append("")

        detailed = [f for f in report.findings if f.suggestion or f.rationale]
        if detailed:
            out.append("### Details")
            out.append("")
            for f in detailed:
                out.append(f"**Line {f.line}: {f.message}**")
                out.append("")
                if f.snippet:
                    out.append(f"```python\n{f.snippet}\n```")
                if f.rationale:
                    out.append(f"*Why:* {f.rationale}")
                    out.append("")
                if f.suggestion:
                    out.append(f"*Fix:* {f.suggestion}")
                    out.append("")

        if report.metrics:
            out.append("### Metrics")
            out.append("")
            for k, v in report.metrics.items():
                out.append(f"- **{k}**: {v}")
            out.append("")
    return "\n".join(out)


def format_json(reports: Sequence[FileReport]) -> str:
    payload = {
        "version": __version__,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": [r.to_dict() for r in reports],
        "summary": {
            "files": len(reports),
            "findings": sum(len(r.findings) for r in reports),
            "by_severity": _aggregate(reports, lambda f: f.severity),
            "by_category": _aggregate(reports, lambda f: f.category),
        },
    }
    return json.dumps(payload, indent=2)


def format_sarif(reports: Sequence[FileReport]) -> str:
    """SARIF 2.1.0, which is what GitHub code scanning ingests."""
    level_map = {"critical": "error", "high": "error", "medium": "warning",
                 "low": "note", "info": "note"}
    rules: Dict[str, Dict[str, Any]] = {}
    results = []

    for report in reports:
        for f in report.findings:
            rules.setdefault(f.rule, {
                "id": f.rule,
                "shortDescription": {"text": f.message[:120]},
                "properties": {"category": f.category},
            })
            results.append({
                "ruleId": f.rule,
                "level": level_map.get(f.severity, "note"),
                "message": {"text": f.message},
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": report.path},
                        "region": {"startLine": max(f.line, 1)},
                    }
                }],
            })

    return json.dumps({
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {
                "name": "Code Review Assistant",
                "version": __version__,
                "rules": list(rules.values()),
            }},
            "results": results,
        }],
    }, indent=2)


def _aggregate(reports: Sequence[FileReport], key) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in reports:
        for f in r.findings:
            k = key(f)
            out[k] = out.get(k, 0) + 1
    return out


# --------------------------------------------------------------------------
# labelled test corpus
# --------------------------------------------------------------------------
#
# Ground truth is written into the snippets themselves with a trailing
# "#@ category" marker on the offending line. Keeping the label next to the
# code means the corpus cannot drift out of sync with a separate answer key,
# which is the usual way these evaluation sets rot.

MARKER = re.compile(r"#@\s*(bug|security|style|optimization)\b")
MARKER_COMMENT = re.compile(r"\s*#@\s*(bug|security|style|optimization)\b.*$")


@dataclass
class TestCase:
    name: str
    code: str
    note: str = ""
    # "static": a defect the rule engine is built to catch.
    # "llm": a logic error no rule can see, there to measure what the model adds.
    kind: str = "static"
    # For "llm" cases: patterns, any one of which must appear in the model's
    # explanation for the finding to count. Matching the line alone credited
    # a model that pointed at the right line for the wrong reason.
    keywords: Tuple[str, ...] = ()

    def source(self) -> str:
        """The code as the reviewer sees it: the answer-key markers removed.

        The markers are comments on the very line that holds the defect, so
        leaving them in would tell the model where to look.
        """
        return "\n".join(MARKER_COMMENT.sub("", line) for line in self.code.split("\n"))

    def expected(self) -> List[Tuple[int, str]]:
        out = []
        for i, line in enumerate(self.code.split("\n"), start=1):
            m = MARKER.search(line)
            if m:
                out.append((i, m.group(1)))
        return out


TEST_CORPUS: List[TestCase] = [
    TestCase(
        name="mutable_default",
        note="classic shared-state bug",
        code='''"""Cart helpers."""


def add_item(item, basket=[]):  #@ bug
    basket.append(item)
    return basket


def add_tag(tag, tags={}):  #@ bug
    tags[tag] = True
    return tags
''',
    ),
    TestCase(
        name="swallowed_exception",
        note="error handling",
        code='''"""Config loading."""
import json


def load(path):
    """Read a JSON config, returning None when it cannot be read."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except:  #@ bug
        pass
    return None
''',
    ),
    TestCase(
        name="sql_injection",
        note="injection via f-string",
        code='''"""User lookup."""


def find_user(cursor, username):
    """Return the row for a username."""
    query = f"SELECT * FROM users WHERE name = '{username}'"
    cursor.execute(query)  #@ security
    return cursor.fetchone()


def find_by_id(cursor, uid):
    """Return the row for an id."""
    cursor.execute("SELECT * FROM users WHERE id = " + str(uid))  #@ security
    return cursor.fetchone()
''',
    ),
    TestCase(
        name="hardcoded_secret",
        note="credential in source",
        code='''"""API client."""
import hashlib

API_KEY = "sk-live-9f8a7b6c5d4e3f2a1b0c"  #@ security


def digest(password):
    """Hash a password."""
    return hashlib.md5(password.encode()).hexdigest()  #@ security
''',
    ),
    TestCase(
        name="shell_and_eval",
        note="command execution",
        code='''"""Deploy helpers."""
import os
import subprocess


def run(cmd):
    """Run a command."""
    os.system(cmd)  #@ security


def run2(cmd):
    """Run a command through subprocess."""
    subprocess.run(cmd, shell=True)  #@ security


def compute(expression):
    """Evaluate an arithmetic expression."""
    return eval(expression)  #@ security
''',
    ),
    TestCase(
        name="pickle_and_yaml",
        note="deserialisation",
        code='''"""Cache layer."""
import pickle

import yaml


def load_cache(blob):
    """Restore a cached object."""
    return pickle.loads(blob)  #@ security


def load_settings(text):
    """Parse a settings document."""
    return yaml.load(text)  #@ security
''',
    ),
    TestCase(
        name="string_concat_loop",
        note="quadratic string building",
        code='''"""Report rendering."""


def render(rows):
    """Render rows into one string."""
    out = ""
    for row in rows:
        out += f"{row}\\n"  #@ optimization
    return out
''',
    ),
    TestCase(
        name="range_len",
        note="indexing instead of enumerate",
        code='''"""Scoring."""


def label(items):
    """Pair each item with its index."""
    result = []
    for i in range(len(items)):  #@ optimization
        result.append((i, items[i]))
    return result
''',
    ),
    TestCase(
        name="redundant_keys",
        note="iteration and membership",
        code='''"""Lookup helpers."""


def names(mapping):
    """List the keys of a mapping."""
    out = []
    for key in mapping.keys():  #@ optimization
        out.append(key)
    return out


def is_weekend(day):
    """Say whether a day name is a weekend."""
    return day in ["sat", "sun", "saturday", "sunday", "weekend"]  #@ optimization
''',
    ),
    TestCase(
        name="comprehension_in_sum",
        note="list built only to be consumed",
        code='''"""Totals."""


def total(values):
    """Sum the positive values."""
    return sum([v for v in values if v > 0])  #@ optimization
''',
    ),
    TestCase(
        name="style_naming",
        note="PEP 8 naming",
        code='''"""Geometry."""


def CalculateArea(width, height):  #@ style
    """Return the area of a rectangle."""
    return width * height


class user_record:  #@ style
    """A user."""

    def __init__(self, name):
        self.name = name
''',
    ),
    TestCase(
        name="style_wildcard_and_lambda",
        note="imports and lambdas",
        code='''"""Maths helpers."""
from math import *  #@ style

square = lambda x: x * x  #@ style


def area(radius):
    """Area of a circle."""
    return pi * square(radius)
''',
    ),
    TestCase(
        name="style_long_line",
        note="line length",
        code='''"""Message building."""


def describe(user, role, team, project, status):
    """Build a one-line description."""
    return "user " + user + " with role " + role + " on team " + team + " working on " + project + " is " + status  #@ style
''',
    ),
    TestCase(
        name="style_shadow_builtin",
        note="shadowing",
        code='''"""Filtering."""


def keep_positive(values):
    """Keep positive values."""
    list = []  #@ style
    for v in values:
        if v > 0:
            list.append(v)
    return list
''',
    ),
    TestCase(
        name="none_comparison",
        note="identity vs equality",
        code='''"""Defaults."""


def resolve(value, fallback):
    """Return value unless it is missing."""
    if value == None:  #@ bug
        return fallback
    return value


def enabled(flag):
    """Say whether a flag is on."""
    return flag == True  #@ bug
''',
    ),
    TestCase(
        name="mutation_while_iterating",
        note="container modified mid-loop",
        code='''"""Pruning."""


def drop_negatives(values):
    """Remove negative values in place."""
    for v in values:
        if v < 0:
            values.remove(v)  #@ bug
    return values
''',
    ),
    TestCase(
        name="duplicate_dict_key",
        note="silently lost value",
        code='''"""Settings."""

CONFIG = {
    "retries": 3,
    "timeout": 30,
    "retries": 5,  #@ bug
}
''',
    ),
    TestCase(
        name="inconsistent_return",
        note="sometimes returns None",
        code='''"""Parsing."""


def parse_port(text):
    """Return the port number in text."""
    if text.isdigit():
        return int(text)
    return  #@ bug
''',
    ),
    TestCase(
        name="float_equality",
        note="exact float compare",
        code='''"""Billing."""


def is_free(amount):
    """Say whether an amount is zero."""
    return amount == 0.0  #@ bug
''',
    ),
    TestCase(
        name="return_in_finally",
        note="exception swallowed",
        code='''"""Transactions."""


def commit(conn):
    """Commit, always closing the connection."""
    try:
        conn.commit()
        return True
    finally:
        conn.close()
        return False  #@ bug
''',
    ),
    TestCase(
        name="unreachable_code",
        note="dead statements",
        code='''"""Validation."""


def check(value):
    """Validate a value."""
    if value < 0:
        raise ValueError("negative")
        print("never runs")  #@ bug
    return value
''',
    ),
    TestCase(
        name="assert_tuple",
        note="always-true assert",
        code='''"""Checks."""


def validate(items):
    """Validate a list."""
    assert (len(items) > 0, "items must not be empty")  #@ bug
    return items
''',
    ),
    TestCase(
        name="unused_import_and_var",
        note="dead names",
        code='''"""Reporting."""
import os
import sys  #@ style


def summarise(rows):
    """Summarise rows."""
    count = len(rows)
    unused_total = sum(rows)  #@ bug
    return os.linesep.join(str(count) for _ in range(1))
''',
    ),
    TestCase(
        name="complex_function",
        note="high branching",
        code='''"""Grading."""


def grade(score, bonus, penalty, attended, late, retake):  #@ style
    """Work out a grade."""
    total = score
    if bonus:
        total += 5
    if penalty:
        total -= 5
    if attended:
        total += 2
    if late:
        total -= 2
    if retake:
        total -= 10
    if total > 90:
        return "A"
    if total > 80:
        return "B"
    if total > 70:
        return "C"
    if total > 60:
        return "D"
    return "F"
''',
    ),
    # Logic errors that no rule in this file can see. They are what the model
    # is for, so they are scored separately: a static-only run should miss
    # all of them, and any it finds come from the LLM stage.
    TestCase(
        name="off_by_one_sum",
        kind="llm",
        keywords=(r"off[- ]by[- ]one", r"\bfirst (price|item|element)", r"range\(1", r"skip", r"index 0", r"start\w* (at|from) 1",),
        note="loop skips the first price",
        code='''"""Invoice totals."""


def invoice_total(prices):
    """Add up every price on the invoice."""
    total = 0
    for i in range(1, len(prices)):  #@ bug
        total += prices[i]
    return total
''',
    ),
    TestCase(
        name="wrong_boundary",
        kind="llm",
        keywords=(r">=", r"\b18\b", r"greater than or equal", r"or over\b", r"boundary",),
        note="> where >= was meant",
        code='''"""Eligibility."""


def can_vote(age):
    """Return True for anyone aged 18 or over."""
    return age > 18  #@ bug
''',
    ),
    TestCase(
        name="empty_average",
        kind="llm",
        keywords=(r"\bzero\b", r"\bempty\b", r"ZeroDivision", r"len\(values\) ?== ?0",),
        note="divides by zero on an empty list",
        code='''"""Statistics."""


def average(values):
    """Mean of the values, or 0.0 when there are none."""
    return sum(values) / len(values)  #@ bug
''',
    ),
    TestCase(
        name="unclosed_file",
        kind="llm",
        keywords=(r"\bclos", r"\bwith open\b", r"context manager", r"\bleak",),
        note="file handle is never closed",
        code='''"""Config reading."""


def read_config(path):
    """Return the text of a config file."""
    handle = open(path)  #@ bug
    text = handle.read()
    return text
''',
    ),
    TestCase(
        name="sign_of_zero",
        kind="llm",
        keywords=(r"\bzero\b", r"x ?== ?0", r"<= ?0", r"(equal to|for|is) 0\b",),
        note="zero is treated as negative",
        code='''"""Number helpers."""


def sign(x):
    """Return 1, -1 or 0 depending on the sign of x."""
    if x > 0:
        return 1
    return -1  #@ bug
''',
    ),
    TestCase(
        name="early_return_search",
        kind="llm",
        keywords=(r"inside the loop", r"outside the loop", r"after the loop", r"first (user|item|element|iteration)",),
        note="gives up after the first user",
        code='''"""User lookup."""


def find_user(users, user_id):
    """Return the user with this id, or None if nobody matches."""
    for user in users:
        if user["id"] == user_id:
            return user
        return None  #@ bug
''',
    ),
    TestCase(
        name="swapped_transfer",
        kind="llm",
        keywords=(r"swap", r"revers", r"wrong (account|direction|way)", r"source\.withdraw", r"target\.deposit", r"withdraw\w* from (the )?source",),
        note="money moves the wrong way",
        code='''"""Banking."""


def transfer(source, target, amount):
    """Move amount from source to target."""
    target.withdraw(amount)  #@ bug
    source.deposit(amount)
''',
    ),
    TestCase(
        name="missing_increment",
        kind="llm",
        keywords=(r"increment", r"never (ends|end|terminat|stops|changes|updat)", r"infinite", r"i \+= 1", r"i = i \+ 1",),
        note="the loop never ends",
        code='''"""Counting."""


def count_up(limit):
    """Return the numbers from 0 up to limit, exclusive."""
    numbers = []
    i = 0
    while i < limit:  #@ bug
        numbers.append(i)
    return numbers
''',
    ),
    TestCase(
        name="floor_division_share",
        kind="llm",
        keywords=(r"floor", r"//", r"true division", r"truncat", r"\bdecimal", r"fraction", r"paise", r"\bround",),
        note="// drops the paise",
        code='''"""Bill splitting."""


def share(total, people):
    """Each person's exact share of the bill, paise included."""
    return total // people  #@ bug
''',
    ),
    TestCase(
        name="latest_zero",
        kind="llm",
        keywords=(r"\bn ?(==|is|equals) ?(0|zero)\b", r"\bn = 0\b", r"\bzero\b", r"-0\b", r"when n is 0",),
        note="n == 0 returns every post",
        code='''"""Feed helpers."""


def latest(posts, n):
    """Return the n most recent posts; an empty list when n is 0."""
    return posts[-n:]  #@ bug
''',
    ),
    # Negative controls: clean code that must produce no findings in the
    # categories we score. Without these, precision is meaningless.
    TestCase(
        name="clean_simple",
        note="negative control",
        code='''"""Temperature conversion."""


def to_celsius(fahrenheit):
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5 / 9


def to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32
''',
    ),
    TestCase(
        name="clean_class",
        note="negative control",
        code='''"""An account balance."""


class Account:
    """A bank account with a running balance."""

    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        """Add an amount to the balance."""
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        """Take an amount off the balance."""
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount
        return self.balance
''',
    ),
    TestCase(
        name="clean_safe_db",
        note="negative control, parameterised SQL",
        code='''"""User lookup done properly."""
import secrets


def find_user(cursor, username):
    """Return the row for a username."""
    cursor.execute("SELECT * FROM users WHERE name = %s", (username,))
    return cursor.fetchone()


def make_token():
    """Return a fresh session token."""
    return secrets.token_urlsafe(32)
''',
    ),
    TestCase(
        name="broken_syntax",
        note="must degrade gracefully, not crash",
        code='''"""Incomplete file."""


def unfinished(a, b
    return a + b
''',
    ),
    TestCase(
        name="empty_file",
        note="edge case",
        code="",
    ),
    TestCase(
        name="comments_only",
        note="edge case",
        code="# just a note\n# and another\n",
    ),
]


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def match_findings(predicted: Sequence[Finding], expected: Sequence[Tuple[int, str]],
                   tolerance: int = 1) -> Tuple[int, int, int, List[Tuple[int, str]],
                                                List[Finding]]:
    """Greedy one-to-one match between predictions and ground truth.

    A prediction counts as correct when the category matches and it lands
    within `tolerance` lines of the labelled line. Each label can only be
    claimed once, so three duplicate predictions on one line still produce
    one true positive and two false positives.
    """
    remaining = list(expected)
    tp = 0
    false_positives: List[Finding] = []

    ordered = sorted(predicted, key=lambda f: SEVERITY_RANK.get(f.severity, 9))
    for f in ordered:
        hit = None
        for cand in remaining:
            if cand[1] == f.category and abs(cand[0] - f.line) <= tolerance:
                hit = cand
                break
        if hit is not None:
            remaining.remove(hit)
            tp += 1
        else:
            false_positives.append(f)

    fp = len(false_positives)
    fn = len(remaining)
    return tp, fp, fn, remaining, false_positives


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn,
    }


def mentions_code(finding: Finding) -> bool:
    """Does an explanation talk about this code, or about code in general?

    A crude proxy: it counts as specific when it names an identifier from the
    flagged line. "Refactor the function" fails; "split loads() so the
    object_hook branch lives in its own helper" passes. It is no substitute
    for a person rating the advice, but it is cheap and it is repeatable.
    """
    names = {w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", finding.snippet)
             if not keyword.iskeyword(w) and w not in {"self", "True", "False", "None"}}
    text = f"{finding.rationale} {finding.suggestion}"
    return any(re.search(rf"\b{re.escape(n)}\b", text) for n in names)


def names_the_bug(finding: Finding, keywords: Sequence[str]) -> bool:
    """Does the model's explanation describe the labelled bug?

    Used only for the rule-blind cases, where the model is the one finding the
    bug. A case with no keywords accepts any finding on the right line.
    """
    if not keywords:
        return True
    text = f"{finding.message} {finding.rationale} {finding.suggestion}"
    return any(re.search(k, text, re.IGNORECASE) for k in keywords)


def confidence_sweep(per_case: Sequence[Dict[str, Any]],
                     thresholds: Sequence[float] = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)
                     ) -> List[Dict[str, Any]]:
    """Precision of the model's own findings, and how many of the rule-blind
    bugs it catches, at each confidence cut-off. This is the evidence for
    where --llm-confidence should sit."""
    scored = [r for r in per_case if not r.get("crashed")]
    blind_total = sum(r["expected"] for r in scored if r["kind"] == "llm")
    rows = []
    for t in thresholds:
        kept = correct = blind_hits = 0
        for r in scored:
            mine = [c for c in r["llm_candidates"] if c["confidence"] >= t]
            right = sum(1 for c in mine if c["correct"])
            kept += len(mine)
            correct += right
            if r["kind"] == "llm":
                blind_hits += min(right, r["expected"])
        rows.append({
            "threshold": t,
            "kept": kept,
            "correct": correct,
            "precision": round(correct / kept, 3) if kept else None,
            "blind_recall": round(blind_hits / blind_total, 3) if blind_total else None,
        })
    return rows


def score_case(assistant: CodeReviewAssistant, case: TestCase,
               tolerance: int = 1) -> Dict[str, Any]:
    """Review one corpus case and score it.

    Everything it needs comes in through the arguments and everything it
    returns is plain data, so it runs the same in the parent process or in a
    worker.
    """
    expected = case.expected()
    try:
        report = assistant.review_text(case.source(), path=f"corpus/{case.name}.py")
    except Exception as exc:  # the whole point of the harness
        return {"name": case.name, "crashed": True, "error": str(exc)}

    # Only score what the corpus labels. P001 syntax errors would otherwise
    # count as false positives on the edge cases built to contain them.
    predicted = [f for f in report.findings if f.rule != "P001"]
    # A model finding on the right line that explains something else is a
    # wrong finding, not a hit: it would send the reader after the wrong fix.
    misdiagnosed = [f for f in predicted
                    if f.source == "llm" and not names_the_bug(f, case.keywords)]
    scorable = [f for f in predicted if f not in misdiagnosed]
    tp, fp, fn, missed, spurious = match_findings(scorable, expected, tolerance)
    spurious = spurious + misdiagnosed
    fp += len(misdiagnosed)

    categories = {}
    for cat in CATEGORIES:
        c_tp, c_fp, c_fn, _, _ = match_findings(
            [f for f in scorable if f.category == cat],
            [e for e in expected if e[1] == cat],
            tolerance,
        )
        c_fp += sum(1 for f in misdiagnosed if f.category == cat)
        categories[cat] = {"tp": c_tp, "fp": c_fp, "fn": c_fn}

    enriched = [f for f in predicted if f.source == "static+llm"]
    candidates = [
        {"line": f.line, "category": f.category, "confidence": f.confidence,
         "message": f.message,
         "on_line": any(c == f.category and abs(l - f.line) <= tolerance for l, c in expected),
         "correct": names_the_bug(f, case.keywords) and any(
             c == f.category and abs(l - f.line) <= tolerance for l, c in expected)}
        for f in report.llm_candidates
    ]

    return {
        "name": case.name,
        "kind": case.kind,
        "note": case.note,
        "expected": len(expected),
        "predicted": len(predicted),
        "tp": tp, "fp": fp, "fn": fn,
        "categories": categories,
        "lines": max(len(case.code.split("\n")), 1),
        "seconds": round(report.total_seconds, 4),
        # what the model did, as opposed to what the rules did
        "static_found": sum(1 for f in predicted if f.source != "llm"),
        "enriched": len(enriched),
        "specific": sum(1 for f in enriched if mentions_code(f)),
        "llm_candidates": candidates,
        "llm_found": sum(1 for f in predicted if f.source == "llm"),
        "llm_wrong": sum(1 for f in spurious if f.source == "llm"),
        "llm_misdiagnosed": len(misdiagnosed),
        "missed": [{"line": l, "category": c} for l, c in missed],
        "false_positives": [
            {"line": f.line, "category": f.category, "rule": f.rule,
             "source": f.source, "message": f.message}
            for f in spurious
        ],
        "errors": report.errors,
    }


def run_evaluation(backend: Optional[LLMBackend] = None, tolerance: int = 1,
                   verbose: bool = False, config: Optional[Dict[str, Any]] = None,
                   workers: int = 1, spec: Optional[Dict[str, Any]] = None,
                   gpus: Sequence[int] = ()) -> Dict[str, Any]:
    """Score the assistant against the labelled corpus.

    With workers > 1 the cases are spread over a process pool. Workers cannot
    be handed a loaded model, so that path takes `spec`, the plain values
    make_backend() needs, and each worker builds its own.
    """
    config = config or {}
    if backend is None and spec:
        backend = make_backend(**spec)

    start = time.perf_counter()
    if workers > 1:
        per_case = run_parallel(
            functools.partial(_task_score_case, tolerance=tolerance),
            range(len(TEST_CORPUS)), workers, spec, config, gpus,
        )
    else:
        assistant = CodeReviewAssistant(backend=backend, config=config)
        per_case = [score_case(assistant, case, tolerance) for case in TEST_CORPUS]
    wall = time.perf_counter() - start

    overall = {"tp": 0, "fp": 0, "fn": 0}
    per_category = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CATEGORIES}
    llm = {"static_findings": 0, "enriched": 0, "specific": 0, "added": 0, "added_wrong": 0,
           "misdiagnosed": 0}
    subsets = {k: {"tp": 0, "fp": 0, "fn": 0} for k in ("static", "llm")}
    latencies: List[float] = []
    total_lines = 0
    crashes = 0

    for r in per_case:
        if r.get("crashed"):
            crashes += 1
            continue
        for key in overall:
            overall[key] += r[key]
            subsets[r["kind"]][key] += r[key]
        for cat in CATEGORIES:
            for key in ("tp", "fp", "fn"):
                per_category[cat][key] += r["categories"][cat][key]
        llm["static_findings"] += r["static_found"]
        llm["enriched"] += r["enriched"]
        llm["specific"] += r["specific"]
        llm["added"] += r["llm_found"]
        llm["added_wrong"] += r["llm_wrong"]
        llm["misdiagnosed"] += r.get("llm_misdiagnosed", 0)
        latencies.append(r["seconds"])
        total_lines += r["lines"]

    result = {
        "backend": backend.describe() if backend else "static only",
        "cases": len(TEST_CORPUS),
        "crashes": crashes,
        "tolerance_lines": tolerance,
        "workers": workers,
        "overall": prf(**overall),
        "per_category": {c: prf(**per_category[c]) for c in CATEGORIES},
        # bugs the rules are built for, versus logic errors only a model can see
        "subsets": {"static_visible": prf(**subsets["static"]),
                    "static_blind": prf(**subsets["llm"])},
        "llm": llm,
        "confidence_sweep": confidence_sweep(per_case),
        "latency": {
            "mean_seconds": round(statistics.fmean(latencies), 4) if latencies else 0.0,
            "median_seconds": round(statistics.median(latencies), 4) if latencies else 0.0,
            "p95_seconds": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 4)
            if len(latencies) >= 2 else 0.0,
            "max_seconds": round(max(latencies), 4) if latencies else 0.0,
            "total_seconds": round(sum(latencies), 4),
            "lines_per_second": round(total_lines / sum(latencies), 1)
            if sum(latencies) > 0 else 0.0,
            # wall clock includes starting workers and loading model weights
            "wall_seconds": round(wall, 3),
        },
        "per_case": per_case,
    }

    if verbose:
        print(format_eval(result))
    return result


def format_eval(result: Dict[str, Any]) -> str:
    out = []
    out.append(f"Backend        : {result['backend']}")
    out.append(f"Cases          : {result['cases']} (crashes: {result['crashes']})")
    out.append(f"Match tolerance: +/-{result['tolerance_lines']} line(s)")
    out.append("")

    o = result["overall"]
    out.append(f"{'category':<14}{'P':>8}{'R':>8}{'F1':>8}{'TP':>6}{'FP':>6}{'FN':>6}")
    out.append("-" * 56)
    out.append(f"{'OVERALL':<14}{o['precision']:>8.3f}{o['recall']:>8.3f}"
               f"{o['f1']:>8.3f}{o['tp']:>6}{o['fp']:>6}{o['fn']:>6}")
    for cat, m in result["per_category"].items():
        out.append(f"{cat:<14}{m['precision']:>8.3f}{m['recall']:>8.3f}"
                   f"{m['f1']:>8.3f}{m['tp']:>6}{m['fp']:>6}{m['fn']:>6}")

    subsets = result.get("subsets") or {}
    if subsets:
        out.append("-" * 56)
        for name, label in (("static_visible", "rule-visible"), ("static_blind", "rule-blind")):
            m = subsets[name]
            out.append(f"{label:<14}{m['precision']:>8.3f}{m['recall']:>8.3f}"
                       f"{m['f1']:>8.3f}{m['tp']:>6}{m['fp']:>6}{m['fn']:>6}")

    lat = result["latency"]
    out.append("")
    out.append(f"Latency: mean {lat['mean_seconds'] * 1000:.1f} ms, "
               f"median {lat['median_seconds'] * 1000:.1f} ms, "
               f"p95 {lat['p95_seconds'] * 1000:.1f} ms, "
               f"max {lat['max_seconds'] * 1000:.1f} ms")
    out.append(f"Throughput: {lat['lines_per_second']:.0f} lines/second per worker")
    out.append(f"Wall clock: {lat['wall_seconds']:.1f}s with {result.get('workers', 1)} worker(s), "
               "including start-up and model loading")

    llm = result.get("llm") or {}
    if llm.get("enriched") or llm.get("added"):
        out.append("")
        out.append(f"LLM stage: explained {llm['enriched']} of {llm['static_findings']} static "
                   f"findings; added {llm['added']} of its own, {llm['added_wrong']} of them wrong")
        if llm.get("misdiagnosed"):
            out.append(f"  ({llm['misdiagnosed']} of the wrong ones were on the right line "
                       "but explained a different problem)")
        if llm["enriched"]:
            out.append(f"Explanations that name something in the code: {llm['specific']} of "
                       f"{llm['enriched']} ({llm['specific'] / llm['enriched']:.0%})")
    sweep = result.get("confidence_sweep") or []
    if any(row["kept"] for row in sweep):
        out.append("")
        out.append("Model's own findings by confidence cut-off")
        out.append(f"{'cut-off':>9}{'kept':>7}{'right':>7}{'precision':>11}{'blind recall':>14}")
        for row in sweep:
            p = "-" if row["precision"] is None else f"{row['precision']:.3f}"
            r = "-" if row["blind_recall"] is None else f"{row['blind_recall']:.3f}"
            out.append(f"{row['threshold']:>9.1f}{row['kept']:>7}{row['correct']:>7}{p:>11}{r:>14}")

    # What the model actually said on each rule-blind case, so the scoring can
    # be checked by eye rather than taken on trust.
    blind = [r for r in result.get("per_case", []) if r.get("kind") == "llm" and not r.get("crashed")]
    if any(r["llm_candidates"] for r in blind):
        out.append("")
        out.append("Model's findings on the rule-blind cases")
        for r in blind:
            if not r["llm_candidates"]:
                out.append(f"  {r['name']:<22} (nothing reported)")
            for c in r["llm_candidates"]:
                verdict = "right" if c["correct"] else ("wrong reason" if c.get("on_line") else "wrong line")
                out.append(f"  {r['name']:<22} {verdict:<13} conf {c['confidence']:.1f}  {c['message'][:70]}")
    return "\n".join(out)


def format_eval_markdown(result: Dict[str, Any]) -> str:
    o = result["overall"]
    lat = result["latency"]
    out = [
        f"**Backend:** {result['backend']}  ",
        f"**Cases:** {result['cases']}  ",
        f"**Line-match tolerance:** ±{result['tolerance_lines']}",
        "",
        "| Category | Precision | Recall | F1 | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| **Overall** | {o['precision']:.3f} | {o['recall']:.3f} | {o['f1']:.3f} "
        f"| {o['tp']} | {o['fp']} | {o['fn']} |",
    ]
    for cat, m in result["per_category"].items():
        out.append(f"| {cat} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
                   f"| {m['tp']} | {m['fp']} | {m['fn']} |")
    for name, label in (("static_visible", "*rule-visible bugs*"), ("static_blind", "*rule-blind bugs*")):
        m = (result.get("subsets") or {}).get(name)
        if m:
            out.append(f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
                       f"| {m['tp']} | {m['fp']} | {m['fn']} |")
    out += [
        "",
        f"Mean latency {lat['mean_seconds'] * 1000:.1f} ms, "
        f"median {lat['median_seconds'] * 1000:.1f} ms, "
        f"p95 {lat['p95_seconds'] * 1000:.1f} ms. "
        f"Throughput {lat['lines_per_second']:.0f} lines/second per worker. "
        f"Wall clock {lat['wall_seconds']:.1f}s with {result.get('workers', 1)} worker(s).",
    ]
    llm = result.get("llm") or {}
    if llm.get("enriched") or llm.get("added"):
        out += [
            "",
            f"The model explained {llm['enriched']} of {llm['static_findings']} static findings "
            f"and added {llm['added']} of its own, {llm['added_wrong']} of them wrong"
            f"{' (' + str(llm['misdiagnosed']) + ' on the right line for the wrong reason)' if llm.get('misdiagnosed') else ''}. "
            f"{llm['specific']} of its explanations name something in the code.",
        ]
    sweep = result.get("confidence_sweep") or []
    if any(row["kept"] for row in sweep):
        out += ["", "| Confidence cut-off | Kept | Correct | Precision | Rule-blind recall |",
                "| ---: | ---: | ---: | ---: | ---: |"]
        for row in sweep:
            p = "-" if row["precision"] is None else f"{row['precision']:.3f}"
            r = "-" if row["blind_recall"] is None else f"{row['blind_recall']:.3f}"
            out.append(f"| {row['threshold']:.1f} | {row['kept']} | {row['correct']} | {p} | {r} |")
    return "\n".join(out)


def dump_corpus(target: str | Path) -> List[Path]:
    """Write the corpus out as real .py files, for demos and manual runs."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for case in TEST_CORPUS:
        p = target / f"{case.name}.py"
        p.write_text(case.source(), encoding="utf-8")   # no answer-key comments in demo files
        written.append(p)
    return written


def bench_file(assistant: CodeReviewAssistant, path: str) -> Dict[str, Any]:
    """Review one file and return only counts, which keeps worker replies small."""
    try:
        report = assistant.review_file(path)
    except Exception as exc:
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}

    by_rule: Dict[str, int] = {}
    by_category: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    for f in report.findings:
        by_rule[f.rule] = by_rule.get(f.rule, 0) + 1
        by_category[f.category] = by_category.get(f.category, 0) + 1
        by_severity[f.severity] = by_severity.get(f.severity, 0) + 1
    return {
        "path": str(path),
        "lines": report.metrics.get("total_lines", 0),
        "findings": len(report.findings),
        "by_rule": by_rule,
        "by_category": by_category,
        "by_severity": by_severity,
        "seconds": report.total_seconds,
    }


def run_benchmark(paths: Sequence[str], config: Optional[Dict[str, Any]] = None,
                  limit: Optional[int] = None, workers: int = 1,
                  use_external: bool = False,
                  tools: Sequence[str] = ExternalTools.DEFAULT_TOOLS) -> Dict[str, Any]:
    """Run over real third-party code to measure noise and throughput.

    The labelled corpus says whether the rules fire when they should. It
    cannot say whether they fire when they should not, because every snippet
    in it was written to contain a defect. Running over a large body of
    reviewed, working code gives the complementary number: how much noise a
    developer has to wade through per hundred lines. Findings here are not
    all false positives -- real code has real problems -- but the rate and
    the mix are what tell you whether the tool is usable day to day.
    """
    config = config or {}
    files = [str(p) for p in iter_python_files(paths)]
    if limit:
        files = files[:limit]

    start = time.perf_counter()
    if workers > 1:
        # files are small, so hand them out in batches to keep the pipe quiet
        results = run_parallel(
            _task_bench_file, files, workers, None, config, (), use_external, tools,
            chunksize=max(1, len(files) // (workers * 8)),
        )
    else:
        assistant = CodeReviewAssistant(config=config, use_external_tools=use_external,
                                        external_tools=tools)
        results = [bench_file(assistant, f) for f in files]
    wall = time.perf_counter() - start

    by_rule: Dict[str, int] = {}
    by_category: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    total_lines = 0
    total_findings = 0
    durations: List[float] = []
    failures: List[str] = []

    for r in results:
        if "error" in r:
            failures.append(f"{r['path']}: {r['error']}")
            continue
        total_lines += r["lines"]
        total_findings += r["findings"]
        durations.append(r["seconds"])
        for source, target in ((r["by_rule"], by_rule), (r["by_category"], by_category),
                               (r["by_severity"], by_severity)):
            for key, count in source.items():
                target[key] = target.get(key, 0) + count

    return {
        "paths": [str(p) for p in paths],
        "workers": workers,
        "files_scanned": len(durations),
        "files_failed": len(failures),
        "failures": failures[:10],
        "total_lines": total_lines,
        "total_findings": total_findings,
        "findings_per_100_lines": round(total_findings / total_lines * 100, 2)
        if total_lines else 0.0,
        "serious_per_100_lines": round(
            sum(by_severity.get(s, 0) for s in ("critical", "high")) / total_lines * 100, 3
        ) if total_lines else 0.0,
        "by_category": dict(sorted(by_category.items(), key=lambda kv: -kv[1])),
        "by_severity": {s: by_severity.get(s, 0) for s in SEVERITY_ORDER},
        "top_rules": dict(sorted(by_rule.items(), key=lambda kv: -kv[1])[:15]),
        "wall_seconds": round(wall, 3),
        "busy_seconds": round(sum(durations), 3),
        "lines_per_second": round(total_lines / wall, 0) if wall else 0.0,
        "mean_file_seconds": round(statistics.fmean(durations), 4) if durations else 0.0,
    }


def format_benchmark(result: Dict[str, Any]) -> str:
    out = [
        f"Scanned        : {result['files_scanned']} files, "
        f"{result['total_lines']:,} lines ({result['files_failed']} failed)",
        f"Findings       : {result['total_findings']:,} "
        f"({result['findings_per_100_lines']} per 100 lines)",
        f"Critical/high  : {result['serious_per_100_lines']} per 100 lines",
        f"Throughput     : {result['lines_per_second']:,.0f} lines/second "
        f"({result['wall_seconds']}s wall, {result.get('workers', 1)} worker(s))",
        "",
        "By severity: " + ", ".join(f"{k} {v}" for k, v in result["by_severity"].items() if v),
        "By category: " + ", ".join(f"{k} {v}" for k, v in result["by_category"].items()),
        "",
        "Most frequent rules:",
    ]
    for rule, count in result["top_rules"].items():
        out.append(f"  {count:>6}  {rule}")
    return "\n".join(out)


def _selftest_adapters(check) -> None:
    """Ruff parsing and merging, the OpenAI-compatible client, the local-only guard."""
    import http.server
    import threading

    raw = json.dumps([
        {"code": "F401", "message": "`os` imported but unused", "location": {"row": 1, "column": 8}},
        {"code": "S608", "message": "Possible SQL injection", "location": {"row": 5, "column": 9}},
        {"code": "PERF401", "message": "Use a list comprehension", "location": {"row": 9, "column": 5}},
        {"code": "E501", "message": "Line too long", "location": {"row": 12, "column": 101}},
        {"code": None, "message": "SyntaxError: Expected ':'", "location": {"row": 3, "column": 1}},
        {"code": "F821", "message": "Undefined name `x`", "location": {"row": 14, "column": 1}},
    ])
    parsed = {f.rule: f for f in parse_ruff_json(raw)}
    check("ruff output is parsed into findings", len(parsed) == 6)
    newer = parse_ruff_json(json.dumps([{"code": "invalid-syntax", "message": "x",
                                         "location": {"row": 2, "column": 1}}]))
    check("newer ruff spells syntax errors 'invalid-syntax'",
          newer[0].rule == "ruff:syntax" and newer[0].severity == "critical")
    p001 = [Finding("P001", "bug", "critical", 9, "does not parse")]
    check("ruff syntax error is dropped when we already reported the parse failure",
          drop_equivalents(newer, p001) == [])
    check("ruff codes map to our categories",
          [parsed[r].category for r in ("ruff:F401", "ruff:S608", "ruff:PERF401",
                                        "ruff:E501", "ruff:syntax", "ruff:F821")]
          == ["style", "security", "optimization", "style", "bug", "bug"])
    check("ruff syntax errors are critical", parsed["ruff:syntax"].severity == "critical")
    check("ruff SIM rules are not mistaken for security rules",
          ruff_severity("SIM108") == "low" and ruff_severity("S307") == "high"
          and ruff_severity("S105") == "medium")
    check("ruff garbage output yields nothing", parse_ruff_json("not json") == [])

    ours = [Finding("Y013", "style", "low", 1, "m"), Finding("S011", "security", "critical", 5, "m"),
            Finding("Y014", "style", "low", 12, "m")]
    kept = {f.rule for f in drop_equivalents(parsed.values(), ours)}
    check("ruff duplicates of our rules are dropped, new findings kept",
          kept == {"ruff:PERF401", "ruff:syntax", "ruff:F821"}, str(sorted(kept)))
    near = [Finding("S011", "security", "critical", 7, "m")]
    check("a twin a couple of lines away still counts as the same issue",
          drop_equivalents([Finding("ruff:S608", "security", "high", 5, "m")], near) == [])
    check("a twin far away does not hide a different problem",
          len(drop_equivalents([Finding("ruff:S608", "security", "high", 40, "m")], near)) == 1)

    # A stand-in `ruff` on PATH that prints the fixture above. This exercises
    # the real plumbing (temp file, subprocess call, parsing, merging) even
    # where the genuine binary cannot be installed.
    bin_dir = tempfile.mkdtemp()
    fake = os.path.join(bin_dir, "ruff")
    with open(fake, "w", encoding="utf-8") as fh:
        fh.write("#!/usr/bin/env python3\nimport sys\nprint(" + repr(raw) + ")\n")
    os.chmod(fake, 0o755)
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = bin_dir + os.pathsep + old_path
    try:
        lines = "import os\n" + "\n" * 15
        r = CodeReviewAssistant(use_external_tools=True, external_tools=("ruff",)).review_text(
            lines, "ext.py")
        rules = {f.rule for f in r.findings}
        check("ruff findings are merged through the subprocess path", "ruff:PERF401" in rules)
        check("ruff duplicate of an existing rule is not shown twice", "ruff:F401" not in rules)
    finally:
        os.environ["PATH"] = old_path
        shutil.rmtree(bin_dir, ignore_errors=True)

    r = CodeReviewAssistant(use_external_tools=True,
                            external_tools=("no-such-tool",)).review_text("x = 1\n", "a.py")
    check("missing external tools are reported, not fatal",
          any("none are installed" in e for e in r.errors))

    class Handler(http.server.BaseHTTPRequestHandler):
        seen: Dict[str, Any] = {}

        def do_POST(self):
            size = int(self.headers.get("Content-Length", 0))
            Handler.seen["path"] = self.path
            Handler.seen["body"] = json.loads(self.rfile.read(size))
            reply = json.dumps({"choices": [{"message": {
                "content": '{"explanations": [], "new_findings": []}'}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        backend = OpenAICompatBackend(model="m", base_url=f"http://127.0.0.1:{server.server_port}/v1")
        text = backend.generate("hello", system="be brief")
        roles = [m["role"] for m in Handler.seen["body"]["messages"]]
        check("OpenAI-compatible backend sends and parses a chat completion",
              extract_json(text) == {"explanations": [], "new_findings": []}
              and Handler.seen["path"] == "/v1/chat/completions" and roles == ["system", "user"])
    finally:
        server.shutdown()

    try:
        OpenAICompatBackend(base_url="https://api.example.com/v1")
        refused = False
    except ValueError:
        refused = True
    check("remote model servers are refused by default", refused)
    check("remote model servers are allowed when asked for",
          OpenAICompatBackend(base_url="https://api.example.com/v1", allow_remote=True) is not None)


def _selftest_parallel(check) -> None:
    """Worker pool: GPU pinning, ordering, agreement with the serial path, errors."""
    info = run_parallel(_task_worker_info, range(4), workers=2, gpus=[3, 5])
    check("each worker is pinned to its own GPU",
          {r["cuda_visible"] for r in info} == {"3", "5"}, str(sorted({r["cuda_visible"] for r in info})))
    check("work is spread over two processes", len({r["pid"] for r in info}) == 2)

    blind = [c for c in TEST_CORPUS if c.kind == "llm"]
    plain = CodeReviewAssistant()
    blind_hits = [f for c in blind for f in plain.review_text(c.source(), "b.py").findings]
    check("rule-blind corpus cases really are invisible to the rules",
          len(blind) >= 10 and not blind_hits, str([(f.rule, f.line) for f in blind_hits][:5]))
    vague = Finding("Y007", "style", "medium", 1, "m", snippet="def loads(s, cls=None):",
                    rationale="High complexity is hard to maintain.")
    exact = Finding("Y007", "style", "medium", 1, "m", snippet="def loads(s, cls=None):",
                    rationale="loads() mixes parsing and the cls hook.")
    check("specificity check tells generic advice from specific advice",
          not mentions_code(vague) and mentions_code(exact))

    serial = run_evaluation(workers=1)
    pooled = run_evaluation(workers=2)
    check("parallel evaluation matches the serial result",
          serial["overall"] == pooled["overall"] and serial["per_category"] == pooled["per_category"])
    check("parallel evaluation keeps case order",
          [c["name"] for c in serial["per_case"]] == [c["name"] for c in pooled["per_case"]])

    try:
        run_parallel(_task_review_file, ["x.py"], workers=2,
                     spec={"kind": "hf", "quantize": "8bit"})
        message = ""
    except RuntimeError as exc:
        message = str(exc)
    check("a worker that cannot start explains why", "worker could not start" in message
          and "unsupported quantize" in message, message[:120])

    check("gpu list parsing", parse_gpus("0,1") == [0, 1] and parse_gpus("none") == []
          and parse_gpus(None) == [])
    check("one worker per GPU for a local model",
          resolve_workers(0, "hf", [0, 1]) == 2 and resolve_workers(0, "hf", []) == 1)
    check("worker count is explicit when asked", resolve_workers(3, "hf", [0]) == 3)
    check("CPU-bound jobs default to every core",
          resolve_workers(0, "none", [], cpu_bound=True) == (os.cpu_count() or 1))


def run_selftest() -> int:
    """Sanity checks that have to pass before anything else is believable."""
    checks: List[Tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, bool(condition), detail))

    assistant = CodeReviewAssistant()

    r = assistant.review_text("def f(x=[]):\n    return x\n", "t.py")
    check("mutable default is caught", any(f.rule == "B001" for f in r.findings))

    r = assistant.review_text("", "empty.py")
    check("empty file does not crash", True, f"{len(r.findings)} findings")

    r = assistant.review_text("def broken(:\n", "bad.py")
    check("syntax error is reported, not raised",
          any(f.rule == "P001" for f in r.findings))
    check("syntax error still yields a report", r.metrics.get("parseable") is False)

    r = assistant.review_text('x = "no newline at end"', "nn.py")
    check("missing final newline is caught", any(f.rule == "Y019" for f in r.findings))

    r = assistant.review_text(
        'import os\n\n\ndef go(c):\n    """Run."""\n    os.system(c)\n', "s.py")
    check("os.system is caught", any(f.rule == "S003" for f in r.findings))

    def rules_in(src, wanted, **kw):
        found = CodeReviewAssistant(**kw).review_text(src).findings
        return [f.rule for f in found if f.rule in wanted]

    check("tuple unpacking is not an unused variable",
          not rules_in("def f(e, s):\n    data, used = e(s)\n    return data\n", {"B013"}))
    check("plain unused assignment is still caught",
          rules_in("def f(x):\n    total = x + 1\n    return x\n", {"B013"}) == ["B013"])
    check("assert on an argument is flagged",
          rules_in("def f(alt):\n    assert len(alt) == 2\n", {"S013"}) == ["S013"])
    check("assert on internal state is not flagged",
          not rules_in("class A:\n    def f(self):\n        assert self.x is None\n", {"S013"}))
    lookup = "def f(d):\n    if 'k' in d:\n        return d['k']\n"
    check("double lookup rule is off by default, on with --enable O008",
          not rules_in(lookup, {"O008"})
          and rules_in(lookup, {"O008"}, config={"enable": {"O008"}}) == ["O008"])

    wrong = Finding("LLM001", "bug", "high", 8, "Loop condition should include limit",
                    suggestion="Change < to <=", source="llm")
    right = Finding("LLM001", "bug", "high", 8, "i is never incremented, so the loop never ends",
                    source="llm")
    case = next(c for c in TEST_CORPUS if c.name == "missing_increment")
    check("right line, wrong reason does not count as finding the bug",
          not names_the_bug(wrong, case.keywords) and names_the_bug(right, case.keywords))
    check("the reviewer never sees the answer-key markers",
          all("#@" not in c.source() and c.source().count("\n") == c.code.count("\n")
              for c in TEST_CORPUS))
    check("every rule-blind case has keywords",
          all(c.keywords for c in TEST_CORPUS if c.kind == "llm"))

    clean = '"""Doc."""\n\n\ndef add(a, b):\n    """Add."""\n    return a + b\n'
    r = assistant.review_text(clean, "clean.py")
    noisy = [f for f in r.findings if f.severity in {"critical", "high", "medium"}]
    check("clean code produces no serious findings", not noisy,
          f"{[f.rule for f in noisy]}")

    data = extract_json('here you go: ```json\n{"a": 1,}\n``` hope that helps')
    check("JSON extraction survives fences and trailing commas", data == {"a": 1})

    data = extract_json("no json at all")
    check("JSON extraction returns None on junk", data is None)

    tp, fp, fn, _, _ = match_findings(
        [Finding("X", "bug", "high", 10, "m"), Finding("Y", "bug", "high", 10, "m")],
        [(10, "bug")],
    )
    check("one label cannot be matched twice", (tp, fp, fn) == (1, 1, 0), f"{tp},{fp},{fn}")

    big = "\n".join(f"x{i} = {i}" for i in range(1200))
    unit = SourceUnit(big, "big.py", max_lines=400)
    check("long files are chunked", len(unit.chunks()) > 1, f"{len(unit.chunks())} chunks")

    check("corpus markers parse", len(TEST_CORPUS[0].expected()) == 2,
          str(TEST_CORPUS[0].expected()))

    r = assistant.review_text("def f(x=[]):  # noqa: B001\n    return x\n", "sup.py")
    check("targeted noqa suppresses its rule", not any(f.rule == "B001" for f in r.findings))

    r = assistant.review_text("def f(x=[]):  # noqa: Y003\n    return x\n", "sup2.py")
    check("noqa for another rule leaves the finding",
          any(f.rule == "B001" for f in r.findings))

    r = assistant.review_text("import os\nx = 1\n", "unused.py")
    check("unused import is caught", any(f.rule == "Y013" for f in r.findings))

    r = assistant.review_text("from __future__ import annotations\nx = 1\n", "fut.py")
    check("__future__ import is not called unused",
          not any(f.rule == "Y013" for f in r.findings))

    semicolon_in_prose = '"""Sorted by key; useful for tests."""\n\nVALUE = 1\n'
    r = assistant.review_text(semicolon_in_prose, "prose.py")
    check("semicolon inside a docstring is not a statement",
          not any(f.rule == "Y016" for f in r.findings))

    r = assistant.review_text('"""Doc."""\n\n__all__ = ["x"]\n\nimport os\n\nx = os.sep\n', "d.py")
    check("imports after __all__ are not flagged",
          not any(f.rule == "Y002" for f in r.findings))

    # The LLM merge path is exercised with a scripted backend so the whole
    # pipeline is testable without a GPU or a running model.
    class ScriptedBackend(LLMBackend):
        name = "scripted"

        def generate(self, prompt: str, system: str = "") -> str:
            return (
                "Sure, here is the review:\n```json\n"
                '{"explanations": [{"line": 2, "rule": "B001",'
                ' "explanation": "The list is shared across calls.",'
                ' "fix": "Default to None."}],'
                ' "new_findings": ['
                '{"line": 3, "category": "bug", "severity": "high",'
                ' "message": "Off-by-one on the slice", "confidence": 0.9},'
                '{"line": 3, "category": "bug", "severity": "high",'
                ' "message": "Possibly wrong, not sure", "confidence": 0.2},'
                '{"line": 3, "category": "nonsense", "severity": "high",'
                ' "message": "Bad category", "confidence": 0.9},'
                '{"line": 9999, "category": "bug", "severity": "high",'
                ' "message": "Outside the file", "confidence": 0.9}'
                "]}\n```\n"
            )

    llm_assistant = CodeReviewAssistant(backend=ScriptedBackend(), min_confidence=0.6)
    r = llm_assistant.review_text("def f(x=[]):\n    y = x[:1]\n    return y\n", "llm.py")
    enriched = [f for f in r.findings if f.rule == "B001"]
    check("LLM explanation is merged into the static finding",
          bool(enriched) and "shared across calls" in enriched[0].rationale)
    check("confident LLM finding is kept",
          any(f.source == "llm" and "Off-by-one" in f.message for f in r.findings))
    check("low-confidence LLM finding is dropped",
          not any("not sure" in f.message for f in r.findings))
    check("invalid LLM category is dropped",
          not any("Bad category" in f.message for f in r.findings))
    check("out-of-range LLM line is dropped",
          not any("Outside the file" in f.message for f in r.findings))
    check("every valid proposal is kept as a candidate, confident or not",
          sorted(f.confidence for f in r.llm_candidates) == [0.2, 0.9])

    class BrokenBackend(LLMBackend):
        name = "broken"

        def generate(self, prompt: str, system: str = "") -> str:
            raise RuntimeError("model server is down")

    r = CodeReviewAssistant(backend=BrokenBackend()).review_text(
        "def f(x=[]):\n    return x\n", "down.py")
    check("static findings survive an LLM failure",
          any(f.rule == "B001" for f in r.findings) and bool(r.errors))

    _selftest_adapters(check)
    _selftest_parallel(check)

    width = max(len(n) for n, _, _ in checks)
    failed = 0
    for name, ok, detail in checks:
        status = "pass" if ok else "FAIL"
        if not ok:
            failed += 1
        suffix = f"  ({detail})" if detail and not ok else ""
        print(f"  {status}  {name.ljust(width)}{suffix}")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


# --------------------------------------------------------------------------
# interactive UI
# --------------------------------------------------------------------------

EXAMPLE_CODE = '''import os


def process(records, cache={}):
    results = ""
    for i in range(len(records)):
        key = records[i]
        if key in cache:
            results += str(cache[key])
        else:
            value = os.popen("lookup " + key).read()
            cache[key] = value
            results += value
    return results
'''


def build_ui(backend: Optional[LLMBackend] = None, config: Optional[Dict[str, Any]] = None):
    """Build the Gradio app without starting it.

    Returns the app and the function behind its button, so the review path
    can be exercised directly in a test or a headless notebook run.
    """
    import gradio as gr

    assistant = CodeReviewAssistant(backend=backend, config=config or {})
    backend_label = backend.describe() if backend else "static only"

    def review(code: str, categories: List[str], min_severity: str):
        if not code.strip():
            return "Paste some Python to review.", "{}", ""

        assistant.config["categories"] = categories or list(CATEGORIES)
        assistant.config["min_severity"] = min_severity
        report = assistant.review_text(code, "pasted_snippet.py")

        md = format_markdown([report])
        metrics = json.dumps(report.metrics, indent=2)
        counts = report.counts_by_severity()
        summary = (
            f"**{len(report.findings)} findings** in {report.total_seconds * 1000:.0f} ms "
            f"(static {report.static_seconds * 1000:.0f} ms, "
            f"LLM {report.llm_seconds * 1000:.0f} ms), backend: {backend_label}\n\n"
            + " · ".join(f"{k}: {v}" for k, v in counts.items() if v)
        )
        return summary, metrics, md

    with gr.Blocks(title="Code Review Assistant") as demo:
        gr.Markdown(
            "# Code Review Assistant\n"
            "Static analysis for bugs, security issues, style and performance, "
            f"with an LLM pass on top. Current backend: **{backend_label}**."
        )
        with gr.Row():
            with gr.Column(scale=3):
                code_box = gr.Code(value=EXAMPLE_CODE, language="python",
                                   label="Python source", lines=24)
                with gr.Row():
                    cats = gr.CheckboxGroup(
                        choices=list(CATEGORIES), value=list(CATEGORIES),
                        label="Categories",
                    )
                    sev = gr.Dropdown(
                        choices=SEVERITY_ORDER, value="info", label="Minimum severity",
                    )
                run_btn = gr.Button("Review", variant="primary")
            with gr.Column(scale=4):
                summary_out = gr.Markdown(label="Summary")
                review_out = gr.Markdown(label="Review")
                metrics_out = gr.Code(label="Metrics", language="json")

        run_btn.click(review, inputs=[code_box, cats, sev],
                      outputs=[summary_out, metrics_out, review_out])

    return demo, review


def launch_ui(backend: Optional[LLMBackend] = None, share: bool = False,
              config: Optional[Dict[str, Any]] = None) -> None:
    try:
        demo, _ = build_ui(backend, config)
    except ImportError:
        print("Gradio is not installed. Install it with:  pip install gradio", file=sys.stderr)
        raise SystemExit(2)
    demo.launch(share=share)


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="code_review_assistant",
        description="AI-assisted code review: static analysis plus a local LLM.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_llm_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--llm", default="none",
                       choices=["none", "ollama", "openai", "hf", "offline"],
                       help="LLM backend for the inference stage (default: none). "
                            "'openai' means any OpenAI-compatible server, e.g. vLLM")
        p.add_argument("--model", default=None, help="model name for the chosen backend")
        p.add_argument("--base-url", default=None,
                       help="server address for the ollama and openai backends")
        p.add_argument("--allow-remote", action="store_true",
                       help="permit a model server that is not on this machine "
                            "(source code will be sent to it)")
        p.add_argument("--quantize", default=None, choices=["4bit"],
                       help="load the hf model in 4-bit so a 7B model fits one T4")
        p.add_argument("--llm-confidence", type=float, default=0.6,
                       help="drop LLM-only findings below this confidence (default: 0.6)")

    def add_parallel_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--workers", type=int, default=0,
                       help="worker processes. 0 (default) picks for you: one per GPU "
                            "for --llm hf, otherwise one")
        p.add_argument("--gpus", default="auto",
                       help="GPUs to spread hf workers over: 'auto', '0,1' or 'none'")

    def add_tool_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--external-tools", action="store_true",
                       help="also run ruff/bandit/pylint when they are installed")
        p.add_argument("--tools", nargs="+", default=list(ExternalTools.DEFAULT_TOOLS),
                       choices=["ruff", "bandit", "pylint"],
                       help="which external tools to run (default: ruff bandit; "
                            "pylint is much slower, so it is opt-in)")

    p_review = sub.add_parser("review", help="review a file or directory")
    p_review.add_argument("paths", nargs="+", help="files or directories")
    p_review.add_argument("--format", default="text",
                          choices=["text", "json", "markdown", "sarif"])
    p_review.add_argument("--output", "-o", default=None, help="write to a file instead of stdout")
    p_review.add_argument("--min-severity", default=None, choices=SEVERITY_ORDER)
    p_review.add_argument("--categories", nargs="+", default=None, choices=list(CATEGORIES))
    p_review.add_argument("--max-line-length", type=int, default=DEFAULT_LINE_LENGTH)
    p_review.add_argument("--max-complexity", type=int, default=10)
    p_review.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    p_review.add_argument("--exclude", nargs="+", default=[], help="regexes to skip")
    p_review.add_argument("--enable", nargs="+", default=[], metavar="RULE",
                          help="turn on opt-in rules (currently: O008)")
    add_tool_flags(p_review)
    p_review.add_argument("--no-colour", action="store_true")
    p_review.add_argument("--fail-on", default=None, choices=SEVERITY_ORDER,
                          help="exit non-zero if a finding at this severity or worse exists")
    add_llm_flags(p_review)
    add_parallel_flags(p_review)

    p_eval = sub.add_parser("eval", help="score the assistant against the labelled corpus")
    p_eval.add_argument("--tolerance", type=int, default=1,
                        help="line tolerance when matching a prediction to a label")
    p_eval.add_argument("--output", "-o", default=None, help="write the full JSON result here")
    p_eval.add_argument("--markdown", action="store_true", help="print a markdown table")
    p_eval.add_argument("--show-errors", action="store_true",
                        help="list every false positive and miss")
    add_llm_flags(p_eval)
    add_parallel_flags(p_eval)

    p_ui = sub.add_parser("ui", help="launch the Gradio interface")
    p_ui.add_argument("--share", action="store_true", help="create a public Gradio link")
    add_llm_flags(p_ui)

    sub.add_parser("selftest", help="run built-in sanity checks")

    p_bench = sub.add_parser(
        "benchmark",
        help="measure noise and throughput over a real codebase",
    )
    p_bench.add_argument("paths", nargs="*", default=None,
                         help="defaults to the Python standard library")
    p_bench.add_argument("--limit", type=int, default=None, help="cap the number of files")
    p_bench.add_argument("--output", "-o", default=None)
    p_bench.add_argument("--max-line-length", type=int, default=DEFAULT_LINE_LENGTH)
    p_bench.add_argument("--max-complexity", type=int, default=10)
    add_tool_flags(p_bench)
    p_bench.add_argument("--workers", type=int, default=0,
                         help="worker processes (default: one per CPU core)")

    p_dump = sub.add_parser("dump-corpus", help="write the labelled corpus out as .py files")
    p_dump.add_argument("target", nargs="?", default="samples")

    return parser


def backend_spec_from_args(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """The plain values a worker needs to build its own backend."""
    kind = getattr(args, "llm", "none")
    if kind == "none":
        return None
    return {
        "kind": kind,
        "model": getattr(args, "model", None),
        "base_url": getattr(args, "base_url", None),
        "allow_remote": getattr(args, "allow_remote", False),
        "quantize": getattr(args, "quantize", None),
    }


def backend_from_args(args: argparse.Namespace) -> Optional[LLMBackend]:
    """Build the backend here too, even when workers will build their own, so
    a bad flag fails immediately instead of inside a pool."""
    spec = backend_spec_from_args(args)
    if spec is None:
        return None
    try:
        return make_backend(**spec)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


def plan_workers(args: argparse.Namespace) -> Tuple[int, List[int]]:
    kind = getattr(args, "llm", "none")
    gpus = parse_gpus(getattr(args, "gpus", "none")) if kind in {"hf", "transformers"} else []
    workers = resolve_workers(getattr(args, "workers", 0), kind, gpus)
    if workers > 1:
        where = f" on GPU(s) {gpus}" if gpus else ""
        print(f"using {workers} worker processes{where}", file=sys.stderr)
    return workers, gpus


def config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "max_line_length": getattr(args, "max_line_length", DEFAULT_LINE_LENGTH),
        "max_complexity": getattr(args, "max_complexity", 10),
        "max_lines": getattr(args, "max_lines", DEFAULT_MAX_LINES),
        "min_severity": getattr(args, "min_severity", None),
        "categories": getattr(args, "categories", None),
        "enable": set(getattr(args, "enable", None) or ()),
    }


def cmd_review(args: argparse.Namespace) -> int:
    backend = backend_from_args(args)
    config = config_from_args(args)
    workers, gpus = plan_workers(args)

    if workers > 1:
        reports = review_paths_parallel(
            args.paths, args.exclude, workers, backend_spec_from_args(args), config, gpus,
            args.external_tools, args.tools, args.llm_confidence,
        )
    else:
        assistant = CodeReviewAssistant(
            backend=backend,
            config=config,
            use_external_tools=args.external_tools,
            min_confidence=args.llm_confidence,
            external_tools=args.tools,
        )
        reports = assistant.review_paths(args.paths, exclude=args.exclude)
    if not reports:
        print("no Python files found", file=sys.stderr)
        return 1

    if args.format == "json":
        text = format_json(reports)
    elif args.format == "markdown":
        text = format_markdown(reports)
    elif args.format == "sarif":
        text = format_sarif(reports)
    else:
        colour = not args.no_colour and sys.stdout.isatty()
        text = format_text(reports, colour=colour)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)

    if args.fail_on:
        cutoff = SEVERITY_RANK[args.fail_on]
        worst = min((SEVERITY_RANK.get(f.severity, 9)
                     for r in reports for f in r.findings), default=99)
        if worst <= cutoff:
            return 1
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    backend = backend_from_args(args)
    workers, gpus = plan_workers(args)
    result = run_evaluation(
        backend=backend if workers == 1 else None, tolerance=args.tolerance,
        workers=workers, spec=backend_spec_from_args(args), gpus=gpus,
    )

    print(format_eval_markdown(result) if args.markdown else format_eval(result))

    if args.show_errors:
        print("\nPer-case detail")
        print("-" * 56)
        for case in result["per_case"]:
            if case.get("crashed"):
                print(f"{case['name']}: CRASHED ({case['error']})")
                continue
            if case["fp"] or case["fn"]:
                print(f"\n{case['name']}  (tp={case['tp']} fp={case['fp']} fn={case['fn']})")
                for m in case["missed"]:
                    print(f"    missed  line {m['line']:>3}  {m['category']}")
                for f in case["false_positives"]:
                    print(f"    extra   line {f['line']:>3}  "
                          f"{f['category']}/{f['rule']}  {f['message'][:70]}")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "review":
        return cmd_review(args)
    if args.command == "eval":
        return cmd_eval(args)
    if args.command == "ui":
        launch_ui(backend_from_args(args), share=args.share)
        return 0
    if args.command == "selftest":
        return run_selftest()
    if args.command == "benchmark":
        paths = args.paths or [sysconfig.get_paths()["stdlib"]]
        workers = resolve_workers(args.workers, "none", [], cpu_bound=True)
        result = run_benchmark(paths, config_from_args(args), limit=args.limit,
                               workers=workers, use_external=args.external_tools,
                               tools=args.tools)
        print(format_benchmark(result))
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"\nwrote {args.output}")
        return 0
    if args.command == "dump-corpus":
        written = dump_corpus(args.target)
        print(f"wrote {len(written)} files to {args.target}/")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

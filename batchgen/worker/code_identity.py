"""Code identity: the source revision a worker cohort imported.

``POST /v1/workers/restart`` replaces the worker processes while the server
process stays up, and every new rank imports BatchGen's Python from disk
again. If the source tree moved in between, the new cohort runs code the
server process never imported, so only a rank can say what its cohort runs.
Local rank 0 of every node computes the identity once, at the top of its
start-up and before the model build imports the model code; it travels with
the deployment-capacity publication, and the server logs it once per
generation (``CodeIdentity.log_line``) and reports it in
``/v1/workers/status``.

The identity names a revision, never a directory:

``source=git``
    The ``batchgen`` package directory (symlinks resolved) sits at the top of
    a git work tree. ``revision`` is its checked-out commit and ``dirty``
    whether the tree differs from it: a tracked change, or an untracked file
    git does not ignore. Both come from ONE ``git status`` call. The commit
    alone could be read from the ``.git`` files, but the dirty verdict needs
    the index compared against every tracked file and an untracked scan that
    honours the ignore rules, which only git does reliably; and a commit
    without that verdict does not identify the code.
``source=none``
    No git metadata beside the package: an installed package, whose build
    records no revision, or a tree copied without its history.

Whatever stops git from answering -- no binary, a timeout, an error exit, an
answer that does not parse, a repository without a commit -- yields revision
``unknown`` with one of the fixed ``reason`` values below, never a guess.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

LOG_TAG = "[CODE_IDENTITY]"

REVISION_UNKNOWN = "unknown"

SOURCE_GIT = "git"
SOURCE_NONE = "none"
SOURCES = (SOURCE_GIT, SOURCE_NONE)

REASON_NO_GIT_METADATA = "no_git_metadata"
REASON_GIT_UNAVAILABLE = "git_unavailable"
REASON_GIT_TIMEOUT = "git_timeout"
REASON_GIT_NO_COMMIT = "git_no_commit"
REASON_GIT_UNPARSEABLE = "git_unparseable"
# An error exit is reported as ``git_exit_<code>``: git's own message names
# the directory, so only its exit status is kept.
_REASON_GIT_EXIT_PREFIX = "git_exit_"
REASONS = (
    REASON_NO_GIT_METADATA,
    REASON_GIT_UNAVAILABLE,
    REASON_GIT_TIMEOUT,
    REASON_GIT_NO_COMMIT,
    REASON_GIT_UNPARSEABLE,
)

# Once per cohort start-up, which already takes minutes; a status that takes
# longer than this is reported as unknown rather than holding the start-up.
GIT_TIMEOUT_S = 30.0

_OBJECT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_GIT_EXIT_REASON = re.compile(r"git_exit_-?\d+")
_BRANCH_OID = "# branch.oid "
_FIELDS = ("source", "revision", "dirty", "reason")


@dataclass(frozen=True)
class CodeIdentity:
    source: str
    revision: str
    # None exactly when the revision is unknown.
    dirty: Optional[bool]
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"code identity source {self.source!r} is not one of {SOURCES}")
        if self.revision == REVISION_UNKNOWN:
            if self.dirty is not None:
                raise ValueError("an unknown revision has no dirty verdict")
            if not _known_reason(self.reason):
                raise ValueError(f"unknown code identity reason {self.reason!r}")
            if self.source == SOURCE_NONE and self.reason != REASON_NO_GIT_METADATA:
                raise ValueError(
                    f"source {SOURCE_NONE!r} carries reason "
                    f"{REASON_NO_GIT_METADATA!r}, not {self.reason!r}"
                )
            return
        if self.source != SOURCE_GIT:
            raise ValueError(f"a revision comes from git, not source {self.source!r}")
        if not isinstance(self.revision, str) or not _OBJECT_ID.fullmatch(self.revision):
            raise ValueError(f"code identity revision {self.revision!r} is not a git object id")
        if not isinstance(self.dirty, bool):
            raise ValueError("a known revision carries a boolean dirty verdict")
        if self.reason is not None:
            raise ValueError("a known revision carries no reason")

    def as_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in _FIELDS}

    @classmethod
    def from_dict(cls, payload: Any) -> "CodeIdentity":
        if not isinstance(payload, dict) or set(payload) != set(_FIELDS):
            raise ValueError(
                f"code identity must be an object with exactly {list(_FIELDS)}, "
                f"got {payload!r:.200}"
            )
        return cls(**payload)

    def log_line(self, generation: int, node: int) -> str:
        """The one server-log line of this identity; parsed by tooling, so
        every field is always present and no value contains a space."""
        dirty = "unknown" if self.dirty is None else str(self.dirty).lower()
        return (
            f"{LOG_TAG} generation={int(generation)} node={int(node)} "
            f"source={self.source} revision={self.revision} dirty={dirty} "
            f"reason={self.reason or '-'}"
        )


def _known_reason(reason: Optional[str]) -> bool:
    return reason in REASONS or (
        isinstance(reason, str) and _GIT_EXIT_REASON.fullmatch(reason) is not None
    )


def _unknown(source: str, reason: str) -> CodeIdentity:
    return CodeIdentity(source=source, revision=REVISION_UNKNOWN, dirty=None, reason=reason)


def package_tree() -> Path:
    """The directory holding the imported ``batchgen`` package, symlinks
    resolved: a source tree's top level, or the installed site directory."""
    import batchgen

    return Path(batchgen.__file__).resolve().parent.parent


def git_status_argv(tree: Path) -> list:
    return [
        "git",
        # A read must never take the index lock an operator's git may hold.
        "--no-optional-locks",
        # The workers execute this tree's code, so trusting it for one
        # read-only status adds nothing; without it git refuses a tree owned
        # by another user.
        "-c", f"safe.directory={tree}",
        "-C", str(tree),
        "status", "--porcelain=v2", "--branch", "--no-ahead-behind",
        "--untracked-files=normal",
    ]


def parse_git_status(stdout: str) -> CodeIdentity:
    """Identity from ``git status --porcelain=v2 --branch`` output: the
    ``# branch.oid`` header is the commit, and any entry line is a change."""
    oid: Optional[str] = None
    dirty = False
    for line in stdout.splitlines():
        if line.startswith(_BRANCH_OID):
            oid = line[len(_BRANCH_OID):].strip()
        elif line and not line.startswith("#"):
            dirty = True
    if oid == "(initial)":
        return _unknown(SOURCE_GIT, REASON_GIT_NO_COMMIT)
    if oid is None or not _OBJECT_ID.fullmatch(oid):
        return _unknown(SOURCE_GIT, REASON_GIT_UNPARSEABLE)
    return CodeIdentity(source=SOURCE_GIT, revision=oid, dirty=dirty)


def compute_code_identity(
    tree: Optional[Path] = None,
    *,
    run: Callable[..., Any] = subprocess.run,
    timeout_s: float = GIT_TIMEOUT_S,
) -> CodeIdentity:
    """The identity of the code this process imported from ``tree``
    (default: ``package_tree()``)."""
    tree = package_tree() if tree is None else Path(tree)
    # Exactly at the package's parent: a git repository further up (say, a
    # virtual environment inside some checkout) does not hold this code.
    if not (tree / ".git").exists():
        return _unknown(SOURCE_NONE, REASON_NO_GIT_METADATA)
    try:
        done = run(
            git_status_argv(tree),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _unknown(SOURCE_GIT, REASON_GIT_TIMEOUT)
    except OSError:
        return _unknown(SOURCE_GIT, REASON_GIT_UNAVAILABLE)
    if done.returncode != 0:
        return _unknown(SOURCE_GIT, f"{_REASON_GIT_EXIT_PREFIX}{done.returncode}")
    return parse_git_status(done.stdout)

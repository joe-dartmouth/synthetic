"""No task set has a companion document under `docs/`, and the control that keeps the guard honest.

ADR `tooling/one-home-per-kind-of-statement`: a task set's graph and status live in its
`tasks/<set>-index.md`, its units in `tasks/`, its plan in `docs/RESEARCH.md` and its dates in
`docs/ROADMAP.md`. A per-set page under `docs/` would be a second home for all four.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._corpus import REPO_ROOT


def task_sets(tasks: Path) -> list[str]:
    """The name of every task set, read from its ``<set>-index.md``."""
    return sorted(path.name.removesuffix("-index.md") for path in tasks.glob("*-index.md"))


def _named_for(stem: str, name: str) -> bool:
    """Whether a document stem names the set: the name itself or ``<name>-...``, in any case."""
    stem, name = stem.lower(), name.lower()
    return stem == name or stem.startswith(f"{name}-")


def refuse_companion_documents(docs: Path, sets: list[str], relative_to: Path) -> None:
    """Fail when a markdown file under ``docs`` is a companion to one of ``sets``.

    A companion is named for a set (its stem is the set's name or starts with ``<set>-``, in any
    case) or sits under a ``tasks`` directory inside ``docs``. A stem that merely shares a prefix,
    such as ``proposalish`` for the set ``proposal``, is not a companion. The assertion
    lives here rather than inline so a synthetic tree can drive it.

    Args:
        docs: The documentation directory to walk.
        sets: The task-set names to refuse companions for.
        relative_to: The path the reported names are made relative to.

    Raises:
        AssertionError: When a companion exists, and separately when ``docs`` holds no markdown at
            all, which is what a wrong or deleted ``docs`` path looks like.
    """
    markdown = sorted(p for p in docs.rglob("*") if p.suffix.lower() in {".md", ".markdown"})
    assert markdown, f"{docs} holds no markdown, so this guard would pass by finding nothing"
    found = sorted(
        str(path.relative_to(relative_to))
        for path in markdown
        if "tasks" in path.relative_to(docs).parts[:-1]
        or any(_named_for(path.stem, name) for name in sets)
    )
    assert not found, (
        f"task-set companion document(s) exist: {found}. Route the content to the home that owns "
        "its kind (ADR `tooling/one-home-per-kind-of-statement`) and delete the file."
    )


def test_no_task_set_has_a_companion_document_under_docs() -> None:
    """`docs/` holds topics; a task set's own pages live in `tasks/`.

    A landed set's index is deleted with its last unit, so a `tasks/` with no index is legitimate
    and refuses nothing. The walk still runs, so the floor on `docs/` holding markdown still holds.
    """
    sets = task_sets(REPO_ROOT / "tasks")
    refuse_companion_documents(REPO_ROOT / "docs", sets, REPO_ROOT)


@pytest.mark.parametrize(
    "companion", ["PROPOSAL.md", "proposal-status.md", "Proposal-Notes.MD", "tasks/units.md"]
)
def test_the_companion_guard_fails_on_a_synthetic_docs_tree(tmp_path: Path, companion: str) -> None:
    """The control the committed tree cannot supply, because a matching file is the defect.

    The clean half keeps non-matching documents, one sharing only a prefix with the set name, so
    the guard is shown walking real markdown and still passing, and the floor is shown refusing an
    empty or missing directory.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ARCHITECTURE.md").write_text("A document with a proper home.\n", encoding="utf-8")
    (docs / "proposalish.md").write_text("Shares a prefix, names no set.\n", encoding="utf-8")
    planted = docs / companion
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text("A companion nobody asked for.\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="companion document"):
        refuse_companion_documents(docs, ["proposal"], tmp_path)

    planted.unlink()
    refuse_companion_documents(docs, ["proposal"], tmp_path)

    for path in sorted(docs.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    with pytest.raises(AssertionError, match="holds no markdown"):
        refuse_companion_documents(docs, ["proposal"], tmp_path)

    docs.rmdir()
    with pytest.raises(AssertionError, match="holds no markdown"):
        refuse_companion_documents(docs, ["proposal"], tmp_path)

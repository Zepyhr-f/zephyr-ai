from pathlib import Path

import pytest

from app.ingestion.markdown import (
    MarkdownChunk,
    MarkdownDocument,
    discover_markdown_files,
    load_markdown_file,
    split_markdown_document,
)


def test_discover_markdown_files_returns_md_and_mdx_recursively(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    nested = docs / "nested"
    nested.mkdir(parents=True)
    (docs / "a.md").write_text("# A", encoding="utf-8")
    (nested / "b.mdx").write_text("# B", encoding="utf-8")
    (docs / "c.txt").write_text("# C", encoding="utf-8")

    paths = discover_markdown_files(docs)

    assert paths == [docs / "a.md", nested / "b.mdx"]


def test_discover_markdown_files_ignores_hidden_and_excluded_directories(
    tmp_path: Path,
) -> None:
    for directory in (
        ".git",
        ".hidden",
        "node_modules",
        "dist",
        "build",
        ".venv",
        "__pycache__",
    ):
        ignored = tmp_path / directory
        ignored.mkdir()
        (ignored / "ignored.md").write_text("# Ignored", encoding="utf-8")

    docs = tmp_path / "docs"
    docs.mkdir()
    kept = docs / "kept.md"
    kept.write_text("# Kept", encoding="utf-8")

    assert discover_markdown_files(tmp_path) == [kept]


def test_discover_markdown_files_returns_stable_sorted_paths(tmp_path: Path) -> None:
    for file_name in ("c.md", "a.mdx", "b.md"):
        (tmp_path / file_name).write_text("# Doc", encoding="utf-8")

    assert discover_markdown_files(tmp_path) == [
        tmp_path / "a.mdx",
        tmp_path / "b.md",
        tmp_path / "c.md",
    ]


def test_discover_markdown_files_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_markdown_files(tmp_path / "missing")


def test_discover_markdown_files_rejects_file_root(tmp_path: Path) -> None:
    file_path = tmp_path / "doc.md"
    file_path.write_text("# Doc", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        discover_markdown_files(file_path)


def test_load_markdown_file_reads_frontmatter_title_and_content(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text(
        "---\n"
        "title: Test Doc\n"
        "tags:\n"
        "  - rag\n"
        "---\n\n"
        "# Heading\n\n"
        "Body\n",
        encoding="utf-8",
    )

    document = load_markdown_file(path)

    assert document == MarkdownDocument(
        file_path=str(path),
        title="Test Doc",
        content="# Heading\n\nBody",
        metadata={
            "title": "Test Doc",
            "tags": ["rag"],
            "source_path": str(path),
            "file_name": "doc.md",
            "file_extension": ".md",
        },
    )


def test_load_markdown_file_uses_first_h1_when_frontmatter_has_no_title(tmp_path: Path) -> None:
    path = tmp_path / "api.md"
    path.write_text("---\ncategory: docs\n---\n\n# API Guide\n\nBody", encoding="utf-8")

    document = load_markdown_file(path)

    assert document.title == "API Guide"
    assert document.metadata["category"] == "docs"


def test_load_markdown_file_uses_file_stem_without_frontmatter_or_h1(tmp_path: Path) -> None:
    path = tmp_path / "plain-notes.mdx"
    path.write_text("plain body", encoding="utf-8")

    document = load_markdown_file(path)

    assert document.title == "plain-notes"


def test_load_markdown_file_works_without_frontmatter(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("# Heading\n\nBody", encoding="utf-8")

    document = load_markdown_file(path)

    assert document.content == "# Heading\n\nBody"
    assert document.metadata == {
        "source_path": str(path),
        "file_name": "doc.md",
        "file_extension": ".md",
    }


def test_load_markdown_file_rejects_non_markdown_file(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("# Doc", encoding="utf-8")

    with pytest.raises(ValueError, match="Markdown"):
        load_markdown_file(path)


def test_split_markdown_document_splits_by_headings() -> None:
    document = MarkdownDocument(
        file_path="docs/project.md",
        title="Project",
        content="# Intro\nA\n\n## Install\nB\n\n## Usage\nC",
        metadata={"source_path": "docs/project.md"},
    )

    chunks = split_markdown_document(document)

    assert [chunk.header_path for chunk in chunks] == ["Intro", "Intro > Install", "Intro > Usage"]
    assert [chunk.content for chunk in chunks] == ["# Intro\nA", "## Install\nB", "## Usage\nC"]


def test_split_markdown_document_preserves_nested_header_path() -> None:
    document = MarkdownDocument(
        file_path="docs/project.md",
        title="Project",
        content="# Project\n## Setup\n### Docker\nContent",
        metadata={"source_path": "docs/project.md"},
    )

    chunks = split_markdown_document(document)

    assert chunks[0].header_path == "Project"
    assert chunks[1].header_path == "Project > Setup"
    assert chunks[2].header_path == "Project > Setup > Docker"


def test_split_markdown_document_handles_content_before_first_heading() -> None:
    document = MarkdownDocument(
        file_path="docs/project.md",
        title="Project",
        content="Intro paragraph.\n\n# Section\nBody",
        metadata={"source_path": "docs/project.md"},
    )

    chunks = split_markdown_document(document)

    assert chunks[0].header_path is None
    assert chunks[0].content == "Intro paragraph."
    assert chunks[1].header_path == "Section"
    assert chunks[1].content == "# Section\nBody"


def test_split_markdown_document_assigns_sequential_chunk_indexes() -> None:
    document = MarkdownDocument(
        file_path="docs/project.md",
        title="Project",
        content="# One\nA\n\n# Two\nB\n\n# Three\nC",
        metadata={"source_path": "docs/project.md"},
    )

    chunks = split_markdown_document(document)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]


def test_split_markdown_document_carries_document_metadata() -> None:
    document = MarkdownDocument(
        file_path="docs/project.md",
        title="Project",
        content="# Intro\nA",
        metadata={"source_path": "docs/project.md", "category": "docs"},
    )

    chunks = split_markdown_document(document)

    assert chunks == [
        MarkdownChunk(
            chunk_index=0,
            content="# Intro\nA",
            header_path="Intro",
            metadata={
                "source_path": "docs/project.md",
                "category": "docs",
                "chunk_index": 0,
                "header_path": "Intro",
            },
        )
    ]


def test_split_markdown_document_splits_large_section_by_chunk_size() -> None:
    document = MarkdownDocument(
        file_path="docs/project.md",
        title="Project",
        content="# Intro\nabcdefghijklmnopqrstuvwxyz",
        metadata={"source_path": "docs/project.md"},
    )

    chunks = split_markdown_document(document, chunk_size=12, chunk_overlap=2)

    assert len(chunks) > 1
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.header_path == "Intro" for chunk in chunks)
    assert all(len(chunk.content) <= 12 for chunk in chunks)


def test_split_markdown_document_applies_chunk_overlap() -> None:
    document = MarkdownDocument(
        file_path="docs/project.md",
        title="Project",
        content="abcdefghijklmnopqrstuv",
        metadata={"source_path": "docs/project.md"},
    )

    chunks = split_markdown_document(document, chunk_size=10, chunk_overlap=3)

    assert chunks[0].content == "abcdefghij"
    assert chunks[1].content.startswith("hij")


def test_split_markdown_document_validates_chunk_size_and_overlap() -> None:
    document = MarkdownDocument(
        file_path="docs/project.md",
        title="Project",
        content="# Intro\nA",
        metadata={"source_path": "docs/project.md"},
    )

    with pytest.raises(ValueError, match="chunk_size"):
        split_markdown_document(document, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_markdown_document(document, chunk_overlap=-1)
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_markdown_document(document, chunk_size=10, chunk_overlap=10)


def test_split_markdown_document_is_deterministic() -> None:
    document = MarkdownDocument(
        file_path="docs/project.md",
        title="Project",
        content="# Intro\nA\n\n## Install\nB",
        metadata={"source_path": "docs/project.md", "category": "docs"},
    )

    first = split_markdown_document(document, chunk_size=20, chunk_overlap=5)
    second = split_markdown_document(document, chunk_size=20, chunk_overlap=5)

    assert first == second

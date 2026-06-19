from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re


MARKDOWN_EXTENSIONS = {".md", ".mdx"}
EXCLUDED_DIRS = {".git", "node_modules", "dist", "build", ".venv", "__pycache__"}
H1_PATTERN = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class MarkdownDocument:
    file_path: str
    title: str
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MarkdownChunk:
    chunk_index: int
    content: str
    header_path: str | None
    metadata: dict[str, Any]


def discover_markdown_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(root_path)
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)

    markdown_files: list[Path] = []
    for path in root_path.rglob("*"):
        if any(part.startswith(".") or part in EXCLUDED_DIRS for part in path.relative_to(root_path).parts[:-1]):
            continue
        if path.is_file() and path.suffix.lower() in MARKDOWN_EXTENSIONS:
            markdown_files.append(path)

    return sorted(markdown_files)


def load_markdown_file(path: str | Path) -> MarkdownDocument:
    file_path = Path(path)
    if file_path.suffix.lower() not in MARKDOWN_EXTENSIONS:
        raise ValueError(f"Markdown file expected, got: {file_path}")

    raw_content = file_path.read_text(encoding="utf-8")
    metadata, content = _parse_frontmatter(raw_content)
    content = content.strip()
    title = _resolve_title(metadata, content, file_path)
    metadata.update(
        {
            "source_path": str(file_path),
            "file_name": file_path.name,
            "file_extension": file_path.suffix.lower(),
        }
    )

    return MarkdownDocument(
        file_path=str(file_path),
        title=title,
        content=content,
        metadata=metadata,
    )


def split_markdown_document(
    document: MarkdownDocument,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[MarkdownChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[MarkdownChunk] = []
    for content, header_path in _iter_markdown_sections(document.content):
        if not content.strip():
            continue
        for piece in _split_content_window(content.strip(), chunk_size, chunk_overlap):
            chunk_index = len(chunks)
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "source_path": document.metadata.get("source_path", document.file_path),
                    "chunk_index": chunk_index,
                    "header_path": header_path,
                }
            )
            chunks.append(
                MarkdownChunk(
                    chunk_index=chunk_index,
                    content=piece,
                    header_path=header_path,
                    metadata=metadata,
                )
            )

    return chunks


def _iter_markdown_sections(content: str) -> list[tuple[str, str | None]]:
    sections: list[tuple[str, str | None]] = []
    current_lines: list[str] = []
    current_header_path: str | None = None
    header_stack: list[str] = []

    for line in content.splitlines():
        heading = HEADING_PATTERN.match(line)
        if heading:
            if current_lines:
                sections.append(("\n".join(current_lines), current_header_path))

            level = len(heading.group(1))
            title = heading.group(2).strip()
            header_stack = header_stack[: level - 1]
            header_stack.append(title)
            current_header_path = " > ".join(header_stack)
            current_lines = [line]
            continue

        current_lines.append(line)

    if current_lines:
        sections.append(("\n".join(current_lines), current_header_path))

    return sections


def _split_content_window(content: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(content) <= chunk_size:
        return [content]

    pieces: list[str] = []
    start = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))
        pieces.append(content[start:end])
        if end == len(content):
            break
        start = end - chunk_overlap

    return pieces


def _parse_frontmatter(raw_content: str) -> tuple[dict[str, Any], str]:
    lines = raw_content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw_content

    closing_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break

    if closing_index is None:
        return {}, raw_content

    metadata = _parse_frontmatter_metadata(lines[1:closing_index])
    content = "\n".join(lines[closing_index + 1 :])
    return metadata, content


def _parse_frontmatter_metadata(lines: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    current_list_key: str | None = None

    for line in lines:
        if not line.strip():
            continue

        stripped = line.strip()
        if stripped.startswith("-") and current_list_key:
            metadata[current_list_key].append(stripped[1:].strip())
            continue

        if ":" not in line:
            current_list_key = None
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            current_list_key = None
            continue

        if value:
            metadata[key] = value
            current_list_key = None
        else:
            metadata[key] = []
            current_list_key = key

    return metadata


def _resolve_title(metadata: dict[str, Any], content: str, file_path: Path) -> str:
    frontmatter_title = metadata.get("title")
    if frontmatter_title:
        return str(frontmatter_title)

    match = H1_PATTERN.search(content)
    if match:
        return match.group(1).strip()

    return file_path.stem

from app.models.conversation import Message
from app.retrieval.service import SemanticSearchResult

SYSTEM_PROMPT = """你是 Zephyr 项目的 RAG 助手。请只基于给定上下文回答问题。
如果上下文不足以回答，请明确说明当前知识库没有足够信息。
回答要简洁、准确，并尽量引用 Zephyr 项目的真实术语。"""


def build_prompt(
    message: str,
    results: list[SemanticSearchResult],
    history: list[Message] | None = None,
) -> str:
    if not results:
        context = "当前没有检索到相关知识库片段。"
    else:
        context = "\n\n".join(format_context(index, result) for index, result in enumerate(results, start=1))

    history_block = ""
    if history:
        formatted_history = "\n".join(format_history(item) for item in history)
        history_block = f"历史对话：\n{formatted_history}\n\n"

    return (
        "请基于以下知识库上下文回答用户问题。\n\n"
        f"{history_block}"
        f"知识库上下文：\n{context}\n\n"
        f"用户问题：\n{message}\n\n"
        "回答要求：\n"
        "1. 优先使用上下文中的信息。\n"
        "2. 如果上下文不足，请说明不足。\n"
        "3. 不要编造未出现在上下文中的部署细节、密钥或连接串。"
    )


def format_context(index: int, result: SemanticSearchResult) -> str:
    title = result.document_title or "Untitled"
    header = result.header_path or ""
    return (
        f"[{index}] title={title}\n"
        f"file={result.document_file_path}\n"
        f"header={header}\n"
        f"score={result.score:.6f}\n"
        f"content={result.content}"
    )


def format_history(message: Message) -> str:
    content = message.content.replace("\n", " ")[:800]
    return f"{message.role}: {content}"


def source_from_result(result: SemanticSearchResult):
    from app.rag.schemas import RagSource

    preview = result.content.replace("\n", " ")[:200]
    return RagSource(
        title=result.document_title,
        file_path=result.document_file_path,
        header_path=result.header_path,
        score=result.score,
        preview=preview,
        chunk_index=chunk_index(result),
    )


def chunk_index(result: SemanticSearchResult) -> int | None:
    metadata = result.metadata or {}
    try:
        return int(metadata.get("chunk_index"))
    except (TypeError, ValueError):
        return None

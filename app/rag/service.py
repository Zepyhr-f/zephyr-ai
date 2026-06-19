import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm_client import ChatMessage, LLMClient
from app.models.conversation import Conversation, Message
from app.rag.schemas import RagChatResponse, RagSource
from app.retrieval.service import SemanticRetrievalService, SemanticSearchResult

SYSTEM_PROMPT = """你是 Zephyr 项目的 RAG 助手。请只基于给定上下文回答问题。
如果上下文不足以回答，请明确说明当前知识库没有足够信息。
回答要简洁、准确，并尽量引用 Zephyr 项目的真实术语。"""


@dataclass(frozen=True)
class RagChatResult:
    conversation_id: str
    answer: str
    sources: list[RagSource]


class RagChatService:
    def __init__(
        self,
        *,
        retrieval_service: SemanticRetrievalService,
        llm_client: LLMClient,
        session: AsyncSession,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm_client = llm_client
        self.session = session

    async def chat(
        self,
        *,
        message: str,
        conversation_id: str | None = None,
        top_k: int = 5,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        include_history: bool = True,
    ) -> RagChatResponse:
        normalized_message = message.strip()
        if not normalized_message:
            raise ValueError("message must not be blank")

        conversation = await self._get_or_create_conversation(
            conversation_id=conversation_id,
            title=normalized_message[:80],
        )
        history = []
        if conversation_id and include_history:
            history = await self._recent_messages(conversation.id)

        results = await self.retrieval_service.search(normalized_message, limit=top_k)
        prompt = self._build_prompt(normalized_message, results, history)
        answer = await self.llm_client.chat(
            [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=prompt),
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        sources = [self._source_from_result(result) for result in results]
        self.session.add(
            Message(
                conversation_id=conversation.id,
                role="user",
                content=normalized_message,
                tool_calls=None,
            )
        )
        self.session.add(
            Message(
                conversation_id=conversation.id,
                role="assistant",
                content=answer,
                tool_calls={"sources": [source.model_dump() for source in sources]},
            )
        )
        await self.session.flush()

        return RagChatResponse(
            conversation_id=conversation.thread_id,
            answer=answer,
            sources=sources,
        )

    async def _get_or_create_conversation(
        self,
        *,
        conversation_id: str | None,
        title: str,
    ) -> Conversation:
        if conversation_id:
            result = await self.session.execute(
                select(Conversation).where(Conversation.thread_id == conversation_id)
            )
            conversation = result.scalar_one_or_none()
            if conversation is not None:
                return conversation

        conversation = Conversation(
            thread_id=conversation_id or str(uuid.uuid4()),
            title=title,
        )
        self.session.add(conversation)
        await self.session.flush()
        return conversation

    async def _recent_messages(self, conversation_db_id: uuid.UUID) -> list[Message]:
        result = await self.session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_db_id)
            .order_by(Message.created_at.desc())
            .limit(6)
        )
        return list(reversed(result.scalars().all()))

    def _build_prompt(
        self,
        message: str,
        results: list[SemanticSearchResult],
        history: list[Message] | None = None,
    ) -> str:
        if not results:
            context = "当前没有检索到相关知识库片段。"
        else:
            context = "\n\n".join(
                self._format_context(index, result)
                for index, result in enumerate(results, start=1)
            )

        history_block = ""
        if history:
            formatted_history = "\n".join(self._format_history(item) for item in history)
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

    def _format_context(self, index: int, result: SemanticSearchResult) -> str:
        title = result.document_title or "Untitled"
        header = result.header_path or ""
        return (
            f"[{index}] title={title}\n"
            f"file={result.document_file_path}\n"
            f"header={header}\n"
            f"score={result.score:.6f}\n"
            f"content={result.content}"
        )

    def _format_history(self, message: Message) -> str:
        content = message.content.replace("\n", " ")[:800]
        return f"{message.role}: {content}"

    def _source_from_result(self, result: SemanticSearchResult) -> RagSource:
        preview = result.content.replace("\n", " ")[:200]
        return RagSource(
            title=result.document_title,
            file_path=result.document_file_path,
            header_path=result.header_path,
            score=result.score,
            preview=preview,
            chunk_index=self._chunk_index(result),
        )

    def _chunk_index(self, result: SemanticSearchResult) -> int | None:
        metadata = result.metadata or {}
        try:
            return int(metadata.get("chunk_index"))
        except (TypeError, ValueError):
            return None

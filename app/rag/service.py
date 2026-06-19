import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm_client import ChatMessage, LLMClient
from app.models.conversation import Conversation, Message
from app.rag.prompt import SYSTEM_PROMPT, build_prompt, chunk_index, source_from_result
from app.rag.schemas import RagChatResponse, RagSource
from app.retrieval.service import SemanticRetrievalService, SemanticSearchResult


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
        prompt = build_prompt(normalized_message, results, history)
        answer = await self.llm_client.chat(
            [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=prompt),
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        sources = [source_from_result(result) for result in results]
        self.session.add(
            Message(
                conversation_id=conversation.id,
                role="user",
                content=normalized_message,
                tool_calls=None,
            )
        )
        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            tool_calls={"sources": [source.model_dump() for source in sources]},
        )
        self.session.add(assistant_message)
        await self.session.flush()

        return RagChatResponse(
            conversation_id=conversation.thread_id,
            answer=answer,
            sources=sources,
            assistant_message_id=str(assistant_message.id),
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
        return build_prompt(message, results, history)

    def _source_from_result(self, result: SemanticSearchResult) -> RagSource:
        return source_from_result(result)

    def _chunk_index(self, result: SemanticSearchResult) -> int | None:
        return chunk_index(result)

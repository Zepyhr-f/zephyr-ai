from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.feedback.schemas import FeedbackCreateRequest, FeedbackResponse, FeedbackStats
from app.feedback.service import FeedbackService

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse, status_code=201)
async def create_feedback(
    request: FeedbackCreateRequest,
    session: AsyncSession = Depends(get_db_session),
) -> FeedbackResponse:
    response = await FeedbackService(session=session).create(request)
    await session.commit()
    return response


@router.get("/stats", response_model=FeedbackStats)
async def feedback_stats(session: AsyncSession = Depends(get_db_session)) -> FeedbackStats:
    return await FeedbackService(session=session).stats()

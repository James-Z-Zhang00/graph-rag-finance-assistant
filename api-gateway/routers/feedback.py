"""
Feedback router — lightweight acknowledgement handler.
"""

from fastapi import APIRouter
from models.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    action = "positive_acknowledged" if request.is_positive else "negative_acknowledged"
    return FeedbackResponse(status="success", action=action)

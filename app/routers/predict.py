
import time
import logging
from fastapi import APIRouter, HTTPException
from app.schemas.schemas import (
    ReviewInput,
    BatchReviewInput,
    PredictionResponse,
    BatchPredictionResponse
)
from app.services.model_service import model_service

logger = APIRouter = APIRouter
router = __import__("fastapi").APIRouter(prefix="/predict", tags=["Predictions"])
logger = logging.getLogger(__name__)


@router.post("/single", response_model=PredictionResponse)
async def predict_single(review: ReviewInput):
    """
    Predict segment + sentiment + retention risk for one review.

    Returns:
    - Customer segment (0-4) with name
    - Predicted sentiment (negative/neutral/positive)
    - Confidence scores for both predictions
    - Retention risk level and priority tier
    - Recommended retention strategy
    """
    if not model_service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = model_service.predict_batch([review.model_dump()])
        return PredictionResponse(**results[0])
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(payload: BatchReviewInput):
    """
    Predict for up to 1000 reviews in one request.
    Returns predictions + total count + processing time.

    Useful for:
    - Bulk scoring of existing customer base
    - Nightly batch jobs
    - Backfilling historical data
    """
    if not model_service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        t0 = time.time()
        feature_dicts = [r.model_dump() for r in payload.reviews]
        results = model_service.predict_batch(feature_dicts)
        elapsed_ms = round((time.time() - t0) * 1000, 2)

        return BatchPredictionResponse(
            predictions       = [PredictionResponse(**r) for r in results],
            total             = len(results),
            processing_time_ms= elapsed_ms,
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

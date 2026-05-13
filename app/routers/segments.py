
from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas.schemas import SegmentSummaryResponse, RiskLevel
from app.services.model_service import model_service, SEGMENT_NAMES, SEGMENT_RISK, SEGMENT_STRATEGY

router = APIRouter(prefix="/segments", tags=["Segments"])


@router.get("/", response_model=List[SegmentSummaryResponse])
async def list_segments():
    """
    Returns a summary of all 5 customer segments.
    Useful for dashboards and business intelligence views.
    """
    summaries = []
    for seg_id, insight in model_service.segment_insights.items():
        k = int(seg_id)
        summaries.append(SegmentSummaryResponse(
            segment_id      = k,
            segment_name    = SEGMENT_NAMES[k],
            risk_level      = RiskLevel(SEGMENT_RISK[k]),
            avg_score       = insight["avg_score"],
            pct_positive    = insight["pct_positive"],
            top_platform    = insight["top_platform"],
            business_insight= insight["business_insight"],
        ))
    return summaries


@router.get("/{segment_id}", response_model=SegmentSummaryResponse)
async def get_segment(segment_id: int):
    """Get details for a specific segment by ID (0-4)."""
    if segment_id not in range(5):
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")

    insight = model_service.segment_insights.get(str(segment_id))
    if not insight:
        raise HTTPException(status_code=404, detail="Segment insight not available")

    return SegmentSummaryResponse(
        segment_id      = segment_id,
        segment_name    = SEGMENT_NAMES[segment_id],
        risk_level      = RiskLevel(SEGMENT_RISK[segment_id]),
        avg_score       = insight["avg_score"],
        pct_positive    = insight["pct_positive"],
        top_platform    = insight["top_platform"],
        business_insight= insight["business_insight"],
    )

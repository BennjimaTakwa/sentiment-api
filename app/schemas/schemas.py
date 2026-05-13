
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum


class SentimentLabel(str, Enum):
    negative = "negative"
    neutral  = "neutral"
    positive = "positive"


class RiskLevel(str, Enum):
    low      = "low"
    medium   = "medium"
    high     = "high"
    critical = "critical"


# ── Single review input
class ReviewInput(BaseModel):
    frustration_score  : float = Field(..., ge=0.0, description="Frustration score 0+")
    engagement_quality : float = Field(..., ge=0.0, le=1.0, description="Engagement 0-1")
    influence_weight   : float = Field(..., ge=0.0, le=1.0, description="Influence 0-1")
    recency_weight     : float = Field(..., ge=0.0, le=1.0, description="Recency 0-1")
    word_count         : int   = Field(..., ge=0, description="Word count")
    has_reply          : int   = Field(..., ge=0, le=1, description="1 if app replied")
    bilstm_prob_neg    : float = Field(..., ge=0.0, le=1.0)
    bilstm_prob_neu    : float = Field(..., ge=0.0, le=1.0)
    bilstm_prob_pos    : float = Field(..., ge=0.0, le=1.0)
    bilstm_confidence  : float = Field(..., ge=0.0, le=1.0)

    @field_validator("bilstm_prob_neg", "bilstm_prob_neu", "bilstm_prob_pos")
    @classmethod
    def probs_valid(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Probability must be between 0 and 1")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "frustration_score": 1.5,
                "engagement_quality": 0.3,
                "influence_weight": 0.6,
                "recency_weight": 0.4,
                "word_count": 25,
                "has_reply": 0,
                "bilstm_prob_neg": 0.6,
                "bilstm_prob_neu": 0.3,
                "bilstm_prob_pos": 0.1,
                "bilstm_confidence": 0.75
            }
        }
    }


# ── Batch input
class BatchReviewInput(BaseModel):
    reviews: List[ReviewInput] = Field(..., min_length=1, max_length=1000)


# ── Prediction response
class PredictionResponse(BaseModel):
    segment_id         : int
    segment_name       : str
    sentiment          : SentimentLabel
    segment_confidence : float
    sentiment_confidence: float
    risk_level         : RiskLevel
    priority_tier      : str
    recommended_strategy: str
    retention_risk_score: float


# ── Batch response
class BatchPredictionResponse(BaseModel):
    predictions : List[PredictionResponse]
    total       : int
    processing_time_ms: float


# ── Health check response
class HealthResponse(BaseModel):
    model_config   = {"protected_namespaces": ()}
    status     : str
    model_loaded: bool
    version    : str
    uptime_seconds: float


# ── Segment summary response
class SegmentSummaryResponse(BaseModel):
    segment_id  : int
    segment_name: str
    risk_level  : RiskLevel
    avg_score   : float
    pct_positive: float
    top_platform: str
    business_insight: str

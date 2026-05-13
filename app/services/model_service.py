
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import joblib
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Model directory (relative to this file inside Docker)
MODEL_DIR = Path(__file__).parent.parent / "models"

# ── Segment metadata (hardcoded — never changes)
SEGMENT_NAMES = {
    0: "Satisfied Loyal Shoppers",
    1: "Frustrated Complainers",
    2: "Neutral Browsers",
    3: "Impulsive Buyers",
    4: "Engaged Brand Advocates",
}

SEGMENT_RISK = {
    0: "low",
    1: "critical",
    2: "medium",
    3: "high",
    4: "medium",
}

SEGMENT_STRATEGY = {
    0: "Reward & deepen loyalty",
    1: "Urgent complaint resolution & proactive outreach",
    2: "Re-engagement campaigns & discovery features",
    3: "Personalized offers & purchase incentives",
    4: "Community building & brand advocacy programs",
}

RISK_TO_PRIORITY = {
    "critical": "P1 - Critical",
    "high"    : "P2 - High",
    "medium"  : "P3 - Medium",
    "low"     : "P4 - Low",
}

SEGMENT_RISK_SCORES = {
    0: 0.0,
    1: 61.1,
    2: 47.1,
    3: 75.6,
    4: 100.0,
}

SENTIMENT_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


# ── Dual-head MLP (identical to nb-07)
class BehaviorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_segments, n_sentiments, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        self.backbone       = nn.Sequential(*layers)
        self.segment_head   = nn.Linear(in_dim, n_segments)
        self.sentiment_head = nn.Linear(in_dim, n_sentiments)

    def forward(self, x):
        shared = self.backbone(x)
        return self.segment_head(shared), self.sentiment_head(shared)


class ModelService:
    """
    Singleton ML service.
    Loads all models once at startup, serves predictions efficiently.
    Thread-safe for concurrent FastAPI requests.
    """

    _instance = None
    _startup_time = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized:
            return

        logger.info("Initializing ModelService...")
        start = time.time()

        # ── Load feature column list
        with open(MODEL_DIR / "feature_cols.json") as f:
            self.feature_cols = json.load(f)

        # ── Load sklearn artifacts
        self.scaler       = joblib.load(MODEL_DIR / "mlp_scaler.pkl")
        self.le_segment   = joblib.load(MODEL_DIR / "le_segment.pkl")
        self.le_sentiment = joblib.load(MODEL_DIR / "le_sentiment.pkl")

        # ── Load segment insights
        with open(MODEL_DIR / "segment_insights.json") as f:
            self.segment_insights = json.load(f)

        # ── Build MLP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = BehaviorMLP(
            input_dim    = len(self.feature_cols),
            hidden_dims  = [256, 128, 64],
            n_segments   = 5,
            n_sentiments = 3,
            dropout      = 0.3,
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(
                MODEL_DIR / "mlp_best.pt",
                map_location=self.device
            )
        )
        self.model.eval()

        self._startup_time = time.time()
        self._initialized  = True

        elapsed = time.time() - start
        logger.info(f"ModelService initialized in {elapsed:.2f}s on {self.device}")

    def predict_batch(self, feature_dicts: List[Dict]) -> List[Dict]:
        """
        Run inference on a list of feature dicts.
        Returns list of prediction dicts ready for API response.
        """
        # ── Build numpy matrix in correct column order
        X = np.array([
            [row[col] for col in self.feature_cols]
            for row in feature_dicts
        ], dtype=np.float32)

        # ── Scale
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        # ── Inference
        with torch.no_grad():
            seg_logits, sent_logits = self.model(X_tensor)
            seg_probs  = torch.softmax(seg_logits,  dim=1).cpu().numpy()
            sent_probs = torch.softmax(sent_logits, dim=1).cpu().numpy()

        seg_ids  = seg_probs.argmax(axis=1)
        sent_ids = sent_probs.argmax(axis=1)

        results = []
        for i in range(len(X)):
            seg_id  = int(seg_ids[i])
            sent_id = int(sent_ids[i])
            risk    = SEGMENT_RISK[seg_id]

            results.append({
                "segment_id"          : seg_id,
                "segment_name"        : SEGMENT_NAMES[seg_id],
                "sentiment"           : SENTIMENT_NAMES[sent_id],
                "segment_confidence"  : round(float(seg_probs[i].max()), 4),
                "sentiment_confidence": round(float(sent_probs[i].max()), 4),
                "risk_level"          : risk,
                "priority_tier"       : RISK_TO_PRIORITY[risk],
                "recommended_strategy": SEGMENT_STRATEGY[seg_id],
                "retention_risk_score": SEGMENT_RISK_SCORES[seg_id],
            })

        return results

    @property
    def uptime_seconds(self) -> float:
        if self._startup_time is None:
            return 0.0
        return round(time.time() - self._startup_time, 2)

    @property
    def is_ready(self) -> bool:
        return self._initialized


# ── Global singleton instance
model_service = ModelService()

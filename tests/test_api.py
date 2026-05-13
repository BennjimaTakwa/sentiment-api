
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.model_service import model_service

# ── Initialize model once for all tests
model_service.initialize()
client = TestClient(app)

# ── Sample valid payload
VALID_PAYLOAD = {
    "frustration_score"  : 1.5,
    "engagement_quality" : 0.3,
    "influence_weight"   : 0.6,
    "recency_weight"     : 0.4,
    "word_count"         : 25,
    "has_reply"          : 0,
    "bilstm_prob_neg"    : 0.6,
    "bilstm_prob_neu"    : 0.3,
    "bilstm_prob_pos"    : 0.1,
    "bilstm_confidence"  : 0.75
}


class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/health/")
        assert r.status_code == 200

    def test_health_model_loaded(self):
        r = client.get("/health/")
        assert r.json()["model_loaded"] is True

    def test_readiness_returns_200(self):
        r = client.get("/health/ready")
        assert r.status_code == 200


class TestPredictSingle:
    def test_single_returns_200(self):
        r = client.post("/predict/single", json=VALID_PAYLOAD)
        assert r.status_code == 200

    def test_single_response_has_required_fields(self):
        r = client.post("/predict/single", json=VALID_PAYLOAD)
        data = r.json()
        required = [
            "segment_id", "segment_name", "sentiment",
            "segment_confidence", "risk_level",
            "priority_tier", "recommended_strategy"
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_segment_id_in_valid_range(self):
        r = client.post("/predict/single", json=VALID_PAYLOAD)
        assert r.json()["segment_id"] in range(5)

    def test_sentiment_valid_label(self):
        r = client.post("/predict/single", json=VALID_PAYLOAD)
        assert r.json()["sentiment"] in ["negative", "neutral", "positive"]

    def test_invalid_payload_returns_422(self):
        r = client.post("/predict/single", json={"frustration_score": -999})
        assert r.status_code == 422


class TestPredictBatch:
    def test_batch_returns_200(self):
        r = client.post("/predict/batch", json={"reviews": [VALID_PAYLOAD] * 5})
        assert r.status_code == 200

    def test_batch_count_matches(self):
        n = 10
        r = client.post("/predict/batch", json={"reviews": [VALID_PAYLOAD] * n})
        assert r.json()["total"] == n

    def test_batch_has_processing_time(self):
        r = client.post("/predict/batch", json={"reviews": [VALID_PAYLOAD]})
        assert "processing_time_ms" in r.json()


class TestSegments:
    def test_list_segments_returns_5(self):
        r = client.get("/segments/")
        assert r.status_code == 200
        assert len(r.json()) == 5

    def test_get_segment_by_id(self):
        r = client.get("/segments/0")
        assert r.status_code == 200
        assert r.json()["segment_id"] == 0

    def test_invalid_segment_returns_404(self):
        r = client.get("/segments/99")
        assert r.status_code == 404

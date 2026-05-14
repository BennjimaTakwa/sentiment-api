#  Customer Sentiment & Retention API

> Production-ready REST API for AI-powered customer segmentation, sentiment analysis, and retention risk scoring — built with FastAPI and deployed via Docker.

🔗 **Live API**: https://sentiment-api-8mdq.onrender.com/docs

---

##  Overview

This API serves predictions from a dual-head MLP neural network trained on 100,000 e-commerce reviews. It classifies customers into behavioral segments, detects sentiment, and computes retention risk scores — all in a single inference call.

---

##  Quick Start

### Health Check
```bash
curl https://sentiment-api-8mdq.onrender.com/health/
```

### Single Prediction
```bash
curl -X POST https://sentiment-api-8mdq.onrender.com/predict/single \
-H "Content-Type: application/json" \
-d '{
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
}'
```

### Response
```json
{
  "segment_id": 3,
  "segment_name": "Impulsive Buyers",
  "sentiment": "negative",
  "segment_confidence": 0.5965,
  "sentiment_confidence": 1.0,
  "risk_level": "high",
  "priority_tier": "P2 - High",
  "recommended_strategy": "Personalized offers & purchase incentives",
  "retention_risk_score": 75.6
}
```

---

##  Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health/` | Liveness probe — model status and uptime |
| GET | `/health/ready` | Readiness probe — returns 503 if model not loaded |
| POST | `/predict/single` | Score one customer review |
| POST | `/predict/batch` | Score up to 1000 reviews in one request |
| GET | `/segments/` | List all 5 customer segments with metadata |
| GET | `/segments/{id}` | Get details for a specific segment (0–4) |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc UI |

Input Features (10)
↓
Linear(10 → 256) + BatchNorm + ReLU + Dropout(0.3)
↓
Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.3)
↓
Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.3)
↓
┌───────────────┐
↓               ↓
Segment Head    Sentiment Head
Linear(64→5)    Linear(64→3)
↓               ↓
5 Segments      3 Sentiments

**Parameters:** 45,384  
**Training data:** 100,000 e-commerce reviews  
**Platforms:** 11 (Amazon, Alibaba, Lazada, Walmart, Shein, etc.)

---

## 📥 Input Features

| Feature | Type | Description |
|---|---|---|
| `frustration_score` | float (0+) | Customer frustration level |
| `engagement_quality` | float (0–1) | Quality of engagement |
| `influence_weight` | float (0–1) | Customer's public influence |
| `recency_weight` | float (0–1) | Recency of interaction |
| `word_count` | int (0+) | Review word count |
| `has_reply` | int (0 or 1) | Whether platform replied |
| `bilstm_prob_neg` | float (0–1) | BiLSTM negative probability |
| `bilstm_prob_neu` | float (0–1) | BiLSTM neutral probability |
| `bilstm_prob_pos` | float (0–1) | BiLSTM positive probability |
| `bilstm_confidence` | float (0–1) | BiLSTM confidence score |

---

##  Customer Segments

| ID | Name | Risk Level | Priority |
|---|---|---|---|
| 0 | Satisfied Loyal Shoppers | 🟢 Low | P4 |
| 1 | Frustrated Complainers | 🟠 High | P2 |
| 2 | Neutral Browsers | 🟡 Medium | P3 |
| 3 | Impulsive Buyers | 🟠 High | P2 |
| 4 | Engaged Brand Advocates | 🔴 Critical | P1 |

---

##  Run Locally with Docker

```bash
git clone https://github.com/BennjimaTakwa/sentiment-api.git
cd sentiment-api
docker-compose up --build
```

API will be available at:

http://localhost:8000/docs
---

##  Model Architecture : 

sentiment-api/
├── app/
│   ├── main.py              ← FastAPI app — lifespan, CORS, middleware
│   ├── models/              ← ML model artifacts (.pt, .pkl, .json)
│   ├── routers/
│   │   ├── health.py        ← Liveness + readiness probes
│   │   ├── predict.py       ← Single + batch inference endpoints
│   │   └── segments.py      ← Segment metadata endpoints
│   ├── schemas/
│   │   └── schemas.py       ← Pydantic v2 request/response models
│   └── services/
│       └── model_service.py ← Singleton ML service — loads model once
├── tests/
│   └── test_api.py          ← 14 pytest tests (100% passing)
├── Dockerfile               ← python:3.11-slim, non-root user
├── docker-compose.yml       ← Service + health check
├── render.yaml              ← Render deployment config
└── requirements.txt         ← Pinned dependencies
---

## 🛠️ Tech Stack

- **Framework** — FastAPI 0.111.0
- **Server** — Uvicorn 0.29.0
- **ML** — PyTorch 2.2.0 (CPU)
- **Validation** — Pydantic v2
- **Serialization** — Joblib, JSON
- **Containerization** — Docker + Docker Compose
- **Deployment** — Render
- **Testing** — pytest 8.4.2

---

## 🔗 Related

| Service | URL |
|---|---|
|  Live Dashboard | https://sentiment-dashboard-f8agdbvg5epa4kt4etasaz.streamlit.app |
|  ML Notebooks | https://github.com/BennjimaTakwa/customer-retention-DL |
|  Dashboard Code | https://github.com/BennjimaTakwa/sentiment-dashboard |

---

##  Authors

**Bennjimatakwa** · **MabroukYahya**  


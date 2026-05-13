Customer Sentiment & Retention API

AI-powered REST API for customer segmentation, sentiment analysis, and retention risk scoring.

========================
QUICK START
========================

docker-compose up --build

Test:
curl http://localhost:8000/health/

curl -X POST http://localhost:8000/predict/single -H "Content-Type: application/json" -d '{"frustration_score":1.5,"engagement_quality":0.3,"influence_weight":0.6,"recency_weight":0.4,"word_count":25,"has_reply":0,"bilstm_prob_neg":0.6,"bilstm_prob_neu":0.3,"bilstm_prob_pos":0.1,"bilstm_confidence":0.75}'

========================
API ENDPOINTS
========================

GET  /health/
GET  /health/ready
POST /predict/single
POST /predict/batch
GET  /segments/
GET  /segments/{id}
GET  /docs

========================
MODEL PERFORMANCE
========================

DistilBERT -> 0.718 F1
BiLSTM     -> 0.610 F1
MLP Sent   -> 0.883 F1
MLP Seg    -> 0.553 F1

========================
TECH STACK
========================

FastAPI
PyTorch
Pydantic v2
Docker
pytest

========================
ARCHITECTURE
========================

Reviews → Features → ML Models → FastAPI → Docker → Deployment

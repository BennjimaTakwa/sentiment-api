
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import health, predict, segments
from app.services.model_service import model_service

# ── Logging
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ── Lifespan: runs once at startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: load models before accepting any request
    logger.info("Starting up — loading ML models...")
    model_service.initialize()
    logger.info("Models loaded. API ready.")
    yield
    # ── Shutdown
    logger.info("Shutting down.")


# ── App definition
app = FastAPI(
    title       = "Customer Sentiment & Retention API",
    description = """
## AI-Powered Customer Profiling API

Predicts customer **segment**, **sentiment**, and **retention risk**
from behavioral features using a dual-head MLP trained on 100k e-commerce reviews.

### Models used
- **DistilBERT** (nb-03): fine-tuned sentiment classifier, test F1=0.718
- **BiLSTM** (nb-04): lightweight sentiment model, test F1=0.610
- **MLP** (nb-07): dual-head behavioral predictor, sentiment acc=88%

### Endpoints
- `POST /predict/single` — score one customer
- `POST /predict/batch`  — score up to 1000 customers
- `GET  /segments/`      — list all 5 customer segments
- `GET  /health/`        — liveness probe
    """,
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)


# ── CORS middleware (allow all origins for dev — tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    elapsed  = round((time.time() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    return response


# ── Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code = 500,
        content     = {"detail": "Internal server error", "type": str(type(exc).__name__)}
    )


# ── Include routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(segments.router)


# ── Root
@app.api_route("/", methods=["GET", "HEAD"], tags=["Root"])
async def root():
    return {
        "name"   : "Customer Sentiment & Retention API",
        "version": "1.0.0",
        "docs"   : "/docs",
        "health" : "/health",
    }

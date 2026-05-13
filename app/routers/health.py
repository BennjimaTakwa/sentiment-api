
import time
from fastapi import APIRouter
from app.schemas.schemas import HealthResponse
from app.services.model_service import model_service

router = APIRouter(prefix="/health", tags=["Health"])

API_VERSION   = "1.0.0"
API_START_TIME = time.time()

@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Liveness probe.
    Returns model load status and uptime.
    Used by Docker HEALTHCHECK and Kubernetes readiness probes.
    """
    return HealthResponse(
        status        = "ok" if model_service.is_ready else "loading",
        model_loaded  = model_service.is_ready,
        version       = API_VERSION,
        uptime_seconds= model_service.uptime_seconds,
    )

@router.get("/ready")
async def readiness_check():
    """
    Readiness probe — returns 503 if model not loaded.
    Kubernetes uses this to decide if pod should receive traffic.
    """
    if not model_service.is_ready:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "ready"}

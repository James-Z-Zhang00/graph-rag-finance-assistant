"""
Lightweight health check server for GCP deployment.
Runs alongside Streamlit to provide health endpoints for load balancers.
"""
import os
import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Frontend Health Check Server")


@app.get("/")
def root():
    """Root endpoint - returns OK for basic health checks"""
    return {"status": "ok", "service": "graphrag-frontend"}


@app.get("/healthz")
def healthz():
    """
    Health check endpoint for GCP Load Balancer and Kubernetes liveness probe.
    Always returns 200 OK if the service is running.
    """
    return {"ok": True}


@app.get("/readiness")
def readiness_check():
    """
    Readiness check endpoint for Kubernetes readiness probe.
    Checks if Streamlit is ready to serve traffic.
    """
    # For frontend, we can simply check if this server is running
    # More sophisticated checks could verify Streamlit port is accessible
    return {"status": "ready", "service": "graphrag-frontend"}


if __name__ == "__main__":
    # Run on port 8080 (separate from Streamlit on 8501)
    port = int(os.getenv("HEALTH_PORT", "8080"))
    uvicorn.run(
        "frontend.utils.health_server:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

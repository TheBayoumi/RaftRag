"""
Health check endpoints.

Endpoints for monitoring system health and metrics.
"""

import platform
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter

router = APIRouter()


@router.get(
    "/health",
    summary="Health check",
    description="Check if the API is running",
)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Dict: Health status.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "RAFT-RAG Server",
    }


@router.get(
    "/metrics",
    summary="System metrics",
    description="Get system metrics and information",
)
async def get_metrics() -> Dict[str, Any]:
    """
    Get system metrics.

    Returns:
        Dict: System metrics.
    """
    return {
        "system": {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "architecture": platform.machine(),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

"""
API v1 router.

Combines all v1 endpoints into a single router.
"""

from fastapi import APIRouter

from .endpoints import fine_tuning, health, raft
from .endpoints import rag as rag_endpoints

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    fine_tuning.router,
    tags=["Fine-tuning"],
)

api_router.include_router(
    raft.router,
    tags=["RAFT"],
)

api_router.include_router(
    rag_endpoints.router,
    tags=["RAG"],
)

api_router.include_router(
    health.router,
    tags=["Health"],
)

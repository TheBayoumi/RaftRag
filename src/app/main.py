"""
FastAPI main application.

This is the entry point for the RAFT-RAG server.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .api.v1.router import api_router
from .core.config import get_settings
from .core.logging import setup_logging

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager.

    Handles startup and shutdown events.

    Args:
        app: FastAPI application.

    Yields:
        None
    """
    # Startup
    logger.info("Starting RAFT-RAG Server...")

    # Setup logging
    setup_logging(
        level=settings.log_level,
        log_file="./logs/raft_rag.log",
    )

    # Ensure directories exist
    settings.ensure_directories()

    logger.success("RAFT-RAG Server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down RAFT-RAG Server...")
    logger.success("RAFT-RAG Server shut down successfully")


# Create FastAPI application
app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="RAFT Fine-tuning with DoRA and RAG Server",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(
    api_router,
    prefix=settings.api_v1_prefix,
)


@app.get("/", tags=["Root"])
async def root() -> dict:
    """
    Root endpoint.

    Returns:
        dict: Welcome message.
    """
    return {
        "message": "Welcome to RAFT-RAG Server",
        "version": settings.version,
        "docs": f"{settings.api_v1_prefix}/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )

"""
RAFT (Retrieval Augmented Fine-Tuning) API endpoints.

Endpoints for managing RAFT fine-tuning jobs with retrieval augmentation.
"""

from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from ....core.exceptions import FineTuningException, RAGException
from ....schemas.raft import RaftFineTuneRequest, RaftFineTuneResponse
from ....services.raft_service import RaftService

router = APIRouter()

# Singleton service instance per worker (lazy-initialized)
_raft_service_instance: Optional[RaftService] = None


def get_raft_service() -> RaftService:
    """
    Get or create RAFT service instance (dependency injection).

    This function provides a singleton RAFT service per worker process.
    Using dependency injection avoids module-level instantiation,
    which prevents pickle errors when using multiple uvicorn workers.

    Returns:
        RaftService: Singleton RAFT service instance.
    """
    global _raft_service_instance
    if _raft_service_instance is None:
        _raft_service_instance = RaftService()
    return _raft_service_instance


@router.post(
    "/raft-fine-tune",
    response_model=RaftFineTuneResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start RAFT fine-tuning job",
    description=(
        "Start a new RAFT (Retrieval Augmented Fine-Tuning) job. "
        "RAFT augments training data with retrieved context documents "
        "to teach models to use retrieval information effectively."
    ),
)
async def start_raft_fine_tuning(
    request: RaftFineTuneRequest,
    raft_service: RaftService = Depends(get_raft_service),
) -> RaftFineTuneResponse:
    """
    Start a new RAFT fine-tuning job.

    RAFT Process:
    1. Loads Q&A pairs from dataset
    2. For each question, retrieves relevant documents from ChromaDB
    3. Adds distractor (irrelevant) documents
    4. Formats context + question + answer for training
    5. Fine-tunes model on augmented dataset

    Args:
        request: RAFT fine-tuning request with Q&A dataset and retrieval config.
        raft_service: RAFT service instance (injected).

    Returns:
        RaftFineTuneResponse: Job information with ID and status.

    Raises:
        HTTPException: If RAFT fine-tuning fails to start.

    Example:
        ```json
        {
          "model_name": "meta-llama/Llama-3.2-3B-Instruct",
          "dataset_path": "./data/datasets/qa_pairs.jsonl",
          "output_dir": "./data/models/raft_model",
          "raft_config": {
            "collection_name": "technical_docs",
            "num_context_docs": 5,
            "num_distractor_docs": 2,
            "context_mode": "oracle"
          },
          "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "use_dora": true
          },
          "training_config": {
            "num_epochs": 3,
            "batch_size": 4
          }
        }
        ```
    """
    try:
        if not raft_service.is_initialized:
            await raft_service.initialize()

        response = await raft_service.start_raft_fine_tuning(request)
        return response

    except FineTuningException as e:
        logger.error(f"RAFT fine-tuning error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": e.error_code,
                "message": e.message,
                "details": e.details,
            },
        )
    except RAGException as e:
        logger.error(f"RAG error during RAFT: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": e.error_code,
                "message": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_ERROR", "message": str(e)},
        )


@router.get(
    "/raft-fine-tune/{job_id}",
    response_model=RaftFineTuneResponse,
    summary="Get RAFT job status",
    description="Get the status of a RAFT fine-tuning job",
)
async def get_raft_job_status(
    job_id: str,
    raft_service: RaftService = Depends(get_raft_service),
) -> RaftFineTuneResponse:
    """
    Get RAFT fine-tuning job status.

    Args:
        job_id: Job identifier (starts with 'raft_').
        raft_service: RAFT service instance (injected).

    Returns:
        RaftFineTuneResponse: Current job status with progress.

    Raises:
        HTTPException: If job not found.

    Example Response:
        ```json
        {
          "job_id": "raft_abc123",
          "status": "running",
          "model_name": "meta-llama/Llama-3.2-3B-Instruct",
          "output_dir": "./data/models/raft_model",
          "collection_name": "technical_docs",
          "num_examples": 1000,
          "num_augmented": 1000,
          "progress": 65.5,
          "created_at": "2025-10-24T10:00:00",
          "updated_at": "2025-10-24T10:15:00"
        }
        ```
    """
    try:
        if not raft_service.is_initialized:
            await raft_service.initialize()

        response = await raft_service.get_job_status(job_id)
        return response

    except FineTuningException as e:
        logger.error(f"RAFT job status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": e.error_code,
                "message": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_ERROR", "message": str(e)},
        )


@router.delete(
    "/raft-fine-tune/{job_id}",
    response_model=RaftFineTuneResponse,
    summary="Cancel RAFT job",
    description="Cancel a running RAFT fine-tuning job",
)
async def cancel_raft_job(
    job_id: str,
    raft_service: RaftService = Depends(get_raft_service),
) -> RaftFineTuneResponse:
    """
    Cancel a RAFT fine-tuning job.

    This will also cancel the underlying standard fine-tuning job
    if it has already started.

    Args:
        job_id: Job identifier (starts with 'raft_').
        raft_service: RAFT service instance (injected).

    Returns:
        RaftFineTuneResponse: Updated job status.

    Raises:
        HTTPException: If job cannot be cancelled.

    Example Response:
        ```json
        {
          "job_id": "raft_abc123",
          "status": "cancelled",
          "model_name": "meta-llama/Llama-3.2-3B-Instruct",
          "progress": 45.5,
          "created_at": "2025-10-24T10:00:00",
          "updated_at": "2025-10-24T10:10:00"
        }
        ```
    """
    try:
        if not raft_service.is_initialized:
            await raft_service.initialize()

        response = await raft_service.cancel_job(job_id)
        return response

    except FineTuningException as e:
        logger.error(f"RAFT job cancellation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": e.error_code,
                "message": e.message,
                "details": e.details,
            },
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_ERROR", "message": str(e)},
        )

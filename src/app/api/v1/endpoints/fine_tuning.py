"""
Fine-tuning API endpoints.

Endpoints for managing fine-tuning jobs.
"""

from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from ....core.exceptions import FineTuningException, ModelNotFoundException
from ....schemas.fine_tuning import FineTuneRequest, FineTuneResponse
from ....services.fine_tuning_service import FineTuningService

router = APIRouter()

# Singleton service instance per worker (lazy-initialized)
_fine_tuning_service_instance: Optional[FineTuningService] = None


def get_fine_tuning_service() -> FineTuningService:
    """
    Get or create Fine-tuning service instance (dependency injection).

    This function provides a singleton fine-tuning service per worker process.
    Using dependency injection avoids module-level instantiation,
    which prevents pickle errors when using multiple uvicorn workers.

    Returns:
        FineTuningService: Singleton fine-tuning service instance.
    """
    global _fine_tuning_service_instance
    if _fine_tuning_service_instance is None:
        _fine_tuning_service_instance = FineTuningService()
    return _fine_tuning_service_instance


@router.post(
    "/fine-tune",
    response_model=FineTuneResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start fine-tuning job",
    description="Start a new fine-tuning job with RAFT and DoRA",
)
async def start_fine_tuning(
    request: FineTuneRequest,
    fine_tuning_service: FineTuningService = Depends(get_fine_tuning_service),
) -> FineTuneResponse:
    """
    Start a new fine-tuning job.

    Args:
        request: Fine-tuning request with configuration.
        fine_tuning_service: Fine-tuning service instance (injected).

    Returns:
        FineTuneResponse: Job information with ID and status.

    Raises:
        HTTPException: If fine-tuning fails to start.
    """
    try:
        if not fine_tuning_service.is_initialized:
            await fine_tuning_service.initialize()

        response = await fine_tuning_service.start_fine_tuning(request)
        return response

    except FineTuningException as e:
        logger.error(f"Fine-tuning error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": e.error_code, "message": e.message, "details": e.details},
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_ERROR", "message": str(e)},
        )


@router.get(
    "/fine-tune/{job_id}",
    response_model=FineTuneResponse,
    summary="Get job status",
    description="Get the status of a fine-tuning job",
)
async def get_job_status(
    job_id: str,
    fine_tuning_service: FineTuningService = Depends(get_fine_tuning_service),
) -> FineTuneResponse:
    """
    Get fine-tuning job status.

    Args:
        job_id: Job identifier.
        fine_tuning_service: Fine-tuning service instance (injected).

    Returns:
        FineTuneResponse: Current job status.

    Raises:
        HTTPException: If job not found.
    """
    try:
        if not fine_tuning_service.is_initialized:
            await fine_tuning_service.initialize()

        response = await fine_tuning_service.get_job_status(job_id)
        return response

    except FineTuningException as e:
        logger.error(f"Job status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": e.error_code, "message": e.message, "details": e.details},
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_ERROR", "message": str(e)},
        )


@router.delete(
    "/fine-tune/{job_id}",
    response_model=FineTuneResponse,
    summary="Cancel job",
    description="Cancel a running fine-tuning job",
)
async def cancel_job(
    job_id: str,
    fine_tuning_service: FineTuningService = Depends(get_fine_tuning_service),
) -> FineTuneResponse:
    """
    Cancel a fine-tuning job.

    Args:
        job_id: Job identifier.
        fine_tuning_service: Fine-tuning service instance (injected).

    Returns:
        FineTuneResponse: Updated job status.

    Raises:
        HTTPException: If job cannot be cancelled.
    """
    try:
        if not fine_tuning_service.is_initialized:
            await fine_tuning_service.initialize()

        response = await fine_tuning_service.cancel_job(job_id)
        return response

    except FineTuningException as e:
        logger.error(f"Job cancellation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": e.error_code, "message": e.message, "details": e.details},
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_ERROR", "message": str(e)},
        )

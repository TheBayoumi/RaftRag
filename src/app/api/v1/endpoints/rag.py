"""
RAG API endpoints.

Endpoints for document management and querying.
Uses the centralized RAG service for all operations.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from ....core.exceptions import RAGException
from ....schemas.rag import (
    DocumentResponse,
    DocumentUpload,
    QueryRequest,
    QueryResponse,
)
from ....services.centralized_rag_service import CentralizedRAGService

router = APIRouter()

# Singleton service instance per worker (lazy-initialized)
_rag_service_instance: Optional[CentralizedRAGService] = None


def get_rag_service() -> CentralizedRAGService:
    """
    Get or create RAG service instance (dependency injection).

    This function provides a singleton RAG service per worker process.
    Using dependency injection avoids module-level instantiation,
    which prevents pickle errors when using multiple uvicorn workers.

    Returns:
        CentralizedRAGService: Singleton RAG service instance.
    """
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = CentralizedRAGService()
    return _rag_service_instance


@router.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload document",
    description="Upload and process a document for RAG",
)
async def upload_document(
    upload: DocumentUpload,
    rag_service: CentralizedRAGService = Depends(get_rag_service),
) -> DocumentResponse:
    """
    Upload and process a document.

    Args:
        upload: Document upload request.
        rag_service: RAG service instance (injected).

    Returns:
        DocumentResponse: Upload confirmation with document info.

    Raises:
        HTTPException: If upload fails.
    """
    try:
        if not rag_service.is_initialized:
            await rag_service.initialize()

        response = await rag_service.upload_document(upload)
        return response

    except RAGException as e:
        logger.error(f"Document upload error: {e}")
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


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="RAG query",
    description="Perform RAG query with retrieval and generation",
)
async def query(
    request: QueryRequest,
    rag_service: CentralizedRAGService = Depends(get_rag_service),
) -> QueryResponse:
    """
    Perform RAG query.

    Args:
        request: Query request with parameters.
        rag_service: RAG service instance (injected).

    Returns:
        QueryResponse: Answer with retrieved sources.

    Raises:
        HTTPException: If query fails.
    """
    try:
        if not rag_service.is_initialized:
            await rag_service.initialize()

        response = await rag_service.query_with_answer(request)
        return response

    except RAGException as e:
        logger.error(f"Query error: {e}")
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


@router.delete(
    "/documents/{doc_id}",
    summary="Delete document",
    description="Delete a document from the collection. Supports deletion by UUID or filename.",
)
async def delete_document(
    doc_id: str,
    collection_name: Optional[str] = None,
    rag_service: CentralizedRAGService = Depends(get_rag_service),
) -> Dict[str, Any]:
    """
    Delete a document.

    Args:
        doc_id: Document identifier (UUID or filename).
        collection_name: Optional collection name. If provided, searches only in this collection (faster).
                        If not provided, searches all collections.
        rag_service: RAG service instance (injected).

    Returns:
        Dict: Deletion confirmation.

    Raises:
        HTTPException: If deletion fails.
    """
    try:
        if not rag_service.is_initialized:
            await rag_service.initialize()

        response = await rag_service.delete_document(doc_id, collection_name=collection_name)
        return response

    except RAGException as e:
        logger.error(f"Document deletion error: {e}")
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



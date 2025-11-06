"""
Custom exception classes for better error handling.

All custom exceptions must inherit from these base classes.
"""

from typing import Any, Dict, Optional


class RAFTRAGException(Exception):
    """Base exception for all custom exceptions."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize exception.

        Args:
            message: Error message.
            error_code: Optional error code for API responses.
            details: Optional additional details.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}


class ValidationException(RAFTRAGException):
    """Exception for validation errors."""

    def __init__(self, message: str, field: Optional[str] = None) -> None:
        """Initialize validation exception."""
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={"field": field} if field else {},
        )


class ModelNotFoundException(RAFTRAGException):
    """Exception when model is not found."""

    def __init__(self, model_name: str) -> None:
        """Initialize model not found exception."""
        super().__init__(
            message=f"Model not found: {model_name}",
            error_code="MODEL_NOT_FOUND",
            details={"model_name": model_name},
        )


class DatasetException(RAFTRAGException):
    """Exception for dataset-related errors."""

    def __init__(self, message: str, dataset_path: Optional[str] = None) -> None:
        """Initialize dataset exception."""
        super().__init__(
            message=message,
            error_code="DATASET_ERROR",
            details={"dataset_path": dataset_path} if dataset_path else {},
        )


class FineTuningException(RAFTRAGException):
    """Exception for fine-tuning errors."""

    def __init__(self, message: str, job_id: Optional[str] = None) -> None:
        """Initialize fine-tuning exception."""
        super().__init__(
            message=message,
            error_code="FINETUNING_ERROR",
            details={"job_id": job_id} if job_id else {},
        )


class RAGException(RAFTRAGException):
    """Exception for RAG operation errors."""

    def __init__(self, message: str, operation: Optional[str] = None) -> None:
        """Initialize RAG exception."""
        super().__init__(
            message=message,
            error_code="RAG_ERROR",
            details={"operation": operation} if operation else {},
        )


class AuthenticationException(RAFTRAGException):
    """Exception for authentication errors."""

    def __init__(self, message: str = "Authentication failed") -> None:
        """Initialize authentication exception."""
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
        )


class AuthorizationException(RAFTRAGException):
    """Exception for authorization errors."""

    def __init__(self, message: str = "Authorization failed") -> None:
        """Initialize authorization exception."""
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
        )


class ResourceNotFoundException(RAFTRAGException):
    """Exception when a resource is not found."""

    def __init__(self, resource_type: str, resource_id: str) -> None:
        """Initialize resource not found exception."""
        super().__init__(
            message=f"{resource_type} not found: {resource_id}",
            error_code="RESOURCE_NOT_FOUND",
            details={"resource_type": resource_type, "resource_id": resource_id},
        )


class GPUMemoryException(RAFTRAGException):
    """Exception for GPU memory errors."""

    def __init__(self, message: str, required_memory: Optional[int] = None) -> None:
        """Initialize GPU memory exception."""
        super().__init__(
            message=message,
            error_code="GPU_MEMORY_ERROR",
            details={"required_memory": required_memory} if required_memory else {},
        )

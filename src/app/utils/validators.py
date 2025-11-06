"""
Input validation utilities with security focus.

All user inputs MUST pass through these validators.
"""

import re
from pathlib import Path
from typing import Any, List, Optional

from ..core.exceptions import ValidationException


class SecurityValidator:
    """Security-focused validation utilities."""

    # Path traversal prevention
    SAFE_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9_\-./\\]+$")
    FORBIDDEN_PATHS = {"..", "~", "/etc", "/root", "/proc", "/sys", "C:\\Windows"}

    # Model name pattern
    MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-/]+$")

    # Safe string pattern
    SAFE_STRING_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\s.,!?]+$")

    @classmethod
    def validate_file_path(
        cls, path: str, allowed_dirs: Optional[List[str]] = None
    ) -> Path:
        """
        Validate file path for security.

        Args:
            path: File path to validate.
            allowed_dirs: Optional list of allowed directory prefixes.

        Returns:
            Path: Validated path object.

        Raises:
            ValidationException: If path is unsafe.
        """
        # Check for path traversal attempts
        if any(forbidden in path for forbidden in cls.FORBIDDEN_PATHS):
            raise ValidationException(
                f"Forbidden path pattern detected: {path}", field="path"
            )

        # Check path format
        if not cls.SAFE_PATH_PATTERN.match(path):
            raise ValidationException(f"Invalid path format: {path}", field="path")

        # Resolve and check if within allowed directories
        try:
            resolved_path = Path(path).resolve()
        except (OSError, RuntimeError) as e:
            raise ValidationException(
                f"Cannot resolve path: {path}. Error: {e}", field="path"
            )

        if allowed_dirs:
            allowed_paths = [Path(d).resolve() for d in allowed_dirs]
            if not any(
                str(resolved_path).startswith(str(allowed_dir))
                for allowed_dir in allowed_paths
            ):
                raise ValidationException(
                    f"Path outside allowed directories: {path}", field="path"
                )

        return resolved_path

    @classmethod
    def validate_model_name(cls, name: str) -> str:
        """
        Validate model name format.

        Args:
            name: Model name to validate.

        Returns:
            str: Validated model name.

        Raises:
            ValidationException: If name is invalid.
        """
        if not cls.MODEL_NAME_PATTERN.match(name):
            raise ValidationException(
                f"Invalid model name: {name}. "
                "Use only alphanumeric, underscore, hyphen, and forward slash.",
                field="model_name",
            )

        if len(name) < 3 or len(name) > 100:
            raise ValidationException(
                f"Model name must be 3-100 characters, got {len(name)}",
                field="model_name",
            )

        return name

    @classmethod
    def sanitize_user_input(cls, text: str, max_length: int = 1000) -> str:
        """
        Sanitize user text input.

        Args:
            text: User input text.
            max_length: Maximum allowed length.

        Returns:
            str: Sanitized text.

        Raises:
            ValidationException: If text is invalid.
        """
        # Strip whitespace
        text = text.strip()

        # Limit length
        if len(text) > max_length:
            raise ValidationException(
                f"Text exceeds maximum length of {max_length}", field="text"
            )

        # Remove control characters
        text = "".join(char for char in text if ord(char) >= 32)

        return text

    @classmethod
    def validate_chunk_size(cls, chunk_size: int, chunk_overlap: int) -> None:
        """
        Validate chunk size and overlap.

        Args:
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.

        Raises:
            ValidationException: If values are invalid.
        """
        if chunk_size <= 0:
            raise ValidationException("Chunk size must be positive", field="chunk_size")

        if chunk_overlap < 0:
            raise ValidationException(
                "Chunk overlap must be non-negative", field="chunk_overlap"
            )

        if chunk_overlap >= chunk_size:
            raise ValidationException(
                "Chunk overlap must be less than chunk size", field="chunk_overlap"
            )

    @classmethod
    def validate_api_key(cls, api_key: str) -> str:
        """
        Validate API key format.

        Args:
            api_key: API key to validate.

        Returns:
            str: Validated API key.

        Raises:
            ValidationException: If API key is invalid.
        """
        if not api_key or len(api_key) < 10:
            raise ValidationException("Invalid API key format", field="api_key")

        return api_key

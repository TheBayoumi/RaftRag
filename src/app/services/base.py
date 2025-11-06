"""
Base service class with common functionality.

All services must inherit from this base class to ensure
consistent error handling and logging.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from loguru import logger


class BaseService(ABC):
    """
    Abstract base service class.

    All services must implement the required abstract methods
    and should use the provided utility methods for consistency.
    """

    def __init__(self, service_name: str) -> None:
        """
        Initialize base service.

        Args:
            service_name: Name of the service for logging.
        """
        self.service_name = service_name
        self.logger = logger.bind(service=service_name)
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize service resources.

        Override this method to perform async initialization.
        """
        if self._initialized:
            self.logger.warning(f"{self.service_name} already initialized")
            return

        self.logger.info(f"Initializing {self.service_name}")
        await self._initialize_impl()
        self._initialized = True
        self.logger.success(f"{self.service_name} initialized successfully")

    @abstractmethod
    async def _initialize_impl(self) -> None:
        """
        Implementation-specific initialization.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    async def cleanup(self) -> None:
        """
        Cleanup service resources.

        Override this method to perform cleanup operations.
        """
        if not self._initialized:
            return

        self.logger.info(f"Cleaning up {self.service_name}")
        await self._cleanup_impl()
        self._initialized = False
        self.logger.success(f"{self.service_name} cleaned up successfully")

    async def _cleanup_impl(self) -> None:
        """
        Implementation-specific cleanup.

        Override in subclasses if needed.
        """
        pass

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Standardized error handling.

        Args:
            error: The exception that occurred.
            context: Optional context information.
        """
        self.logger.error(
            f"Error in {self.service_name}: {error}",
            context=context or {},
            exc_info=True,
        )

    @property
    def is_initialized(self) -> bool:
        """
        Check if service is initialized.

        Returns:
            bool: True if service is initialized.
        """
        return self._initialized

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare service for pickling.

        Removes non-picklable logger object before serialization.
        The logger will be reconstructed in __setstate__.

        Returns:
            Dict[str, Any]: Picklable state dictionary.
        """
        state = self.__dict__.copy()
        # Remove the logger since it contains file handles
        state.pop("logger", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore service after unpickling.

        Reconstructs the logger with the same service name binding.

        Args:
            state: State dictionary from pickle.
        """
        self.__dict__.update(state)
        # Reconstruct the logger with the same service name binding
        self.logger = logger.bind(service=self.service_name)

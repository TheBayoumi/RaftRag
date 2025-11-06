"""
Configuration management using Pydantic Settings.

This module handles all configuration for the RAFT-RAG server,
following 12-factor app principles.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with validation.

    Attributes:
        environment: Current environment (development/staging/production).
        debug: Enable debug mode.
        log_level: Logging level.
        api_v1_prefix: API version 1 prefix.
        project_name: Name of the project.
        version: Application version.
        host: Server host.
        port: Server port.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),  # Disable protected namespace warnings for model_ fields
    )

    # General Settings
    environment: str = Field(
        default="development",
        description="Current environment",
    )
    debug: bool = Field(
        default=False,
        description="Debug mode flag",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    # API Settings
    api_v1_prefix: str = Field(
        default="/api/v1",
        description="API v1 route prefix",
    )
    project_name: str = Field(
        default="RAFT-RAG Server",
        description="Project name for documentation",
    )
    version: str = Field(
        default="1.0.0",
        description="API version",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Server host",
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Server port",
    )

    # Model Settings
    default_base_model: Optional[str] = Field(
        default=None,
        description="Default base model for fine-tuning (user must specify)",
    )
    model_cache_dir: Path = Field(
        default=Path("./data/models"),
        description="Directory for model storage",
    )
    max_model_cache_size: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of models to keep in cache",
    )

    # Database Settings
    database_url: str = Field(
        default="sqlite:///./data/metadata/raft_rag.db",
        description="Database connection URL",
    )

    # Vector Store Settings
    chroma_persist_dir: Path = Field(
        default=Path("./data/vectorstore"),
        description="ChromaDB persistence directory",
    )
    bm25_indices_dir: Path = Field(
        default=Path("./data/bm25_indices"),
        description="BM25 keyword search indices storage directory",
    )
    default_embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Default embedding model",
    )

    # GPU Settings
    cuda_visible_devices: str = Field(
        default="0",
        description="CUDA devices to use",
    )
    torch_cuda_arch_list: str = Field(
        default="7.0;7.5;8.0;8.6",
        description="CUDA architectures",
    )

    # HuggingFace Settings
    HF_HOME: Path = Field(
        default=Path("./data/huggingface"),
        description="HuggingFace cache directory (replaces deprecated TRANSFORMERS_CACHE)",
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="HuggingFace API token",
    )

    # Security Settings
    secret_key: str = Field(
        default="your-secret-key-here-change-in-production",
        description="Secret key for JWT encoding",
    )
    api_key: str = Field(
        default="your-api-key-here",
        description="API key for authentication",
    )

    # Training Defaults
    default_batch_size: int = Field(
        default=4,
        ge=1,
        le=128,
        description="Default training batch size",
    )
    default_epochs: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Default number of epochs",
    )
    default_learning_rate: float = Field(
        default=2e-4,
        gt=0.0,
        lt=1.0,
        description="Default learning rate",
    )
    default_lora_rank: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Default LoRA rank",
    )
    default_lora_alpha: int = Field(
        default=32,
        ge=1,
        description="Default LoRA alpha",
    )

    # RAG Defaults
    default_chunk_size: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Default document chunk size",
    )
    default_chunk_overlap: int = Field(
        default=100,  # Increased from 50 to 20% overlap (industry standard)
        ge=0,
        le=512,
        description="Default chunk overlap (should be ~20% of chunk_size)",
    )
    default_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of retrieved documents",
    )
    default_similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Default similarity threshold (0.3-0.5 recommended for balance)",
    )

    # RAG Service Settings (REMOVED DUPLICATES - See RAG Pipeline Settings below)

    # RAG Pipeline Settings (raganything)
    rag_working_dir: Path = Field(
        default=Path("./data/rag_storage"),
        description="Working directory for raganything RAG pipeline",
    )
    rag_parser: str = Field(
        default="mineru",
        description="Document parser for raganything (auto, mineru, docling)",
    )
    rag_parse_method: str = Field(
        default="auto",
        description="Parse method (auto, ocr, txt)",
    )
    enable_rag_image_processing: bool = Field(
        default=False,
        description="Enable image processing in RAG",
    )
    enable_rag_table_processing: bool = Field(
        default=False,
        description="Enable table processing in RAG",
    )
    enable_rag_equation_processing: bool = Field(
        default=False,
        description="Enable equation processing in RAG",
    )
    rag_batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Batch size for document processing",
    )
    rag_max_file_size_mb: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum file size for upload in MB",
    )
    rag_context_window: int = Field(
        default=3,  # Increased from 1 to retrieve more surrounding context
        ge=1,
        le=10,
        description="Context window for multimodal content (surrounding chunks)",
    )
    rag_max_context_tokens: int = Field(
        default=4000,  # FIXED: Balanced for complex queries without memory issues
        ge=100,
        le=10000,
        description="Maximum context tokens for RAG retrieval (sum of all retrieved chunks)",
    )

    # Local RAG Models
    # Note: RAG_LLM_MODEL env var can be set to override the default
    # If not set, we use default_base_model as fallback via the property below
    rag_llm_model_env: Optional[str] = Field(
        default=None,
        description="RAG LLM model from RAG_LLM_MODEL env var (use rag_llm_model property).",
    )
    rag_embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Embedding model for RAG (used for document and query embeddings)",
    )
    rag_vision_model: Optional[str] = Field(
        default=None,
        description="Optional vision model for multimodal RAG",
    )
    rag_max_tokens: int = Field(
        default=1024,  # FIXED: Optimal for complete answers without memory issues (~700-800 words)
        ge=128,
        le=8192,
        description="Maximum tokens for RAG LLM generation",
    )
    rag_temperature: float = Field(
        default=0.1,  # LOW for factual, grounded answers (prevent hallucination)
        ge=0.0,
        le=2.0,
        description="Temperature for RAG LLM generation (0.1-0.2 for factual, 0.7+ for creative)",
    )

    # System prompt selection for RAG queries
    # Allowed values: standard, concise, strict
    # Prompts always place citations in a bottom '## Sources' section
    rag_system_prompt_mode: str = Field(
        default="standard",
        description=(
            "Which system prompt to use for RAG queries. "
            "Options: 'standard', 'concise', 'strict'. "
            "All variants place citations in a bottom '## Sources' section."
        ),
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        """Validate environment value."""
        allowed = {"development", "staging", "production"}
        if value not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Validate log level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if value.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return value.upper()

    @property
    def rag_llm_model(self) -> str:
        """
        Get the RAG LLM model to use, with fallback logic.

        Priority:
        1. RAG_LLM_MODEL from .env (if set via rag_llm_model_env field)
        2. DEFAULT_BASE_MODEL from .env (if set)
        3. Default 3B model

        This allows users to set DEFAULT_BASE_MODEL in .env and have it
        automatically used for RAG queries.

        Note: To explicitly set the RAG model, use RAG_LLM_MODEL in .env.
        Otherwise, DEFAULT_BASE_MODEL will be used as fallback.
        """
        if self.rag_llm_model_env:
            return self.rag_llm_model_env
        if self.default_base_model:
            return self.default_base_model
        return "meta-llama/Llama-3.2-3B-Instruct"

    @field_validator("rag_system_prompt_mode")
    @classmethod
    def validate_rag_system_prompt_mode(cls, value: str) -> str:
        allowed = {"standard", "concise", "strict"}
        v = value.lower()
        if v not in allowed:
            raise ValueError(f"rag_system_prompt_mode must be one of {allowed}")
        return v

    def ensure_directories(self) -> None:
        """
        Ensure all required directories exist.

        Returns:
            None
        """
        directories = [
            self.model_cache_dir,  # For fine-tuned model outputs
            self.chroma_persist_dir,
            self.HF_HOME,  # HuggingFace cache (models downloaded here)
            self.rag_working_dir,
            self.bm25_indices_dir,  # BM25 keyword search indices
            Path("./data/datasets"),
            Path("./data/uploads"),
            Path("./data/metadata"),
            Path("./logs"),
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings.
    """
    return Settings()

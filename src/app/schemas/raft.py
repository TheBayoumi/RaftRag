"""
Pydantic schemas for RAFT (Retrieval Augmented Fine-Tuning) operations.

RAFT combines retrieval with fine-tuning by including relevant context
documents during training to teach models to use retrieved information.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContextMode(str, Enum):
    """Context inclusion modes for RAFT."""

    ORACLE = "oracle"  # Include ground-truth document
    RETRIEVAL_ONLY = "retrieval_only"  # Only use retrieved docs
    HYBRID = "hybrid"  # Mix of oracle and retrieval


class RaftConfig(BaseModel):
    """
    RAFT-specific configuration for retrieval augmented fine-tuning.

    Attributes:
        collection_name: ChromaDB collection to retrieve documents from.
        num_context_docs: Number of relevant documents to retrieve per question.
        num_distractor_docs: Number of irrelevant documents to include.
        context_template: Template for formatting retrieved context.
        context_mode: How to include ground-truth documents.
        similarity_threshold: Minimum similarity score for retrieved docs.
        use_reranking: Whether to rerank retrieved documents.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "collection_name": "technical_docs",
                "num_context_docs": 5,
                "num_distractor_docs": 2,
                "context_template": "Document {idx}:\n{content}\n\n",
                "context_mode": "oracle",
                "similarity_threshold": 0.7,
                "use_reranking": False,
            }
        },
    )

    collection_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="ChromaDB collection name to retrieve from",
    )

    num_context_docs: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant documents to include in context",
    )

    num_distractor_docs: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Number of irrelevant (distractor) documents to include",
    )

    context_template: str = Field(
        default="Document {idx}:\n{content}\n\n",
        description="Template for formatting each document. Use {idx} for index, {content} for text.",
    )

    context_mode: ContextMode = Field(
        default=ContextMode.ORACLE,
        description="How to include ground-truth documents in training",
    )

    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for retrieved documents",
    )

    use_reranking: bool = Field(
        default=False,
        description="Whether to rerank retrieved documents before including",
    )

    max_context_length: int = Field(
        default=2048,
        ge=512,
        le=8192,
        description="Maximum total length of context in tokens",
    )

    @field_validator("context_template")
    @classmethod
    def validate_template(cls, value: str) -> str:
        """Validate context template has required placeholders."""
        if "{content}" not in value:
            raise ValueError("context_template must contain '{content}' placeholder")
        return value


class RaftFineTuneRequest(BaseModel):
    """
    Request schema for starting RAFT fine-tuning job.

    Attributes:
        model_name: Base model to fine-tune.
        dataset_path: Path to Q&A pairs dataset (JSON/JSONL).
        output_dir: Directory to save fine-tuned model.
        raft_config: RAFT-specific configuration.
        lora_config: LoRA/DoRA configuration.
        training_config: Training hyperparameters.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                "dataset_path": "./data/datasets/qa_pairs.jsonl",
                "output_dir": "./data/models/raft_model",
                "raft_config": {
                    "collection_name": "technical_docs",
                    "num_context_docs": 5,
                    "num_distractor_docs": 2,
                    "context_mode": "oracle",
                },
                "lora_config": {
                    "r": 16,
                    "lora_alpha": 32,
                    "use_dora": True,
                },
                "training_config": {
                    "num_epochs": 3,
                    "batch_size": 4,
                    "learning_rate": 2e-4,
                },
            }
        },
    )

    model_name: str = Field(
        ...,
        description="HuggingFace model identifier or path to local model",
    )

    dataset_path: str = Field(
        ...,
        description="Path to Q&A pairs dataset (JSON/JSONL format)",
    )

    output_dir: str = Field(
        ...,
        description="Directory to save the fine-tuned model",
    )

    raft_config: RaftConfig = Field(
        ...,
        description="RAFT-specific configuration",
    )

    lora_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "use_dora": True,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        },
        description="LoRA/DoRA configuration",
    )

    training_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 2e-4,
            "warmup_steps": 100,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 512,
        },
        description="Training hyperparameters",
    )

    @field_validator("dataset_path", "output_dir")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        """Validate paths don't contain dangerous patterns."""
        dangerous_patterns = ["..", "~", "/etc", "/root"]
        if any(pattern in value for pattern in dangerous_patterns):
            raise ValueError(f"Path contains forbidden pattern: {value}")
        return value


class RaftFineTuneResponse(BaseModel):
    """
    Response schema for RAFT fine-tuning operations.

    Attributes:
        job_id: Unique identifier for the RAFT job.
        status: Current job status.
        model_name: Base model being fine-tuned.
        output_dir: Where the model will be saved.
        collection_name: ChromaDB collection used for retrieval.
        num_examples: Total number of training examples.
        num_augmented: Number of examples augmented with context.
        created_at: Job creation timestamp.
        updated_at: Last update timestamp.
        error: Error message if failed.
        progress: Training progress percentage.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "raft_abc123",
                "status": "running",
                "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                "output_dir": "./data/models/raft_model",
                "collection_name": "technical_docs",
                "num_examples": 1000,
                "num_augmented": 1000,
                "created_at": "2025-10-24T10:00:00",
                "updated_at": "2025-10-24T10:05:00",
                "progress": 45.5,
            }
        }
    )

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(
        ..., description="Job status (pending/running/completed/failed)"
    )
    model_name: str = Field(..., description="Base model name")
    output_dir: str = Field(..., description="Output directory path")
    collection_name: str = Field(..., description="ChromaDB collection used")
    num_examples: int = Field(default=0, description="Total training examples")
    num_augmented: int = Field(default=0, description="Examples augmented with context")
    created_at: datetime = Field(..., description="Job creation time")
    updated_at: datetime = Field(..., description="Last update time")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    progress: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Progress percentage"
    )


class QAPair(BaseModel):
    """
    Schema for question-answer pairs in RAFT dataset.

    Attributes:
        question: The question text.
        answer: The answer text.
        ground_truth_doc: Optional ground-truth document for oracle mode.
        metadata: Optional metadata (e.g., source, difficulty).
    """

    question: str = Field(..., min_length=1, description="Question text")
    answer: str = Field(..., min_length=1, description="Answer text")
    ground_truth_doc: Optional[str] = Field(
        default=None, description="Ground-truth document for oracle mode"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata"
    )


class AugmentedExample(BaseModel):
    """
    Schema for RAFT-augmented training example.

    Attributes:
        original_question: The original question.
        original_answer: The original answer.
        context_docs: Retrieved context documents.
        distractor_docs: Distractor documents.
        formatted_text: Final formatted text for training.
    """

    original_question: str
    original_answer: str
    context_docs: List[str]
    distractor_docs: List[str]
    formatted_text: str

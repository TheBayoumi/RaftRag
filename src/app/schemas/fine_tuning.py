"""
Pydantic schemas for fine-tuning operations.

All schemas follow strict validation rules and provide
comprehensive documentation for API consumers.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelType(str, Enum):
    """Supported model types for fine-tuning."""

    LLAMA2_7B = "meta-llama/Llama-2-7b-hf"
    LLAMA2_13B = "meta-llama/Llama-2-13b-hf"
    LLAMA3_8B = "meta-llama/Llama-3-8b-hf"
    MISTRAL_7B = "mistralai/Mistral-7B-v0.1"
    MISTRAL_7B_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.2"


class TrainingStatus(str, Enum):
    """Training job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LoRAConfig(BaseModel):
    """
    LoRA/DoRA configuration parameters.

    Attributes:
        r: LoRA rank (attention dimension).
        lora_alpha: LoRA scaling parameter.
        lora_dropout: Dropout probability for LoRA layers.
        use_dora: Enable DoRA (Weight-Decomposed Low-Rank Adaptation).
        target_modules: Modules to apply LoRA to.
        modules_to_save: Additional modules to save.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "use_dora": True,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            }
        },
    )

    r: int = Field(
        default=16,
        ge=1,
        le=128,
        description="LoRA rank (attention dimension)",
    )
    lora_alpha: int = Field(
        default=32,
        ge=1,
        description="LoRA scaling parameter (typically 2x rank)",
    )
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Dropout probability for LoRA layers",
    )
    use_dora: bool = Field(
        default=True,
        description="Enable DoRA for better weight updates",
    )
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"],
        min_length=1,
        description="Modules to apply LoRA/DoRA to",
    )
    modules_to_save: Optional[List[str]] = Field(
        default=None,
        description="Additional modules to save during training",
    )

    @field_validator("r")
    @classmethod
    def validate_rank(cls, value: int) -> int:
        """
        Validate LoRA rank is a power of 2 for optimal performance.

        Args:
            value: LoRA rank value.

        Returns:
            int: Validated rank.

        Raises:
            ValueError: If rank is not a power of 2.
        """
        valid_ranks = {1, 2, 4, 8, 16, 32, 64, 128}
        if value not in valid_ranks:
            raise ValueError(f"LoRA rank should be a power of 2. Valid: {valid_ranks}")
        return value

    @field_validator("lora_alpha")
    @classmethod
    def validate_alpha_ratio(cls, value: int, info: Any) -> int:
        """
        Validate alpha is reasonable relative to rank.

        Args:
            value: LoRA alpha value.
            info: Validation context.

        Returns:
            int: Validated alpha.
        """
        if "r" in info.data:
            rank = info.data["r"]
            if value < rank:
                raise ValueError(f"lora_alpha ({value}) should be >= rank ({rank})")
        return value


class TrainingConfig(BaseModel):
    """
    Training configuration parameters.

    Attributes:
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        warmup_steps: Number of warmup steps.
        max_seq_length: Maximum sequence length.
        gradient_accumulation_steps: Gradient accumulation steps.
        fp16: Use mixed precision training.
        save_steps: Save checkpoint every N steps.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    num_epochs: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of training epochs",
    )
    batch_size: int = Field(
        default=4,
        ge=1,
        le=128,
        description="Training batch size per device",
    )
    learning_rate: float = Field(
        default=2e-4,
        gt=0.0,
        lt=1.0,
        description="Learning rate",
    )
    warmup_steps: int = Field(
        default=100,
        ge=0,
        description="Number of warmup steps",
    )
    max_seq_length: int = Field(
        default=512,
        ge=128,
        le=4096,
        description="Maximum sequence length",
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Gradient accumulation steps",
    )
    fp16: bool = Field(
        default=True,
        description="Use mixed precision training",
    )
    save_steps: int = Field(
        default=500,
        ge=1,
        description="Save checkpoint every N steps",
    )


class FineTuneRequest(BaseModel):
    """
    Fine-tuning job request schema.

    Attributes:
        model_name: Base model to fine-tune.
        dataset_path: Path to training dataset.
        output_dir: Output directory for fine-tuned model.
        lora_config: LoRA/DoRA configuration.
        training_config: Training configuration.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    model_name: str = Field(
        ...,
        description="Base model identifier",
    )
    dataset_path: str = Field(
        ...,
        description="Path to training dataset",
    )
    output_dir: str = Field(
        ...,
        description="Output directory for fine-tuned model",
    )
    lora_config: LoRAConfig = Field(
        default_factory=LoRAConfig,
        description="LoRA/DoRA configuration",
    )
    training_config: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training configuration",
    )


class FineTuneResponse(BaseModel):
    """
    Fine-tuning job response schema.

    Attributes:
        job_id: Unique job identifier.
        status: Current job status.
        model_name: Base model being fine-tuned.
        created_at: Job creation timestamp.
        message: Status message.
        progress: Training progress percentage (0-100).
        error: Error message if job failed.
        metrics: Training metrics if available.
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    job_id: str = Field(
        ...,
        description="Unique job identifier",
    )
    status: TrainingStatus = Field(
        ...,
        description="Current job status",
    )
    model_name: str = Field(
        ...,
        description="Base model identifier",
    )
    created_at: datetime = Field(
        ...,
        description="Job creation timestamp",
    )
    message: str = Field(
        default="",
        description="Status message",
    )
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Training progress percentage",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if job failed",
    )
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Training metrics (loss, steps, etc.)",
    )


class TrainingMetrics(BaseModel):
    """
    Training metrics schema.

    Attributes:
        epoch: Current epoch.
        step: Current step.
        loss: Training loss.
        learning_rate: Current learning rate.
        grad_norm: Gradient norm.
    """

    epoch: int = Field(
        ...,
        description="Current epoch",
    )
    step: int = Field(
        ...,
        description="Current step",
    )
    loss: float = Field(
        ...,
        description="Training loss",
    )
    learning_rate: float = Field(
        ...,
        description="Current learning rate",
    )
    grad_norm: Optional[float] = Field(
        default=None,
        description="Gradient norm",
    )

"""
Fine-tuning service with RAFT and DoRA support.

This service handles model fine-tuning using LoRA/DoRA with RAFT dataset preparation.
"""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from loguru import logger
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from ..core.config import get_settings
from ..core.exceptions import (
    DatasetException,
    FineTuningException,
    ModelNotFoundException,
)
from ..schemas.fine_tuning import FineTuneRequest, FineTuneResponse, TrainingStatus
from ..utils.local_models import LocalEmbeddingWrapper, LocalLLMWrapper
from ..utils.resource_detector import ResourceDetector
from .base import BaseService

settings = get_settings()


class FineTuningService(BaseService):
    """
    Fine-tuning service for RAFT with DoRA.

    This service manages the complete fine-tuning pipeline including:
    - Model loading and preparation
    - LoRA/DoRA configuration
    - RAFT dataset preparation
    - Training execution
    - Model saving and versioning
    """

    def __init__(self) -> None:
        """Initialize fine-tuning service."""
        super().__init__("FineTuningService")
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.unloaded_models: Dict[str, Any] = {}

        # Detect available resources intelligently
        self.resources = ResourceDetector.detect_available_resources()
        self.device = self.resources["device"]
        self.logger.info(f"Using device: {self.device}")
        self.loaded_models: Dict[str, Any] = {}

        # Reference to RAG service for model coordination (lazy-loaded)
        self._rag_service: Optional[Any] = None

    async def _initialize_impl(self) -> None:
        """
        Initialize fine-tuning service resources.

        Returns:
            None
        """
        # CRITICAL: Check if RAG model is loaded and unload it before fine-tuning
        # This ensures we have clean memory state for fine-tuning
        await self._unload_rag_model_if_loaded()
        
        # HF_HOME directory is created in ensure_directories() on startup
        # No need to create it here (redundant)
        self.logger.info(f"HuggingFace cache directory: {settings.HF_HOME}")

        # Print comprehensive resource summary
        ResourceDetector.print_resource_summary(self.resources)

        # Log recommendations
        self.logger.info(f"Recommended model: {self.resources['recommended_model']}")
        self.logger.info(
            f"Recommended batch size: {self.resources['recommended_batch_size']}"
        )


    async def _unload_rag_model_if_loaded(self) -> None:
        """
        Unload RAG models (LLM and embedding) if they're currently loaded to free memory before fine-tuning.

        Uses LocalLLMWrapper.unload_model() and LocalEmbeddingWrapper.unload_model() to properly
        free memory and clear caches, preventing memory conflicts and "duplicate template name"
        errors when switching from RAG to fine-tuning pipeline.

        This is critical because:
        1. RAG LLM model may be loaded in memory (via LocalLLMWrapper)
        2. RAG embedding model may be loaded in memory (via LocalEmbeddingWrapper)
        3. Fine-tuning needs to load a different model (or same model with different config)
        4. Both models in memory can cause OOM or template conflicts
        5. switch_model() or unload_model() properly cleans up Jinja2 template cache

        Strategy:
        - First tries to access the singleton RAG service instance from the endpoint module
        - Falls back to creating a new instance if singleton is not available
        - Checks if models are loaded before unloading (both LLM and embedding)
        - Unloads both models to maximize available memory for fine-tuning

        Returns:
            None
        """
        try:
            # Try to get the singleton RAG service instance from endpoint module
            # This is the instance that actually has the loaded models
            rag_service = None
            
            # Method 1: Try to access singleton from endpoint module
            try:
                from ..api.v1.endpoints.rag import get_rag_service
                rag_service = get_rag_service()
                self.logger.debug("Accessed RAG service singleton from endpoint module")
            except (ImportError, AttributeError, Exception) as e:
                self.logger.debug(
                    f"Could not access RAG service singleton from endpoint: {e}. "
                    "Trying to create new instance..."
                )
                
                # Method 2: Fallback - create new instance (may not have loaded model)
                try:
                    from .centralized_rag_service import CentralizedRAGService
                    rag_service = CentralizedRAGService()
                    # Check if this instance is initialized (has loaded models)
                    if not hasattr(rag_service, "llm_wrapper") or rag_service.llm_wrapper is None:
                        self.logger.debug(
                            "New RAG service instance created but not initialized. "
                            "No models to unload."
                        )
                        return
                except Exception as e2:
                    self.logger.debug(f"Could not create RAG service instance: {e2}")
                    return

            # Store reference for potential reuse
            if rag_service:
                self._rag_service = rag_service

            # Unload LLM model if loaded
            if (
                rag_service
                and hasattr(rag_service, "llm_wrapper")
                and rag_service.llm_wrapper is not None
            ):
                llm_wrapper: LocalLLMWrapper = rag_service.llm_wrapper

                # CRITICAL: Check if model is actually loaded using is_model_loaded()
                # This is the authoritative check - don't rely on just checking if wrapper exists
                if llm_wrapper.is_model_loaded():
                    current_model = llm_wrapper.model_name
                    self.logger.info(
                        f"RAG LLM model '{current_model}' is loaded. "
                        "Using unload_model() to free memory before fine-tuning..."
                    )

                    # CRITICAL: Use unload_model() from LocalLLMWrapper
                    # This properly:
                    # 1. Deletes model and tokenizer (sets to None)
                    # 2. Runs aggressive garbage collection twice (clears Jinja2 template cache)
                    # 3. Clears CUDA cache if available
                    # This prevents "duplicate template name" errors and memory conflicts
                    llm_wrapper.unload_model()

                    # Verify unload was successful
                    if not llm_wrapper.is_model_loaded():
                        self.logger.success(
                            "RAG LLM model unloaded successfully. "
                            "Memory freed and caches cleared."
                        )
                    else:
                        self.logger.warning(
                            "RAG LLM model unload called but model still appears loaded. "
                            "This may indicate an issue with the unload logic."
                        )
                else:
                    self.logger.debug("RAG LLM model not loaded, no need to unload")
            else:
                self.logger.debug("RAG service not initialized or no LLM wrapper")

            # Unload embedding model if loaded
            if (
                rag_service
                and hasattr(rag_service, "embedding_wrapper")
                and rag_service.embedding_wrapper is not None
            ):
                embedding_wrapper: LocalEmbeddingWrapper = rag_service.embedding_wrapper

                # Check if embedding model is loaded
                if embedding_wrapper.model is not None:
                    embedding_model = embedding_wrapper.model_name
                    self.logger.info(
                        f"RAG embedding model '{embedding_model}' is loaded. "
                        "Using unload_model() to free memory before fine-tuning..."
                    )

                    # Unload embedding model
                    embedding_wrapper.unload_model()

                    # Verify unload was successful
                    if embedding_wrapper.model is None:
                        self.logger.success(
                            "RAG embedding model unloaded successfully. "
                            "Memory freed for fine-tuning."
                        )
                    else:
                        self.logger.warning(
                            "RAG embedding model unload called but model still appears loaded. "
                            "This may indicate an issue with the unload logic."
                        )
                else:
                    self.logger.debug("RAG embedding model not loaded, no need to unload")
            else:
                self.logger.debug("RAG service not initialized or no embedding wrapper")

        except Exception as e:
            # Don't fail fine-tuning if we can't unload RAG models
            # Log warning but continue (graceful degradation)
            self.logger.warning(
                f"Could not unload RAG models before fine-tuning: {e}. "
                "Continuing anyway - this may cause memory issues or template conflicts."
            )

    def _load_model_and_tokenizer(
        self, model_name: str, use_quantization: bool = True
    ) -> tuple[Any, Any]:
        """
        Load model and tokenizer with intelligent resource detection (FIXED!).

        FIXES:
        - Uses ResourceDetector for optimal config
        - Prevents "meta tensor" errors
        - Validates model fits in memory
        - Automatic fallback strategies

        Args:
            model_name: HuggingFace model identifier.
            use_quantization: Whether to use quantization (recommended: True).

        Returns:
            tuple: (model, tokenizer)

        Raises:
            ModelNotFoundException: If model cannot be loaded.
        """
        try:
            self.logger.info(f"Loading model: {model_name}")

            # Load tokenizer - explicitly use fast tokenizer
            # Use HF_HOME for HuggingFace cache (set as environment variable)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=str(settings.HF_HOME),
                use_fast=True,  # Force fast tokenizer (requires sentencepiece for Mistral)
            )

            # Set pad token if not exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            # Get optimal model config from resource detector
            model_config = ResourceDetector.get_optimal_model_config(
                model_name, self.resources
            )

            self.logger.info(f"Loading model with config: {model_config}")

            # Attempt to load model with optimal config
            # Use HF_HOME for HuggingFace cache (set as environment variable)
            # Allow download if model not cached (local_files_only=False)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=str(settings.HF_HOME),
                    **model_config,
                )

                # Prepare for k-bit training if quantized
                if "quantization_config" in model_config:
                    model = prepare_model_for_kbit_training(model)
                    self.logger.success(
                        "Model loaded with quantization and prepared for training"
                    )

                # CRITICAL: Validate model fits in memory
                if not ResourceDetector.validate_model_fits(model, self.resources):
                    raise ModelNotFoundException(
                        f"{model_name} - Model too large! "
                        f"Try: {self.resources['recommended_model']}"
                    )

            except Exception as load_error:
                self.logger.error(f"Failed to load with optimal config: {load_error}")

                # Fallback strategy: try CPU offload
                if self.device == "cuda" and "max_memory" in model_config:
                    self.logger.warning(
                        "Attempting fallback with aggressive CPU offload"
                    )

                    model_config["max_memory"] = {
                        0: f"{self.resources['available_vram_gb'] * 0.7:.1f}GB",
                        "cpu": "24GB",
                    }

                    # Use HF_HOME for HuggingFace cache (set as environment variable)
                    # Allow download if model not cached (local_files_only=False)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        cache_dir=str(settings.HF_HOME),
                        **model_config,
                    )

                    if "quantization_config" in model_config:
                        model = prepare_model_for_kbit_training(model)

                    # Validate again
                    if not ResourceDetector.validate_model_fits(model, self.resources):
                        raise ModelNotFoundException(
                            f"{model_name} - Still too large with CPU offload! "
                            f"Recommended: {self.resources['recommended_model']}"
                        )

                    self.logger.success("Model loaded with aggressive CPU offload")
                else:
                    raise

            # Configure model for training
            model.config.use_cache = False
            model.config.pretraining_tp = 1

            # Explicitly disable gradient checkpointing
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
                self.logger.info("Gradient checkpointing disabled")

            self.logger.success(
                f"✅ Model loaded successfully: {model_name} "
                f"(fits in {self.resources['available_vram_gb']:.1f}GB VRAM)"
            )
            return model, tokenizer

        except Exception as e:
            import traceback

            self.logger.error(
                f"Failed to load model {model_name}: {e}\n{traceback.format_exc()}"
            )
            raise ModelNotFoundException(model_name)

    def _prepare_raft_dataset(
        self,
        dataset_path: str,
        tokenizer: Any,
        max_length: int = 512,
    ) -> Dataset:
        """
        Prepare RAFT dataset for training.

        RAFT format includes:
        - question: The query
        - context: Retrieved documents
        - answer: Ground truth answer

        Args:
            dataset_path: Path to dataset file (JSON/JSONL).
            tokenizer: Tokenizer for the model.
            max_length: Maximum sequence length.

        Returns:
            Dataset: Prepared HuggingFace dataset.

        Raises:
            DatasetException: If dataset preparation fails.
        """
        try:
            self.logger.info(f"Loading dataset from: {dataset_path}")

            dataset_file = Path(dataset_path)
            if not dataset_file.exists():
                raise DatasetException(
                    f"Dataset file not found: {dataset_path}",
                    dataset_path=dataset_path,
                )

            # Load dataset based on file extension
            if dataset_path.endswith(".jsonl"):
                dataset = load_dataset("json", data_files=dataset_path, split="train")
            elif dataset_path.endswith(".json"):
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                dataset = Dataset.from_list(data)
            else:
                raise DatasetException(
                    f"Unsupported dataset format: {dataset_path}",
                    dataset_path=dataset_path,
                )

            self.logger.info(f"Loaded {len(dataset)} examples")

            # Format examples for RAFT training
            def format_raft_example(example: Dict[str, Any]) -> Dict[str, str]:
                """Format a single example for RAFT training."""
                question = example.get("question", "")
                context = example.get("context", "")
                answer = example.get("answer", "")

                # Create instruction-following format
                prompt = f"""### Instruction:
Answer the question based on the provided context.

### Context:
{context}

### Question:
{question}

### Answer:
{answer}"""

                return {"text": prompt}

            # Apply formatting
            formatted_dataset = dataset.map(
                format_raft_example,
                remove_columns=dataset.column_names,
            )

            # Tokenize dataset (NO padding - will be done dynamically)
            def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, List]:
                """Tokenize examples and create labels for causal LM."""
                tokenized = tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=max_length,
                    padding=False,  # ✅ CHANGED: Dynamic padding per batch
                )
                # For causal language modeling, labels are the same as input_ids
                tokenized["labels"] = tokenized["input_ids"].copy()
                return tokenized

            tokenized_dataset = formatted_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
            )

            self.logger.success(f"Dataset prepared: {len(tokenized_dataset)} examples")
            return tokenized_dataset

        except Exception as e:
            self.logger.error(f"Dataset preparation failed: {e}")
            raise DatasetException(
                f"Failed to prepare dataset: {e}",
                dataset_path=dataset_path,
            )

    def _configure_peft(
        self,
        model: Any,
        lora_config: Dict[str, Any],
    ) -> Any:
        """
        Configure PEFT with LoRA/DoRA.

        Args:
            model: Base model to apply PEFT to.
            lora_config: LoRA configuration parameters.

        Returns:
            Any: Model with PEFT applied.
        """
        self.logger.info("Configuring PEFT with LoRA/DoRA")

        # Create PEFT config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
            use_dora=lora_config.get("use_dora", True),
            modules_to_save=lora_config.get("modules_to_save"),
        )

        # Apply PEFT to model
        peft_model = get_peft_model(model, peft_config)

        # Explicitly disable gradient checkpointing on PEFT model
        if hasattr(peft_model, "gradient_checkpointing_disable"):
            peft_model.gradient_checkpointing_disable()
        if hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()

        # Log trainable parameters
        trainable_params = sum(
            p.numel() for p in peft_model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in peft_model.parameters())
        trainable_percentage = 100 * trainable_params / total_params

        self.logger.info(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({trainable_percentage:.2f}%)"
        )

        return peft_model

    def _execute_training(
        self,
        job_id: str,
        model: Any,
        tokenizer: Any,
        dataset: Dataset,
        training_config: Dict[str, Any],
        output_dir: str,
    ) -> None:
        """
        Execute the training loop.

        Args:
            job_id: Job identifier.
            model: PEFT model to train.
            tokenizer: Tokenizer for the model.
            dataset: Prepared dataset.
            training_config: Training configuration.
            output_dir: Directory to save checkpoints.
        """
        try:
            self.logger.info(f"Starting training for job: {job_id}")

            # Update job status
            self.active_jobs[job_id]["status"] = TrainingStatus.RUNNING
            self.active_jobs[job_id]["start_time"] = datetime.utcnow()

            # Configure training arguments
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Determine optimizer based on BitsAndBytes availability
            try:
                import bitsandbytes as bnb

                optimizer = "paged_adamw_32bit"
                self.logger.info("Using paged_adamw_32bit optimizer (BitsAndBytes)")
            except Exception:
                optimizer = "adamw_torch"
                self.logger.warning(
                    "BitsAndBytes not available, using adamw_torch optimizer"
                )

            # Check if model is using device_map (for offloading)
            # If so, don't let Trainer move the model
            has_device_map = hasattr(model, "hf_device_map") and model.hf_device_map

            training_args = TrainingArguments(
                output_dir=str(output_path),
                num_train_epochs=training_config.get("num_epochs", 3),
                per_device_train_batch_size=training_config.get("batch_size", 4),
                gradient_accumulation_steps=training_config.get(
                    "gradient_accumulation_steps", 4
                ),
                learning_rate=training_config.get("learning_rate", 2e-4),
                warmup_steps=training_config.get("warmup_steps", 100),
                logging_steps=10,
                save_steps=500,
                save_total_limit=3,
                fp16=torch.cuda.is_available(),
                optim=optimizer,
                lr_scheduler_type="cosine",
                max_grad_norm=0.3,
                weight_decay=0.001,
                remove_unused_columns=False,
                report_to="none",  # Disable wandb/tensorboard
                # ✅ PERFORMANCE OPTIMIZATIONS
                dataloader_num_workers=2,  # Async data loading (2-4 recommended)
                dataloader_pin_memory=True,  # Faster CPU→GPU transfer
                gradient_checkpointing=False,  # Explicitly disable
                gradient_checkpointing_kwargs={
                    "use_reentrant": False
                },  # If enabled anyway
                group_by_length=True,  # Group similar lengths for efficiency
                ddp_find_unused_parameters=False,  # Optimization
                # ✅ FIX: Don't move model if using device_map (prevents meta tensor error)
                use_cpu=not torch.cuda.is_available(),
            )

            if has_device_map:
                self.logger.warning(
                    "Model is using device_map for offloading. "
                    "Training will use model's existing device placement."
                )

            # ✅ Create data collator for dynamic padding
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # Causal LM, not masked LM
            )

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                processing_class=tokenizer,
                data_collator=data_collator,  # ✅ Dynamic padding per batch
            )

            # Train model
            self.logger.info("Training started...")
            train_result = trainer.train()

            # Save adapter weights (LoRA/DoRA adapters only)
            adapter_path = output_path / "adapter"
            trainer.save_model(str(adapter_path))
            tokenizer.save_pretrained(str(adapter_path))
            self.logger.success(f"Adapter weights saved to: {adapter_path}")

            # Save FULL merged model weights (adapter + base model)
            self.logger.info("Merging adapter with base model for full weights...")
            try:
                # Merge LoRA weights with base model
                merged_model = model.merge_and_unload()

                # Save merged model
                merged_path = output_path / "merged_model"
                merged_model.save_pretrained(str(merged_path))
                tokenizer.save_pretrained(str(merged_path))

                self.logger.success(f"✅ Full merged model saved to: {merged_path}")

                # Update job with both paths
                self.active_jobs[job_id]["adapter_path"] = str(adapter_path)
                self.active_jobs[job_id]["merged_model_path"] = str(merged_path)
                self.active_jobs[job_id]["model_path"] = str(
                    merged_path
                )  # Default to merged

            except Exception as merge_error:
                self.logger.warning(
                    f"Could not merge model (adapter-only saved): {merge_error}"
                )
                # Fallback: just use adapter path
                self.active_jobs[job_id]["adapter_path"] = str(adapter_path)
                self.active_jobs[job_id]["model_path"] = str(adapter_path)

            # Update job with completion info
            self.active_jobs[job_id]["status"] = TrainingStatus.COMPLETED
            self.active_jobs[job_id]["end_time"] = datetime.utcnow()
            self.active_jobs[job_id]["metrics"] = {
                "train_loss": float(train_result.training_loss),
                "train_steps": train_result.global_step,
            }

            self.logger.success(f"Training completed for job: {job_id}")

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            self.logger.error(f"Training failed for job {job_id}: {e}\n{error_details}")
            self.active_jobs[job_id]["status"] = TrainingStatus.FAILED
            self.active_jobs[job_id]["error"] = str(e)
            self.active_jobs[job_id]["error_details"] = error_details

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def start_fine_tuning(self, request: FineTuneRequest) -> FineTuneResponse:
        """
        Start a fine-tuning job.

        Args:
            request: Fine-tuning job request.

        Returns:
            FineTuneResponse: Job response with job ID and status.

        Raises:
            FineTuningException: If fine-tuning fails to start.
        """
        job_id = str(uuid.uuid4())

        try:
            # CRITICAL: Unload RAG model before fine-tuning to free memory
            # This prevents memory conflicts and "duplicate template name" errors
            await self._unload_rag_model_if_loaded()

            # Create job record
            job_info = {
                "job_id": job_id,
                "status": TrainingStatus.PENDING,
                "model_name": request.model_name,
                "dataset_path": request.dataset_path,
                "output_dir": request.output_dir,
                "created_at": datetime.utcnow(),
                "lora_config": request.lora_config.model_dump(),
                "training_config": request.training_config.model_dump(),
            }

            self.active_jobs[job_id] = job_info
            self.logger.info(f"Created fine-tuning job: {job_id}")

            # Load model and tokenizer
            model, tokenizer = self._load_model_and_tokenizer(
                request.model_name,
                use_quantization=True,
            )

            # Prepare RAFT dataset
            dataset = self._prepare_raft_dataset(
                request.dataset_path,
                tokenizer,
                max_length=request.training_config.max_seq_length,
            )

            # Configure PEFT (LoRA/DoRA)
            peft_model = self._configure_peft(
                model,
                request.lora_config.model_dump(),
            )

            # Start training in background
            asyncio.create_task(
                asyncio.to_thread(
                    self._execute_training,
                    job_id,
                    peft_model,
                    tokenizer,
                    dataset,
                    request.training_config.model_dump(),
                    request.output_dir,
                )
            )

            return FineTuneResponse(
                job_id=job_id,
                status=TrainingStatus.PENDING,
                model_name=request.model_name,
                created_at=job_info["created_at"],
                message="Fine-tuning job started successfully",
                progress=0.0,
                error=None,
                metrics=None,
            )

        except Exception as e:
            self.handle_error(e, {"job_id": job_id})
            if job_id in self.active_jobs:
                self.active_jobs[job_id]["status"] = TrainingStatus.FAILED
            raise FineTuningException(
                f"Failed to start fine-tuning: {e}", job_id=job_id
            )

    async def get_job_status(self, job_id: str) -> FineTuneResponse:
        """
        Get fine-tuning job status.

        Args:
            job_id: Job identifier.

        Returns:
            FineTuneResponse: Current job status with metrics.

        Raises:
            FineTuningException: If job not found.
        """
        if job_id not in self.active_jobs:
            raise FineTuningException(f"Job not found: {job_id}", job_id=job_id)

        job_info = self.active_jobs[job_id]

        # Build status message
        status = job_info["status"]
        message = ""

        if status == TrainingStatus.PENDING:
            message = "Job is pending, waiting to start training"
        elif status == TrainingStatus.RUNNING:
            start_time = job_info.get("start_time")
            if start_time:
                elapsed = (datetime.utcnow() - start_time).total_seconds() / 60
                message = f"Training in progress (elapsed: {elapsed:.1f} minutes)"
            else:
                message = "Training in progress"
        elif status == TrainingStatus.COMPLETED:
            metrics = job_info.get("metrics", {})
            model_path = job_info.get("model_path", "")
            message = (
                f"Training completed successfully. "
                f"Loss: {metrics.get('train_loss', 'N/A')}, "
                f"Steps: {metrics.get('train_steps', 'N/A')}. "
                f"Model saved at: {model_path}"
            )
        elif status == TrainingStatus.FAILED:
            error = job_info.get("error", "Unknown error")
            message = f"Training failed: {error}"
        elif status == TrainingStatus.CANCELLED:
            message = "Training was cancelled by user"

        return FineTuneResponse(
            job_id=job_id,
            status=status,
            model_name=job_info["model_name"],
            created_at=job_info["created_at"],
            message=message,
            progress=job_info.get("progress", 0.0),
            error=job_info.get("error"),
            metrics=job_info.get("metrics"),
        )

    async def cancel_job(self, job_id: str) -> FineTuneResponse:
        """
        Cancel a fine-tuning job.

        Args:
            job_id: Job identifier.

        Returns:
            FineTuneResponse: Updated job status.

        Raises:
            FineTuningException: If job cannot be cancelled.
        """
        if job_id not in self.active_jobs:
            raise FineTuningException(f"Job not found: {job_id}", job_id=job_id)

        job_info = self.active_jobs[job_id]

        if job_info["status"] == TrainingStatus.COMPLETED:
            raise FineTuningException("Cannot cancel completed job", job_id=job_id)

        job_info["status"] = TrainingStatus.CANCELLED
        job_info["message"] = "Job cancelled by user"

        self.logger.info(f"Cancelled job: {job_id}")

        return FineTuneResponse(
            job_id=job_id,
            status=TrainingStatus.CANCELLED,
            model_name=job_info["model_name"],
            created_at=job_info["created_at"],
            message="Job cancelled successfully",
            progress=job_info.get("progress", 0.0),
            error=None,
            metrics=job_info.get("metrics"),
        )

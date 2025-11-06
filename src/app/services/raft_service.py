"""
RAFT (Retrieval Augmented Fine-Tuning) service.

This service implements RAFT methodology by augmenting training data
with retrieved context documents before fine-tuning.
"""

import asyncio
import json
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from ..core.config import get_settings
from ..core.exceptions import FineTuningException, RAGException
from ..schemas.fine_tuning import FineTuneRequest
from ..schemas.raft import (
    AugmentedExample,
    ContextMode,
    QAPair,
    RaftFineTuneRequest,
    RaftFineTuneResponse,
)
from .base import BaseService
from .centralized_rag_service import CentralizedRAGService
from .fine_tuning_service import FineTuningService

settings = get_settings()


class RaftService(BaseService):
    """
    RAFT service for retrieval augmented fine-tuning.

    This service:
    1. Loads Q&A pairs from dataset
    2. Retrieves relevant documents from ChromaDB for each question
    3. Adds distractor documents
    4. Formats context according to template
    5. Creates augmented dataset
    6. Delegates to FineTuningService for actual training

    Uses composition pattern to leverage existing services.
    """

    def __init__(self) -> None:
        """Initialize RAFT service."""
        super().__init__("RaftService")
        self.fine_tuning_service = FineTuningService()
        self.rag_service = CentralizedRAGService()
        self.jobs: Dict[str, Dict[str, Any]] = {}

    async def _initialize_impl(self) -> None:
        """
        Initialize RAFT service resources.

        Returns:
            None
        """
        # Initialize dependent services
        if not self.fine_tuning_service.is_initialized:
            await self.fine_tuning_service.initialize()

        if not self.rag_service.is_initialized:
            await self.rag_service.initialize()

        self.logger.success("RAFT service initialized with dependent services")

    async def start_raft_fine_tuning(
        self, request: RaftFineTuneRequest
    ) -> RaftFineTuneResponse:
        """
        Start RAFT fine-tuning job.

        Args:
            request: RAFT fine-tuning request.

        Returns:
            RaftFineTuneResponse: Job information.

        Raises:
            FineTuningException: If RAFT fine-tuning fails to start.
        """
        job_id = f"raft_{uuid.uuid4().hex[:8]}"
        self.logger.info(f"Starting RAFT fine-tuning job: {job_id}")

        try:
            # Create job record
            job_info = {
                "job_id": job_id,
                "status": "pending",
                "model_name": request.model_name,
                "output_dir": request.output_dir,
                "collection_name": request.raft_config.collection_name,
                "num_examples": 0,
                "num_augmented": 0,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "error": None,
                "progress": 0.0,
            }
            self.jobs[job_id] = job_info

            # Start RAFT augmentation and training in background
            asyncio.create_task(self._run_raft_pipeline(job_id, request))

            return RaftFineTuneResponse(**job_info)

        except Exception as e:
            self.handle_error(e, {"job_id": job_id})
            raise FineTuningException(f"Failed to start RAFT job: {e}")

    async def _run_raft_pipeline(
        self, job_id: str, request: RaftFineTuneRequest
    ) -> None:
        """
        Run the complete RAFT pipeline.

        Args:
            job_id: Job identifier.
            request: RAFT fine-tuning request.

        Returns:
            None
        """
        try:
            self._update_job_status(job_id, "running", progress=5.0)

            # Step 1: Load Q&A pairs
            self.logger.info(
                f"[{job_id}] Loading Q&A pairs from {request.dataset_path}"
            )
            qa_pairs = await self._load_qa_pairs(request.dataset_path)
            self.jobs[job_id]["num_examples"] = len(qa_pairs)
            self._update_job_status(job_id, "running", progress=10.0)

            # Step 2: Augment dataset with retrieved context
            self.logger.info(
                f"[{job_id}] Augmenting {len(qa_pairs)} examples with RAFT"
            )
            augmented_dataset = await self._augment_dataset(
                qa_pairs, request.raft_config, job_id
            )
            self.jobs[job_id]["num_augmented"] = len(augmented_dataset)
            self._update_job_status(job_id, "running", progress=40.0)

            # Step 3: Save augmented dataset
            augmented_dataset_path = await self._save_augmented_dataset(
                augmented_dataset, request.output_dir, job_id
            )
            self._update_job_status(job_id, "running", progress=50.0)

            # Step 4: Create fine-tuning request with augmented dataset
            fine_tune_request = FineTuneRequest(
                model_name=request.model_name,
                dataset_path=str(augmented_dataset_path),
                output_dir=request.output_dir,
                lora_config=request.lora_config,
                training_config=request.training_config,
            )

            # Step 5: Delegate to fine-tuning service
            self.logger.info(f"[{job_id}] Starting fine-tuning with augmented dataset")
            fine_tune_response = await self.fine_tuning_service.start_fine_tuning(
                fine_tune_request
            )

            # Store the underlying fine-tuning job ID for status tracking
            self.jobs[job_id]["fine_tune_job_id"] = fine_tune_response.job_id
            self._update_job_status(job_id, "running", progress=60.0)

            # Monitor fine-tuning progress
            await self._monitor_fine_tuning(job_id, fine_tune_response.job_id)

        except Exception as e:
            self.handle_error(e, {"job_id": job_id})
            self._update_job_status(
                job_id, "failed", error=str(e), progress=self.jobs[job_id]["progress"]
            )

    async def _load_qa_pairs(self, dataset_path: str) -> List[QAPair]:
        """
        Load Q&A pairs from dataset file.

        Args:
            dataset_path: Path to Q&A dataset (JSON/JSONL).

        Returns:
            List[QAPair]: Loaded Q&A pairs.

        Raises:
            FineTuningException: If loading fails.
        """
        path = Path(dataset_path)
        if not path.exists():
            raise FineTuningException(f"Dataset not found: {dataset_path}")

        qa_pairs: List[QAPair] = []

        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix == ".jsonl":
                    # JSONL format
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            qa_pairs.append(QAPair(**data))
                else:
                    # JSON format
                    data = json.load(f)
                    if isinstance(data, list):
                        qa_pairs = [QAPair(**item) for item in data]
                    else:
                        raise ValueError("JSON must contain a list of Q&A pairs")

            self.logger.info(f"Loaded {len(qa_pairs)} Q&A pairs from {dataset_path}")
            return qa_pairs

        except Exception as e:
            raise FineTuningException(f"Failed to load Q&A pairs: {e}")

    async def _augment_dataset(
        self, qa_pairs: List[QAPair], raft_config: Any, job_id: str
    ) -> List[AugmentedExample]:
        """
        Augment Q&A pairs with retrieved context documents.

        Args:
            qa_pairs: Original Q&A pairs.
            raft_config: RAFT configuration.
            job_id: Job identifier for progress tracking.

        Returns:
            List[AugmentedExample]: Augmented examples.
        """
        augmented_examples: List[AugmentedExample] = []
        total = len(qa_pairs)

        for idx, qa_pair in enumerate(qa_pairs):
            try:
                # Retrieve relevant documents
                context_docs = await self._retrieve_context_docs(
                    qa_pair.question, raft_config
                )

                # Get distractor documents
                distractor_docs = await self._get_distractor_docs(
                    qa_pair.question, context_docs, raft_config
                )

                # Handle oracle mode
                if raft_config.context_mode == ContextMode.ORACLE:
                    if qa_pair.ground_truth_doc:
                        # Ensure ground-truth doc is in context
                        if qa_pair.ground_truth_doc not in context_docs:
                            context_docs = [qa_pair.ground_truth_doc] + context_docs[
                                :-1
                            ]

                # Format the training example
                formatted_text = self._format_raft_example(
                    qa_pair, context_docs, distractor_docs, raft_config
                )

                augmented_example = AugmentedExample(
                    original_question=qa_pair.question,
                    original_answer=qa_pair.answer,
                    context_docs=context_docs,
                    distractor_docs=distractor_docs,
                    formatted_text=formatted_text,
                )
                augmented_examples.append(augmented_example)

                # Update progress
                progress = 10.0 + (30.0 * (idx + 1) / total)
                self._update_job_status(job_id, "running", progress=progress)

            except Exception as e:
                self.logger.warning(f"Failed to augment example {idx}: {e}. Skipping.")
                continue

        self.logger.info(
            f"Successfully augmented {len(augmented_examples)}/{len(qa_pairs)} examples"
        )
        return augmented_examples

    async def _retrieve_context_docs(
        self, question: str, raft_config: Any
    ) -> List[str]:
        """
        Retrieve relevant context documents for a question.

        Args:
            question: The question to retrieve context for.
            raft_config: RAFT configuration.

        Returns:
            List[str]: Retrieved document texts.
        """
        self.logger.debug(
            f"Retrieving {raft_config.num_context_docs} docs for: {question[:50]}..."
        )

        # Determine how many documents to retrieve
        retrieve_k = raft_config.num_context_docs
        if raft_config.use_reranking:
            retrieve_k = raft_config.num_context_docs * settings.rerank_top_k_multiplier
            self.logger.debug(
                f"Reranking enabled for RAFT: retrieving {retrieve_k} docs "
                f"(will rerank to {raft_config.num_context_docs})"
            )

        # Use centralized RAG service for retrieval
        try:
            # Get initial documents
            context_docs_raw = await self.rag_service.retrieve_documents(
                query=question,
                collection_name=raft_config.collection_name,
                top_k=retrieve_k,
                similarity_threshold=raft_config.similarity_threshold,
            )

            # Apply reranking if enabled (NEW!)
            if (
                raft_config.use_reranking
                and self.rag_service.reranker_service
                and len(context_docs_raw) > 0
            ):
                self.logger.debug("Reranking RAFT context documents")

                # Convert strings to format expected by reranker
                docs_for_reranking = [
                    {
                        "content": doc,
                        "score": 1.0,
                        "metadata": {},
                        "doc_id": f"doc_{i}",
                    }
                    for i, doc in enumerate(context_docs_raw)
                ]

                # Rerank
                reranked_docs = self.rag_service.reranker_service.rerank_documents(
                    query=question,
                    documents=docs_for_reranking,
                    top_k=raft_config.num_context_docs,
                )

                # Extract text content
                context_docs = [doc["content"] for doc in reranked_docs]
                self.logger.success(
                    f"Reranked {len(context_docs_raw)} → {len(context_docs)} docs for RAFT"
                )
            else:
                # No reranking, just take top_k
                context_docs = context_docs_raw[: raft_config.num_context_docs]

            return context_docs

        except Exception as e:
            self.logger.warning(f"RAG retrieval failed: {e}. Using empty context.")
            return []

    async def _get_distractor_docs(
        self, question: str, context_docs: List[str], raft_config: Any
    ) -> List[str]:
        """
        Get distractor (irrelevant) documents.

        Args:
            question: The question.
            context_docs: Already selected context documents.
            raft_config: RAFT configuration.

        Returns:
            List[str]: Distractor document texts.
        """
        if raft_config.num_distractor_docs == 0:
            return []

        # Use centralized RAG service to get distractors
        try:
            distractors = await self.rag_service.get_distractor_documents(
                query=question,
                collection_name=raft_config.collection_name,
                exclude_docs=context_docs,
                count=raft_config.num_distractor_docs,
            )
            return distractors
        except Exception as e:
            self.logger.warning(
                f"Distractor retrieval failed: {e}. Using empty distractors."
            )
            return []

    def _format_raft_example(
        self,
        qa_pair: QAPair,
        context_docs: List[str],
        distractor_docs: List[str],
        raft_config: Any,
    ) -> str:
        """
        Format RAFT training example with context, question, and answer.

        Args:
            qa_pair: Original Q&A pair.
            context_docs: Retrieved context documents.
            distractor_docs: Distractor documents.
            raft_config: RAFT configuration.

        Returns:
            str: Formatted training text.
        """
        # Combine and shuffle context + distractors
        all_docs = context_docs + distractor_docs
        random.shuffle(all_docs)

        # Format each document using template
        formatted_docs = []
        for idx, doc_content in enumerate(all_docs, 1):
            formatted_doc = raft_config.context_template.format(
                idx=idx, content=doc_content
            )
            formatted_docs.append(formatted_doc)

        # Build final training text
        context_section = "Context:\n" + "".join(formatted_docs)
        question_section = f"Question: {qa_pair.question}\n"
        answer_section = f"Answer: {qa_pair.answer}"

        formatted_text = f"{context_section}\n{question_section}{answer_section}"

        return formatted_text

    async def _save_augmented_dataset(
        self, augmented_examples: List[AugmentedExample], output_dir: str, job_id: str
    ) -> Path:
        """
        Save augmented dataset to disk.

        Args:
            augmented_examples: Augmented training examples.
            output_dir: Output directory.
            job_id: Job identifier.

        Returns:
            Path: Path to saved dataset.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dataset_path = output_path / f"raft_augmented_{job_id}.jsonl"

        with open(dataset_path, "w", encoding="utf-8") as f:
            for example in augmented_examples:
                # Save in format expected by fine-tuning service
                record = {"text": example.formatted_text}
                f.write(json.dumps(record) + "\n")

        self.logger.success(f"Saved augmented dataset to {dataset_path}")
        return dataset_path

    async def _monitor_fine_tuning(
        self, raft_job_id: str, fine_tune_job_id: str
    ) -> None:
        """
        Monitor the underlying fine-tuning job and update RAFT job status.

        Args:
            raft_job_id: RAFT job identifier.
            fine_tune_job_id: Underlying fine-tuning job identifier.

        Returns:
            None
        """
        while True:
            try:
                # Get fine-tuning job status
                ft_response = await self.fine_tuning_service.get_job_status(
                    fine_tune_job_id
                )

                # Map progress (60-100% for fine-tuning phase)
                ft_progress = 60.0 + (40.0 * ft_response.progress / 100.0)

                if ft_response.status == "completed":
                    self._update_job_status(raft_job_id, "completed", progress=100.0)
                    self.logger.success(
                        f"RAFT job {raft_job_id} completed successfully"
                    )
                    break
                elif ft_response.status == "failed":
                    self._update_job_status(
                        raft_job_id,
                        "failed",
                        error=ft_response.error,
                        progress=ft_progress,
                    )
                    self.logger.error(
                        f"RAFT job {raft_job_id} failed: {ft_response.error}"
                    )
                    break
                elif ft_response.status == "cancelled":
                    self._update_job_status(
                        raft_job_id, "cancelled", progress=ft_progress
                    )
                    self.logger.info(f"RAFT job {raft_job_id} was cancelled")
                    break
                else:
                    self._update_job_status(
                        raft_job_id, "running", progress=ft_progress
                    )

                # Wait before next check
                await asyncio.sleep(5)

            except Exception as e:
                self.logger.error(f"Error monitoring fine-tuning job: {e}")
                await asyncio.sleep(10)

    def _update_job_status(
        self,
        job_id: str,
        status: str,
        error: Optional[str] = None,
        progress: Optional[float] = None,
    ) -> None:
        """
        Update job status.

        Args:
            job_id: Job identifier.
            status: New status.
            error: Optional error message.
            progress: Optional progress percentage.

        Returns:
            None
        """
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
            self.jobs[job_id]["updated_at"] = datetime.utcnow()
            if error is not None:
                self.jobs[job_id]["error"] = error
            if progress is not None:
                self.jobs[job_id]["progress"] = progress

    async def get_job_status(self, job_id: str) -> RaftFineTuneResponse:
        """
        Get RAFT job status.

        Args:
            job_id: Job identifier.

        Returns:
            RaftFineTuneResponse: Current job status.

        Raises:
            FineTuningException: If job not found.
        """
        if job_id not in self.jobs:
            raise FineTuningException(f"RAFT job not found: {job_id}")

        return RaftFineTuneResponse(**self.jobs[job_id])

    async def cancel_job(self, job_id: str) -> RaftFineTuneResponse:
        """
        Cancel a RAFT job.

        Args:
            job_id: Job identifier.

        Returns:
            RaftFineTuneResponse: Updated job status.

        Raises:
            FineTuningException: If job cannot be cancelled.
        """
        if job_id not in self.jobs:
            raise FineTuningException(f"RAFT job not found: {job_id}")

        job_info = self.jobs[job_id]

        # If there's an underlying fine-tuning job, cancel it
        if "fine_tune_job_id" in job_info:
            try:
                await self.fine_tuning_service.cancel_job(job_info["fine_tune_job_id"])
            except Exception as e:
                self.logger.warning(f"Failed to cancel underlying fine-tune job: {e}")

        self._update_job_status(job_id, "cancelled")
        self.logger.info(f"Cancelled RAFT job: {job_id}")

        return RaftFineTuneResponse(**self.jobs[job_id])

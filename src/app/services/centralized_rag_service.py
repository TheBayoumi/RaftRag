"""
Centralized RAG Service using custom ChromaDB ingestion + RagAnything orchestration.

FIXED: RagAnything's document ingestion was broken (producing 0 chunks).
SOLUTION: Use custom ChromaDB ingestion + RagAnything for retrieval/reasoning only.

This service provides a unified RAG pipeline using:
- CUSTOM ChromaDB ingestion (bypasses RagAnything's broken parser)
- RagAnything for query orchestration, retrieval, evaluation, reasoning
- Local models (LocalLLMWrapper + LocalEmbeddingWrapper, no external APIs)
- Separate RAFT and RAG databases

For both:
1. Standalone RAG operations (document ingestion, querying)
2. RAFT fine-tuning (document retrieval, distractor selection)
"""

import asyncio
import re
import time
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np
from lightrag.utils import EmbeddingFunc
from loguru import logger
from raganything import RAGAnything, RAGAnythingConfig

from ..core.config import get_settings
from ..core.exceptions import RAGException
from ..prompts.rag_system_prompt import (
    RAG_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT_CONCISE,
    RAG_SYSTEM_PROMPT_STRICT,
)
from ..schemas.rag import (
    DocumentResponse,
    DocumentUpload,
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
)
from ..storage.chroma_vector_storage import ChromaVectorDBStorage
from ..utils.local_models import LocalEmbeddingWrapper, LocalLLMWrapper
from .base import BaseService
from .document_ingestion_service import DocumentIngestionService

settings = get_settings()

# RAG System Prompt for guiding LLM behavior during retrieval-augmented answering


class CentralizedRAGService(BaseService):
    """
    Centralized RAG service with custom ChromaDB ingestion.

    This service provides:
    - CUSTOM document ingestion (direct ChromaDB storage - FIX for 0 chunks issue!)
    - RagAnything for query orchestration and reasoning (NOT ingestion!)
    - Separate RAFT and RAG databases
    - Semantic retrieval with local embeddings
    - Query answering with local LLM
    - Distractor document selection for RAFT

    Architecture: Custom ChromaDB Ingestion + RagAnything Orchestration + Local Models.
    """

    def __init__(self) -> None:
        """Initialize centralized RAG service."""
        super().__init__("CentralizedRAGService")

        # Custom document ingestion service (REPLACES RagAnything's broken ingestion)
        self.ingestion_service: Optional[DocumentIngestionService] = None

        # RAGAnything components (for orchestration/reasoning only, NOT ingestion)
        self.rag_engine: Optional[RAGAnything] = None

        # Local model wrappers
        self.llm_wrapper: Optional[LocalLLMWrapper] = None
        self.embedding_wrapper: Optional[LocalEmbeddingWrapper] = None
        self._llm_func: Optional[Callable[..., Awaitable[str]]] = None
        self._embedding_func: Optional[EmbeddingFunc] = None

        # Document and collection tracking
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.collections: Dict[str, Any] = {}

    async def _initialize_impl(self) -> None:
        """
        Initialize RAG service with custom ingestion and RagAnything orchestration.

        Returns:
            None
        """
        # Directories are created in ensure_directories() on startup
        # No need to create here (redundant)

        # Initialize custom document ingestion service (CRITICAL FIX!)
        self.logger.info("Initializing custom document ingestion service")
        self.ingestion_service = DocumentIngestionService()
        await self.ingestion_service.initialize()
        self.logger.success("Custom document ingestion service initialized")

        # Initialize embedding model
        self.logger.info("Initializing embedding model")
        self.embedding_wrapper = LocalEmbeddingWrapper()
        self.embedding_wrapper.load_model()
        self.logger.success("Embedding model initialized")

        async def embedding_func(
            texts: List[str], *_args: Any, **kwargs: Any
        ) -> np.ndarray:
            """Asynchronous wrapper for embeddings compatible with LightRAG."""
            _ = kwargs.pop("_priority", None)
            _ = kwargs.pop("_timeout", None)
            batch = texts if isinstance(texts, list) else [str(texts)]
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                None,
                self.embedding_wrapper.embed_documents,
                batch,
            )
            return np.asarray(embeddings, dtype=np.float32)

        self._embedding_func = EmbeddingFunc(
            embedding_dim=self.embedding_wrapper.embedding_dim,
            func=embedding_func,
        )

        # Initialize LLM
        self.logger.info("Initializing LLM")
        self.llm_wrapper = LocalLLMWrapper()
        self.llm_wrapper.load_model()
        self.logger.success("LLM initialized")

        async def llm_func(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: Optional[List[Dict[str, str]]] = None,
            **kwargs: Any,
        ) -> str:
            """Run local LLM calls in an executor for async compatibility."""

            loop = asyncio.get_running_loop()
            call_llm = partial(
                self.llm_wrapper,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )
            return await loop.run_in_executor(None, call_llm)

        self._llm_func = llm_func

        # Create RAGAnything config
        self.logger.info("Creating RAGAnything configuration")
        rag_config = RAGAnythingConfig(
            working_dir=str(settings.rag_working_dir),
            parser=settings.rag_parser,  # docling for multimodal
            parse_method=settings.rag_parse_method,
            enable_image_processing=settings.enable_rag_image_processing,
            enable_table_processing=settings.enable_rag_table_processing,
            enable_equation_processing=settings.enable_rag_equation_processing,
            context_window=settings.rag_context_window,
            max_context_tokens=settings.rag_max_context_tokens,
        )

        # Configure global config for ChromaVectorDBStorage
        global_config = {
            "ingestion_service": self.ingestion_service,
            "database_type": "rag",  # Default to RAG database
        }

        # CRITICAL FIX: Create LightRAG instance with custom storage
        # RAGAnything validates vector_storage against a whitelist, so we need to:
        # 1. Create LightRAG with a valid default storage name
        # 2. Replace the storage with our custom implementation after LightRAG init
        # 3. Ensure all storage instances (entities, relationships, chunks) use our custom storage
        from lightrag import LightRAG
        from lightrag.kg.shared_storage import initialize_pipeline_status

        self.logger.info("Creating LightRAG with ChromaVectorDBStorage adapter")

        # Create LightRAG with default storage to pass validation
        # Disable LLM caching to avoid pickle errors with closures that capture self
        lightrag_instance = LightRAG(
            working_dir=str(settings.rag_working_dir),
            llm_model_func=self._llm_func,
            embedding_func=self._embedding_func,
            vector_storage="NanoVectorDBStorage",  # Use default to pass validation
            graph_storage="NetworkXStorage",
            enable_llm_cache=False,  # Disable caching to avoid pickle errors
            enable_llm_cache_for_entity_extract=False,  # Disable entity extraction cache too
        )

        # ✅ CRITICAL FIX: Initialize LightRAG storages BEFORE using them
        # This initializes internal asyncio.Lock objects that prevent
        # "'NoneType' object does not support the asynchronous context manager protocol" errors
        # See: https://github.com/HKUDS/LightRAG/issues/1933
        self.logger.info("Initializing LightRAG storages and pipeline...")
        await lightrag_instance.initialize_storages()
        await initialize_pipeline_status()
        self.logger.success("LightRAG storages initialized successfully")

        # Now replace the vector storage with our custom implementation
        # LightRAG uses the storage object directly, so we need to replace it properly
        # Get the namespace from the existing storage
        existing_storage = lightrag_instance.vector_storage

        # Check if it's already an object (not a string)
        if hasattr(existing_storage, "namespace"):
            default_namespace = existing_storage.namespace
            default_meta_fields = getattr(existing_storage, "meta_fields", None)
            self.logger.debug(
                f"Extracted namespace from existing storage: {default_namespace}"
            )
        else:
            # If it's a string, use a default namespace
            # This shouldn't happen, but handle it gracefully
            default_namespace = "lightrag"
            default_meta_fields = None
            self.logger.warning(
                f"LightRAG vector_storage is a string ({existing_storage}), "
                f"using default namespace: {default_namespace}"
            )

        # Create our custom storage instance
        # Get workspace from LightRAG instance
        workspace = getattr(
            lightrag_instance, "workspace", str(settings.rag_working_dir)
        )

        # Get cosine_better_than_threshold from existing storage if available
        cosine_threshold = (
            getattr(
                existing_storage,
                "cosine_better_than_threshold",
                0.2,
            )
            if hasattr(existing_storage, "cosine_better_than_threshold")
            else 0.2
        )

        custom_storage = ChromaVectorDBStorage(
            namespace=default_namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=self._embedding_func,
            cosine_better_than_threshold=cosine_threshold,
            meta_fields=default_meta_fields,
        )

        # Initialize the custom storage
        await custom_storage.initialize()

        # Replace the vector storage - ensure it's replaced in the instance
        # Verify the replacement worked
        lightrag_instance.vector_storage = custom_storage

        # Verify replacement succeeded
        if not hasattr(lightrag_instance.vector_storage, "namespace"):
            raise RuntimeError(
                "Failed to replace vector_storage - it's still a string or invalid object"
            )

        self.logger.debug(
            f"Successfully replaced vector_storage with ChromaVectorDBStorage "
            f"(namespace: {lightrag_instance.vector_storage.namespace})"
        )

        # Also check if LightRAG has internal storage references that need updating
        # LightRAG might have separate storage for entities, relationships, chunks
        if hasattr(lightrag_instance, "_vector_storage_entities"):
            lightrag_instance._vector_storage_entities = custom_storage
        if hasattr(lightrag_instance, "_vector_storage_relationships"):
            lightrag_instance._vector_storage_relationships = custom_storage
        if hasattr(lightrag_instance, "_vector_storage_chunks"):
            lightrag_instance._vector_storage_chunks = custom_storage

        self.logger.success(
            "LightRAG instance created with custom ChromaVectorDBStorage"
        )

        # Initialize RAGAnything with pre-initialized LightRAG
        self.logger.info("Initializing RAGAnything with pre-initialized LightRAG")
        self.rag_engine = RAGAnything(
            lightrag=lightrag_instance,  # Pass our custom LightRAG instance
            llm_model_func=self._llm_func,
            embedding_func=self._embedding_func,
            config=rag_config,
        )

        # After RAGAnything init, verify our storage is still in place
        # RAGAnything might have re-validated and reset it
        if hasattr(self.rag_engine, "lightrag") and hasattr(
            self.rag_engine.lightrag, "vector_storage"
        ):
            if isinstance(self.rag_engine.lightrag.vector_storage, str):
                self.logger.warning(
                    "RAGAnything reset vector_storage to string, replacing again"
                )
                self.rag_engine.lightrag.vector_storage = custom_storage

        self.logger.success(
            "RAG service initialized with ChromaVectorDBStorage (hybrid search)"
        )

    # ==================== STANDALONE RAG METHODS ====================

    async def upload_document(self, upload: DocumentUpload) -> DocumentResponse:
        """
        Upload and process document using CUSTOM ingestion (FIXED!).

        CRITICAL FIX: Uses DocumentIngestionService instead of RagAnything's
        broken ingestion that was producing 0 chunks!

        Args:
            upload: Document upload request.

        Returns:
            DocumentResponse: Upload response with ACCURATE document info.

        Raises:
            RAGException: If document ingestion fails.
        """
        self.logger.info(f"Uploading document: {upload.file_path}")

        try:
            # Ensure service is properly initialized (critical for multi-worker)
            if not self._initialized or self.ingestion_service is None:
                self.logger.warning(
                    "Service not initialized (likely after unpickling), reinitializing now..."
                )
                # Clean up old instances
                if self.rag_engine is not None:
                    del self.rag_engine
                    self.rag_engine = None

                await self._initialize_impl()
                self._initialized = True
                self.logger.success(
                    "Service reinitialized successfully for this worker"
                )

            # Verify ingestion service is available
            if self.ingestion_service is None:
                raise RAGException(
                    "Ingestion service not initialized. Service initialization failed.",
                    operation="upload",
                )

            # Track collection
            if upload.collection_name not in self.collections:
                self.collections[upload.collection_name] = {
                    "created_at": datetime.utcnow(),
                    "document_count": 0,
                }

            # Use custom ingestion service (BYPASSES RagAnything's broken parser!)
            self.logger.debug("Using custom ChromaDB ingestion (FIXED!)")

            response = await self.ingestion_service.ingest_document(
                file_path=upload.file_path,
                collection_name=upload.collection_name,
                database_type="rag",  # Store in RAG database
                chunk_size=getattr(upload, "chunk_size", 512),
                chunk_overlap=getattr(upload, "chunk_overlap", 50),
            )

            # Store document metadata
            self.documents[response.doc_id] = {
                "doc_id": response.doc_id,
                "filename": response.filename,
                "collection_name": response.collection_name,
                "file_path": upload.file_path,
                "file_size_mb": response.file_size_mb,
                "num_chunks": response.num_chunks,
                "created_at": response.created_at,
            }

            # Update collection count
            self.collections[upload.collection_name]["document_count"] += 1

            self.logger.success(
                f"Document uploaded successfully: {response.doc_id} "
                f"({response.num_chunks} chunks, {response.file_size_mb:.1f}MB) "
                f"✅ FIXED - num_chunks > 0!"
            )

            return response

        except RAGException:
            raise
        except Exception as e:
            self.handle_error(e, {"file": upload.file_path})
            raise RAGException(f"Failed to upload document: {e}", operation="upload")

    async def query_with_answer(self, request: QueryRequest) -> QueryResponse:
        """
        Perform RAG query using raganything's aquery() with custom storage.

        APPROACH A (Full raganything Integration):
        - raganything orchestrates the entire pipeline
        - ChromaVectorDBStorage adapter uses our hybrid BM25 + Semantic search
        - Local LLM generates answers with optimal zero-shot system prompt
        - Clean, simple, leverages raganything's capabilities

        Args:
            request: Query request with query, collection_name, top_k, etc.

        Returns:
            QueryResponse: Query response with answer and metadata.

        Raises:
            RAGException: If query fails.
        """
        start_time = time.time()
        self.logger.info(
            f"Processing RAG query via raganything: {request.query[:50]}..."
        )

        try:
            # Ensure service is properly initialized (critical for multi-worker)
            # Check not just if objects exist, but if their internal state is valid
            needs_reinit = (
                not self._initialized
                or self.rag_engine is None
                or self.ingestion_service is None
                or self._llm_func is None
                or self._embedding_func is None
            )

            # Additional check: verify rag_engine's internal state is valid
            if not needs_reinit and self.rag_engine is not None:
                try:
                    # Check if lightrag's internal functions are valid
                    lightrag = getattr(self.rag_engine, "lightrag", None)
                    if (
                        lightrag is None
                        or getattr(lightrag, "llm_model_func", None) is None
                    ):
                        self.logger.warning(
                            "RAG engine has invalid internal state, needs reinitialization"
                        )
                        needs_reinit = True
                except Exception:
                    needs_reinit = True

            if needs_reinit:
                self.logger.warning(
                    "Service components not properly initialized (likely after unpickling), "
                    "forcing complete reinitialization..."
                )
                # Clean up old instances before reinitializing
                if self.rag_engine is not None:
                    del self.rag_engine
                    self.rag_engine = None

                await self._initialize_impl()
                self._initialized = True
                self.logger.success(
                    "Service reinitialized successfully for this worker"
                )

            # Verify critical components are available after initialization
            if self.rag_engine is None:
                raise RAGException(
                    "RAG engine not initialized. Service initialization failed.",
                    operation="query",
                )
            if self.ingestion_service is None:
                raise RAGException(
                    "Ingestion service not initialized. Service initialization failed.",
                    operation="query",
                )

            # Prepare query parameters for raganything
            # RAGAnything.aquery() signature: (query: str, mode: str = 'mix', **kwargs)
            # CRITICAL: Use "naive" mode instead of "hybrid" because:
            # - Documents were ingested directly into ChromaDB (not through LightRAG pipeline)
            # - LightRAG's graph storage (entities/relationships) is empty
            # - "naive" mode uses basic vector search on chunks (traditional RAG)
            # - "hybrid" mode requires entities/relationships (which we don't have)
            mode = "naive"  # Use naive mode for direct vector search
            top_k = request.top_k or settings.default_top_k

            # Add collection name if specified
            collection_name = (
                request.collection_name
                if hasattr(request, "collection_name")
                else "default"
            )

            self.logger.info(
                f"Querying collection '{collection_name}' with top_k={top_k}"
            )

            # ✅ CRITICAL FIX: Bypass LightRAG's query since its storages are empty
            # Instead, use our working ChromaDB retrieval + direct LLM generation
            # This avoids "no-context" errors from empty LightRAG graph storage

            # Step 1: Retrieve documents from ChromaDB (hybrid search: BM25 + Semantic)
            self.logger.debug("Retrieving documents from ChromaDB")
            retrieved_docs = await self.ingestion_service.query_documents(
                query=request.query,
                collection_name=collection_name,
                database_type="rag",
                top_k=top_k,
                similarity_threshold=request.similarity_threshold,
            )

            # Convert to RetrievedDocument schema
            sources: List[RetrievedDocument] = []
            for doc in retrieved_docs:
                sources.append(
                    RetrievedDocument(
                        content=doc["content"],
                        metadata=doc["metadata"],
                        score=doc["score"],
                        doc_id=doc["doc_id"],
                    )
                )

            # Step 2: Build context from retrieved documents
            if not retrieved_docs:
                self.logger.warning("No documents retrieved from ChromaDB")
                answer = "No relevant documents found to answer your query."
                citations_text = ""
            else:
                self.logger.info(
                    f"Building context from {len(retrieved_docs)} documents"
                )

                # Build citation mapping: citation number -> actual filename
                # Use first top_k documents for context
                citation_map: Dict[int, str] = {}
                context_parts = []

                for idx, doc in enumerate(retrieved_docs[:top_k], 1):
                    # Get actual filename from metadata (preferred) or extract from source
                    filename = doc["metadata"].get("filename")
                    if not filename:
                        # Fallback: extract filename from source path
                        source = doc["metadata"].get("source", "Unknown")
                        if source != "Unknown":
                            filename = source.split("/")[-1].split("\\")[-1]
                        else:
                            filename = f"Document {idx}"

                    # Store citation mapping (citation number -> filename)
                    citation_map[idx] = filename

                    # Format context with numbered citations
                    context_parts.append(f"[{idx}] {doc['content']}")

                context = "\n\n".join(context_parts)

                # Log context stats for debugging
                unique_filenames = list(set(citation_map.values()))
                self.logger.info(
                    f"Context built: {len(context)} chars from {len(retrieved_docs)} documents. "
                    f"Sources: {', '.join(unique_filenames[:5])}..."
                )

                # Step 3: Generate answer using LLM with standard prompts
                # Select system prompt based on configuration
                mode = getattr(settings, "rag_system_prompt_mode", "standard")
                if mode == "concise":
                    system_prompt = RAG_SYSTEM_PROMPT_CONCISE
                elif mode == "strict":
                    system_prompt = RAG_SYSTEM_PROMPT_STRICT
                else:
                    # Default to standard mode
                    system_prompt = RAG_SYSTEM_PROMPT

                user_prompt = f"""Context Documents (numbered for citation):
{context}

Question: {request.query}

Instructions: Answer the question using ONLY the context above. Use numbered citations [1], [2], [3], etc. after every factual claim based on the document numbers shown in the context."""

                self.logger.debug("Generating answer with LLM")
                try:
                    # Call LLM with proper system_prompt and user prompt separation
                    raw_answer = await self._llm_func(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        max_tokens=settings.rag_max_tokens,
                    )

                    # Handle empty or None response
                    if not raw_answer or raw_answer.strip() == "":
                        self.logger.warning("LLM returned empty answer")
                        answer = "Unable to generate answer. Please try rephrasing your query."
                        citations_text = ""
                    else:
                        # Step 4: Extract citations and format response
                        answer, citations_text = self._extract_and_format_citations(
                            raw_answer, citation_map
                        )

                except Exception as llm_error:
                    self.logger.error(f"LLM generation failed: {llm_error}")
                    answer = "Error generating answer. Please try again."
                    citations_text = ""

            processing_time = time.time() - start_time

            # Append citations section to answer if citations exist
            if citations_text:
                answer = f"{answer}\n\n{citations_text}"

            self.logger.success(
                f"Query completed in {processing_time:.2f}s "
                f"with {len(sources)} sources"
            )

            return QueryResponse(
                query=request.query,
                answer=answer,
                sources=sources,
                num_sources=len(sources),
                processing_time=processing_time,
            )

        except RAGException:
            raise
        except Exception as error:
            self.handle_error(error, {"query": request.query})
            raise RAGException(f"Failed to process query: {error}", operation="query")

    async def delete_document(
        self, doc_id: str, collection_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Delete a document using custom ChromaDB service.

        Supports deletion by either:
        - Document ID (UUID): e.g., "e0132492-81c9-4938-a66e-ef78367e62b3"
        - Document filename: e.g., "rag_sample_qas_from_kis.csv"

        Args:
            doc_id: Document identifier (UUID or filename).
            collection_name: Optional collection name. If provided, searches only in this collection (faster).
                           If not provided, searches all collections.

        Returns:
            Dict: Deletion confirmation with chunk count.

        Raises:
            RAGException: If deletion fails.
        """
        try:
            # Ensure service is properly initialized (critical for multi-worker)
            if not self._initialized or self.ingestion_service is None:
                self.logger.warning(
                    "Service not initialized (likely after unpickling), reinitializing now..."
                )
                # Clean up old instances
                if self.rag_engine is not None:
                    del self.rag_engine
                    self.rag_engine = None

                await self._initialize_impl()
                self._initialized = True
                self.logger.success(
                    "Service reinitialized successfully for this worker"
                )

            # Verify ingestion service is available
            if self.ingestion_service is None:
                raise RAGException(
                    "Ingestion service not initialized. Service initialization failed.",
                    operation="delete",
                )

            # Determine if identifier is UUID or filename
            actual_doc_id: Optional[str] = None
            found_collection_name: Optional[str] = None
            chunks_deleted: int = 0

            # Check if it's a UUID (doc_id) format
            try:
                uuid.UUID(doc_id)
                is_uuid = True
            except (ValueError, AttributeError):
                is_uuid = False

            if is_uuid:
                # Try to find by doc_id in memory
                if doc_id in self.documents:
                    actual_doc_id = doc_id
                    doc_info = self.documents[doc_id]
                    found_collection_name = doc_info["collection_name"]
                else:
                    # doc_id not in memory, try to find in ChromaDB
                    self.logger.info(
                        f"Document {doc_id} not in memory, searching ChromaDB..."
                    )
                    
                    # If collection_name provided, search in that collection first (faster)
                    if collection_name:
                        try:
                            chroma_collection = (
                                self.ingestion_service._get_or_create_collection(
                                    collection_name, "rag"
                                )
                            )
                            results = chroma_collection.get(
                                where={"doc_id": doc_id}, limit=1
                            )
                            if results["ids"]:
                                actual_doc_id = doc_id
                                found_collection_name = collection_name
                                self.logger.info(
                                    f"Found document in specified collection: {doc_id} "
                                    f"(collection: {collection_name})"
                                )
                            else:
                                # Not found in specified collection, fall back to searching all collections
                                self.logger.warning(
                                    f"Document {doc_id} not found in specified collection "
                                    f"{collection_name}, searching all collections..."
                                )
                                collection_name = None  # Clear to trigger full search
                        except Exception as e:
                            self.logger.warning(
                                f"Error searching collection {collection_name}: {e}, "
                                f"falling back to full search"
                            )
                            collection_name = None  # Clear to trigger full search
                    
                    # If not found in specified collection or collection_name not provided, search all
                    if not found_collection_name:
                        # Search all collections
                        found_in_search = False
                        try:
                            rag_client = self.ingestion_service.chroma_clients.get("rag")
                            if rag_client:
                                chroma_collections = rag_client.list_collections()
                                self.logger.debug(
                                    f"Found {len(chroma_collections)} collections in ChromaDB"
                                )
                                
                                for chroma_collection in chroma_collections:
                                    coll_name = chroma_collection.name
                                    try:
                                        # Query for chunks with matching doc_id
                                        results = chroma_collection.get(
                                            where={"doc_id": doc_id}, limit=1
                                        )
                                        if results["ids"]:
                                            actual_doc_id = doc_id
                                            found_collection_name = coll_name
                                            found_in_search = True
                                            self.logger.info(
                                                f"Found document in ChromaDB: {doc_id} "
                                                f"(collection: {coll_name})"
                                            )
                                            break
                                    except Exception as e:
                                        self.logger.debug(
                                            f"Error searching collection {coll_name}: {e}"
                                        )
                                        continue
                                
                                if not found_in_search:
                                    raise RAGException(
                                        f"Document not found: {doc_id}",
                                        operation="delete",
                                    )
                            else:
                                raise RAGException(
                                    f"RAG ChromaDB client not available. Document not found: {doc_id}",
                                    operation="delete",
                                )
                        except RAGException:
                            raise
                        except Exception as e:
                            self.logger.warning(f"Error listing ChromaDB collections: {e}")
                            raise RAGException(
                                f"Document not found: {doc_id}. Error: {e}",
                                operation="delete",
                            )
            else:
                # It's a filename - search for matching document
                self.logger.debug(f"Searching for document by filename: {doc_id}")
                found = False

                # First, search in memory
                for stored_doc_id, doc_info in self.documents.items():
                    if doc_info.get("filename") == doc_id:
                        # If collection_name provided, verify it matches
                        if collection_name and doc_info.get("collection_name") != collection_name:
                            continue
                        actual_doc_id = stored_doc_id
                        found_collection_name = doc_info["collection_name"]
                        found = True
                        self.logger.debug(
                            f"Found document in memory: {doc_id} -> {actual_doc_id}"
                        )
                        break

                # If not found in memory, search in ChromaDB by filename
                if not found:
                    self.logger.info(
                        f"Document {doc_id} not in memory, searching ChromaDB by filename..."
                    )
                    
                    # If collection_name provided, search in that collection first (faster)
                    if collection_name:
                        try:
                            chroma_collection = (
                                self.ingestion_service._get_or_create_collection(
                                    collection_name, "rag"
                                )
                            )
                            # Try exact match first
                            results = chroma_collection.get(
                                where={"filename": doc_id}, limit=1
                            )
                            if not results["ids"]:
                                # Try case-insensitive search
                                all_results = chroma_collection.get()
                                if all_results["metadatas"]:
                                    doc_ids_seen = set()
                                    for meta in all_results["metadatas"]:
                                        stored_filename = meta.get("filename", "")
                                        stored_doc_id = meta.get("doc_id")
                                        if stored_doc_id in doc_ids_seen:
                                            continue
                                        doc_ids_seen.add(stored_doc_id)
                                        if stored_filename.lower() == doc_id.lower():
                                            actual_doc_id = stored_doc_id
                                            found_collection_name = collection_name
                                            found = True
                                            self.logger.info(
                                                f"Found document (case-insensitive): {doc_id} -> {actual_doc_id} "
                                                f"(collection: {collection_name}, stored as: {stored_filename})"
                                            )
                                            break
                            else:
                                metadata = results["metadatas"][0] if results["metadatas"] else {}
                                actual_doc_id = metadata.get("doc_id")
                                if actual_doc_id:
                                    found_collection_name = collection_name
                                    found = True
                                    self.logger.info(
                                        f"Found document in ChromaDB: {doc_id} -> {actual_doc_id} "
                                        f"(collection: {collection_name})"
                                    )
                            
                            if not found:
                                # Not found in specified collection, fall back to searching all collections
                                self.logger.warning(
                                    f"Document {doc_id} not found in specified collection "
                                    f"{collection_name}, searching all collections..."
                                )
                                collection_name = None  # Clear to trigger full search
                        except Exception as e:
                            self.logger.warning(
                                f"Error searching collection {collection_name}: {e}, "
                                f"falling back to full search"
                            )
                            collection_name = None  # Clear to trigger full search
                    
                    # If not found in specified collection or collection_name not provided, search all
                    if not found:
                        # Search all collections
                        try:
                            rag_client = self.ingestion_service.chroma_clients.get("rag")
                            if rag_client:
                                chroma_collections = rag_client.list_collections()
                                self.logger.debug(
                                    f"Found {len(chroma_collections)} collections in ChromaDB"
                                )
                                
                                for chroma_collection in chroma_collections:
                                    coll_name = chroma_collection.name
                                    try:
                                        # Query for chunks with matching filename (exact match)
                                        results = chroma_collection.get(
                                            where={"filename": doc_id}, limit=1
                                        )
                                        if results["ids"]:
                                            # Found it! Get the doc_id from metadata
                                            metadata = results["metadatas"][0] if results["metadatas"] else {}
                                            actual_doc_id = metadata.get("doc_id")
                                            if actual_doc_id:
                                                found_collection_name = coll_name
                                                found = True
                                                self.logger.info(
                                                    f"Found document in ChromaDB: {doc_id} -> {actual_doc_id} "
                                                    f"(collection: {coll_name})"
                                                )
                                                break
                                        
                                        # If exact match failed, try case-insensitive search
                                        if not found:
                                            self.logger.debug(
                                                f"Exact match failed for {doc_id}, "
                                                f"trying case-insensitive search in {coll_name}..."
                                            )
                                            all_results = chroma_collection.get()
                                            if all_results["metadatas"]:
                                                # Group by doc_id to find unique documents
                                                doc_ids_seen = set()
                                                for meta in all_results["metadatas"]:
                                                    stored_filename = meta.get("filename", "")
                                                    stored_doc_id = meta.get("doc_id")
                                                    
                                                    # Skip if we've already checked this doc_id
                                                    if stored_doc_id in doc_ids_seen:
                                                        continue
                                                    doc_ids_seen.add(stored_doc_id)
                                                    
                                                    # Case-insensitive comparison
                                                    if stored_filename.lower() == doc_id.lower():
                                                        actual_doc_id = stored_doc_id
                                                        if actual_doc_id:
                                                            found_collection_name = coll_name
                                                            found = True
                                                            self.logger.info(
                                                                f"Found document (case-insensitive): "
                                                                f"{doc_id} -> {actual_doc_id} "
                                                                f"(collection: {coll_name}, "
                                                                f"stored as: {stored_filename})"
                                                            )
                                                            break
                                            
                                            if found:
                                                break
                                    except Exception as e:
                                        self.logger.debug(
                                            f"Error searching collection {coll_name}: {e}"
                                        )
                                        continue
                            else:
                                self.logger.warning("RAG ChromaDB client not available")
                        except Exception as e:
                            self.logger.warning(f"Error listing ChromaDB collections: {e}")
                            # Fallback: try collections in memory
                            collections_to_search = set(self.collections.keys())
                            if hasattr(self.ingestion_service, "collections"):
                                for coll_key in self.ingestion_service.collections.keys():
                                    if coll_key.startswith("rag_"):
                                        coll_name = coll_key[4:]
                                        collections_to_search.add(coll_name)
                            
                            for coll_name in collections_to_search:
                                try:
                                    collection = (
                                        self.ingestion_service._get_or_create_collection(
                                            coll_name, "rag"
                                        )
                                    )
                                    results = collection.get(
                                        where={"filename": doc_id}, limit=1
                                    )
                                    if results["ids"]:
                                        metadata = results["metadatas"][0] if results["metadatas"] else {}
                                        actual_doc_id = metadata.get("doc_id")
                                        if actual_doc_id:
                                            found_collection_name = coll_name
                                            found = True
                                            self.logger.info(
                                                f"Found document in fallback search: {doc_id} -> {actual_doc_id}"
                                            )
                                            break
                                except Exception as e:
                                    self.logger.debug(f"Error in fallback search for {coll_name}: {e}")
                                    continue

                if not found:
                    raise RAGException(
                        f"Document not found: {doc_id}",
                        operation="delete",
                    )

            # Now delete the document using the actual doc_id
            if actual_doc_id is None or found_collection_name is None:
                raise RAGException(
                    f"Could not determine document ID or collection for: {doc_id}",
                    operation="delete",
                )

            self.logger.debug(
                f"Deleting document from ChromaDB: {actual_doc_id} "
                f"(collection: {found_collection_name})"
            )

            # Get chunk count before deletion
            try:
                collection = self.ingestion_service._get_or_create_collection(
                    found_collection_name, "rag"
                )
                results = collection.get(where={"doc_id": actual_doc_id})
                chunks_deleted = len(results["ids"]) if results["ids"] else 0
            except Exception as e:
                self.logger.warning(f"Could not get chunk count: {e}")
                chunks_deleted = 0

            # Delete document using custom ingestion service
            await self.ingestion_service.delete_document(
                doc_id=actual_doc_id,
                collection_name=found_collection_name,
                database_type="rag",
            )

            # Update collection count
            if found_collection_name in self.collections:
                self.collections[found_collection_name]["document_count"] -= 1

            # Remove document metadata from memory if it exists
            if actual_doc_id in self.documents:
                del self.documents[actual_doc_id]

            self.logger.success(
                f"Deleted document: {actual_doc_id} ({chunks_deleted} chunks)"
            )

            return {
                "message": f"Document deleted successfully: {doc_id}",
                "doc_id": actual_doc_id,
                "collection_name": found_collection_name,
                "chunks_deleted": chunks_deleted,
            }

        except RAGException:
            raise
        except Exception as e:
            self.handle_error(e, {"doc_id": doc_id})
            raise RAGException(f"Failed to delete document: {e}", operation="delete")

    def _extract_and_format_citations(
        self, raw_answer: str, citation_map: Dict[int, str]
    ) -> Tuple[str, str]:
        """
        Extract numbered citations from answer and format with actual filenames.

        This method:
        1. Extracts all citation numbers (e.g., [1], [2], [1,2], [1][2]) from the answer
        2. Removes inline citations from the answer text
        3. Creates a grouped citations section at the end with actual filenames

        Args:
            raw_answer: LLM-generated answer with inline numbered citations.
            citation_map: Mapping of citation number to actual filename.

        Returns:
            Tuple[str, str]: (cleaned_answer, citations_text)
                - cleaned_answer: Answer text with citations removed
                - citations_text: Formatted citations section to append
        """
        # Remove any LLM-provided '## Sources' or 'Sources:' block at the end
        # so we can generate a consistent, canonical sources section.
        no_sources_answer = re.sub(
            r"\n##+\s*Sources[\s\S]*$", "", raw_answer, flags=re.IGNORECASE
        )
        no_sources_answer = re.sub(
            r"\nSources:\s*[\s\S]*$", "", no_sources_answer, flags=re.IGNORECASE
        )

        # Remove inline citation markers like [1], [1,2], [1][2]
        citation_pattern = r"\[(\d+(?:\s*,\s*\d+)*)\]"
        cleaned_answer = re.sub(citation_pattern, "", no_sources_answer)

        # Clean up extra spaces/newlines
        cleaned_answer = re.sub(r"[ \t]+", " ", cleaned_answer)
        cleaned_answer = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned_answer)
        cleaned_answer = re.sub(r"\s+([.,;:!?])", r"\1", cleaned_answer)
        lines = [line.strip() for line in cleaned_answer.split("\n")]
        cleaned_answer = "\n".join(line for line in lines if line).strip()

        # Build canonical citations section using the provided citation_map.
        # This places ALL context sources at the bottom (not inline).
        if citation_map:
            sorted_items = sorted(citation_map.items())
            citations_list = [f"[{num}] {name}" for num, name in sorted_items]
            citations_text = "## Sources\n\n" + "\n".join(citations_list)
            self.logger.debug(f"Prepared {len(citations_list)} canonical sources")
        else:
            citations_text = ""

        return cleaned_answer, citations_text

    # ==================== RAFT SUPPORT METHODS ====================

    async def retrieve_documents(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[str]:
        """
        Retrieve relevant documents for RAFT using custom ChromaDB (FIXED!).

        Args:
            query: Query text.
            collection_name: Collection to search in (NOTE: Uses RAFT database!).
            top_k: Number of documents to retrieve.
            similarity_threshold: Minimum similarity score.

        Returns:
            List[str]: List of retrieved document texts.

        Raises:
            RAGException: If retrieval fails.
        """
        self.logger.debug(f"Retrieving {top_k} docs for RAFT: {query[:50]}...")

        try:
            # Use custom ChromaDB for retrieval (RAFT database, not RAG!)
            retrieved_docs = await self.ingestion_service.query_documents(
                query=query,
                collection_name=collection_name,
                database_type="raft",  # Use RAFT database!
                top_k=top_k,
            )

            # Extract document texts and filter by similarity
            documents = []
            for doc in retrieved_docs:
                score = doc.get("score", 1.0)
                if score >= similarity_threshold:
                    documents.append(doc["content"])

            self.logger.debug(f"Retrieved {len(documents)} documents for RAFT")
            return documents

        except Exception as e:
            self.handle_error(e, {"query": query, "collection": collection_name})
            # Return empty list instead of failing (graceful degradation for RAFT)
            self.logger.warning(f"Retrieval failed, returning empty: {e}")
            return []

    async def get_distractor_documents(
        self,
        query: str,
        collection_name: str,
        exclude_docs: List[str],
        count: int = 2,
    ) -> List[str]:
        """
        Get distractor (irrelevant) documents for RAFT using custom ChromaDB (FIXED!).

        Strategy: Query custom ChromaDB, get low-scoring documents (distractors).

        Args:
            query: Original query.
            collection_name: Collection to sample from (Uses RAFT database!).
            exclude_docs: Documents to exclude (already selected as relevant).
            count: Number of distractors to get.

        Returns:
            List[str]: List of distractor document texts.

        Raises:
            RAGException: If retrieval fails.
        """
        self.logger.debug(f"Getting {count} distractor docs (low-scoring)")

        try:
            # Query custom ChromaDB for many documents (get more than needed)
            retrieved_docs = await self.ingestion_service.query_documents(
                query=query,
                collection_name=collection_name,
                database_type="raft",  # Use RAFT database!
                top_k=count * 5,  # Get 5x to ensure we have options after filtering
            )

            # Sort by score (ascending = low similarity = good distractors)
            # Lower scores are less relevant, which is what we want for distractors
            sorted_docs = sorted(retrieved_docs, key=lambda x: x.get("score", 0.0))

            # Filter out excluded docs and collect distractors
            distractors = []
            for doc in sorted_docs:
                doc_text = doc["content"]
                # Exclude if document is in excluded list or already added
                if (
                    doc_text
                    and doc_text not in exclude_docs
                    and doc_text not in distractors
                ):
                    distractors.append(doc_text)
                    if len(distractors) >= count:
                        break

            self.logger.debug(f"Retrieved {len(distractors)} distractor documents")
            return distractors[:count]

        except Exception as e:
            self.handle_error(e, {"query": query, "collection": collection_name})
            # Return empty list instead of failing (graceful degradation for RAFT)
            self.logger.warning(f"Distractor retrieval failed, returning empty: {e}")
            return []

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare service for pickling (multi-worker support).

        Removes non-picklable objects before serialization:
        - RAGAnything engine (contains closures and async functions)
        - LLM/embedding wrappers and functions (contain model references)
        - Ingestion service (will be recreated on initialization)

        Returns:
            Dict[str, Any]: Picklable state dictionary.
        """
        state = super().__getstate__()

        # Remove RAGAnything engine (will be recreated on reinitialization)
        state.pop("rag_engine", None)

        # Remove model wrappers and async functions (contain closures)
        state.pop("llm_wrapper", None)
        state.pop("embedding_wrapper", None)
        state.pop("_llm_func", None)
        state.pop("_embedding_func", None)

        # Remove ingestion service (will be recreated on initialization)
        state.pop("ingestion_service", None)

        # documents and collections dicts can be pickled (contain only basic types)

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore service after unpickling.

        Sets non-picklable objects to None and marks service as uninitialized.
        The service will be properly reinitialized on first use.

        Args:
            state: State dictionary from pickle.
        """
        super().__setstate__(state)

        # Set non-picklable objects to None (will be recreated on initialization)
        self.rag_engine = None
        self.llm_wrapper = None
        self.embedding_wrapper = None
        self._llm_func = None
        self._embedding_func = None
        self.ingestion_service = None

        # Mark as uninitialized to force reinitialization
        self._initialized = False

        self.logger.warning(
            "CentralizedRAGService was unpickled and needs reinitialization. "
            "It will be automatically reinitialized on first use."
        )


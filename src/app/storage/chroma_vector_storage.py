"""
Custom ChromaDB Vector Storage adapter for LightRAG/raganything.

This adapter bridges the existing DocumentIngestionService (ChromaDB + BM25 hybrid)
with LightRAG's BaseVectorStorage interface, enabling rag.aquery() to work
with our custom retrieval pipeline.
"""

import asyncio
import hashlib
from typing import Any, Dict, List, Optional

import numpy as np
from lightrag.base import BaseVectorStorage
from loguru import logger


class ChromaVectorDBStorage(BaseVectorStorage):
    """
    Custom ChromaDB vector storage adapter for LightRAG.

    This adapter implements LightRAG's BaseVectorStorage interface while
    using the existing DocumentIngestionService for actual storage and retrieval.

    Attributes:
        namespace: Storage namespace (collection name).
        global_config: Global configuration dictionary.
        embedding_func: Function to generate embeddings.
        meta_fields: Metadata field names to preserve.
        ingestion_service: Reference to DocumentIngestionService.
        database_type: Type of database ("raft" or "rag").
    """

    def __init__(
        self,
        namespace: str = "default",
        workspace: str = "./rag_workspace",
        global_config: Optional[Dict[str, Any]] = None,
        embedding_func: Optional[Any] = None,
        cosine_better_than_threshold: float = 0.2,
        meta_fields: Optional[set] = None,
    ) -> None:
        """
        Initialize ChromaDB vector storage.

        Args:
            namespace: Collection name for storage.
            workspace: Working directory for storage.
            global_config: Configuration dictionary with ingestion_service.
            embedding_func: Embedding generation function.
            cosine_better_than_threshold: Cosine similarity threshold.
            meta_fields: Set of metadata field names to preserve.
        """
        # Provide defaults to prevent "missing positional argument" errors
        # This can happen when LightRAG tries to recreate storage
        if global_config is None:
            global_config = {}

        super().__init__(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=embedding_func,
            cosine_better_than_threshold=cosine_better_than_threshold,
            meta_fields=meta_fields,
        )

        self.logger = logger.bind(storage="ChromaVectorDBStorage")

        # Get DocumentIngestionService from global config
        self.ingestion_service = global_config.get("ingestion_service")
        if not self.ingestion_service:
            self.logger.warning(
                "ingestion_service not provided in global_config. "
                "Storage will need to be reinitialized before use."
            )

        # Database type (raft or rag)
        self.database_type = global_config.get("database_type", "rag")

        # Collection name is the namespace
        self.collection_name = namespace

        self.logger.info(
            f"Initialized ChromaVectorDBStorage for collection: "
            f"{self.collection_name} (database: {self.database_type})"
        )

    async def initialize(self) -> None:
        """
        Initialize the storage.

        The DocumentIngestionService is already initialized,
        so this is a no-op unless we need to reconnect after unpickling.
        """
        # Check if ingestion_service needs to be restored
        if not self.ingestion_service and hasattr(self, "global_config"):
            self.ingestion_service = self.global_config.get("ingestion_service")

        if not self.ingestion_service:
            self.logger.error(
                "Cannot initialize ChromaVectorDBStorage: ingestion_service is None. "
                "Storage needs to be properly reinitialized."
            )
            return

        self.logger.info(
            f"ChromaVectorDBStorage initialized for {self.collection_name}"
        )

    async def upsert(self, data: Dict[str, Dict[str, Any]]) -> None:
        """
        Insert or update vectors in ChromaDB.

        Args:
            data: Dictionary mapping IDs to documents with content and metadata.

        Format:
            {
                "id1": {
                    "content": "document text",
                    "metadata": {"key": "value", ...}
                },
                ...
            }
        """
        if not data:
            return

        # Defensive check for ingestion_service
        if not self.ingestion_service:
            self.logger.error(
                "Cannot upsert: ingestion_service is None. "
                "Storage not properly initialized."
            )
            raise RuntimeError("ChromaVectorDBStorage not properly initialized")

        self.logger.info(f"Upserting {len(data)} documents to ChromaDB")

        try:
            # Convert data to list of documents
            documents = []
            for doc_id, doc_data in data.items():
                content = doc_data.get("content", "")
                metadata = doc_data.get("metadata", {})

                # Add document ID to metadata
                metadata["__id__"] = doc_id
                metadata["__source__"] = "lightrag"

                documents.append(
                    {
                        "content": content,
                        "metadata": metadata,
                    }
                )

            # Use ingestion service to store documents
            # Note: This uses the existing chunking and embedding pipeline
            for doc in documents:
                # For raganything integration, we store pre-chunked content
                # directly without re-chunking (chunk_size = very large)
                await self.ingestion_service.ingest_document_from_text(
                    text=doc["content"],
                    collection_name=self.collection_name,
                    database_type=self.database_type,
                    metadata=doc["metadata"],
                    chunk_size=100000,  # Large size to avoid re-chunking
                    chunk_overlap=0,
                )

            self.logger.success(f"Upserted {len(documents)} documents successfully")

        except Exception as error:
            self.logger.error(f"Failed to upsert documents: {error}")
            raise

    async def query(
        self,
        query: str,
        top_k: int = 5,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query similar vectors using hybrid search.

        Args:
            query: Query string.
            top_k: Number of results to return.
            query_embedding: Pre-computed query embedding (optional).

        Returns:
            List[Dict[str, Any]]: List of similar documents with metadata.
        """
        # Defensive check for ingestion_service
        if not self.ingestion_service:
            self.logger.error(
                "Cannot query: ingestion_service is None. "
                "Storage not properly initialized."
            )
            raise RuntimeError("ChromaVectorDBStorage not properly initialized")

        self.logger.info(f"Querying ChromaDB: '{query[:50]}...' (top_k={top_k})")

        try:
            # Use the existing hybrid search (BM25 + Semantic + RRF)
            results = await self.ingestion_service.query_documents(
                query=query,
                collection_name=self.collection_name,
                database_type=self.database_type,
                top_k=top_k,
                similarity_threshold=0.0,  # Return all top_k results
            )

            # Convert to LightRAG format
            lightrag_results = []
            for result in results:
                lightrag_results.append(
                    {
                        "id": result["metadata"].get(
                            "__id__", self._generate_id(result["content"])
                        ),
                        "content": result["content"],
                        "metadata": result["metadata"],
                        "score": result["score"],
                    }
                )

            self.logger.info(f"Query returned {len(lightrag_results)} results")
            return lightrag_results

        except Exception as error:
            self.logger.error(f"Query failed: {error}")
            raise

    async def delete(self, ids: List[str]) -> None:
        """
        Delete vectors by IDs.

        Args:
            ids: List of document IDs to delete.
        """
        self.logger.info(f"Deleting {len(ids)} documents from ChromaDB")

        try:
            # Use ingestion service to delete documents
            for doc_id in ids:
                await self.ingestion_service.delete_document_by_metadata(
                    collection_name=self.collection_name,
                    database_type=self.database_type,
                    metadata_filter={"__id__": doc_id},
                )

            self.logger.success(f"Deleted {len(ids)} documents")

        except Exception as error:
            self.logger.warning(f"Delete operation failed: {error}")

    async def delete_entity(self, entity_name: str) -> None:
        """
        Delete entity-specific vectors.

        Args:
            entity_name: Name of the entity to delete.
        """
        self.logger.info(f"Deleting entity: {entity_name}")

        # Generate entity ID using same hash method as LightRAG
        entity_id = self._generate_id(entity_name)
        await self.delete([entity_id])

    async def delete_entity_relation(self, entity_name: str) -> None:
        """
        Delete relation records for an entity.

        Args:
            entity_name: Name of the entity.
        """
        self.logger.info(f"Deleting relations for entity: {entity_name}")

        try:
            # Delete documents with entity relation metadata
            await self.ingestion_service.delete_document_by_metadata(
                collection_name=self.collection_name,
                database_type=self.database_type,
                metadata_filter={"entity": entity_name},
            )
        except Exception as error:
            self.logger.warning(f"Delete relation operation failed: {error}")

    async def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single document by ID.

        Args:
            doc_id: Document ID.

        Returns:
            Optional[Dict[str, Any]]: Document data or None.
        """
        try:
            # Query by exact ID match using metadata filter
            results = await self.ingestion_service.query_documents(
                query="",  # Empty query - we're filtering by metadata
                collection_name=self.collection_name,
                database_type=self.database_type,
                top_k=1,
                metadata_filter={"__id__": doc_id},
            )

            if results:
                result = results[0]
                return {
                    "id": doc_id,
                    "content": result["content"],
                    "metadata": result["metadata"],
                }

            return None

        except Exception as error:
            self.logger.error(f"Failed to get document by ID: {error}")
            return None

    async def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple documents by IDs.

        Args:
            ids: List of document IDs.

        Returns:
            List[Dict[str, Any]]: List of documents.
        """
        documents = []
        for doc_id in ids:
            doc = await self.get_by_id(doc_id)
            if doc:
                documents.append(doc)

        return documents

    async def get_vectors_by_ids(self, ids: List[str]) -> Dict[str, List[float]]:
        """
        Get vectors for multiple IDs.

        Args:
            ids: List of document IDs.

        Returns:
            Dict[str, List[float]]: Mapping of ID to vector.
        """
        # Note: This requires embedding extraction from ChromaDB
        # For now, return empty dict as vectors are managed internally
        self.logger.warning(
            "get_vectors_by_ids not fully implemented - " "vectors managed by ChromaDB"
        )
        return {}

    async def index_done_callback(self) -> bool:
        """
        Callback after indexing is complete.

        Returns:
            bool: True if successful.
        """
        self.logger.info("Index done callback - persisting changes")

        # ChromaDB persists automatically, so this is a no-op
        return True

    async def drop(self) -> Dict[str, str]:
        """
        Drop the entire collection.

        Returns:
            Dict[str, str]: Status message.
        """
        self.logger.warning(f"Dropping collection: {self.collection_name}")

        try:
            # Delete entire collection
            await self.ingestion_service.delete_collection(
                collection_name=self.collection_name,
                database_type=self.database_type,
            )

            return {"status": "success", "message": "Collection dropped"}

        except Exception as error:
            self.logger.error(f"Failed to drop collection: {error}")
            return {"status": "error", "message": str(error)}

    @staticmethod
    def _generate_id(content: str) -> str:
        """
        Generate deterministic ID from content using hash.

        Args:
            content: Content to hash.

        Returns:
            str: Hexadecimal hash ID.
        """
        return hashlib.md5(content.encode()).hexdigest()

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare storage for pickling.

        Removes non-picklable objects (logger, ingestion_service reference)
        before serialization.

        Returns:
            Dict[str, Any]: Picklable state dictionary.
        """
        state = self.__dict__.copy()

        # Remove the logger (contains bindings and file handles)
        state.pop("logger", None)

        # Remove ingestion_service reference (will be restored via global_config)
        state.pop("ingestion_service", None)

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore storage after unpickling.

        Reconstructs the logger and ingestion_service reference.

        Args:
            state: State dictionary from pickle.
        """
        self.__dict__.update(state)

        # Reconstruct the logger
        self.logger = logger.bind(storage="ChromaVectorDBStorage")

        # Restore ingestion_service from global_config
        if hasattr(self, "global_config") and self.global_config:
            self.ingestion_service = self.global_config.get("ingestion_service")
        else:
            # Set to None if global_config is not available
            # It will be re-initialized when needed
            self.ingestion_service = None
            self.logger.warning(
                "ChromaVectorDBStorage unpickled without ingestion_service. "
                "It must be re-initialized before use."
            )

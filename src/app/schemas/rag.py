"""
Pydantic schemas for RAG operations.

Schemas for document management, querying, and retrieval.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentUpload(BaseModel):
    """
    Document upload schema.

    Attributes:
        file_path: Path to uploaded file.
        collection_name: Name of the collection.
        chunk_size: Size of text chunks.
        chunk_overlap: Overlap between chunks.
        metadata: Optional metadata.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    file_path: str = Field(
        ...,
        description="Path to uploaded file",
    )
    collection_name: str = Field(
        default="default",
        min_length=1,
        max_length=100,
        description="Collection name for documents",
    )
    chunk_size: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Document chunk size",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=512,
        description="Chunk overlap size",
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Optional document metadata",
    )


class DocumentResponse(BaseModel):
    """
    Document response schema.

    Attributes:
        doc_id: Document identifier.
        filename: Original filename.
        collection_name: Collection name.
        num_chunks: Number of chunks created.
        file_size_mb: File size in megabytes.
        created_at: Upload timestamp.
        status: Processing status.
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    doc_id: str = Field(
        ...,
        description="Document identifier",
    )
    filename: str = Field(
        ...,
        description="Original filename",
    )
    collection_name: str = Field(
        ...,
        description="Collection name",
    )
    num_chunks: int = Field(
        ...,
        description="Number of chunks created",
    )
    file_size_mb: float = Field(
        ...,
        description="File size in megabytes",
    )
    created_at: datetime = Field(
        ...,
        description="Upload timestamp",
    )
    status: str = Field(
        default="completed",
        description="Processing status",
    )


class QueryRequest(BaseModel):
    """
    RAG query request schema.

    Attributes:
        query: User query text.
        collection_name: Collection to search.
        top_k: Number of documents to retrieve.
        similarity_threshold: Minimum similarity score.
        model_name: Model to use for this query (overrides default).
        temperature: Temperature for generation (optional, defaults to 0.2).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "query": "What is LoRA fine-tuning?",
                "collection_name": "default",
                "top_k": 5,
                "similarity_threshold": 0.3,
                "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                "temperature": 0.2,
            }
        },
    )

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User query text",
    )
    collection_name: str = Field(
        default="default",
        min_length=1,
        max_length=100,
        description="Collection to search",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve",
    )
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0.3-0.5 recommended for balance)",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model to use for this query (overrides default from settings)",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Temperature for generation (0.0-0.3 factual, 0.7-1.0 creative). Defaults to 0.2 for RAG.",
    )


class RetrievedDocument(BaseModel):
    """
    Retrieved document schema.

    Attributes:
        content: Document content.
        score: Similarity score.
        metadata: Document metadata.
        doc_id: Document identifier.
    """

    content: str = Field(
        ...,
        description="Document content",
    )
    score: float = Field(
        ...,
        description="Similarity score",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Document metadata",
    )
    doc_id: str = Field(
        ...,
        description="Document identifier",
    )


class QueryResponse(BaseModel):
    """
    RAG query response schema.

    Attributes:
        query: Original query.
        answer: Generated answer.
        sources: Retrieved source documents.
        num_sources: Number of sources used.
        processing_time: Query processing time in seconds.
    """

    query: str = Field(
        ...,
        description="Original query text",
    )
    answer: str = Field(
        ...,
        description="Generated answer",
    )
    sources: List[RetrievedDocument] = Field(
        default_factory=list,
        description="Retrieved source documents",
    )
    num_sources: int = Field(
        ...,
        description="Number of sources retrieved",
    )
    processing_time: float = Field(
        ...,
        description="Processing time in seconds",
    )


class EmbeddingConfig(BaseModel):
    """
    Embedding configuration schema.

    Attributes:
        model_name: Embedding model identifier.
        batch_size: Batch size for embedding generation.
        normalize: Normalize embeddings.
    """

    model_name: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Embedding model identifier",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for embeddings",
    )
    normalize: bool = Field(
        default=True,
        description="Normalize embeddings",
    )


class CollectionInfo(BaseModel):
    """
    Collection information schema.

    Attributes:
        name: Collection name.
        num_documents: Number of documents.
        embedding_model: Embedding model used.
        created_at: Collection creation timestamp.
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    name: str = Field(
        ...,
        description="Collection name",
    )
    num_documents: int = Field(
        ...,
        description="Number of documents in collection",
    )
    embedding_model: str = Field(
        ...,
        description="Embedding model identifier",
    )
    created_at: datetime = Field(
        ...,
        description="Collection creation timestamp",
    )

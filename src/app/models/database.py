"""
SQLAlchemy database models for tracking resources.

All database models use SQLAlchemy ORM for type safety and migrations.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class FineTunedModel(Base):
    """
    Fine-tuned model tracking.

    Attributes:
        id: Primary key.
        model_id: Unique model identifier.
        base_model: Base model identifier.
        job_id: Training job identifier.
        model_path: Path to saved model.
        status: Model status.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """

    __tablename__ = "fine_tuned_models"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(255), unique=True, index=True, nullable=False)
    base_model = Column(String(255), nullable=False)
    job_id = Column(String(255), index=True, nullable=False)
    model_path = Column(String(500), nullable=False)
    status = Column(String(50), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Dataset(Base):
    """
    Dataset tracking.

    Attributes:
        id: Primary key.
        dataset_id: Unique dataset identifier.
        name: Dataset name.
        file_path: Path to dataset file.
        num_samples: Number of samples.
        created_at: Creation timestamp.
    """

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    num_samples = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingJob(Base):
    """
    Training job tracking.

    Attributes:
        id: Primary key.
        job_id: Unique job identifier.
        model_name: Base model identifier.
        dataset_id: Dataset identifier.
        status: Job status.
        config: Training configuration (JSON).
        metrics: Training metrics (JSON).
        error_message: Error message if failed.
        created_at: Creation timestamp.
        started_at: Start timestamp.
        completed_at: Completion timestamp.
    """

    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(255), unique=True, index=True, nullable=False)
    model_name = Column(String(255), nullable=False)
    dataset_id = Column(String(255), nullable=True)
    status = Column(String(50), default="pending")
    config = Column(Text, nullable=True)
    metrics = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)


class Document(Base):
    """
    Document tracking for RAG.

    Attributes:
        id: Primary key.
        doc_id: Unique document identifier.
        filename: Original filename.
        collection_name: Collection name.
        file_path: Path to document.
        num_chunks: Number of chunks.
        metadata: Document metadata (JSON).
        created_at: Upload timestamp.
    """

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(String(255), unique=True, index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    collection_name = Column(String(100), index=True, nullable=False)
    file_path = Column(String(500), nullable=False)
    num_chunks = Column(Integer, default=0)
    metadata = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class QueryLog(Base):
    """
    Query logging for RAG.

    Attributes:
        id: Primary key.
        query_id: Unique query identifier.
        query_text: Query text.
        collection_name: Collection queried.
        num_results: Number of results returned.
        processing_time: Processing time in seconds.
        created_at: Query timestamp.
    """

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(String(255), unique=True, index=True, nullable=False)
    query_text = Column(Text, nullable=False)
    collection_name = Column(String(100), nullable=False)
    num_results = Column(Integer, default=0)
    processing_time = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db(database_url: str) -> sessionmaker:
    """
    Initialize database and create tables.

    Args:
        database_url: Database connection URL.

    Returns:
        sessionmaker: SQLAlchemy session factory.
    """
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

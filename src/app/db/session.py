"""
Database session management.

This module provides database session handling and dependency injection.
"""

from typing import Generator

from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..models.database import init_db

settings = get_settings()

# Initialize database session maker
SessionLocal = init_db(settings.database_url)


def get_db() -> Generator[Session, None, None]:
    """
    Get database session.

    Yields:
        Session: SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

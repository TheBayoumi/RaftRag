"""
System prompts for RAG operations.

This module contains carefully crafted zero-shot prompts for reliable
retrieval-augmented generation with minimal hallucinations.
"""

from .rag_system_prompt import (
    RAG_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT_CONCISE,
    RAG_SYSTEM_PROMPT_STRICT,
)

__all__ = [
    "RAG_SYSTEM_PROMPT",
    "RAG_SYSTEM_PROMPT_CONCISE",
    "RAG_SYSTEM_PROMPT_STRICT",
]

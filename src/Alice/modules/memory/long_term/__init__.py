"""Long-term memory management submodule.

This submodule provides functionality for storing and retrieving long-term memories
using vector storage and semantic search capabilities.
"""

from .memory_manager import MemoryManager, get_memory_manager
from .vector_store import Memory, get_vector_store

__all__ = ["MemoryManager", "get_memory_manager", "Memory", "get_vector_store"]

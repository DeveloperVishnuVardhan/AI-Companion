"""Memory manager module for handling long-term memory operations.

This module provides functionality for analyzing, storing, and retrieving memories.
It uses a language model to determine the importance of messages and formats them for storage.
The module integrates with a vector store for efficient memory retrieval and similarity search.
"""

# fmt: off
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir =  os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(parent_dir)

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from Alice.core.prompts import MEMORY_ANALYSIS_PROMPT
from Alice.modules.memory.long_term.vector_store import get_vector_store
from Alice.settings import settings
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# fmt: on


class MemoryAnalysis(BaseModel):
    """Result of analyzing a message for memory-worthy content.

    This class represents the analysis of a message to determine if it should be stored as a memory.
    It uses Pydantic for data validation and provides structured output for memory analysis.

    Attributes:
        is_important (bool): Whether the message is important enough to be stored
        formatted_memory (Optional[str]): The formatted version of the memory if important
    """

    is_important: bool = Field(
        ...,
        description="Whether the message is important enough to be stored as a memory",
    )
    formatted_memory: Optional[str] = Field(...,
                                            description="The formatted memory to be stored")


class MemoryManager:
    """Manager class for handling long-term memory operations.

    This class manages the lifecycle of memories, from analysis to storage and retrieval.
    It uses a language model to analyze message importance and a vector store for memory storage.
    The class implements methods for memory extraction, storage, and context-aware retrieval.

    Attributes:
        vector_store: Instance of VectorStore for memory storage
        logger: Logger instance for logging operations
        llm: Language model for memory analysis
    """

    def __init__(self):
        self.vector_store = get_vector_store()
        self.logger = logging.getLogger(__name__)
        self.llm = ChatGroq(
            model=settings.SMALL_TEXT_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=0.1,
            max_retries=2,
        ).with_structured_output(MemoryAnalysis)

    async def _analyze_memory(self, message: str) -> MemoryAnalysis:
        """Analyze a message to determine importance and format if needed."""
        prompt = MEMORY_ANALYSIS_PROMPT.format(message=message)
        return await self.llm.ainvoke(prompt)

    async def extract_and_store_memories(self, message: BaseMessage) -> None:
        """Extract important information from a message and store in vector store."""
        if message.type != "human":
            return

        # Analyze the message for importance and formatting
        analysis = await self._analyze_memory(message.content)
        if analysis.is_important and analysis.formatted_memory:
            # Check if similar memory exists
            similar = self.vector_store.find_similar_memory(
                analysis.formatted_memory)
            if similar:
                # Skip storage if we already have a similar memory
                self.logger.info(
                    f"Similar memory already exists: '{analysis.formatted_memory}'")
                return

            # Store new memory
            print("Storing new memory:", analysis.formatted_memory)
            self.logger.info(
                f"Storing new memory: '{analysis.formatted_memory}'")
            self.vector_store.store_memory(
                text=analysis.formatted_memory,
                metadata={
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                },
            )

    def get_relevant_memories(self, context: str) -> List[str]:
        """Retrieve relevant memories based on the current context."""
        memories = self.vector_store.search_memories(
            context, k=settings.MEMORY_TOP_K)
        if memories:
            for memory in memories:
                self.logger.debug(
                    f"Memory: '{memory.text}' (score: {memory.score:.2f})")
        return [memory.text for memory in memories]

    def format_memories_for_prompt(self, memories: List[str]) -> str:
        """Format retrieved memories as bullet points."""
        if not memories:
            return ""
        return "\n".join(f"- {memory}" for memory in memories)


def get_memory_manager() -> MemoryManager:
    """Get a MemoryManager instance."""
    return MemoryManager()

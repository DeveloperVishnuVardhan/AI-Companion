"""
This module defines the edge functions that control the flow between nodes in the conversation graph.
These functions determine the next step in the conversation processing pipeline based on the current state.
"""

# fmt: off
import sys
import os
from langgraph.graph import END

from Alice.settings import settings
from Alice.modules.schedules.context_generation import ScheduleContextGenerator

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir =  os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(parent_dir)
from typing_extensions import Literal

from Alice.graph.state import AICompanionState
# fmt: on


def select_workflow_edge(state: AICompanionState) -> Literal["conversation_node", "image_node", "audio_node"]:
    """
    Determines the next workflow node based on the current state.

    Args:
        state (AICompanionState): The current state of the conversation

    Returns:
        Literal["conversation_node", "image_node", "audio_node"]: The next node to process
            - "conversation_node" for text-based responses
            - "image_node" for image generation
            - "audio_node" for voice responses
    """
    workflow = state["workflow"]
    if workflow == "conversation":
        return "conversation_node"
    elif workflow == "image":
        return "image_node"
    elif workflow == "audio":
        return "audio_node"


def should_summarize_conversation(
    state: AICompanionState,
) -> Literal["summarize_conversation_node", "__end__"]:
    """
    Determines whether the conversation should be summarized based on message count.

    Args:
        state (AICompanionState): The current state of the conversation

    Returns:
        Literal["summarize_conversation_node", "__end__"]: 
            - "summarize_conversation_node" if conversation needs summarization
            - "__end__" if conversation should continue without summarization
    """
    messages = state["messages"]

    if len(messages) > settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return "summarize_conversation_node"

    return END

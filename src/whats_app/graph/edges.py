# fmt: off
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(parent_dir)
from langgraph.graph import END
from typing_extensions import Literal

from whats_app.graph.state import AICompanionState
from whats_app.settings import settings

# fmt: on


def select_workflow_edge(state: AICompanionState) -> Literal["conversation", "image", "audio"]:
    workflow = state["workflow"]
    if workflow == "conversation":
        return "conversation"
    elif workflow == "image":
        return "image"
    elif workflow == "audio":
        return "audio"


def should_summarize_conversation(
    state: AICompanionState,
) -> Literal["summarize_conversation_node", "__end__"]:
    messages = state["messages"]

    if len(messages) > settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return "summarize_conversation_node"

    return END

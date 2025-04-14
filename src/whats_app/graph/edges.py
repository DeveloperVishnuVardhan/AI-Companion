# fmt: off
import sys
import os
from langgraph.graph import END

from whats_app.settings import settings
from whats_app.modules.schedules.context_generation import ScheduleContextGenerator

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir =  os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(parent_dir)
from typing_extensions import Literal

from whats_app.graph.state import AICompanionState
# fmt: on


def select_workflow_edge(state: AICompanionState) -> Literal["conversation_node", "image_node", "audio_node"]:
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
    messages = state["messages"]

    if len(messages) > settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return "summarize_conversation_node"

    return END

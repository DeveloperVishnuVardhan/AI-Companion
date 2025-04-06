# fmt: off
import os
import sys

from whats_app.graph.utils.helpers import get_chat_model

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(parent_dir)

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from whats_app import settings
from whats_app.graph.state import AICompanionState
from whats_app.graph.utils.chains import get_character_response_chain, get_router_chain
from whats_app.modules.schedules.context_generation import ScheduleContextGenerator
# fmt: on


def agent_schedule_node(state: AICompanionState):
    """
    This node is responsible for generating the current schedule of AI companion.
    """
    current_activity = ScheduleContextGenerator.get_current_activity()
    if current_activity != state.get("current_activity", ""):
        apply_activity = True
    else:
        apply_activity = False

    return {
        "current_activity": current_activity,
        "apply_activity": apply_activity,
    }


async def router_node(state: AICompanionState):
    """This node is responsible for routing the user's message to the appropriate node.

    Args:
        state (AICompanionState): The state of the AI companion.
    """
    router_chain = get_router_chain()
    response = await router_chain.ainvoke({"messages": state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE:]})
    return {
        "workflow": response.response_type
    }


async def conversation_node(state: AICompanionState, config: RunnableConfig):
    """This node is responsible for generating the response of the AI companion.

    Args:
        state (AICompanionState): The state of the AI companion.
        config (RunnableConfig): The configuration of the runnable.
    """
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")
    character_response_chain = get_character_response_chain(
        state.get("summary", ""))
    response = await character_response_chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context
        },
        config
    )
    return {"messages": AIMessage(content=response)}


async def summary_node(state: AICompanionState):
    """This node is responsible for summarizing the conversation.

    Args:
        state (AICompanionState): The state of the AI companion.
    """
    chat_model = get_chat_model()
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is summary of the conversation to date between Ava and the user: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = (
            "Create a summary of the conversation above between Ava and the user. "
            "The summary must be a short description of the conversation so far, "
            "but that captures all the relevant information shared between Ava and the user:"
        )

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await chat_model.ainvoke(messages)

    delete_messages = [RemoveMessage(
        id=m.id) for m in state["messages"][: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]]
    return {"summary": response.content, "messages": delete_messages}

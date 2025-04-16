# fmt: off
import sys
import os
from uuid import uuid4
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir =  os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(parent_dir)


from Alice.settings import settings
from Alice.modules.memory.long_term.memory_manager import get_memory_manager
from Alice.modules.schedules.context_generation import ScheduleContextGenerator
from Alice.graph.state import AICompanionState
from Alice.graph.utils.chains import get_character_response_chain, get_router_chain
from Alice.graph.utils.helpers import (
    get_chat_model,
    get_text_to_speech_model,
    get_text_to_image_model,
)


from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
# fmt: on


async def router_node(state: AICompanionState) -> AICompanionState:
    router_chain = get_router_chain()
    router_response = router_chain.invoke({"messages": state["messages"]})
    return {"workflow": router_response.response_type}


def context_injection_node(state: AICompanionState) -> AICompanionState:
    """This node is responsible for injecting the current activity into the state.

    Args:
        state (AICompanionState): The current state of the AI companion.

    Returns:
        AICompanionState: The updated state of the AI companion.
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


async def store_longterm_node(state: AICompanionState) -> AICompanionState:
    """This node is responsible for storing Long Term Memory from the state.

    Args:
        state (AICompanionState): The current state of the AI companion.

    Returns:
        AICompanionState: The updated state of the AI companion."""
    if not state["messages"]:
        return {}

    memory_manager = get_memory_manager()
    await memory_manager.extract_and_store_memories(state["messages"][-1])
    return {}


def memory_injection_node(state: AICompanionState) -> AICompanionState:
    """This node is responsible for injecting relevant memories into Charecter_prompt"""
    memory_manager = get_memory_manager()
    recent_context = " ".join([m.content for m in state["messages"][-3:]])
    memories = memory_manager.get_relevant_memories(recent_context)

    # format memories into the character prompt
    memory_context = memory_manager.format_memories_for_prompt(memories)
    return {"memory_context": memory_context}


async def conversation_node(state: AICompanionState, config: RunnableConfig) -> AICompanionState:
    """This node is responsible for generating a response to the user's message."""
    current_activity = state["current_activity"]
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))
    response = chain.invoke(
        {"messages": state["messages"], "current_activity": current_activity, "memory_context": memory_context}, config)
    return {"messages": AIMessage(content=response)}


async def audio_node(state: AICompanionState, config: RunnableConfig) -> AICompanionState:
    """This node is responsible for generating a response to the user's message."""
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))
    text_to_speech = get_text_to_speech_model()
    response = chain.invoke({
        "messages": state["messages"],
        "current_activity": current_activity,
        "memory_context": memory_context
    }, config)

    output_audio = await text_to_speech.synthesize(response)
    return {"messages": AIMessage(content=response), "audio_buffer": output_audio}


async def image_node(state: AICompanionState, config: RunnableConfig) -> AICompanionState:
    """This node is responsible for generating an image based on the user's message."""
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))
    text_to_image = get_text_to_image_model()

    # Pass the last 5 messages (or all if less than 5) as a list
    messages_to_use = state["messages"][-5:] if len(
        state["messages"]) >= 5 else state["messages"]
    scenario = await text_to_image.create_scenario(messages_to_use)

    os.makedirs("generated_images", exist_ok=True)
    img_path = f"generated_images/image_{str(uuid4())}.png"
    await text_to_image.generate_image(scenario.image_prompt, img_path)

    # Inject the image prompt information as an AI message
    scenario_message = HumanMessage(
        content=f"<image attached by Ava generated from prompt: {scenario.image_prompt}>")
    updated_messages = state["messages"] + [scenario_message]

    response = await chain.ainvoke(
        {
            "messages": updated_messages,
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )

    return {"messages": AIMessage(content=response), "image_path": img_path}


async def summarize_conversation_node(state: AICompanionState) -> AICompanionState:
    """This node is responsible for summarizing the conversation."""
    model = get_chat_model()
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
    response = model.invoke(messages)

    delete_messages = [RemoveMessage(
        id=m.id) for m in state["messages"][: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]]
    return {"summary": response.content, "messages": delete_messages}

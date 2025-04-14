# fmt: off
import sys
import os
from uuid import uuid4

from whats_app.graph.edges import select_workflow_edge, should_summarize_conversation
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir =  os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(parent_dir)

from functools import lru_cache
from langgraph.graph import StateGraph, END, START

from whats_app.graph.state import AICompanionState
from whats_app.graph.nodes import (
    router_node,
    context_injection_node,
    store_longterm_node,
    memory_injection_node,
    conversation_node,
    audio_node,
    image_node,
    summarize_conversation_node,

)
# fmt: on


@lru_cache(maxsize=1)
def create_graph():
    graph_builder = StateGraph(AICompanionState)

    graph_builder.add_node("store_longterm_node", store_longterm_node)
    graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("context_injection_node", context_injection_node)
    graph_builder.add_node("memory_injection_node", memory_injection_node)
    graph_builder.add_node("conversation_node", conversation_node)
    graph_builder.add_node("audio_node", audio_node)
    graph_builder.add_node("image_node", image_node)
    graph_builder.add_node("summarize_conversation_node",
                           summarize_conversation_node)

    # Store to longterm memory.
    graph_builder.add_edge(START, "store_longterm_node")

    # Router node.
    graph_builder.add_edge("store_longterm_node", "router_node")

    # inject both context and longterm memory.
    graph_builder.add_edge("router_node", "context_injection_node")
    graph_builder.add_edge("context_injection_node", "memory_injection_node")

    # Proceed to appropriate workflow.
    graph_builder.add_conditional_edges(
        "memory_injection_node", select_workflow_edge)

    # Add conditional edges for conversation summarization
    graph_builder.add_conditional_edges(
        "conversation_node", should_summarize_conversation)
    graph_builder.add_conditional_edges(
        "image_node", should_summarize_conversation)
    graph_builder.add_conditional_edges(
        "audio_node", should_summarize_conversation)

    # Summarize conversation.
    graph_builder.add_edge("summarize_conversation_node", END)

    return graph_builder


graph = create_graph().compile()

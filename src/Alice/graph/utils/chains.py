# fmt: off
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir =  os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(parent_dir)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from Alice.core.prompts import CHARACTER_CARD_PROMPT, ROUTER_PROMPT
from Alice.graph.utils.helpers import AsteriskRemovalParser, get_chat_model
# fmt: on


class RouterResponse(BaseModel):
    response_type: str = Field(
        description="The type of response to generate. It must be one of conversation, image or audio")


def get_router_chain():
    """
    This chain is responsible for determining the right workflow to execute based on the user's message.
    """
    model = get_chat_model(
        temperature=0.3).with_structured_output(RouterResponse)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt | model


def get_character_response_chain(summary: str = ""):
    model = get_chat_model()
    system_message = CHARACTER_CARD_PROMPT

    if summary:
        system_message += f"\n\nSummary of conversation earlier between Ava and the user: {summary}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | model | AsteriskRemovalParser()

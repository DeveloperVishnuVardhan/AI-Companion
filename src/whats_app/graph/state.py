from langgraph.graph import MessagesState


class AICompanionState(MessagesState):
    """
    State for the AI companion.

    Attributes:
        summary: A string containing a summarized context of the conversation.
        workflow:  The current workflow Ava is in. Can be “conversation”, “image” or “audio”.
        audio_buffer: The buffer containing audio data for voice messages.
        image_path: Path to the current image being generated.
        current_activity: The activity that the AI companion is currently performing.
        apply_activity: Flag indicating whether to apply or update the current activity.
        memory_context: - Long Term Memory context for the AI companion.
    """

    summary: str
    workflow: str
    audio_buffer: bytes
    image_path: str
    current_activity: str
    apply_activity: bool
    memory_context: str

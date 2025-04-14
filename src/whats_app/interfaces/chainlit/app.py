# fmt: off
from io import BytesIO
import os
import sys

import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir =  os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(parent_dir)

from whats_app.graph import graph_builder
from whats_app.modules.image import ImageToText
from whats_app.modules.speech import TextToSpeech, SpeechToText
from whats_app.settings import settings
# fmt: on

# global modules instance.
speech_to_text = SpeechToText()
text_to_speech = TextToSpeech()
image_to_text = ImageToText()


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    # thread_id = cl.user_session.get("id")
    cl.user_session.set("thread_id", 1)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle text messages and images"""
    msg = cl.Message(content="")

    # Process any attached images
    content = message.content
    if message.elements:
        for elem in message.elements:
            if isinstance(elem, cl.Image):
                # Read image file content
                with open(elem.path, "rb") as f:
                    image_bytes = f.read()

                # Analyze image and add to message content
                try:
                    # Use global ImageToText instance
                    description = await image_to_text.analyze_image(
                        image_bytes,
                        "Please describe what you see in this image in the context of our conversation.",
                    )
                    content += f"\n[Image Analysis: {description}]"
                except Exception as e:
                    cl.logger.warning(f"Failed to analyze image: {e}")

    # Process through graph with enriched message content
    thread_id = cl.user_session.get("thread_id")

    async with cl.Step(type="run"):
        async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
            graph = graph_builder.compile(checkpointer=short_term_memory)
            async for chunk in graph.astream(
                {"messages": [HumanMessage(content=content)]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            ):
                if chunk[1]["langgraph_node"] == "conversation_node" and isinstance(chunk[0], AIMessageChunk):
                    await msg.stream_token(chunk[0].content)

            output_state = await graph.aget_state(config={"configurable": {"thread_id": thread_id}})

    if output_state.values.get("workflow") == "audio":
        response = output_state.values["messages"][-1].content
        audio_buffer = output_state.values["audio_buffer"]
        output_audio_el = cl.Audio(
            name="Audio",
            auto_play=True,
            mime="audio/mpeg3",
            content=audio_buffer,
        )
        await cl.Message(content=response, elements=[output_audio_el]).send()
    elif output_state.values.get("workflow") == "image":
        response = output_state.values["messages"][-1].content
        image = cl.Image(
            path=output_state.values["image_path"], display="inline")
        await cl.Message(content=response, elements=[image]).send()
    else:
        await msg.send()


@cl.on_audio_chunk
# Use cl.types.AudioChunk if available, or adjust if needed
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    """Handle incoming audio chunks"""
    cl.logger.info(f"Received audio chunk: {chunk}")
    if chunk.isStart:
        buffer = BytesIO()
        # Attempt to get mimeType, default if unavailable
        mime_type = getattr(chunk, 'mimeType', 'audio/mpeg').split('/')[1]
        buffer.name = f"input_audio.{mime_type}"
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", getattr(
            chunk, 'mimeType', 'audio/mpeg'))
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: list):
    """Process completed audio input"""
    # Get audio data
    audio_buffer = cl.user_session.get("audio_buffer")
    if not audio_buffer:
        cl.logger.warning("Audio buffer not found in session.")
        return
    audio_buffer.seek(0)
    audio_data = audio_buffer.read()

    # Show user's audio message
    mime_type = cl.user_session.get("audio_mime_type", "audio/mpeg")
    input_audio_el = cl.Audio(mime=mime_type, content=audio_data)
    await cl.Message(author="You", content="", elements=[input_audio_el, *elements]).send()

    # Use global SpeechToText instance
    transcription = await speech_to_text.transcribe(audio_data)

    thread_id = cl.user_session.get("thread_id")

    async with AsyncSqliteSaver.from_conn_string(settings.SHORT_TERM_MEMORY_DB_PATH) as short_term_memory:
        graph = graph_builder.compile(checkpointer=short_term_memory)
        output_state = await graph.ainvoke(
            {"messages": [HumanMessage(content=transcription)]},
            {"configurable": {"thread_id": thread_id}},
        )

    # Use global TextToSpeech instance
    audio_buffer_out = await text_to_speech.synthesize(output_state["messages"][-1].content)

    output_audio_el = cl.Audio(
        name="Audio",
        auto_play=True,
        mime="audio/mpeg",  # Assuming TTS outputs mpeg, adjust if needed
        content=audio_buffer_out,
    )
    await cl.Message(content=output_state["messages"][-1].content, elements=[output_audio_el]).send()

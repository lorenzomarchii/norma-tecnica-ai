"""Anthropic API client for generating answers."""

from __future__ import annotations

import anthropic

from config import settings
from generation.prompts import SYSTEM_PROMPT, build_user_message


def generate_answer(
    query: str,
    chunks: list[dict],
    conversation_history: list[dict] | None = None,
) -> str:
    """Generate an answer using Claude Sonnet based on retrieved chunks.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        conversation_history: Optional previous messages for multi-turn chat.

    Returns:
        The generated answer text.
    """
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    user_message = build_user_message(query, chunks)

    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=4096,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    return response.content[0].text


def generate_answer_stream(
    query: str,
    chunks: list[dict],
    conversation_history: list[dict] | None = None,
):
    """Stream the answer token by token for real-time UI display.

    Yields text chunks as they arrive.
    """
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    user_message = build_user_message(query, chunks)

    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    with client.messages.stream(
        model=settings.anthropic_model,
        max_tokens=4096,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text

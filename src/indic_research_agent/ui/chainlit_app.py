"""Chainlit UI adapter."""

from __future__ import annotations

import logging
import time

import chainlit as cl

from indic_research_agent.controllers import ChatController

logger = logging.getLogger(__name__)


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("chat_controller", ChatController())


@cl.on_message
async def on_message(message: cl.Message) -> None:
    controller = cl.user_session.get("chat_controller")
    if controller is None:
        controller = ChatController()
        cl.user_session.set("chat_controller", controller)

    response = cl.Message(content="")
    await response.send()
    start = time.perf_counter()
    logger.info("ui.message.start chars=%s", len(message.content))
    try:
        answer = await controller.handle_message(message.content)
    except Exception as exc:
        logger.exception(
            "ui.message.error elapsed_seconds=%.2f",
            time.perf_counter() - start,
        )
        response.content = f"Request failed: {exc}"
    else:
        logger.info(
            "ui.message.end elapsed_seconds=%.2f tool_calls=%s",
            time.perf_counter() - start,
            answer.tool_call_count,
        )
        response.content = answer.answer
    await response.update()

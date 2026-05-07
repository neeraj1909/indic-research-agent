"""Chainlit UI adapter."""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from uuid import uuid4

import chainlit as cl

from indic_research_agent.controllers import ChatController
from indic_research_agent.services.auth_service import AuthService
from indic_research_agent.services.chainlit_data_layer import get_chainlit_data_layer
from indic_research_agent.services.chat_history import (
    history_to_json,
    normalize_history,
)
from indic_research_agent.ui.chainlit_stream_renderer import ChainlitStreamRenderer

logger = logging.getLogger(__name__)

_HISTORY_KEY = "message_history"
_APP_SESSION_KEY = "app_session_id"
_THREAD_KEY = "chainlit_thread_id"
_USER_KEY = "user_identifier"


@cl.data_layer
def data_layer():
    return get_chainlit_data_layer()


@cl.password_auth_callback
async def password_auth_callback(username: str, password: str) -> cl.User | None:
    principal = AuthService().verify_credentials(username, password)
    if principal is None:
        return None
    return cl.User(
        identifier=principal.identifier,
        display_name=principal.display_name,
        metadata=principal.metadata,
    )


@cl.on_chat_start
async def on_chat_start() -> None:
    _initialize_session_state()


@cl.on_chat_resume
async def on_chat_resume(thread: dict[str, Any]) -> None:
    _initialize_session_state(
        thread_id=str(thread.get("id") or _current_thread_id() or uuid4()),
        message_history=_history_from_thread(thread),
    )


@cl.on_message
async def on_message(message: cl.Message) -> None:
    controller = _controller_from_session()
    response = cl.Message(content="")
    await response.send()
    renderer = ChainlitStreamRenderer(response)
    start = time.perf_counter()
    logger.info("ui.message.start chars=%s", len(message.content))
    try:
        async for event in controller.stream_message(message.content):
            await renderer.render(event)
    except Exception as exc:
        logger.exception(
            "ui.message.error elapsed_seconds=%.2f",
            time.perf_counter() - start,
        )
        response.content = f"Request failed: {exc}"
        response.is_error = True
        await response.update()
    else:
        _save_session_state(controller)
        logger.info(
            "ui.message.end elapsed_seconds=%.2f history_turns=%s",
            time.perf_counter() - start,
            len(controller.message_history),
        )


def _initialize_session_state(
    *,
    thread_id: str | None = None,
    message_history: list[dict[str, Any]] | None = None,
) -> None:
    thread_id = thread_id or _current_thread_id()
    user_identifier = _current_user_identifier()
    history = normalize_history(message_history)
    app_session_id = f"chainlit:{thread_id}" if thread_id else f"chat-{uuid4().hex}"
    cl.user_session.set(_HISTORY_KEY, history_to_json(history))
    cl.user_session.set(_APP_SESSION_KEY, app_session_id)
    cl.user_session.set(_THREAD_KEY, thread_id)
    cl.user_session.set(_USER_KEY, user_identifier)


def _controller_from_session() -> ChatController:
    thread_id = _current_thread_id() or cl.user_session.get(_THREAD_KEY)
    user_identifier = _current_user_identifier() or cl.user_session.get(_USER_KEY)
    controller = ChatController(
        message_history=cl.user_session.get(_HISTORY_KEY) or [],
        app_session_id=cl.user_session.get(_APP_SESSION_KEY),
        chainlit_thread_id=thread_id,
        user_identifier=user_identifier,
        persist_queries=True,
    )
    return controller


def _save_session_state(controller: ChatController) -> None:
    cl.user_session.set(_HISTORY_KEY, history_to_json(controller.message_history))
    cl.user_session.set(_APP_SESSION_KEY, controller.app_session_id)
    cl.user_session.set(_THREAD_KEY, controller.chainlit_thread_id)
    cl.user_session.set(_USER_KEY, controller.user_identifier)


def _history_from_thread(thread: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = thread.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if isinstance(metadata, dict):
        metadata_history = metadata.get(_HISTORY_KEY)
        normalized = normalize_history(
            metadata_history if isinstance(metadata_history, list) else []
        )
        if normalized:
            return history_to_json(normalized)
    return _history_from_thread_steps(thread.get("steps") or [])


def _history_from_thread_steps(steps: list[dict[str, Any]]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for step in steps:
        step_type = step.get("type")
        role = "user" if step_type == "user_message" else None
        if step_type == "assistant_message":
            role = "assistant"
        if role is None:
            continue
        content = step.get("output") or step.get("input") or ""
        if isinstance(content, str) and content.strip():
            turns.append({"role": role, "content": content.strip()})
    return history_to_json(normalize_history(turns))


def _current_user_identifier() -> str | None:
    user = cl.user_session.get("user")
    return str(getattr(user, "identifier", "") or "") or None


def _current_thread_id() -> str | None:
    try:
        return str(cl.context.session.thread_id)
    except Exception:
        return None

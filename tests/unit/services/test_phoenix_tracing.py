from __future__ import annotations

import pytest

from indic_research_agent.config import AppSettings
from indic_research_agent.services import phoenix_tracing

pytestmark = pytest.mark.unit


def test_configure_phoenix_is_noop_when_disabled() -> None:
    assert (
        phoenix_tracing.configure_phoenix(AppSettings(phoenix_enabled=False)) is False
    )


def test_http_endpoint_adds_otlp_trace_path() -> None:
    assert (
        phoenix_tracing._trace_endpoint(  # noqa: SLF001 - intentional helper coverage
            "http://10.20.30.1:16006",
            protocol="http/protobuf",
        )
        == "http://10.20.30.1:16006/v1/traces"
    )


def test_session_attributes_use_openinference_keys() -> None:
    attrs = phoenix_tracing.session_attributes(
        session_id="thread-1",
        user_identifier="test",
        metadata={"persist_queries": True},
        tags=["chainlit", "agent"],
    )

    assert attrs["session.id"] == "thread-1"
    assert attrs["user.id"] == "test"
    assert "persist_queries" in attrs["metadata"]
    assert "chainlit" in attrs["tag.tags"]

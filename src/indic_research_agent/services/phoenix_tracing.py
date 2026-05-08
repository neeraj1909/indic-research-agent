"""Phoenix/OpenTelemetry tracing helpers for agent observability."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode

from indic_research_agent.config import AppSettings, get_settings

logger = logging.getLogger(__name__)

try:  # pragma: no cover - constants are from a small optional semantic package.
    from openinference.semconv.resource import ResourceAttributes
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
except Exception:  # pragma: no cover
    ResourceAttributes = None  # type: ignore[assignment]
    OpenInferenceSpanKindValues = None  # type: ignore[assignment]
    SpanAttributes = None  # type: ignore[assignment]

_CONFIGURED = False


def configure_phoenix(settings: AppSettings | None = None) -> bool:
    """Configure OTLP tracing to the Phoenix collector once per process."""

    global _CONFIGURED
    settings = settings or get_settings()
    if not settings.phoenix_enabled:
        return False
    if _running_under_pytest():
        return False
    if _CONFIGURED:
        return True

    endpoint = _trace_endpoint(
        settings.phoenix_collector_endpoint,
        protocol=settings.phoenix_protocol,
    )
    try:
        resource_attrs: dict[str, str] = {
            "service.name": settings.app_name,
            "service.environment": settings.environment,
        }
        if ResourceAttributes is not None:
            resource_attrs[ResourceAttributes.PROJECT_NAME] = (
                settings.phoenix_project_name
            )
        else:
            resource_attrs["openinference.project.name"] = settings.phoenix_project_name

        provider = TracerProvider(resource=Resource.create(resource_attrs))
        exporter = OTLPSpanExporter(endpoint=endpoint)
        processor = (
            BatchSpanProcessor(exporter)
            if settings.phoenix_batch_spans
            else SimpleSpanProcessor(exporter)
        )
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        _instrument_openinference(provider, enabled=settings.phoenix_auto_instrument)
        _CONFIGURED = True
        logger.info(
            "phoenix.tracing.enabled endpoint=%s project=%s protocol=%s",
            endpoint,
            settings.phoenix_project_name,
            settings.phoenix_protocol,
        )
        return True
    except Exception:
        logger.exception("phoenix.tracing.configure_failed endpoint=%s", endpoint)
        return False


@contextmanager
def trace_span(
    name: str,
    *,
    kind: str = "CHAIN",
    input_value: Any = None,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Span]:
    """Start an OpenInference-compatible span if tracing is configured."""

    tracer = trace.get_tracer("indic_research_agent")
    with tracer.start_as_current_span(name) as span:
        if span.is_recording():
            set_span_kind(span, kind)
            if input_value is not None:
                set_span_input(span, input_value)
            set_span_attributes(span, attributes or {})
        try:
            yield span
        except Exception as exc:
            record_span_exception(span, exc)
            raise
        else:
            if span.is_recording():
                span.set_status(Status(StatusCode.OK))


def set_span_kind(span: Span, kind: str) -> None:
    if not span.is_recording():
        return
    normalized = kind.upper()
    if OpenInferenceSpanKindValues is not None:
        values = {item.value for item in OpenInferenceSpanKindValues}
        if normalized not in values:
            normalized = OpenInferenceSpanKindValues.CHAIN.value
    span.set_attribute(_span_attr("OPENINFERENCE_SPAN_KIND"), normalized)


def set_span_input(span: Span, value: Any) -> None:
    if not span.is_recording():
        return
    serialized, mime_type = _serialize_io(value)
    span.set_attribute(_span_attr("INPUT_VALUE"), serialized)
    span.set_attribute(_span_attr("INPUT_MIME_TYPE"), mime_type)


def set_span_output(span: Span, value: Any) -> None:
    if not span.is_recording():
        return
    serialized, mime_type = _serialize_io(value)
    span.set_attribute(_span_attr("OUTPUT_VALUE"), serialized)
    span.set_attribute(_span_attr("OUTPUT_MIME_TYPE"), mime_type)


def set_span_attributes(span: Span, attributes: Mapping[str, Any]) -> None:
    if not span.is_recording():
        return
    for key, value in attributes.items():
        if value is None:
            continue
        span.set_attribute(str(key), _attribute_value(value))


def add_span_event(
    name: str,
    attributes: Mapping[str, Any] | None = None,
) -> None:
    span = trace.get_current_span()
    if not span.is_recording():
        return
    span.add_event(
        name, {k: _attribute_value(v) for k, v in (attributes or {}).items()}
    )


def record_span_exception(span: Span, exc: BaseException) -> None:
    if not span.is_recording():
        return
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


def force_flush_traces(timeout_millis: int = 5000) -> None:
    provider = trace.get_tracer_provider()
    force_flush = getattr(provider, "force_flush", None)
    if not callable(force_flush):
        return
    try:
        force_flush(timeout_millis=timeout_millis)
    except Exception:
        logger.debug("phoenix.tracing.force_flush_failed", exc_info=True)


def session_attributes(
    *,
    session_id: str | None = None,
    user_identifier: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if session_id:
        attrs[_span_attr("SESSION_ID")] = session_id
    if user_identifier:
        attrs[_span_attr("USER_ID")] = user_identifier
    if metadata:
        attrs[_span_attr("METADATA")] = _json_dumps(dict(metadata))
    if tags:
        attrs[_span_attr("TAG_TAGS")] = _json_dumps(list(tags))
    return attrs


def llm_attributes(
    *,
    model_name: str | None = None,
    invocation_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if model_name:
        attrs[_span_attr("LLM_MODEL_NAME")] = model_name
    if invocation_parameters:
        attrs[_span_attr("LLM_INVOCATION_PARAMETERS")] = _json_dumps(
            dict(invocation_parameters)
        )
    return attrs


def tool_attributes(
    *,
    name: str,
    parameters: Mapping[str, Any] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {_span_attr("TOOL_NAME"): name}
    if parameters is not None:
        attrs[_span_attr("TOOL_PARAMETERS")] = _json_dumps(dict(parameters))
    if description:
        attrs[_span_attr("TOOL_DESCRIPTION")] = description
    return attrs


def _instrument_openinference(provider: TracerProvider, *, enabled: bool) -> None:
    if not enabled:
        return
    for import_path, class_name in (
        ("openinference.instrumentation.langchain", "LangChainInstrumentor"),
        ("openinference.instrumentation.litellm", "LiteLLMInstrumentor"),
    ):
        try:
            module = __import__(import_path, fromlist=[class_name])
            instrumentor = getattr(module, class_name)()
            instrumentor.instrument(tracer_provider=provider)
            logger.info("phoenix.tracing.instrumented %s.%s", import_path, class_name)
        except Exception:
            logger.debug(
                "phoenix.tracing.instrumentation_failed import_path=%s class=%s",
                import_path,
                class_name,
                exc_info=True,
            )


def _trace_endpoint(endpoint: str, *, protocol: str) -> str:
    cleaned = endpoint.rstrip("/")
    if protocol == "http/protobuf" and not cleaned.endswith("/v1/traces"):
        return f"{cleaned}/v1/traces"
    return cleaned


def _running_under_pytest() -> bool:
    return "pytest" in sys.modules


def _serialize_io(value: Any) -> tuple[str, str]:
    if isinstance(value, str):
        return value, "text/plain"
    return _json_dumps(value), "application/json"


def _attribute_value(value: Any) -> str | bool | int | float | Sequence[str]:
    if isinstance(value, str | bool | int | float):
        return value
    if (
        isinstance(value, Sequence)
        and not isinstance(value, bytes | bytearray)
        and all(isinstance(item, str) for item in value)
    ):
        return list(value)
    return _json_dumps(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _span_attr(name: str) -> str:
    if SpanAttributes is None:
        fallback = {
            "OPENINFERENCE_SPAN_KIND": "openinference.span.kind",
            "INPUT_VALUE": "input.value",
            "INPUT_MIME_TYPE": "input.mime_type",
            "OUTPUT_VALUE": "output.value",
            "OUTPUT_MIME_TYPE": "output.mime_type",
            "SESSION_ID": "session.id",
            "USER_ID": "user.id",
            "METADATA": "metadata",
            "TAG_TAGS": "tag.tags",
            "LLM_MODEL_NAME": "llm.model_name",
            "LLM_INVOCATION_PARAMETERS": "llm.invocation_parameters",
            "TOOL_NAME": "tool.name",
            "TOOL_PARAMETERS": "tool.parameters",
            "TOOL_DESCRIPTION": "tool.description",
        }
        return fallback[name]
    return str(getattr(SpanAttributes, name))

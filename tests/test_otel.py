from typing import Any, Generator, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import SERVICE_NAME
from opentelemetry.sdk.trace.export import BatchSpanProcessor as _BatchSpanProcessor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor as _SimpleSpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter

from middleware.llmobs.otel import (
    PROJECT_NAME,
    BatchSpanProcessor,
    HTTPSpanExporter,
    OTLPTransportProtocol,
    SimpleSpanProcessor,
    TracerProvider,
    _normalized_endpoint,
    register,
)

MW_ENDPOINT = "https://uid.middleware.io:443"


def _get_exporter_from_processor(span_processor: Any) -> Optional[SpanExporter]:
    """Get the exporter from a span processor, handling old and new OpenTelemetry versions."""
    return getattr(getattr(span_processor, "_batch_processor", None), "_exporter", None) or getattr(
        span_processor, "span_exporter", None
    )


@pytest.fixture(autouse=True)
def reset_env() -> Generator[None, None, None]:
    """Clear environment for test isolation."""
    with patch.dict("os.environ", {}, clear=True):
        yield


class TestRegister:
    def test_register_basic_uses_http(self) -> None:
        providers = register(verbose=False, set_global_tracer_provider=False)

        assert isinstance(providers.tracer, TracerProvider)
        assert providers.tracer._default_processor
        # Eval logger/meter are always constructed (independent of the tracer-global flag).
        assert providers.logger is not None and providers.meter is not None

        processors = providers.tracer._active_span_processor._span_processors
        assert len(processors) == 1
        # Default batch=True
        assert isinstance(processors[0], _BatchSpanProcessor)

        exporter = _get_exporter_from_processor(processors[0])
        assert isinstance(exporter, HTTPSpanExporter)

    def test_register_simple_processor(self) -> None:
        providers = register(batch=False, verbose=False, set_global_tracer_provider=False)

        processors = providers.tracer._active_span_processor._span_processors
        assert isinstance(processors[0], _SimpleSpanProcessor)
        exporter = _get_exporter_from_processor(processors[0])
        assert isinstance(exporter, HTTPSpanExporter)

    def test_register_sets_service_and_project_name(self) -> None:
        providers = register(
            service_name="my-app",
            project_name="my-proj",
            verbose=False,
            set_global_tracer_provider=False,
        )

        assert providers.tracer.resource.attributes.get(SERVICE_NAME) == "my-app"
        assert providers.tracer.resource.attributes.get(PROJECT_NAME) == "my-proj"

    def test_register_project_name_defaults_to_service_name(self) -> None:
        providers = register(service_name="my-app", verbose=False, set_global_tracer_provider=False)

        assert providers.tracer.resource.attributes.get(SERVICE_NAME) == "my-app"
        assert providers.tracer.resource.attributes.get(PROJECT_NAME) == "my-app"

    def test_register_with_middleware_endpoint(self) -> None:
        providers = register(endpoint=MW_ENDPOINT, verbose=False, set_global_tracer_provider=False)

        processors = providers.tracer._active_span_processor._span_processors
        exporter = _get_exporter_from_processor(processors[0])
        assert isinstance(exporter, HTTPSpanExporter)
        assert exporter._endpoint == "https://uid.middleware.io:443/v1/traces"

    def test_register_with_authorization_header(self) -> None:
        providers = register(
            endpoint=MW_ENDPOINT,
            headers={"Authorization": "my-key"},
            verbose=False,
            set_global_tracer_provider=False,
        )

        processors = providers.tracer._active_span_processor._span_processors
        exporter = _get_exporter_from_processor(processors[0])
        headers_dict = {h.lower(): v for h, v in exporter._headers.items()}
        # Passed through verbatim — no Bearer rewriting.
        assert headers_dict.get("authorization") == "my-key"

    def test_register_grpc_protocol_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            register(protocol="grpc", verbose=False, set_global_tracer_provider=False)

    def test_register_http_protocol_ok(self) -> None:
        providers = register(
            protocol="http/protobuf", verbose=False, set_global_tracer_provider=False
        )
        processors = providers.tracer._active_span_processor._span_processors
        exporter = _get_exporter_from_processor(processors[0])
        assert isinstance(exporter, HTTPSpanExporter)

    def test_register_without_global_tracer(self) -> None:
        providers = register(set_global_tracer_provider=False, verbose=False)
        assert trace_api.get_tracer_provider() != providers.tracer

    @patch("middleware.llmobs.otel._auto_instrument_installed_openinference_libraries")
    def test_register_auto_instrument(self, mock_auto: Any) -> None:
        providers = register(auto_instrument=True, verbose=False, set_global_tracer_provider=False)
        mock_auto.assert_called_once_with(providers.tracer)

    def test_register_reads_endpoint_from_env(self) -> None:
        with patch.dict("os.environ", {"OTEL_EXPORTER_OTLP_ENDPOINT": MW_ENDPOINT}, clear=True):
            providers = register(verbose=False, set_global_tracer_provider=False)
        processors = providers.tracer._active_span_processor._span_processors
        exporter = _get_exporter_from_processor(processors[0])
        assert exporter._endpoint == "https://uid.middleware.io:443/v1/traces"

    def test_register_passes_through_kwargs(self) -> None:
        from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

        providers = register(sampler=ALWAYS_OFF, verbose=False, set_global_tracer_provider=False)
        assert providers.tracer.sampler == ALWAYS_OFF

    def test_register_global_flags_default_to_true(self) -> None:
        # The set_global_*_provider flags accept None to mean "default", which is True.
        providers = register(
            verbose=False,
            set_global_tracer_provider=False,
            set_global_logger_provider=None,
            set_global_meter_provider=None,
        )
        assert providers.logger is not None and providers.meter is not None

    def test_register_can_disable_each_global_independently(self) -> None:
        providers = register(
            verbose=False,
            set_global_tracer_provider=False,
            set_global_logger_provider=False,
            set_global_meter_provider=False,
        )
        # Providers are still constructed for force_flush(), just not installed globally.
        assert providers.logger is not None and providers.meter is not None


class TestTracerProvider:
    def test_creation_defaults_to_http(self) -> None:
        tracer_provider = TracerProvider(verbose=False)
        processors = tracer_provider._active_span_processor._span_processors
        assert len(processors) == 1
        exporter = _get_exporter_from_processor(processors[0])
        assert isinstance(exporter, HTTPSpanExporter)

    def test_grpc_protocol_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            TracerProvider(protocol="grpc", verbose=False)

    def test_add_span_processor_replaces_default(self) -> None:
        tracer_provider = TracerProvider(verbose=False)
        assert tracer_provider._default_processor

        custom_processor = Mock(spec=_SimpleSpanProcessor)
        tracer_provider.add_span_processor(custom_processor)

        assert not tracer_provider._default_processor
        assert custom_processor in tracer_provider._active_span_processor._span_processors


class TestSpanProcessors:
    def test_simple_span_processor_http(self) -> None:
        processor = SimpleSpanProcessor(endpoint=MW_ENDPOINT)
        assert isinstance(processor, _SimpleSpanProcessor)
        assert isinstance(processor.span_exporter, HTTPSpanExporter)

    def test_simple_span_processor_grpc_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            SimpleSpanProcessor(endpoint=MW_ENDPOINT, protocol="grpc")

    def test_simple_span_processor_with_exporter(self) -> None:
        mock_exporter = MagicMock()
        processor = SimpleSpanProcessor(span_exporter=mock_exporter)
        assert processor.span_exporter == mock_exporter

    def test_batch_span_processor_http(self) -> None:
        processor = BatchSpanProcessor(endpoint=MW_ENDPOINT)
        assert isinstance(processor, _BatchSpanProcessor)
        exporter = _get_exporter_from_processor(processor)
        assert isinstance(exporter, HTTPSpanExporter)

    def test_batch_span_processor_grpc_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            BatchSpanProcessor(endpoint=MW_ENDPOINT, protocol="grpc")


class TestSpanExporters:
    def test_http_exporter_env_headers(self) -> None:
        with patch.dict(
            "os.environ",
            {"OTEL_EXPORTER_OTLP_HEADERS": "Authorization=key,X-Custom=value"},
            clear=True,
        ):
            exporter = HTTPSpanExporter(endpoint=MW_ENDPOINT)
        headers_dict = {h.lower(): v for h, v in exporter._headers.items()}
        assert headers_dict.get("authorization") == "key"
        assert headers_dict.get("x-custom") == "value"

    def test_http_exporter_explicit_headers(self) -> None:
        exporter = HTTPSpanExporter(endpoint=MW_ENDPOINT, headers={"Authorization": "key"})
        headers_dict = {h.lower(): v for h, v in exporter._headers.items()}
        assert headers_dict.get("authorization") == "key"


class TestEndpointNormalization:
    def test_middleware_endpoint_gets_traces_path(self) -> None:
        parsed, endpoint = _normalized_endpoint(MW_ENDPOINT, use_http=True)
        assert parsed.scheme == "https"
        assert parsed.netloc == "uid.middleware.io:443"
        assert parsed.path == "/v1/traces"
        assert endpoint == "https://uid.middleware.io:443/v1/traces"

    def test_endpoint_already_has_traces_path(self) -> None:
        url = "https://uid.middleware.io:443/v1/traces"
        _, endpoint = _normalized_endpoint(url, use_http=True)
        assert endpoint == url

    def test_endpoint_from_env_default(self) -> None:
        with patch.dict("os.environ", {"OTEL_EXPORTER_OTLP_ENDPOINT": MW_ENDPOINT}, clear=True):
            _, endpoint = _normalized_endpoint(None, use_http=True)
        assert endpoint == "https://uid.middleware.io:443/v1/traces"


class TestOTLPTransportProtocol:
    def test_valid_protocols(self) -> None:
        assert OTLPTransportProtocol("http/protobuf") == OTLPTransportProtocol.HTTP_PROTOBUF
        assert OTLPTransportProtocol("grpc") == OTLPTransportProtocol.GRPC
        assert OTLPTransportProtocol(None) == OTLPTransportProtocol.INFER

    def test_invalid_protocol_http(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            OTLPTransportProtocol("http")
        assert "Did you mean 'http/protobuf'?" in str(exc_info.value)

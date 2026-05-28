from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import entry_points
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from urllib.parse import ParseResult, urlparse

from openinference.instrumentation import TracerProvider as _TracerProvider
from openinference.semconv.resource import ResourceAttributes as _ResourceAttributes
from opentelemetry import trace as trace_api
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as _GRPCSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter as _HTTPLogExporter,
)
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter as _HTTPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as _HTTPSpanExporter,
)
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor as _BatchSpanProcessor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor as _SimpleSpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter

from .settings import (
    get_env_client_headers,
    get_env_collector_endpoint,
    get_env_grpc_port,
    get_env_project_name,
    get_env_service_name,
)

PROJECT_NAME = _ResourceAttributes.PROJECT_NAME

_GRPC_UNSUPPORTED_MSG = (
    "gRPC transport is not supported by middleware-llmobs. "
    "Use the HTTP/protobuf transport (protocol='http/protobuf') and an HTTP endpoint, "
    "e.g. OTEL_EXPORTER_OTLP_ENDPOINT='https://uid.middleware.io:443'."
)


class OTLPTransportProtocol(str, Enum):
    HTTP_PROTOBUF = "http/protobuf"
    GRPC = "grpc"
    INFER = "infer"

    @classmethod
    def _missing_(cls, value: object) -> "OTLPTransportProtocol":
        if not isinstance(value, (str, type(None))):
            raise ValueError(f"Invalid protocol: {value}. Must be a string.")
        if value is None:
            return cls.INFER
        elif "http" in value:
            raise ValueError(
                (
                    f"Invalid protocol: {value}. Must be one of {cls._valid_protocols_str()}. "
                    "Did you mean 'http/protobuf'?"
                )
            )
        else:
            raise ValueError(
                (f"Invalid protocol: {value}. Must one of {cls._valid_protocols_str()}.")
            )

    @classmethod
    def _valid_protocols_str(cls) -> str:
        return "[" + ", ".join([f"'{protocol.value}'" for protocol in cls]) + "]"


@dataclass(frozen=True)
class Providers:
    """The OTel providers ``register()`` wires up.

    ``tracer`` is the OpenInference ``TracerProvider`` (set as the global tracer provider unless
    ``set_global_tracer_provider=False``). ``logger`` and ``meter`` are the providers used for
    eval logs (``/v1/logs``) and gauge metrics (``/v1/metrics``); both are also set globally so
    ``submit_evaluation`` and ``flush_evaluations`` find them via the OTel API.

    They are returned together so callers can ``force_flush`` each at shutdown if needed.
    """

    tracer: TracerProvider
    logger: Optional[LoggerProvider]
    meter: Optional[MeterProvider]


def register(
    *,
    endpoint: Optional[str] = None,
    service_name: Optional[str] = None,
    project_name: Optional[str] = None,
    batch: bool = True,
    set_global_tracer_provider: bool = True,
    set_global_logger_provider: Optional[bool] = True,
    set_global_meter_provider: Optional[bool] = True,
    headers: Optional[Dict[str, str]] = None,
    protocol: Optional[Literal["http/protobuf", "grpc"]] = None,
    verbose: bool = True,
    auto_instrument: bool = False,
    **kwargs: Any,
) -> Providers:
    """
    Creates an OpenTelemetry TracerProvider for exporting OpenInference traces to Middleware.

    Only the HTTP/protobuf transport is supported. Requesting gRPC (either explicitly via
    ``protocol="grpc"`` or by supplying a gRPC-style endpoint) raises ``NotImplementedError``.

    Args:
        endpoint (str, optional): The collector endpoint to which spans will be exported. If not
            provided, the `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable will be used, e.g.
            ``https://uid.middleware.io:443``. The ``/v1/traces`` path is appended automatically.
        service_name (str, optional): The service name reported in traces (`service.name`). If not
            provided, the `OTEL_SERVICE_NAME` environment variable is used.
        project_name (str, optional): The OpenInference project name. If not provided, the
            `MW_PROJECT_NAME` environment variable is used, falling back to the service name.
        batch (bool): If True (default), spans are processed with a BatchSpanProcessor. If False,
            spans are processed one at a time with a SimpleSpanProcessor.
        set_global_tracer_provider (bool): If False, the TracerProvider will not be set as the
            global tracer provider. Defaults to True.
        set_global_logger_provider (bool, optional): If False, the eval LoggerProvider will not be
            set as the global logger provider. Defaults to True. Independent of
            ``set_global_tracer_provider``.
        set_global_meter_provider (bool, optional): If False, the eval MeterProvider will not be
            set as the global meter provider. Defaults to True. Independent of
            ``set_global_tracer_provider``.
        headers (dict, optional): Optional headers to include in the export request. If not
            provided, the `OTEL_EXPORTER_OTLP_HEADERS` environment variable will be used. This is
            where the Middleware ``Authorization`` key is supplied.
        protocol (str, optional): The transport protocol. Only "http/protobuf" is supported.
        verbose (bool): If True, configuration details will be printed to stdout.
        auto_instrument (bool): If True, automatically instruments all installed OpenInference
            libraries with GenAI semantic conventions enabled.
        **kwargs: Additional keyword arguments passed to the TracerProvider constructor.

    Returns a :class:`Providers` triple — ``providers.tracer`` is the OpenInference
    ``TracerProvider``; ``providers.logger`` / ``providers.meter`` are the OTel providers used for
    eval logs/metrics. The logger/meter are always constructed (so callers can ``force_flush``
    them) but are only installed as OTel globals when their respective ``set_global_*`` flag is
    true.

    Examples:
        Basic setup with automatic instrumentation::

            from middleware.llmobs import register
            providers = register(auto_instrument=True)

        Explicit configuration::

            providers = register(
                endpoint="https://uid.middleware.io:443",
                headers={"Authorization": "<your-key>"},
                service_name="my-llm-app",
            )

        Using environment variables::

            # export OTEL_EXPORTER_OTLP_ENDPOINT="https://uid.middleware.io:443"
            # export OTEL_EXPORTER_OTLP_HEADERS="Authorization=<key>"
            # export OTEL_SERVICE_NAME="my-llm-app"
            providers = register()
    """
    # gRPC is not supported; fail fast before constructing anything.
    if OTLPTransportProtocol(protocol) == OTLPTransportProtocol.GRPC:
        raise NotImplementedError(_GRPC_UNSUPPORTED_MSG)

    tracer_provider_kwargs = kwargs.copy()
    service_name = service_name or get_env_service_name()
    # Project name falls back to the explicit env value, then to the resolved service name so an
    # explicitly-passed service_name also identifies the project.
    project_name = project_name or get_env_project_name() or service_name
    identity_attributes = {SERVICE_NAME: service_name, PROJECT_NAME: project_name}

    if "resource" not in tracer_provider_kwargs:
        # No resource provided, create one with identity attributes.
        tracer_provider_kwargs["resource"] = Resource.create(identity_attributes)
    else:
        # Resource provided, merge identity attributes into it without overriding user attributes.
        existing_resource = tracer_provider_kwargs["resource"]
        identity_resource = Resource(attributes=identity_attributes)
        tracer_provider_kwargs["resource"] = existing_resource.merge(identity_resource)

    # Ensure TracerProvider verbose is False (register handles its own verbose output)
    tracer_provider_kwargs["verbose"] = False

    tracer_provider = TracerProvider(protocol=protocol, **tracer_provider_kwargs)
    span_processor: SpanProcessor
    if batch:
        span_processor = BatchSpanProcessor(endpoint=endpoint, headers=headers, protocol=protocol)
    else:
        span_processor = SimpleSpanProcessor(endpoint=endpoint, headers=headers, protocol=protocol)
    tracer_provider.add_span_processor(span_processor)
    tracer_provider._default_processor = True

    if set_global_tracer_provider:
        trace_api.set_tracer_provider(tracer_provider)
        global_provider_msg = (
            "|  \n"
            "|  `register` has set this TracerProvider as the global OpenTelemetry default.\n"
            "|  To disable this behavior, call `register` with "
            "`set_global_tracer_provider=False`.\n"
        )
    else:
        global_provider_msg = ""

    # Evaluations are emitted as OTel logs + metrics. Each provider's global registration is
    # controlled by its own flag, independent of the tracer's. ``None`` is treated as the default.
    logger_provider, meter_provider = _register_eval_providers(
        resource=tracer_provider_kwargs["resource"],
        endpoint=endpoint,
        headers=headers,
        set_global_logger_provider=(
            True if set_global_logger_provider is None else set_global_logger_provider
        ),
        set_global_meter_provider=(
            True if set_global_meter_provider is None else set_global_meter_provider
        ),
    )

    if auto_instrument:
        _auto_instrument_installed_openinference_libraries(tracer_provider)

    details = tracer_provider._tracing_details()
    if verbose:
        print(f"{details}{global_provider_msg}")
    return Providers(tracer=tracer_provider, logger=logger_provider, meter=meter_provider)


class TracerProvider(_TracerProvider):
    """
    An extension of `opentelemetry.sdk.trace.TracerProvider` with Middleware-aware defaults.

    Only the HTTP/protobuf transport is supported.

    Args:
        endpoint (str, optional): The collector endpoint to which spans will be exported. If
            specified, a default SpanProcessor will be created and added to this TracerProvider.
            If not provided, the `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable is used.
        protocol (str, optional): The transport protocol. Only "http/protobuf" is supported.
        verbose (bool): If True, configuration details will be printed to stdout.
    """

    def __init__(
        self,
        *args: Any,
        endpoint: Optional[str] = None,
        protocol: Optional[Literal["http/protobuf", "grpc"]] = None,
        verbose: bool = True,
        **kwargs: Any,
    ):
        sig = _get_class_signature(_TracerProvider)
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        if bound_args.arguments.get("resource") is None:
            service_name = get_env_service_name()
            bound_args.arguments["resource"] = Resource.create(
                {
                    SERVICE_NAME: service_name,
                    PROJECT_NAME: get_env_project_name() or service_name,
                }
            )
        super().__init__(*bound_args.args, **bound_args.kwargs)

        validated_protocol = OTLPTransportProtocol(protocol)
        if validated_protocol == OTLPTransportProtocol.GRPC:
            raise NotImplementedError(_GRPC_UNSUPPORTED_MSG)
        parsed_url, endpoint = _normalized_endpoint(endpoint, use_http=True)
        if _maybe_grpc_endpoint(parsed_url):
            raise NotImplementedError(_GRPC_UNSUPPORTED_MSG)
        self._default_processor = False

        http_exporter: SpanExporter = HTTPSpanExporter(endpoint=endpoint)
        self.add_span_processor(SimpleSpanProcessor(span_exporter=http_exporter))
        self._default_processor = True
        if verbose:
            print(self._tracing_details())

    def add_span_processor(
        self, *args: Any, replace_default_processor: bool = True, **kwargs: Any
    ) -> None:
        """
        Registers a new `SpanProcessor` for this `TracerProvider`.

        If this `TracerProvider` has a default processor, it will be removed.
        """

        if self._default_processor and replace_default_processor:
            self._active_span_processor.shutdown()
            self._active_span_processor._span_processors = tuple()  # remove default processors
            self._default_processor = False
        return super().add_span_processor(*args, **kwargs)

    def _tracing_details(self) -> str:
        service = self.resource.attributes.get(SERVICE_NAME)
        project = self.resource.attributes.get(PROJECT_NAME)
        processor_name: Optional[str] = None
        endpoint: Optional[str] = None
        transport: Optional[str] = None
        headers: Optional[Union[Dict[str, str], str]] = None
        span_processor: Optional[SpanProcessor] = None

        if self._active_span_processor:
            if processors := self._active_span_processor._span_processors:
                if len(processors) == 1:
                    span_processor = self._active_span_processor._span_processors[0]
                    # Handle both old and new attribute locations for OpenTelemetry compatibility
                    # OpenTelemetry v1.34.0+ moved exporter from span_exporter to
                    # _batch_processor._exporter
                    # https://github.com/open-telemetry/opentelemetry-python/pull/4580
                    exporter = getattr(
                        getattr(span_processor, "_batch_processor", None), "_exporter", None
                    ) or getattr(span_processor, "span_exporter", None)
                    if exporter:
                        processor_name = span_processor.__class__.__name__
                        endpoint = exporter._endpoint
                        transport = _exporter_transport(exporter)
                        headers = _printable_headers(exporter._headers)
                else:
                    processor_name = "Multiple Span Processors"
                    endpoint = "Multiple Span Exporters"
                    transport = "Multiple Span Exporters"
                    headers = "Multiple Span Exporters"

        details_header = "🔭 Middleware LLMObs Tracing Details 🔭"

        configuration_msg = (
            "|  Using a default SpanProcessor. `add_span_processor` will overwrite this default.\n"
        )

        using_simple_processor = span_processor is not None and isinstance(
            span_processor, _SimpleSpanProcessor
        )
        span_processor_warning = (
            "|  \n"
            "|  ⚠️ WARNING: It is strongly advised to use a BatchSpanProcessor in production "
            "environments.\n"
        )

        details_msg = (
            f"{details_header}\n"
            f"|  Service Name: {service}\n"
            f"|  Project: {project}\n"
            f"|  Span Processor: {processor_name}\n"
            f"|  Collector Endpoint: {endpoint}\n"
            f"|  Transport: {transport}\n"
            f"|  Transport Headers: {headers}\n"
            "|  \n"
            f"{configuration_msg if self._default_processor else ''}"
            f"{span_processor_warning if using_simple_processor else ''}"
        )
        return details_msg


class SimpleSpanProcessor(_SimpleSpanProcessor):
    """
    Simple SpanProcessor implementation.

    SimpleSpanProcessor passes ended spans directly to the configured `SpanExporter`.

    Args:
        span_exporter (SpanExporter, optional): The `SpanExporter` to which ended spans will be
            passed.
        endpoint (str, optional): The collector endpoint. If not provided, the
            `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable will be used.
        headers (dict, optional): Optional headers to include in the export request. If not
            provided, the `OTEL_EXPORTER_OTLP_HEADERS` environment variable will be used.
        protocol (str, optional): The transport protocol. Only "http/protobuf" is supported.
    """

    def __init__(
        self,
        span_exporter: Optional[SpanExporter] = None,
        endpoint: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        protocol: Optional[Literal["http/protobuf", "grpc"]] = None,
    ):
        if span_exporter is None:
            span_exporter = _default_http_exporter(
                endpoint=endpoint, headers=headers, protocol=protocol
            )
        super().__init__(span_exporter)


class BatchSpanProcessor(_BatchSpanProcessor):
    """
    Batch SpanProcessor implementation (recommended for production).

    `BatchSpanProcessor` batches ended spans and pushes them to the configured `SpanExporter`.

    Configurable with the following environment variables which correspond to constructor
    parameters:

    - :envvar:`OTEL_BSP_SCHEDULE_DELAY`
    - :envvar:`OTEL_BSP_MAX_QUEUE_SIZE`
    - :envvar:`OTEL_BSP_MAX_EXPORT_BATCH_SIZE`
    - :envvar:`OTEL_BSP_EXPORT_TIMEOUT`

    Args:
        span_exporter (SpanExporter, optional): The `SpanExporter` to which ended spans will be
            passed.
        endpoint (str, optional): The collector endpoint. If not provided, the
            `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable will be used.
        headers (dict, optional): Optional headers to include in the export request. If not
            provided, the `OTEL_EXPORTER_OTLP_HEADERS` environment variable will be used.
        protocol (str, optional): The transport protocol. Only "http/protobuf" is supported.
    """

    def __init__(
        self,
        span_exporter: Optional[SpanExporter] = None,
        endpoint: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        protocol: Optional[Literal["http/protobuf", "grpc"]] = None,
    ):
        if span_exporter is None:
            span_exporter = _default_http_exporter(
                endpoint=endpoint, headers=headers, protocol=protocol
            )
        super().__init__(span_exporter)


def _default_http_exporter(
    endpoint: Optional[str],
    headers: Optional[Dict[str, str]],
    protocol: Optional[Literal["http/protobuf", "grpc"]],
) -> SpanExporter:
    """Build the default HTTP span exporter, rejecting any gRPC request."""
    validated_protocol = OTLPTransportProtocol(protocol)
    if validated_protocol == OTLPTransportProtocol.GRPC:
        raise NotImplementedError(_GRPC_UNSUPPORTED_MSG)
    parsed_url, endpoint = _normalized_endpoint(endpoint, use_http=True)
    if _maybe_grpc_endpoint(parsed_url):
        raise NotImplementedError(_GRPC_UNSUPPORTED_MSG)
    return HTTPSpanExporter(endpoint=endpoint, headers=headers)


class HTTPSpanExporter(_HTTPSpanExporter):
    """
    OTLP span exporter using HTTP.

    For more information, see:
    - `opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter`

    Args:
        endpoint (str, optional): OpenTelemetry Collector receiver endpoint. If not provided, the
            `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable will be used.
        headers: Headers to send when exporting. If not provided, the
            `OTEL_EXPORTER_OTLP_HEADERS` environment variable will be used. The Middleware
            ``Authorization`` value is passed through verbatim.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        sig = _get_class_signature(_HTTPSpanExporter)
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        if not bound_args.arguments.get("headers"):
            env_headers = get_env_client_headers()
            bound_args.arguments["headers"] = env_headers if env_headers else None
        else:
            headers: Dict[str, str] = dict()
            for header_field, value in bound_args.arguments["headers"].items():
                headers[header_field.lower()] = value
            bound_args.arguments["headers"] = headers

        if bound_args.arguments.get("endpoint") is None:
            _, endpoint = _normalized_endpoint(None, use_http=True)
            bound_args.arguments["endpoint"] = endpoint
        super().__init__(*bound_args.args, **bound_args.kwargs)


class GRPCSpanExporter(_GRPCSpanExporter):
    """
    OTLP span exporter using gRPC.

    .. warning::
        gRPC transport is **not supported** by middleware-llmobs. This class is retained for
        potential future use and is not wired into any default code path. Construct it only if
        you intend to manage gRPC export yourself; ``register`` and the default processors will
        never instantiate it.

    For more information, see:
    - `opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter`

    Args:
        endpoint (str, optional): OpenTelemetry Collector receiver endpoint. If not provided, the
            `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable will be used.
        insecure: Connection type
        credentials: Credentials object for server authentication
        headers: Headers to send when exporting. If not provided, the
            `OTEL_EXPORTER_OTLP_HEADERS` environment variable will be used.
        timeout: Backend request timeout in seconds
        compression: gRPC compression method to use
    """

    def __init__(self, *args: Any, **kwargs: Any):
        sig = _get_class_signature(_GRPCSpanExporter)
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        if not bound_args.arguments.get("headers"):
            env_headers = get_env_client_headers()
            bound_args.arguments["headers"] = env_headers if env_headers else None
        else:
            headers: Dict[str, str] = dict()
            for header_field, value in bound_args.arguments["headers"].items():
                headers[header_field.lower()] = value
            bound_args.arguments["headers"] = headers

        if bound_args.arguments.get("endpoint") is None:
            _, endpoint = _normalized_endpoint(None)
            bound_args.arguments["endpoint"] = endpoint
        super().__init__(*bound_args.args, **bound_args.kwargs)


def _maybe_http_endpoint(parsed_endpoint: ParseResult) -> bool:
    if parsed_endpoint.path == "/v1/traces":
        return True
    return False


def _maybe_grpc_endpoint(parsed_endpoint: ParseResult) -> bool:
    if not parsed_endpoint.path and parsed_endpoint.port == get_env_grpc_port():
        return True
    return False


def _exporter_transport(exporter: SpanExporter) -> str:
    if isinstance(exporter, _HTTPSpanExporter):
        return "HTTP + protobuf"
    if isinstance(exporter, _GRPCSpanExporter):
        return "gRPC"
    else:
        return exporter.__class__.__name__


def _printable_headers(headers: Union[List[Tuple[str, str]], Dict[str, str]]) -> Dict[str, str]:
    """
    Mask header values for safe printing/logging.

    Args:
        headers (Union[List[Tuple[str, str]], Dict[str, str]]): Headers as either
            a list of key-value tuples or a dictionary.

    Returns:
        Dict[str, str]: Dictionary with header keys preserved but values masked as "****".
    """
    if isinstance(headers, dict):
        return {key: "****" for key, _ in headers.items()}
    return {key: "****" for key, _ in headers}


def _construct_http_endpoint(parsed_endpoint: ParseResult) -> ParseResult:
    """Construct HTTP endpoint URL with traces path.

    Args:
        parsed_endpoint (ParseResult): Parsed URL endpoint.

    Returns:
        ParseResult: Modified endpoint with "/v1/traces" path.
    """
    traces_suffix = "/v1/traces"
    if parsed_endpoint.path.endswith(traces_suffix):
        return parsed_endpoint
    return parsed_endpoint._replace(path=traces_suffix)


def _construct_grpc_endpoint(parsed_endpoint: ParseResult) -> ParseResult:
    return parsed_endpoint._replace(netloc=f"{parsed_endpoint.hostname}:{get_env_grpc_port()}")


def _signal_http_endpoint(endpoint: Optional[str], signal_path: str) -> str:
    """Build a ``{base}{signal_path}`` HTTP endpoint (e.g. ``/v1/logs``, ``/v1/metrics``).

    Mirrors ``_construct_http_endpoint`` for non-trace signals, reusing the same base resolution
    (explicit ``endpoint`` or ``OTEL_EXPORTER_OTLP_ENDPOINT``, defaulting to localhost).
    """
    base = endpoint or get_env_collector_endpoint() or "http://localhost:9320"
    parsed = urlparse(base if _has_scheme(base) else f"//{base}")
    scheme = parsed.scheme or "http"
    return f"{scheme}://{parsed.netloc}{signal_path}"


def _register_eval_providers(
    *,
    resource: Resource,
    endpoint: Optional[str],
    headers: Optional[Dict[str, str]],
    set_global_logger_provider: bool = True,
    set_global_meter_provider: bool = True,
) -> Tuple[LoggerProvider, MeterProvider]:
    """Create the LoggerProvider + MeterProvider used for evaluations.

    Evaluations are emitted as OTel logs (``/v1/logs``) and gauge metrics (``/v1/metrics``) over the
    same HTTP/protobuf transport as traces. The eval resource carries ``service.name`` (from
    ``register()``) and ``mw.llm.source="sdk-eval"`` so every emitted log/metric is identifiable
    without per-call attribute work.

    Each provider is also installed as the OTel global only when its respective flag is true;
    callers that want to manage globals themselves can disable either. Providers are returned
    either way so callers can ``force_flush`` them at shutdown.
    """
    log_headers = headers or get_env_client_headers() or None
    # ``resource`` already carries ``service.name`` from ``register()``. Just add ``mw.llm.source``.
    eval_resource = resource.merge(Resource(attributes={"mw.llm.source": "sdk-eval"}))

    logger_provider = LoggerProvider(resource=eval_resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            _HTTPLogExporter(
                endpoint=_signal_http_endpoint(endpoint, "/v1/logs"), headers=log_headers
            )
        )
    )
    if set_global_logger_provider:
        set_logger_provider(logger_provider)

    metric_reader = PeriodicExportingMetricReader(
        _HTTPMetricExporter(
            endpoint=_signal_http_endpoint(endpoint, "/v1/metrics"), headers=log_headers
        )
    )
    meter_provider = MeterProvider(resource=eval_resource, metric_readers=[metric_reader])
    if set_global_meter_provider:
        set_meter_provider(meter_provider)

    return logger_provider, meter_provider


def _has_scheme(s: str) -> bool:
    return "//" in s


def _normalized_endpoint(endpoint: Optional[str], use_http: bool = True) -> Tuple[ParseResult, str]:
    if endpoint is None:
        base_endpoint = get_env_collector_endpoint() or "http://localhost:9320"
        parsed = urlparse(base_endpoint)
        if use_http:
            parsed = _construct_http_endpoint(parsed)
        else:
            parsed = _construct_grpc_endpoint(parsed)
    else:
        if not _has_scheme(endpoint):
            # Use // to indicate an "authority" to properly parse the URL
            # https://en.wikipedia.org/wiki/Uniform_Resource_Identifier#Syntax
            # However, return the original endpoint to avoid overspecifying the URL scheme
            return urlparse(f"//{endpoint}"), endpoint
        parsed = urlparse(endpoint)
        if use_http:
            parsed = _construct_http_endpoint(parsed)
    return parsed, parsed.geturl()


def _get_class_signature(fn: Type[Any]) -> inspect.Signature:
    return inspect.signature(fn)


def _auto_instrument_installed_openinference_libraries(tracer_provider: TracerProvider) -> None:
    openinference_entry_points = entry_points(group="openinference_instrumentor")
    if not openinference_entry_points:
        warnings.warn(
            "No OpenInference instrumentors found. "
            "Maybe you need to update your OpenInference version? "
            "Skipping auto-instrumentation."
        )
        return
    trace_config = _make_genai_trace_config()
    for entry_point in openinference_entry_points:
        instrumentor_cls = entry_point.load()
        instrumentor = instrumentor_cls()
        if trace_config is not None:
            try:
                instrumentor.instrument(tracer_provider=tracer_provider, config=trace_config)
                continue
            except TypeError:
                # Instrumentor does not accept a `config` argument; fall back below.
                pass
        instrumentor.instrument(tracer_provider=tracer_provider)


def _make_genai_trace_config() -> Optional[Any]:
    """Build a TraceConfig with GenAI semantic conventions enabled, if supported."""
    try:
        from openinference.instrumentation import TraceConfig
    except ImportError:
        return None
    try:
        return TraceConfig(enable_genai_semconv=True)
    except TypeError:
        # Installed openinference-instrumentation predates `enable_genai_semconv`.
        return TraceConfig()

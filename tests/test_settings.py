from unittest.mock import patch

import pytest

from middleware.llmobs.settings import (
    GRPC_PORT,
    get_env_client_headers,
    get_env_collector_endpoint,
    get_env_grpc_port,
    get_env_project_name,
    get_env_service_name,
    parse_env_headers,
)


class TestCollectorEndpoint:
    def test_reads_otel_endpoint(self) -> None:
        with patch.dict(
            "os.environ",
            {"OTEL_EXPORTER_OTLP_ENDPOINT": "https://uid.middleware.io:443"},
            clear=True,
        ):
            assert get_env_collector_endpoint() == "https://uid.middleware.io:443"

    def test_none_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert get_env_collector_endpoint() is None


class TestServiceAndProjectName:
    def test_service_name_from_env(self) -> None:
        with patch.dict("os.environ", {"OTEL_SERVICE_NAME": "my-app"}, clear=True):
            assert get_env_service_name() == "my-app"

    def test_service_name_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert get_env_service_name() == "default"

    def test_project_name_from_env(self) -> None:
        with patch.dict("os.environ", {"MW_PROJECT_NAME": "proj"}, clear=True):
            assert get_env_project_name() == "proj"

    def test_project_name_none_when_unset(self) -> None:
        # Service-name fallback is the caller's responsibility, not this function's.
        with patch.dict("os.environ", {"OTEL_SERVICE_NAME": "my-app"}, clear=True):
            assert get_env_project_name() is None


class TestClientHeaders:
    def test_parses_authorization_header(self) -> None:
        with patch.dict(
            "os.environ",
            {"OTEL_EXPORTER_OTLP_HEADERS": "Authorization=secret-key"},
            clear=True,
        ):
            headers = get_env_client_headers()
            assert headers == {"authorization": "secret-key"}

    def test_none_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert get_env_client_headers() is None


class TestParseEnvHeaders:
    def test_single_header(self) -> None:
        assert parse_env_headers("Authorization=key") == {"authorization": "key"}

    def test_multiple_headers(self) -> None:
        result = parse_env_headers("Authorization=key,X-Custom=value")
        assert result == {"authorization": "key", "x-custom": "value"}

    def test_value_with_equals_sign(self) -> None:
        # Tokens may contain '=' (e.g. base64). Only the first '=' splits key/value.
        result = parse_env_headers("Authorization=abc%3D%3D")
        assert result == {"authorization": "abc=="}


class TestGrpcPort:
    def test_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert get_env_grpc_port() == GRPC_PORT

    def test_custom(self) -> None:
        with patch.dict("os.environ", {"MW_GRPC_PORT": "5555"}, clear=True):
            assert get_env_grpc_port() == 5555

    def test_invalid_raises(self) -> None:
        with patch.dict("os.environ", {"MW_GRPC_PORT": "abc"}, clear=True):
            with pytest.raises(ValueError):
                get_env_grpc_port()

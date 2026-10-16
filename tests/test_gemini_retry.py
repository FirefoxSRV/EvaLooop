"""Tests for which failures the Gemini client retries.

This classification decides whether a multi-hour run survives a hiccup or throws
away its results. Two real incidents are pinned here: four `Connection reset by
peer` errors out of ~1800 calls aborted two otherwise healthy runs because a
dropped TCP connection was not recognised as transient, and a 404 saying a model
is "no longer available to new users" matches the word "unavailable", so without
an explicit fatal check it would consume the entire retry budget before
surfacing.
"""

import pytest

from evaloop.models.gemini import GeminiLLM


class Boom(Exception):
    pass


class TestRetryable:
    @pytest.mark.parametrize("message", [
        "[Errno 54] Connection reset by peer",
        "Connection aborted",
        "Broken pipe",
        "[Errno 61] Connection refused",
        "Server disconnected without sending a response",
        "Remote end closed connection without response",
        "SSL: WRONG_VERSION_NUMBER",
        "EOF occurred in violation of protocol",
    ])
    def test_dropped_connections_are_transient(self, message):
        assert GeminiLLM._is_retryable(Boom(message)) is True

    @pytest.mark.parametrize("message", [
        "429 RESOURCE_EXHAUSTED",
        "Quota exceeded for metric",
        "rate limit reached",
    ])
    def test_throttling_is_transient(self, message):
        assert GeminiLLM._is_retryable(Boom(message)) is True

    @pytest.mark.parametrize("message", [
        "500 Internal error", "502 Bad Gateway", "503 Service Unavailable",
        "504 Deadline exceeded", "The model is overloaded", "Read timed out",
    ])
    def test_server_side_and_timeouts_are_transient(self, message):
        assert GeminiLLM._is_retryable(Boom(message)) is True


class TestFatal:
    @pytest.mark.parametrize("message", [
        "400 INVALID_ARGUMENT",
        "403 PERMISSION_DENIED",
        "401 UNAUTHENTICATED",
        "API key not valid. Please pass a valid API key.",
    ])
    def test_client_errors_are_not_retried(self, message):
        assert GeminiLLM._is_retryable(Boom(message)) is False

    def test_a_404_mentioning_unavailable_is_still_fatal(self):
        # The exact message that cost a retry budget: "unavailable" is a
        # transient marker, but the 404 must win.
        message = ("404 NOT_FOUND. This model models/gemini-2.5-flash is no longer "
                   "available to new users.")
        assert GeminiLLM._is_retryable(Boom(message)) is False

    def test_fatal_markers_take_precedence_over_transient_ones(self):
        assert GeminiLLM._is_retryable(Boom("400 INVALID_ARGUMENT: connection reset")) is False


class TestDailyQuota:
    def test_per_day_exhaustion_is_detected(self):
        exc = Boom("{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}")
        assert GeminiLLM._is_daily_quota(exc) is True

    def test_per_minute_throttling_is_not_daily_exhaustion(self):
        exc = Boom("{'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'}")
        assert GeminiLLM._is_daily_quota(exc) is False

    @pytest.mark.parametrize("message,expected", [
        ("{'retryDelay': '59s'}", 59.0),
        ('"retryDelay": "17s"', 17.0),
        ("retryDelay: 5.5s", 5.5),
        ("no hint at all", None),
    ])
    def test_server_retry_hint_is_parsed(self, message, expected):
        assert GeminiLLM._server_retry_delay(Boom(message)) == expected


class TestRequestTimeout:
    """A request with no deadline can wedge; one run sat five hours on one."""

    def test_a_default_timeout_is_set(self):
        llm = GeminiLLM.__new__(GeminiLLM)
        llm.request_timeout = 180.0
        assert llm._http_options().timeout == 180_000  # the SDK takes milliseconds

    def test_the_timeout_is_configurable(self):
        llm = GeminiLLM.__new__(GeminiLLM)
        llm.request_timeout = 45.0
        assert llm._http_options().timeout == 45_000

    def test_a_timeout_is_retryable_rather_than_fatal(self):
        # The whole point: a wedged connection must become an ordinary retry.
        for message in ("Request timed out", "ReadTimeout", "deadline exceeded"):
            assert GeminiLLM._is_retryable(Boom(message)) is True

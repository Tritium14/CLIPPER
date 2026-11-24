"""Tests for analysis/llm_client.py (no network; a fake session is used)."""

import json

import pytest
import requests

from analysis.analysis_errors import LLMAuthError, LLMError
from analysis.llm_client import LLMClient
from config import AIConfig, ConfigError


class FakeResponse:
    def __init__(self, status=200, data=None, text="", headers=None):
        self.status_code = status
        self._data = data
        self.text = text or (json.dumps(data) if data is not None else "")
        self.headers = headers or {}

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ok(content, finish_reason="stop"):
    return FakeResponse(200, {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]})


def make_client(responses, **config_overrides):
    settings = dict(
        provider="lmstudio",
        model="main-model",
        min_seconds_between_requests=0.0,
        max_retries=2,
        retry_backoff_seconds=1.0,
    )
    settings.update(config_overrides)
    session = FakeSession(responses)
    sleeps = []
    client = LLMClient(AIConfig(**settings), session=session, sleep=sleeps.append)
    return client, session, sleeps


def test_successful_request_sends_expected_payload():
    client, session, _ = make_client([ok("hello")])
    assert client.chat("sys", "usr") == "hello"
    call = session.calls[0]
    assert call["url"] == "http://localhost:1234/v1/chat/completions"
    assert call["json"]["model"] == "main-model"
    assert call["json"]["max_tokens"] == AIConfig().max_output_tokens
    assert call["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert call["json"]["messages"][1] == {"role": "user", "content": "usr"}
    assert "Authorization" not in call["headers"]


def test_api_key_is_sent_as_bearer_token(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-1234567890")
    client, session, _ = make_client([ok("hi")], provider="openrouter", model="m:free")
    client.chat("s", "u")
    assert session.calls[0]["headers"]["Authorization"] == "Bearer sk-or-test-1234567890"


def test_missing_key_raises_auth_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client, _, _ = make_client([], provider="openrouter", model="m:free")
    with pytest.raises(LLMAuthError):
        client.chat("s", "u")


def test_rate_limit_waits_for_retry_after_then_succeeds():
    client, _, sleeps = make_client(
        [FakeResponse(429, text="slow down", headers={"Retry-After": "7"}), ok("done")]
    )
    assert client.chat("s", "u") == "done"
    assert sleeps == [7.0]


def test_server_errors_use_exponential_backoff():
    client, _, sleeps = make_client([FakeResponse(500), FakeResponse(503), ok("fine")])
    assert client.chat("s", "u") == "fine"
    assert sleeps == [1.0, 2.0]


def test_retries_are_exhausted():
    client, session, _ = make_client([FakeResponse(500)] * 3, max_retries=2)
    with pytest.raises(LLMError):
        client.chat("s", "u")
    assert len(session.calls) == 3


def test_timeout_is_retried():
    client, _, sleeps = make_client([requests.Timeout("too slow"), ok("ok")])
    assert client.chat("s", "u") == "ok"
    assert sleeps == [1.0]


def test_auth_failure_is_not_retried_and_hides_the_key(monkeypatch):
    secret = "sk-or-secret-abcdefghij"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    client, session, _ = make_client(
        [FakeResponse(401, text=f"invalid key {secret}")], provider="openrouter", model="m:free"
    )
    with pytest.raises(LLMAuthError) as info:
        client.chat("s", "u")
    assert secret not in str(info.value)
    assert len(session.calls) == 1


def test_missing_model_falls_back_to_next_model():
    client, session, _ = make_client(
        [FakeResponse(404, text="no such model"), ok("from backup")], fallback_models=["backup-model"]
    )
    assert client.chat("s", "u") == "from backup"
    assert session.calls[1]["json"]["model"] == "backup-model"


def test_empty_reply_is_retried_then_fails():
    client, _, _ = make_client([ok("")] * 3, max_retries=2)
    with pytest.raises(LLMError) as info:
        client.chat("s", "u")
    assert "empty" in str(info.value)


def test_content_given_as_parts_is_joined():
    client, _, _ = make_client([ok([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])])
    assert client.chat("s", "u") == "ab"


def test_error_object_in_200_response_is_handled():
    body = FakeResponse(200, {"error": {"message": "overloaded", "code": 503}})
    client, _, sleeps = make_client([body, ok("recovered")])
    assert client.chat("s", "u") == "recovered"
    assert sleeps == [1.0]


def test_requests_are_throttled():
    state = {"t": 0.0}
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        state["t"] += seconds

    session = FakeSession([ok("one"), ok("two")])
    config = AIConfig(provider="lmstudio", model="m", min_seconds_between_requests=3.0)
    client = LLMClient(config, session=session, sleep=fake_sleep, clock=lambda: state["t"])
    client.chat("s", "u")
    client.chat("s", "u")
    assert sleeps == [3.0]


def test_no_model_configured_raises():
    client, _, _ = make_client([], provider="openai_compatible", model="", base_url="http://x/v1")
    with pytest.raises(LLMError):
        client.chat("s", "u")


def test_models_list_has_no_duplicates():
    client, _, _ = make_client([], fallback_models=["main-model", "other"])
    assert client.models == ["main-model", "other"]


def test_new_ai_settings_are_validated():
    AIConfig().validate()
    with pytest.raises(ConfigError):
        AIConfig(max_output_tokens=10).validate()
    with pytest.raises(ConfigError):
        AIConfig(candidates_per_chunk=0).validate()
    with pytest.raises(ConfigError):
        AIConfig(min_seconds_between_requests=-1).validate()
    with pytest.raises(ConfigError):
        AIConfig(fallback_models=["ok", ""]).validate()
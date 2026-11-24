"""Chat client for OpenAI-compatible providers (Ollama, LM Studio, OpenRouter).

Features: request throttling for free-tier rate limits, retries with backoff,
Retry-After support, and automatic fallback to other models. API keys are read
from the environment at call time and never logged.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from analysis.analysis_errors import (
    LLMAuthError,
    LLMError,
    LLMModelError,
    LLMTransientError,
)
from config import AIConfig, ConfigError
from logging_setup import redact_text

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 60.0
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _retry_after_seconds(response: Any) -> Optional[float]:
    headers = getattr(response, "headers", None) or {}
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        return min(max(float(value), 0.0), MAX_BACKOFF_SECONDS)
    except (TypeError, ValueError):
        return None


class LLMClient:
    """Sends chat requests to the configured provider."""

    def __init__(
        self,
        ai_config: AIConfig,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = ai_config
        self._session = session
        self._sleep = sleep
        self._clock = clock
        self._last_request: Optional[float] = None

    @property
    def models(self) -> List[str]:
        """The main model followed by any fallback models (no duplicates)."""
        models: List[str] = []
        primary = self._config.resolved_model()
        if primary:
            models.append(primary)
        for name in self._config.fallback_models:
            if name and name not in models:
                models.append(name)
        return models

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send one system+user exchange and return the reply text.

        Raises:
            LLMAuthError: for key or billing problems.
            LLMError: when every configured model failed.
        """
        models = self.models
        if not models:
            raise LLMError("No AI model is configured. Set ai.model in config.json.")

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_error: Optional[LLMError] = None
        for model in models:
            try:
                return self._chat_with_retries(model, messages)
            except LLMAuthError:
                raise
            except LLMError as exc:
                last_error = exc
                logger.warning("Model '%s' failed: %s", model, exc)
        raise LLMError(f"All configured models failed. Last error: {last_error}")

    def _chat_with_retries(self, model: str, messages: List[Dict[str, str]]) -> str:
        attempts = self._config.max_retries + 1
        for attempt in range(attempts):
            try:
                return self._request_once(model, messages)
            except LLMTransientError as exc:
                if attempt == attempts - 1:
                    raise
                delay = exc.retry_after
                if delay is None:
                    delay = min(
                        self._config.retry_backoff_seconds * (2**attempt), MAX_BACKOFF_SECONDS
                    )
                logger.warning(
                    "%s Retrying in %.0f s (attempt %d of %d).", exc, delay, attempt + 1, attempts - 1
                )
                self._sleep(delay)
        raise LLMError("Request failed.")  # unreachable; keeps type checkers satisfied

    def _throttle(self) -> None:
        interval = self._config.min_seconds_between_requests
        if self._last_request is not None and interval > 0:
            wait = interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _api_key(self) -> Optional[str]:
        try:
            return self._config.get_api_key()
        except ConfigError as exc:
            raise LLMAuthError(str(exc)) from exc

    def _request_once(self, model: str, messages: List[Dict[str, str]]) -> str:
        try:
            import requests
        except ImportError as exc:
            raise LLMError("The 'requests' package is not installed. Run: pip install requests") from exc

        session = self._session
        if session is None:
            session = self._session = requests.Session()

        config = self._config
        headers = {"Content-Type": "application/json"}
        key = self._api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_output_tokens,
        }
        if config.request_json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{config.resolved_base_url()}/chat/completions"
        self._throttle()
        logger.debug("Sending request to model '%s'", model)
        try:
            response = session.post(
                url, headers=headers, json=payload, timeout=config.request_timeout_seconds
            )
        except requests.RequestException as exc:
            raise LLMTransientError(
                f"Network problem contacting {config.resolved_base_url()}: {redact_text(str(exc))[:200]}"
            ) from exc

        status = response.status_code
        if status == 200:
            return self._extract_content(response)

        body = redact_text((getattr(response, "text", "") or "")[:300]).strip()
        if status in (401, 403):
            raise LLMAuthError(
                f"The AI provider rejected the request (HTTP {status}). Check your API key. {body}"
            )
        if status == 402:
            raise LLMAuthError(f"The AI provider reports insufficient credits (HTTP 402). {body}")
        if status in _TRANSIENT_STATUS:
            raise LLMTransientError(
                f"The AI provider is busy or unavailable (HTTP {status}). {body}",
                retry_after=_retry_after_seconds(response),
            )
        if status == 404:
            raise LLMModelError(f"Model '{model}' was not found (HTTP 404). Check ai.model. {body}")
        raise LLMModelError(f"The AI request failed (HTTP {status}). {body}")

    def _extract_content(self, response: Any) -> str:
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMTransientError("The AI provider returned a response that is not valid JSON.") from exc

        if isinstance(data, dict) and data.get("error"):
            error = data["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            code = error.get("code") if isinstance(error, dict) else None
            text = redact_text(str(message))[:300]
            try:
                transient = int(code) in _TRANSIENT_STATUS
            except (TypeError, ValueError):
                transient = False
            if transient:
                raise LLMTransientError(f"The AI provider returned an error: {text}")
            raise LLMModelError(f"The AI provider returned an error: {text}")

        try:
            choice = data["choices"][0]
            content = (choice.get("message") or {}).get("content")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMTransientError("The AI provider returned an unexpected response format.") from exc

        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        content = (content or "").strip()
        if not content:
            raise LLMTransientError(
                "The model returned an empty reply. Reasoning models can run out of tokens; "
                "try raising ai.max_output_tokens."
            )
        if choice.get("finish_reason") == "length":
            logger.warning("The model's reply was cut off at the token limit; salvaging what is complete.")
        return content
import asyncio

import pytest

import litellm
from litellm import completion_with_retries
from litellm.exceptions import OpenAIError, RateLimitError
from litellm.litellm_core_utils.fallback_utils import async_completion_with_fallbacks


def test_completion_with_retries_skips_openai_error():
    attempts = 0

    def _failing_completion(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise OpenAIError()

    with pytest.raises(OpenAIError):
        completion_with_retries(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            num_retries=3,
            original_function=_failing_completion,
        )

    assert attempts == 1


def test_completion_with_retries_skips_rate_limit_error():
    attempts = 0

    def _failing_completion(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise RateLimitError("throttle", "openai", "gpt-3.5-turbo")

    with pytest.raises(RateLimitError):
        completion_with_retries(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            num_retries=3,
            original_function=_failing_completion,
        )

    assert attempts == 1


def test_async_fallback_skips_on_openai_error(monkeypatch):
    called_models = []

    async def _failing_acompletion(*args, **kwargs):
        called_models.append(kwargs.get("model"))
        raise OpenAIError()

    monkeypatch.setattr(litellm, "acompletion", _failing_acompletion)

    async def _invoke():
        await async_completion_with_fallbacks(
            model="primary",
            kwargs={
                "messages": [],
                "fallbacks": ["secondary"],
            },
        )

    with pytest.raises(OpenAIError):
        asyncio.run(_invoke())

    assert called_models == ["primary"]


def test_async_fallback_continues_on_rate_limit(monkeypatch):
    called_models = []

    async def _acompletion_with_rate_limit(*args, **kwargs):
        called_models.append(kwargs.get("model"))
        if len(called_models) == 1:
            raise RateLimitError("throttle", "openai", kwargs.get("model"))
        return "ok"

    monkeypatch.setattr(litellm, "acompletion", _acompletion_with_rate_limit)

    async def _invoke():
        return await async_completion_with_fallbacks(
            model="primary",
            kwargs={
                "messages": [],
                "fallbacks": ["secondary"],
            },
        )

    response = asyncio.run(_invoke())

    assert response == "ok"
    assert called_models == ["primary", "secondary"]

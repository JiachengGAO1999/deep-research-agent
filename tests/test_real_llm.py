"""Integration tests against the real vLLM Qwen3-8B instance.

These tests require the SSH tunnel to be active:
    ssh -L 18004:127.0.0.1:8004 sjtu-a800

They are automatically skipped if the vLLM is not reachable.
"""

import os
import pytest
import httpx
from pydantic import BaseModel, Field


# Real LLM config
REAL_LLM_URL = "http://127.0.0.1:18004/v1"
REAL_MODEL_ID = "qwen3-8b-budget"


def _is_vllm_available() -> bool:
    """Check if the vLLM instance is reachable."""
    try:
        import httpx
        resp = httpx.get(f"{REAL_LLM_URL}/models", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


# Skip all tests in this module if vLLM is not available
pytestmark = pytest.mark.skipif(
    not _is_vllm_available(),
    reason="vLLM not reachable at 127.0.0.1:18004 — start SSH tunnel first",
)


class TestVLLMModels:
    """Test /v1/models endpoint."""

    def test_models_endpoint_returns_200(self):
        resp = httpx.get(f"{REAL_LLM_URL}/models", timeout=10.0)
        assert resp.status_code == 200

    def test_model_id_matches(self):
        resp = httpx.get(f"{REAL_LLM_URL}/models", timeout=10.0)
        data = resp.json()
        model_ids = [m["id"] for m in data.get("data", [])]
        assert REAL_MODEL_ID in model_ids, f"Expected {REAL_MODEL_ID} in {model_ids}"

    def test_model_has_max_model_len(self):
        resp = httpx.get(f"{REAL_LLM_URL}/models", timeout=10.0)
        data = resp.json()
        for m in data.get("data", []):
            if m["id"] == REAL_MODEL_ID:
                assert m.get("max_model_len", 0) > 0
                return
        pytest.fail(f"Model {REAL_MODEL_ID} not found")


class TestVLLMChatCompletions:
    """Test /v1/chat/completions basic functionality."""

    def test_simple_completion_returns_content(self):
        """Minimal request should return non-null content."""
        payload = {
            "model": REAL_MODEL_ID,
            "messages": [{"role": "user", "content": "Reply with exactly: hello"}],
            "max_tokens": 200,
            "temperature": 0,
        }
        resp = httpx.post(
            f"{REAL_LLM_URL}/chat/completions",
            json=payload,
            timeout=30.0,
        )
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        assert content is not None, (
            "content is null — model may have been truncated during reasoning. "
            "Increase max_tokens or set enable_thinking=false."
        )
        assert "hello" in content.lower()

    def test_no_api_key_required(self):
        """Server should accept requests without Authorization header."""
        payload = {
            "model": REAL_MODEL_ID,
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 50,
            "temperature": 0,
        }
        resp = httpx.post(
            f"{REAL_LLM_URL}/chat/completions",
            json=payload,
            timeout=30.0,
        )
        assert resp.status_code == 200

    def test_thinking_false_produces_content(self):
        """With enable_thinking=false, content should be populated directly."""
        payload = {
            "model": REAL_MODEL_ID,
            "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            "max_tokens": 100,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = httpx.post(
            f"{REAL_LLM_URL}/chat/completions",
            json=payload,
            timeout=30.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        assert content is not None, "content should not be null when thinking is disabled"


class TestStructuredOutput:
    """Test structured JSON output with the real LLM."""

    def test_simple_structured_output(self):
        """LLM should produce parseable JSON for a simple structured task."""
        # Define a minimal schema inline
        class SentimentResult(BaseModel):
            sentiment: str = Field(description="positive, negative, or neutral")
            confidence: float = Field(ge=0, le=1)

        import json
        schema_json = json.dumps(SentimentResult.model_json_schema(), ensure_ascii=False)

        system_prompt = (
            "You are a classifier. Respond ONLY with a JSON object matching the specified schema. "
            "No markdown fences, no explanatory text.\n"
            f"Schema: {schema_json}"
        )
        user_prompt = "Text: 'The new model architecture shows promising results on benchmark tasks.'"

        payload = {
            "model": REAL_MODEL_ID,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 200,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"},
        }
        resp = httpx.post(
            f"{REAL_LLM_URL}/chat/completions",
            json=payload,
            timeout=30.0,
        )
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        assert content is not None, "content is null"

        # Should be parseable as JSON
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            pytest.fail(f"Failed to parse JSON: {e}\nContent: {content[:300]}")

        # Should match schema
        result = SentimentResult.model_validate(parsed)
        assert result.sentiment in ("positive", "negative", "neutral")
        assert 0 <= result.confidence <= 1

    @pytest.mark.asyncio
    async def test_structured_output_via_client(self):
        """Test through our LLMClient abstraction layer."""
        import app.core.config as config_mod
        from app.llm.client import LLMClient

        # Patch settings to point at real vLLM
        config_mod.Settings.LLM_BASE_URL = REAL_LLM_URL
        config_mod.Settings.LLM_API_KEY = "EMPTY"
        config_mod.Settings.LLM_MODEL_FAST = REAL_MODEL_ID

        settings = config_mod.Settings()
        client = LLMClient(settings=settings)

        class TestOutput(BaseModel):
            answer: str

        try:
            result, usage = await client.generate_structured(
                system_prompt="You are helpful. Respond ONLY with valid JSON.",
                user_prompt='What is the capital of France? Reply with {"answer": "..."}',
                output_model=TestOutput,
                model=REAL_MODEL_ID,
                max_tokens=200,
                enable_thinking=False,
            )

            assert result is not None, "Structured output should parse successfully"
            assert isinstance(result, TestOutput)
            assert "paris" in result.answer.lower()
            assert usage.get("total_tokens", 0) > 0, "Should report token usage"
        finally:
            await client.close()

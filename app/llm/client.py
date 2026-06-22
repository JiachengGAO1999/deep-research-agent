"""LLM client abstraction — OpenAI-compatible chat completions."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional, Type

import httpx
from pydantic import BaseModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Async client for OpenAI-compatible chat completions API.

    Supports vLLM-specific chat_template_kwargs for controlling reasoning mode.
    """

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            # Only add auth header if API key is non-empty
            if self._settings.LLM_API_KEY and self._settings.LLM_API_KEY != "EMPTY":
                headers["Authorization"] = f"Bearer {self._settings.LLM_API_KEY}"

            self._client = httpx.AsyncClient(
                base_url=self._settings.LLM_BASE_URL,
                timeout=httpx.Timeout(self._settings.LLM_TIMEOUT),
                headers=headers,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None,
        enable_thinking: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Send a chat completion request.

        Args:
            messages: Chat messages.
            model: Model ID.
            temperature: Sampling temperature.
            max_tokens: Max completion tokens.
            response_format: Optional response_format dict (e.g. {"type": "json_object"}).
            enable_thinking: If set, adds chat_template_kwargs to control Qwen3 reasoning.
        """
        client = await self._get_client()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        # vLLM-specific: control reasoning mode via chat_template_kwargs
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        logger.debug(
            f"LLM request: model={model}, msgs={len(messages)}, "
            f"max_tokens={max_tokens}, thinking={enable_thinking}"
        )

        for attempt in range(self._settings.LLM_MAX_RETRIES + 1):
            try:
                response = await client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                logger.debug(f"LLM response: tokens={data.get('usage', {})}")
                return data
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < self._settings.LLM_MAX_RETRIES:
                    wait = 2 ** attempt
                    logger.warning(f"LLM rate limited, retrying in {wait}s")
                    import asyncio
                    await asyncio.sleep(wait)
                    continue
                logger.error(
                    f"LLM HTTP error: {e.response.status_code} {e.response.text[:500]}"
                )
                raise
            except Exception as e:
                if attempt < self._settings.LLM_MAX_RETRIES:
                    wait = 2 ** attempt
                    logger.warning(f"LLM error, retrying in {wait}s: {e}")
                    import asyncio
                    await asyncio.sleep(wait)
                    continue
                raise

        raise RuntimeError("LLM request failed after all retries")

    @staticmethod
    def _extract_content(response: dict) -> str:
        """Extract ONLY the 'content' field from chat completion response.

        Never reads 'reasoning' or 'reasoning_content' — those are
        the model's internal thinking, not the final answer.
        """
        try:
            content = response["choices"][0]["message"]["content"]
            if content is None:
                # Model didn't produce final content (e.g., ran out of tokens
                # during reasoning phase). Return empty string — never fall
                # back to reasoning content.
                logger.warning("LLM returned null content (possibly truncated during reasoning)")
                return ""
            return content
        except (KeyError, IndexError):
            logger.error(f"Unexpected LLM response format, keys: {list(response.keys())}")
            return ""

    @staticmethod
    def _extract_usage(response: dict) -> dict:
        """Extract token usage from response."""
        return response.get("usage", {})

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[str, dict]:
        """Generate text from prompts. Returns (content, usage_info)."""
        model = model or self._settings.model_fast
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        started = time.perf_counter()
        response = await self.chat(
            messages, model, temperature,
            max_tokens=max_tokens or 4096,
            enable_thinking=enable_thinking,
        )
        usage = self._extract_usage(response)
        usage.update(
            {"model": model, "latency_seconds": time.perf_counter() - started}
        )
        return self._extract_content(response), usage

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[Optional[BaseModel], dict]:
        """Generate structured output parsed into a Pydantic model.

        Disables thinking by default for structured output (reasoning
        text mixed with JSON breaks parsing).

        Returns (parsed_model_or_None, usage_info).
        """
        model_name = model or self._settings.model_fast
        max_tok = max_tokens or 4096

        # For structured output, reasoning text before JSON breaks parsing.
        # Default to thinking=False when not explicitly set.
        if enable_thinking is None:
            enable_thinking = False

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        schema_json = json.dumps(output_model.model_json_schema(), ensure_ascii=False)
        enhanced_system = (
            system_prompt
            + f"\n\nYou MUST respond ONLY with valid JSON matching this schema:\n```json\n{schema_json}\n```\n"
            + "Do NOT include any text outside the JSON object. Do NOT wrap in markdown code fences."
            + "\nOutput raw JSON directly, no explanation."
        )
        messages[0] = {"role": "system", "content": enhanced_system}

        for attempt in range(3):
            try:
                started = time.perf_counter()
                # Try with response_format json_object
                try:
                    response = await self.chat(
                        messages,
                        model_name,
                        temperature,
                        max_tok,
                        response_format={"type": "json_object"},
                        enable_thinking=enable_thinking,
                    )
                except Exception:
                    response = await self.chat(
                        messages, model_name, temperature, max_tok,
                        enable_thinking=enable_thinking,
                    )

                content = self._extract_content(response)
                usage = self._extract_usage(response)
                usage.update(
                    {
                        "model": model_name,
                        "latency_seconds": time.perf_counter() - started,
                    }
                )

                parsed = self._parse_json_response(content, output_model)
                if parsed is not None:
                    return parsed, usage

                logger.warning(
                    f"JSON parse attempt {attempt + 1} failed, "
                    f"content preview: {content[:200]}"
                )
                if attempt < 2:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Respond ONLY with a valid JSON object matching the schema. "
                            "No markdown fences, no explanatory text."
                        ),
                    })
            except Exception as e:
                logger.error(f"Structured generation attempt {attempt + 1} error: {e}")
                if attempt >= 2:
                    return None, {}

        return None, {}

    @staticmethod
    def _parse_json_response(
        content: str, model: Type[BaseModel]
    ) -> Optional[BaseModel]:
        """Parse JSON from LLM response, handling markdown fences."""
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
            return model.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"JSON parse failed: {e}")
            return None


# Global client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client

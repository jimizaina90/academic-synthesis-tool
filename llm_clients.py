from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

Provider = Literal["openai", "anthropic", "gemini"]


@dataclass
class LLMConfig:
    provider: Provider
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_output_tokens: int = 4000


class LLMError(RuntimeError):
    pass


class BaseClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.api_key = config.api_key or self._get_env_key()
        if not self.api_key:
            raise LLMError(
                f"Missing API key for provider '{config.provider}'. "
                "Pass it in the UI or define the relevant environment variable."
            )

    def _get_env_key(self) -> Optional[str]:
        mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        return os.getenv(mapping[self.config.provider])

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OpenAIClient(BaseClient):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "The 'openai' package is not installed. Install requirements first."
            ) from exc

        client = OpenAI(api_key=self.api_key)
        response = client.responses.create(
            model=self.config.model,
            instructions=system_prompt,
            input=user_prompt,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )

        text = getattr(response, "output_text", None)
        if not text:
            raise LLMError("OpenAI returned no text output.")
        return text.strip()


class AnthropicClient(BaseClient):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise LLMError(
                "The 'anthropic' package is not installed. Install requirements first."
            ) from exc

        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.config.model,
            system=system_prompt,
            max_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": user_prompt}],
        )

        blocks = []
        for block in getattr(response, "content", []):
            text = getattr(block, "text", None)
            if text:
                blocks.append(text)

        if not blocks:
            raise LLMError("Anthropic returned no text blocks.")
        return "\n".join(blocks).strip()


class GeminiClient(BaseClient):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from google import genai
        except ImportError as exc:
            raise LLMError(
                "The 'google-genai' package is not installed. Install requirements first."
            ) from exc

        client = genai.Client(api_key=self.api_key)
        prompt = f"[INSTRUÇÕES DE SISTEMA]\n{system_prompt}\n\n[PEDIDO]\n{user_prompt}"
        response = client.models.generate_content(
            model=self.config.model,
            contents=prompt,
        )

        text = getattr(response, "text", None)
        if not text:
            raise LLMError("Gemini returned no text output.")
        return text.strip()


def get_client(config: LLMConfig) -> BaseClient:
    if config.provider == "openai":
        return OpenAIClient(config)
    if config.provider == "anthropic":
        return AnthropicClient(config)
    if config.provider == "gemini":
        return GeminiClient(config)
    raise LLMError(f"Unsupported provider: {config.provider}")

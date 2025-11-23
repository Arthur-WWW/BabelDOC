from __future__ import annotations

import logging
import os
from typing import Iterable

from babeldoc.translator.translator import BaseTranslator

logger = logging.getLogger(__name__)


class PassthroughTranslator(BaseTranslator):
    """Fallback translator that simply echoes the input text."""

    name = "bridge_passthrough"

    def __init__(self, lang_in: str, lang_out: str) -> None:
        super().__init__(lang_in, lang_out, ignore_cache=True)
        self.model = "identity"

    def do_translate(self, text, rate_limit_params: dict | None = None):
        return text

    def do_llm_translate(self, text, rate_limit_params: dict | None = None):
        return text


def build_translator(lang_in: str, lang_out: str) -> BaseTranslator:
    """Factory that picks a translator implementation based on env vars.

    Supported settings:
        - DEEPBRIDGE_TRANSLATOR=openai -> uses OpenAI chat completions
        - Anything else -> passthrough translator
    """
    name = os.environ.get("DEEPBRIDGE_TRANSLATOR", "passthrough").lower()
    if name == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY missing, falling back to passthrough translator."
            )
        else:
            from babeldoc.translator.translator import OpenAITranslator

            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            base_url = os.environ.get("OPENAI_BASE_URL")
            reasoning = os.environ.get("OPENAI_REASONING")
            logger.info("Using OpenAI translator model=%s base_url=%s", model, base_url)
            return OpenAITranslator(
                lang_in=lang_in,
                lang_out=lang_out,
                model=model,
                base_url=base_url,
                api_key=api_key,
                ignore_cache=True,
                enable_json_mode_if_requested=False,
                send_dashscope_header=False,
                send_temperature=True,
                reasoning=reasoning,
            )

    logger.info("Using passthrough translator (no translation applied).")
    return PassthroughTranslator(lang_in, lang_out)


def translate_blocks(
    blocks: Iterable["MarkerBlock"], translator: BaseTranslator
) -> None:
    """Translate blocks in-place using the provided translator."""
    for block in blocks:
        text = block.text.strip()
        if not text:
            block.translation = ""
            continue
        try:
            try:
                block.translation = translator.llm_translate(text)
            except NotImplementedError:
                block.translation = translator.translate(text)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Translation failed for block %s: %s", block.block_id, exc)
            block.translation = text

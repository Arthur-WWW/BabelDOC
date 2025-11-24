from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TranslationImporter:
    """Load translated paragraphs from a text file and update the IL."""

    def import_translations(self, ordered_paragraphs: list, translation_path: Path) -> None:
        translation_path = Path(translation_path)
        if not translation_path.exists():
            raise FileNotFoundError(f"Translation file not found: {translation_path}")

        with translation_path.open("r", encoding="utf-8") as f:
            lines = [line.rstrip("\n\r") for line in f]

        if len(lines) != len(ordered_paragraphs):
            raise ValueError(
                f"Translation line count ({len(lines)}) does not match paragraphs ({len(ordered_paragraphs)})."
            )

        for meta, text in zip(ordered_paragraphs, lines):
            meta.paragraph.unicode = text

        logger.info("Imported %d translated paragraphs from %s", len(lines), translation_path)

from __future__ import annotations

import json
import logging
from pathlib import Path

from babeldoc.format.pdf.document_il import il_version_1

logger = logging.getLogger(__name__)


class TranslationExporter:
    """Dump PdfParagraph texts in assigned read order for external translation."""

    def __init__(self, output_dir: Path, base_name: str):
        self.output_dir = Path(output_dir)
        self.base_name = base_name

    def export(
        self,
        ordered_paragraphs: list,
    ) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        text_path = self.output_dir / f"{self.base_name}.translation.txt"
        index_path = self.output_dir / f"{self.base_name}.translation.index.jsonl"

        with text_path.open("w", encoding="utf-8") as text_file, index_path.open(
            "w", encoding="utf-8"
        ) as index_file:
            for line_no, meta in enumerate(ordered_paragraphs):
                content = meta.paragraph.unicode or ""
                text_file.write(content.rstrip() + "\n")
                index_entry = {
                    "line": line_no,
                    "page": meta.page_number + 1,
                    "page_index": meta.index_on_page,
                    "read_order": meta.read_order,
                    "paragraph_debug_id": getattr(meta.paragraph, "debug_id", None),
                }
                index_file.write(json.dumps(index_entry, ensure_ascii=False) + "\n")

        logger.info(
            "Exported translation text to %s (index: %s)",
            text_path,
            index_path,
        )
        return text_path, index_path

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
        marker_blocks: list | None = None,
    ) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        text_path = self.output_dir / f"{self.base_name}.translation.txt"
        index_path = self.output_dir / f"{self.base_name}.translation.index.jsonl"
        debug_path = self.output_dir / f"{self.base_name}.debug.txt"

        # Create a map for quick marker block lookup
        marker_map = {}
        if marker_blocks:
            for block in marker_blocks:
                marker_map[block.order] = block

        with text_path.open("w", encoding="utf-8") as text_file, index_path.open(
            "w", encoding="utf-8"
        ) as index_file, debug_path.open("w", encoding="utf-8") as debug_file:
            for line_no, meta in enumerate(ordered_paragraphs):
                content = meta.paragraph.unicode or ""
                text_file.write(content.rstrip() + "\n")
                
                # Write to debug file with clear delimiters
                debug_file.write(f"--- [Page {meta.page_number + 1} | Order {meta.read_order}] ---\n")
                
                # Add Marker Block Info if available
                if meta.read_order is not None and meta.read_order in marker_map:
                    block = marker_map[meta.read_order]
                    debug_file.write(f"[Marker Block: {block.block_type} | ID: {block.block_id}]\n")
                    debug_file.write(f"[Marker Text]: {block.text[:100]}...\n") # Preview first 100 chars
                
                debug_file.write(f"[BabelDOC]: {content.rstrip()}\n\n")

                index_entry = {
                    "line": line_no,
                    "page": meta.page_number + 1,
                    "page_index": meta.index_on_page,
                    "read_order": meta.read_order,
                    "paragraph_debug_id": getattr(meta.paragraph, "debug_id", None),
                }
                index_file.write(json.dumps(index_entry, ensure_ascii=False) + "\n")

        logger.info(
            "Exported translation text to %s (index: %s, debug: %s)",
            text_path,
            index_path,
            debug_path,
        )
        return text_path, index_path

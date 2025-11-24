from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
import json

from marker.config.parser import ConfigParser
from marker.models import create_model_dict

logger = logging.getLogger(__name__)

# Marker exposes many structural block types. Only these types should be
# rendered as text inside BabelDOC.
TEXT_BLOCK_TYPES = {
    "Caption",
    "Code",
    "Equation",
    "Footnote",
    "Handwriting",
    "ListItem",
    "PageFooter",
    "PageHeader",
    "Reference",
    "SectionHeader",
    "TableCell",
    "TableOfContents",
    "Text",
}

_MARKER_MODELS: dict[str, Any] | None = None


def _get_marker_models() -> dict[str, Any]:
    global _MARKER_MODELS
    if _MARKER_MODELS is None:
        logger.info("Loading Marker models ...")
        _MARKER_MODELS = create_model_dict()
    return _MARKER_MODELS


@dataclass(slots=True)
class MarkerBlock:
    """Lightweight representation of a textual block emitted by Marker."""

    order: int
    block_id: str
    block_type: str
    page_index: int
    bbox: tuple[float, float, float, float]
    text: str
    translation: str | None = None

    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())


class _HTMLTextExtractor(HTMLParser):
    """Simplistic HTML→text converter that preserves line breaks."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str]]) -> None:
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")
        elif tag == "br":
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "li"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if data:
            self._parts.append(data)

    def get_text(self) -> str:
        text = unescape("".join(self._parts))
        # Collapse excessive blank lines while keeping deliberate line breaks.
        lines = [line.strip() for line in text.splitlines()]
        normalized = "\n".join(line for line in lines if line)
        return normalized.strip()


class MarkerJSONExtractor:
    """Runs Marker in-process and returns flattened textual blocks."""

    def __init__(self, working_dir: Path) -> None:
        self.working_dir = working_dir
        self.marker_output_dir = working_dir / "marker-output"
        self.marker_output_dir.mkdir(parents=True, exist_ok=True)

    def extract_blocks(
        self, pdf_path: str, page_indices: list[int] | None = None
    ) -> list[MarkerBlock]:
        """Run Marker and flatten the JSON output into MarkerBlock objects."""
        logger.info("Running Marker on %s", pdf_path)
        json_output = self._run_marker(pdf_path, page_indices)
        blocks = self._flatten_blocks(json_output)
        logger.info("Marker extracted %d candidate blocks", len(blocks))
        return [block for block in blocks if block.has_text()]

    # ---- internal helpers -------------------------------------------------
    def _run_marker(
        self, pdf_path: str, page_indices: list[int] | None = None
    ) -> dict[str, Any]:
        cli_options = {
            "output_dir": str(self.marker_output_dir),
            "output_format": "json",
            "disable_multiprocessing": True,
            "use_llm": False,
            # Allow users to set a custom Marker config through env if needed.
            "config_json": os.environ.get("DEEPBRIDGE_MARKER_CONFIG"),
        }
        if page_indices:
            cli_options["page_range"] = ",".join(str(i) for i in page_indices)
        config_parser = ConfigParser(cli_options)
        converter_cls = config_parser.get_converter_cls()
        converter = converter_cls(
            config=config_parser.generate_config_dict(),
            artifact_dict=_get_marker_models(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        rendered = converter(pdf_path)
        if hasattr(rendered, "model_dump"):
            raw_data = rendered.model_dump()
        elif isinstance(rendered, dict):
            raw_data = rendered
        else:
            raise RuntimeError("Unexpected Marker output type: %s" % type(rendered))

        json_path = self.marker_output_dir / f"{Path(pdf_path).stem}.marker.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)

        return raw_data

    def _flatten_blocks(self, json_data: dict[str, Any]) -> list[MarkerBlock]:
        blocks: list[MarkerBlock] = []
        for page in json_data.get("children", []) or []:
            try:
                page_index = self._parse_page_index(page.get("id", ""))
            except ValueError:
                logger.debug("Skip page without index: %s", page.get("id"))
                continue
            self._walk_block_tree(page.get("children") or [], page_index, blocks)
        return blocks

    def _walk_block_tree(
        self,
        nodes: Iterable[dict[str, Any]],
        page_index: int,
        blocks: list[MarkerBlock],
    ) -> None:
        for node in nodes:
            children = node.get("children")
            if children:
                self._walk_block_tree(children, page_index, blocks)
                continue

            block_type = node.get("block_type")
            if block_type not in TEXT_BLOCK_TYPES:
                continue

            html_content = node.get("html", "")
            text = self._html_to_text(html_content)
            if not text:
                continue
            bbox = node.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            order = len(blocks) + 1
            blocks.append(
                MarkerBlock(
                    order=order,
                    block_id=node.get("id", f"node-{order}"),
                    block_type=block_type,
                    page_index=page_index,
                    bbox=(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                    text=text,
                )
            )

    def _parse_page_index(self, block_id: str) -> int:
        # IDs look like "/page/0/Text/123"
        parts = block_id.strip("/").split("/")
        if len(parts) < 2 or parts[0] != "page":
            raise ValueError(f"Unknown block id format: {block_id}")
        return int(parts[1])

    def _html_to_text(self, html: str) -> str:
        if not html:
            return ""
        parser = _HTMLTextExtractor()
        try:
            parser.feed(html)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to parse HTML block: %s", exc)
            return ""
        text = parser.get_text()
        parser.close()
        return text

from __future__ import annotations

import logging
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from babeldoc.format.pdf.document_il import Box
from babeldoc.format.pdf.document_il import PdfParagraph
from babeldoc.format.pdf.document_il import PdfParagraphComposition
from babeldoc.format.pdf.document_il import PdfSameStyleUnicodeCharacters
from babeldoc.format.pdf.document_il import PdfStyle
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.translation_config import TranslationConfig

from .marker_adapter import MarkerBlock

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ParagraphFactoryConfig:
    min_font_size: float = 8.0
    max_font_size: float = 28.0
    default_font_id: str = "base"


class ParagraphBuilder:
    """Convert Marker blocks into BabelDOC PdfParagraph objects."""

    def __init__(
        self,
        translation_config: TranslationConfig,
        factory_config: ParagraphFactoryConfig | None = None,
    ) -> None:
        self.translation_config = translation_config
        self.factory_config = factory_config or ParagraphFactoryConfig()

    def populate_document(
        self, document: il_version_1.Document, blocks: Iterable[MarkerBlock]
    ) -> None:
        blocks_by_page: dict[int, list[MarkerBlock]] = defaultdict(list)
        for block in blocks:
            if not block.translation:
                block.translation = block.text
            if block.translation:
                blocks_by_page[block.page_index].append(block)

        for page in document.page:
            page_blocks = blocks_by_page.get(page.page_number, [])
            if not page_blocks:
                continue
            page_blocks.sort(key=lambda b: b.order)
            updated = self._apply_blocks_to_page(page, page_blocks)
            page.pdf_paragraph = updated
            page.pdf_character = []

    # ---- helpers ---------------------------------------------------------
    def _convert_box(
        self, block: MarkerBlock, page: il_version_1.Page
    ) -> Box | None:
        if not page.cropbox or not page.cropbox.box:
            return None
        x1, y1, x2, y2 = block.bbox
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        crop = page.cropbox.box
        height = max(crop.y2 - crop.y, 1e-3)
        pdf_x1 = crop.x + x1
        pdf_x2 = crop.x + x2
        pdf_y1 = crop.y + (height - y2)
        pdf_y2 = crop.y + (height - y1)

        pdf_x1, pdf_x2 = self._clamp(pdf_x1, pdf_x2, crop.x, crop.x2)
        pdf_y1, pdf_y2 = self._clamp(pdf_y1, pdf_y2, crop.y, crop.y2)

        if pdf_x2 <= pdf_x1 or pdf_y2 <= pdf_y1:
            logger.debug("Degenerated bbox for block %s: %s", block.block_id, block.bbox)
            return None
        return Box(x=pdf_x1, y=pdf_y1, x2=pdf_x2, y2=pdf_y2)

    def _apply_blocks_to_page(
        self,
        page: il_version_1.Page,
        blocks: list[MarkerBlock],
    ) -> list[il_version_1.PdfParagraph]:
        paragraphs = list(page.pdf_paragraph or [])
        used_indices: set[int] = set()
        new_order: list[il_version_1.PdfParagraph] = []

        for block in blocks:
            target_box = self._convert_box(block, page)
            if target_box is None:
                continue
            match_idx = self._match_paragraph(paragraphs, target_box, used_indices)
            if match_idx is not None:
                para = paragraphs[match_idx]
                self._update_paragraph_with_block(para, block, target_box)
                used_indices.add(match_idx)
                new_order.append(para)
            else:
                new_para = self._create_new_paragraph(block, target_box)
                new_order.append(new_para)

        for idx, para in enumerate(paragraphs):
            if idx not in used_indices:
                new_order.append(para)
        return new_order

    def _match_paragraph(
        self,
        paragraphs: list[il_version_1.PdfParagraph],
        target_box: Box,
        used_indices: set[int],
        iou_threshold: float = 0.3,
    ) -> int | None:
        best_idx = None
        best_score = 0.0
        for idx, para in enumerate(paragraphs):
            if idx in used_indices or not para.box:
                continue
            score = self._overlap_ratio(para.box, target_box)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None and best_score >= 0.1:
            return best_idx

        best_idx = None
        best_dist = float("inf")
        for idx, para in enumerate(paragraphs):
            if idx in used_indices or not para.box:
                continue
            dist = self._center_distance(para.box, target_box)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None and best_dist < 15:  # roughly 15pt tolerance
            return best_idx
        return None

    def _update_paragraph_with_block(
        self,
        paragraph: il_version_1.PdfParagraph,
        block: MarkerBlock,
        target_box: Box,
    ):
        normalized_text = block.translation.replace("\r", "").replace("\n", " ")
        chars = self._collect_characters(paragraph)
        if not chars or len(normalized_text) > len(chars):
            self._replace_paragraph_with_same_style(paragraph, block, target_box)
            return

        paragraph.box = target_box
        paragraph.unicode = block.translation
        for idx, char in enumerate(chars):
            if idx < len(normalized_text):
                char.char_unicode = normalized_text[idx]
            else:
                char.char_unicode = " "
        paragraph.render_order = block.order
        paragraph.first_line_indent = False
        paragraph.vertical = False
        paragraph.layout_label = block.block_type
        paragraph.debug_id = block.block_id

    def _create_new_paragraph(
        self,
        block: MarkerBlock,
        target_box: Box,
    ) -> il_version_1.PdfParagraph:
        style = self._default_style(target_box)
        same_style = PdfSameStyleUnicodeCharacters(
            unicode=block.translation,
            pdf_style=style,
        )
        return il_version_1.PdfParagraph(
            first_line_indent=False,
            box=target_box,
            vertical=False,
            pdf_style=style,
            unicode=block.translation,
            pdf_paragraph_composition=[
                PdfParagraphComposition(
                    pdf_same_style_unicode_characters=same_style,
                )
            ],
            xobj_id=-1,
            layout_label=block.block_type,
            debug_id=block.block_id,
            render_order=block.order,
        )

    def _default_style(self, box: Box | None = None) -> PdfStyle:
        font_size = (
            self._estimate_font_size(None, box)
            if box is not None
            else self.factory_config.min_font_size
        )
        return PdfStyle(
            font_id=self.factory_config.default_font_id,
            font_size=font_size,
            graphic_state=il_version_1.GraphicState(),
        )

    def _extract_style_from_paragraph(
        self, paragraph: il_version_1.PdfParagraph
    ) -> PdfStyle:
        styles: list[PdfStyle] = []
        for comp in paragraph.pdf_paragraph_composition:
            if comp.pdf_line:
                for char in comp.pdf_line.pdf_character:
                    if char.pdf_style:
                        styles.append(char.pdf_style)
            elif comp.pdf_same_style_characters and comp.pdf_same_style_characters.pdf_style:
                styles.append(comp.pdf_same_style_characters.pdf_style)
            elif (
                comp.pdf_same_style_unicode_characters
                and comp.pdf_same_style_unicode_characters.pdf_style
            ):
                styles.append(comp.pdf_same_style_unicode_characters.pdf_style)

        if styles:
            style = copy.deepcopy(styles[0])
            if style.font_id is None:
                for s in styles:
                    if s.font_id:
                        style.font_id = s.font_id
                        break
            if style.font_size is None:
                for s in styles:
                    if s.font_size:
                        style.font_size = s.font_size
                        break
            if style.font_id is None:
                style.font_id = self.factory_config.default_font_id
            if style.font_size is None:
                style.font_size = self.factory_config.min_font_size
            return style

        return self._default_style(paragraph.box)

    def _collect_characters(
        self, paragraph: il_version_1.PdfParagraph
    ) -> list[il_version_1.PdfCharacter]:
        chars: list[il_version_1.PdfCharacter] = []
        for comp in paragraph.pdf_paragraph_composition:
            if comp.pdf_line and comp.pdf_line.pdf_character:
                chars.extend(comp.pdf_line.pdf_character)
            elif comp.pdf_character:
                chars.append(comp.pdf_character)
            elif (
                comp.pdf_same_style_characters
                and comp.pdf_same_style_characters.pdf_character
            ):
                chars.extend(comp.pdf_same_style_characters.pdf_character)
        return chars

    def _replace_paragraph_with_same_style(
        self,
        paragraph: il_version_1.PdfParagraph,
        block: MarkerBlock,
        target_box: Box,
    ):
        style = self._extract_style_from_paragraph(paragraph)
        same_style = PdfSameStyleUnicodeCharacters(
            unicode=block.translation,
            pdf_style=style,
        )
        paragraph.box = target_box
        paragraph.unicode = block.translation
        paragraph.pdf_paragraph_composition = [
            PdfParagraphComposition(
                pdf_same_style_unicode_characters=same_style,
            )
        ]
        paragraph.render_order = block.order
        paragraph.first_line_indent = False
        paragraph.vertical = False
        paragraph.layout_label = block.block_type
        paragraph.debug_id = block.block_id

    def _calc_iou(self, box_a: Box, box_b: Box) -> float:
        x_left = max(box_a.x, box_b.x)
        y_top = max(box_a.y, box_b.y)
        x_right = min(box_a.x2, box_b.x2)
        y_bottom = min(box_a.y2, box_b.y2)
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area_a = (box_a.x2 - box_a.x) * (box_a.y2 - box_a.y)
        area_b = (box_b.x2 - box_b.x) * (box_b.y2 - box_b.y)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _overlap_ratio(self, box_a: Box, box_b: Box) -> float:
        x_left = max(box_a.x, box_b.x)
        y_top = max(box_a.y, box_b.y)
        x_right = min(box_a.x2, box_b.x2)
        y_bottom = min(box_a.y2, box_b.y2)
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area_a = (box_a.x2 - box_a.x) * (box_a.y2 - box_a.y)
        area_b = (box_b.x2 - box_b.x) * (box_b.y2 - box_b.y)
        min_area = min(area_a, area_b)
        return intersection / min_area if min_area > 0 else 0.0

    def _center_distance(self, box_a: Box, box_b: Box) -> float:
        ax = (box_a.x + box_a.x2) / 2
        ay = (box_a.y + box_a.y2) / 2
        bx = (box_b.x + box_b.x2) / 2
        by = (box_b.y + box_b.y2) / 2
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def _clamp(self, a: float, b: float, lower: float, upper: float) -> tuple[float, float]:
        a = max(lower, min(a, upper))
        b = max(lower, min(b, upper))
        return (a, b)

    def _estimate_font_size(self, block: MarkerBlock | None, box: Box | None) -> float:
        if not box:
            return self.factory_config.min_font_size
        height = max(box.y2 - box.y, 1.0)
        line_count = (
            block.translation.count("\n") + 1 if block and block.translation else 1
        )
        approx_line_height = height / max(line_count, 1)
        font_size = approx_line_height * 0.9
        return float(
            max(
                self.factory_config.min_font_size,
                min(font_size, self.factory_config.max_font_size),
            )
        )

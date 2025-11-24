from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable
from difflib import SequenceMatcher
from bs4 import BeautifulSoup

from babeldoc.format.pdf.document_il import Box
from babeldoc.format.pdf.document_il import il_version_1

from .marker_adapter import MarkerBlock

logger = logging.getLogger(__name__)


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    # Replace newlines with spaces and collapse repeated whitespace.
    return " ".join(text.replace("\u00A0", " ").split())


@dataclass(slots=True)
class ParagraphMeta:
    paragraph: il_version_1.PdfParagraph
    page_number: int
    index_on_page: int
    normalized_text: str
    box: Box | None
    original_sort_key: tuple[int, int]
    read_order: int | None = None


class ParagraphOrderer:
    """Assign reading-order indices to PdfParagraphs using Marker blocks."""

    similarity_threshold: float = 0.6
    similarity_tolerance: float = 0.02
    overlap_threshold: float = 0.2

    def assign_read_order(
        self,
        document: il_version_1.Document,
        blocks: Iterable[MarkerBlock],
        allowed_pages: set[int] | None = None,
    ) -> list[ParagraphMeta]:
        metas = self._collect_paragraphs(document, allowed_pages)
        metas_by_page: dict[int, list[ParagraphMeta]] = {}
        for meta in metas:
            metas_by_page.setdefault(meta.page_number, []).append(meta)

        order_counter = 0
        for block in blocks:
            normalized_block_text = normalize_text(self._extract_html_text(block))
            if not normalized_block_text:
                continue
            candidates = metas_by_page.get(block.page_index, [])
            unused = [c for c in candidates if c.read_order is None]
            if not unused:
                continue

            overlap_hits = self._collect_overlaps(block, unused)
            if overlap_hits:
                for meta in overlap_hits:
                    meta.read_order = order_counter
                    order_counter += 1
                continue

            match = self._match_by_text(block, normalized_block_text, unused)
            if match is None:
                logger.debug(
                    "No paragraph matched Marker block %s (page %s).",
                    block.block_id,
                    block.page_index,
                )
                continue
            match.read_order = order_counter
            order_counter += 1

        unmatched = [m for m in metas if m.read_order is None]
        if unmatched:
            logger.info(
                "Paragraphs without Marker matches: %d",
                len(unmatched),
            )
            for meta in unmatched[:10]:
                snippet = (meta.paragraph.unicode or "")[:120].replace("\n", " ")
                logger.debug(
                    "Unmatched paragraph page=%s idx=%s text=%r",
                    meta.page_number + 1,
                    meta.index_on_page,
                    snippet,
                )
        return metas

    def _collect_paragraphs(
        self, document: il_version_1.Document, allowed_pages: set[int] | None
    ) -> list[ParagraphMeta]:
        metas: list[ParagraphMeta] = []
        for page in document.page:
            if allowed_pages is not None and page.page_number not in allowed_pages:
                continue
            for idx, paragraph in enumerate(page.pdf_paragraph or []):
                normalized = normalize_text(paragraph.unicode)
                meta = ParagraphMeta(
                    paragraph=paragraph,
                    page_number=page.page_number or 0,
                    index_on_page=idx,
                    normalized_text=normalized,
                    box=paragraph.box,
                    original_sort_key=(page.page_number or 0, idx),
                )
                metas.append(meta)
        return metas

    def _collect_overlaps(
        self,
        block: MarkerBlock,
        candidates: list[ParagraphMeta],
    ) -> list[ParagraphMeta]:
        overlaps: list[tuple[float, ParagraphMeta]] = []
        for c in candidates:
            if not c.box or not c.normalized_text:
                continue
            ratio = self._overlap_ratio(c.box, block.bbox)
            if ratio >= self.overlap_threshold:
                overlaps.append((ratio, c))
        if not overlaps:
            return []
        overlaps.sort(key=lambda x: (-x[0], x[1].box.y if x[1].box else 0))
        return [c for _, c in overlaps]

    def _match_by_text(
        self,
        block: MarkerBlock,
        normalized_block_text: str,
        candidates: list[ParagraphMeta],
    ) -> ParagraphMeta | None:
        # Step 1: exact normalized text match.
        exact_matches = [
            c for c in candidates if c.normalized_text == normalized_block_text
        ]
        if exact_matches:
            return self._select_best_by_bbox(block, exact_matches)

        # Step 2: substring or containment match.
        contains_matches: list[ParagraphMeta] = []
        for c in candidates:
            if not c.normalized_text:
                continue
            if normalized_block_text in c.normalized_text or c.normalized_text in normalized_block_text:
                contains_matches.append(c)
        if contains_matches:
            return self._select_best_by_bbox(block, contains_matches)

        # Step 3: similarity fallback.
        similarity_hits: list[tuple[ParagraphMeta, float]] = []
        for c in candidates:
            if not c.normalized_text:
                continue
            ratio = SequenceMatcher(None, c.normalized_text, normalized_block_text).ratio()
            if ratio >= self.similarity_threshold:
                similarity_hits.append((c, ratio))
        if similarity_hits:
            similarity_hits.sort(key=lambda x: x[1], reverse=True)
            best_ratio = similarity_hits[0][1]
            top_candidates = [
                c for c, ratio in similarity_hits if best_ratio - ratio <= self.similarity_tolerance
            ]
            return self._select_best_by_bbox(block, top_candidates, require_overlap=False)

        # Step 3: fallback to highest IoU.
        return self._select_best_by_bbox(block, candidates, require_overlap=False)

    def _select_best_by_bbox(
        self,
        block: MarkerBlock,
        candidates: list[ParagraphMeta],
        require_overlap: bool = True,
    ) -> ParagraphMeta | None:
        best_meta = None
        best_score = -1.0
        for candidate in candidates:
            if not candidate.box:
                continue
            score = self._overlap_ratio(candidate.box, block.bbox)
            if require_overlap and score <= 0:
                continue
            if score > best_score:
                best_score = score
                best_meta = candidate
        if best_meta:
            return best_meta
        if require_overlap:
            return None
        # If we reach here, bbox info was unusable; return first candidate.
        return candidates[0]

    def _overlap_ratio(self, para_box: Box, block_box: tuple[float, float, float, float]) -> float:
        bx1, by1, bx2, by2 = block_box
        x_left = max(para_box.x, bx1)
        y_top = max(para_box.y, by1)
        x_right = min(para_box.x2, bx2)
        y_bottom = min(para_box.y2, by2)
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        para_area = max((para_box.x2 - para_box.x) * (para_box.y2 - para_box.y), 1e-6)
        block_area = max((bx2 - bx1) * (by2 - by1), 1e-6)
        min_area = min(para_area, block_area)
        return intersection / min_area

    def _extract_html_text(self, block: MarkerBlock) -> str:
        if not block or not block.text:
            return ""
        soup = BeautifulSoup(block.text, "html.parser")
        return soup.get_text(separator=" ")

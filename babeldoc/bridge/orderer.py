from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from babeldoc.format.pdf.document_il import Box
from babeldoc.format.pdf.document_il import il_version_1

from .marker_adapter import MarkerBlock

logger = logging.getLogger(__name__)


def normalize_text_for_matching(text: str | None) -> str:
    """
    Aggressively normalize text for matching:
    - Remove all whitespace (spaces, newlines, tabs)
    - This handles differences in word wrapping and spacing between engines.
    """
    if not text:
        return ""
    return "".join(text.split())


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
    """
    Assign reading-order indices to PdfParagraphs using Marker blocks.
    Strategy: Global Text Alignment.
    """

    def assign_read_order(
        self,
        document: il_version_1.Document,
        blocks: Iterable[MarkerBlock],
        allowed_pages: set[int] | None = None,
    ) -> list[ParagraphMeta]:
        # 1. Collect BabelDOC paragraphs
        metas = self._collect_paragraphs(document, allowed_pages)
        
        # 2. Build Marker Reference Sequence
        # We need a list of blocks to access by index
        block_list = sorted(blocks, key=lambda b: b.order)
        
        # Construct the "Global Text" from Marker
        # We map each character range in the global text back to a block index.
        marker_full_text = ""
        # List of (start_index, end_index, block_order, page_index)
        marker_text_map: list[tuple[int, int, int, int]] = []
        
        current_idx = 0
        for block in block_list:
            norm_text = normalize_text_for_matching(block.text)
            if not norm_text:
                continue
            
            start = current_idx
            end = current_idx + len(norm_text)
            marker_full_text += norm_text
            marker_text_map.append((start, end, block.order, block.page_index))
            current_idx = end
            
        logger.info(
            "Built Marker reference text: %d chars from %d blocks", 
            len(marker_full_text), 
            len(marker_text_map)
        )

        # 3. Align BabelDOC Paragraphs
        matched_count = 0
        
        for meta in metas:
            if not meta.normalized_text:
                continue
                
            # --- Strategy A: Exact Text Match ---
            search_start = 0
            candidates = []
            
            while True:
                idx = marker_full_text.find(meta.normalized_text, search_start)
                if idx == -1:
                    break
                
                match_center = idx + len(meta.normalized_text) // 2
                matched_block_order = None
                matched_page_index = -1
                
                for m_start, m_end, m_order, m_page in marker_text_map:
                    if m_start <= match_center < m_end:
                        matched_block_order = m_order
                        matched_page_index = m_page
                        break
                
                if matched_block_order is not None:
                    candidates.append((matched_block_order, matched_page_index))
                
                search_start = idx + 1
            
            best_order = None
            
            if candidates:
                # Disambiguate by Page Number
                page_matches = [c for c in candidates if c[1] == meta.page_number]
                if page_matches:
                    if len(page_matches) == 1:
                        best_order = page_matches[0][0]
                    else:
                        # Multiple matches on same page (repeated text).
                        # Use Geometric Disambiguation
                        best_order = self._disambiguate_by_geometry(meta, page_matches, block_list)
                else:
                    # Fallback: Pick closest page
                    candidates.sort(key=lambda c: abs(c[1] - meta.page_number))
                    best_order = candidates[0][0]
            
            # --- Strategy B: Fuzzy Text Match (if Exact failed) ---
            if best_order is None:
                # Only search blocks on the same page to save time
                page_blocks = [b for b in block_list if b.page_index == meta.page_number]
                best_fuzzy_score = 0.0
                best_fuzzy_order = None
                
                from difflib import SequenceMatcher
                
                for block in page_blocks:
                    block_norm = normalize_text_for_matching(block.text)
                    if not block_norm: continue
                    
                    # Check if meta text is roughly in block text
                    # Quick check: is it a substring with minor errors?
                    if len(meta.normalized_text) < len(block_norm):
                        # Use 'real_quick_ratio' first
                        matcher = SequenceMatcher(None, meta.normalized_text, block_norm)
                        if matcher.real_quick_ratio() > 0.5:
                            # Look for best matching block
                            match = matcher.find_longest_match(0, len(meta.normalized_text), 0, len(block_norm))
                            if match.size > len(meta.normalized_text) * 0.8: # 80% match
                                best_fuzzy_order = block.order
                                break
                
                if best_fuzzy_order is not None:
                    best_order = best_fuzzy_order

            # --- Strategy C: Geometric Fallback (if Text failed) ---
            if best_order is None:
                 # Find the Marker block that best overlaps with this paragraph
                 # Only consider blocks on the same page
                 page_blocks = [b for b in block_list if b.page_index == meta.page_number]
                 best_ios = 0.0
                 best_geo_order = None
                 
                 for block in page_blocks:
                     ios = self._calculate_ios(meta.box, block.bbox)
                     if ios > best_ios:
                         best_ios = ios
                         best_geo_order = block.order
                 
                 # Threshold for geometric match: 50% of paragraph must be inside block
                 if best_ios > 0.5:
                     best_geo_order = best_geo_order
                     best_order = best_geo_order
                     logger.debug(f"Geometric match: IoS={best_ios:.2f} for '{meta.normalized_text[:20]}...' -> Order {best_geo_order}")
                 else:
                     if meta.box:
                         logger.debug(f"Geometric failed: Best IoS={best_ios:.2f} for '{meta.normalized_text[:20]}...' at {meta.box}")

            if best_order is not None:
                meta.read_order = best_order
                matched_count += 1

        logger.info(
            "Assigned read order to %d/%d paragraphs (%.1f%%)",
            matched_count,
            len(metas),
            (matched_count / len(metas) * 100) if metas else 0
        )

        return metas

    def _disambiguate_by_geometry(self, meta: ParagraphMeta, candidates: list[tuple[int, int]], block_list: list[MarkerBlock]) -> int:
        """
        If text appears multiple times on the same page, pick the instance 
        whose Marker block is closest to the BabelDOC paragraph.
        """
        best_order = candidates[0][0]
        best_score = -1.0
        
        for order, _ in candidates:
            if 0 < order <= len(block_list):
                block = block_list[order-1]
                if block.order == order:
                    # Use Intersection over Self (IoS) to favor containment
                    score = self._calculate_ios(meta.box, block.bbox)
                    if score > best_score:
                        best_score = score
                        best_order = order
        
        return best_order

    def _calculate_ios(self, box1: Box | None, box2: tuple[float, float, float, float]) -> float:
        """
        Calculate Intersection over Self (Area of Box1).
        This is better than IoU when Box1 (paragraph) is much smaller than Box2 (block).
        Returns fraction of Box1 that is inside Box2.
        """
        if not box1: return 0.0
        
        # BabelDOC Box: x, y, x2, y2
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.x, box1.y, box1.x2, box1.y2
        
        # Marker Box: x1, y1, x2, y2
        b2_x1, b2_y1, b2_x2, b2_y2 = box2
        
        # Intersection
        x_left = max(b1_x1, b2_x1)
        y_bottom = max(b1_y1, b2_y1)
        x_right = min(b1_x2, b2_x2)
        y_top = min(b1_y2, b2_y2)
        
        if x_right < x_left or y_top < y_bottom:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_top - y_bottom)
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        
        if b1_area <= 1e-6: return 0.0
        
        return intersection_area / b1_area

    def _collect_paragraphs(
        self, document: il_version_1.Document, allowed_pages: set[int] | None
    ) -> list[ParagraphMeta]:
        metas: list[ParagraphMeta] = []
        for page in document.page:
            if allowed_pages is not None and page.page_number not in allowed_pages:
                continue
            for idx, paragraph in enumerate(page.pdf_paragraph or []):
                # Use the aggressive normalization
                normalized = normalize_text_for_matching(paragraph.unicode)
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


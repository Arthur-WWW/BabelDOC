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
                
            # Search for the paragraph text in the global marker text
            # Strategy: Find all occurrences, then filter by Page Number constraint
            
            search_start = 0
            candidates = []
            
            while True:
                idx = marker_full_text.find(meta.normalized_text, search_start)
                if idx == -1:
                    break
                
                # Found a match at idx. Determine which block it belongs to.
                # We use the center of the match to decide ownership if it spans blocks
                match_center = idx + len(meta.normalized_text) // 2
                
                matched_block_order = None
                matched_page_index = -1
                
                # Binary search or linear scan (linear is fine for <10k blocks)
                # Optimization: The map is sorted by start_index
                for m_start, m_end, m_order, m_page in marker_text_map:
                    if m_start <= match_center < m_end:
                        matched_block_order = m_order
                        matched_page_index = m_page
                        break
                
                if matched_block_order is not None:
                    candidates.append((matched_block_order, matched_page_index))
                
                search_start = idx + 1
            
            # Select the best candidate
            best_order = None
            
            if not candidates:
                # No exact match found. 
                # TODO: Implement fuzzy matching fallback here if needed.
                # For now, we log and skip.
                pass
            elif len(candidates) == 1:
                # Unique match
                best_order = candidates[0][0]
            else:
                # Multiple matches. Disambiguate by Page Number.
                # BabelDOC page_number is 0-based in our meta (we fixed it in _collect)
                # Marker page_index is 0-based.
                
                # Filter by exact page match
                page_matches = [c for c in candidates if c[1] == meta.page_number]
                
                if page_matches:
                    # If multiple matches on the same page, this is tricky (repeated text on same page).
                    # We could use relative order of previous paragraphs, but for now take the first.
                    best_order = page_matches[0][0]
                else:
                    # No match on the correct page? This implies a page mismatch between engines.
                    # Fallback: Pick the candidate on the closest page.
                    candidates.sort(key=lambda c: abs(c[1] - meta.page_number))
                    best_order = candidates[0][0]
            
            if best_order is not None:
                meta.read_order = best_order
                matched_count += 1

        logger.info(
            "Assigned read order to %d/%d paragraphs (%.1f%%)",
            matched_count,
            len(metas),
            (matched_count / len(metas) * 100) if metas else 0
        )

        # 4. Interpolation (Optional but recommended)
        # If we have [Match(10), Unmatched, Match(12)], the Unmatched is likely 11.
        # Simple forward-fill or linear interpolation can help.
        # For now, we leave them None, which means they sort to the end (or original position).
        
        return metas

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


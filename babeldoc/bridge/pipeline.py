from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Tuple

import pymupdf

from babeldoc.docvision.base_doclayout import DocLayoutModel
from babeldoc.format.pdf.document_il.backend.pdf_creater import PDFCreater
from babeldoc.format.pdf.document_il.frontend.il_creater import ILCreater
from babeldoc.format.pdf.document_il.midend.typesetting import Typesetting
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.high_level import (
    TRANSLATE_STAGES,
    add_metadata,
    fix_cmap,
    fix_filter,
    fix_media_box,
    fix_null_page_content,
    fix_null_xref,
    safe_save,
    start_parse_il,
)
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.format.pdf.translation_config import WatermarkOutputMode
from babeldoc.progress_monitor import ProgressMonitor
from babeldoc.const import close_process_pool

from .marker_adapter import MarkerJSONExtractor
from .orderer import ParagraphOrderer
from .translation_exporter import TranslationExporter
from .translation_importer import TranslationImporter
from .translator import PassthroughTranslator, build_translator, translate_blocks

logger = logging.getLogger(__name__)


class NoOpDocLayoutModel(DocLayoutModel):
    """Placeholder layout model to avoid loading the heavy YOLO weights."""

    @property
    def stride(self) -> int:
        return 32

    def handle_document(self, *args, **kwargs):
        if False:  # pragma: no cover - generator requirement
            yield None


def run_hybrid_pipeline(
    input_pdf: str,
    output_pdf: str | None = None,
    pages: str | None = None,
    enable_dual: bool = False,
    translation_input: str | None = None,
) -> Path:
    """Entry point invoked by main.py."""
    output_path = Path(output_pdf) if output_pdf else Path(input_pdf).with_suffix(
        ".translated.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lang_in = os.environ.get("DEEPBRIDGE_LANG_IN", "auto")
    lang_out = os.environ.get("DEEPBRIDGE_LANG_OUT", "zh-cn")

    progress_monitor = ProgressMonitor(TRANSLATE_STAGES)
    should_translate = _should_run_translation()
    translator = (
        build_translator(lang_in, lang_out)
        if should_translate
        else PassthroughTranslator(lang_in, lang_out)
    )
    translation_config = TranslationConfig(
        translator=translator,
        input_file=input_pdf,
        lang_in=lang_in,
        lang_out=lang_out,
        doc_layout_model=NoOpDocLayoutModel(),
        output_dir=output_path.parent,
        working_dir=output_path.parent / ".deepbridge-cache",
        debug=False,
        no_dual=not enable_dual,
        watermark_output_mode=WatermarkOutputMode.NoWatermark,
        progress_monitor=progress_monitor,
        skip_scanned_detection=True,
        auto_extract_glossary=False,
        skip_translation=True,
        pages=pages,
        only_include_translated_page=bool(pages),
    )

    docs = None
    temp_pdf_path = None
    doc_pdf2zh = None
    selected_indices = _get_zero_based_page_list(translation_config)

    try:
        docs, mediabox_data, temp_pdf_path, doc_pdf2zh = _parse_pdf_to_il(
            translation_config
        )
        if not docs.page:
            raise RuntimeError("Parsed IL contains no pages.")

        marker = MarkerJSONExtractor(Path(translation_config.working_dir))
        blocks = marker.extract_blocks(input_pdf, selected_indices)
        if not blocks:
            raise RuntimeError("Marker did not return any textual blocks.")
        marker_json_path = (
            marker.marker_output_dir / f"{Path(input_pdf).stem}.marker.json"
        )
        logger.info("Marker JSON saved to %s", marker_json_path)
        if should_translate:
            translate_blocks(blocks, translator)
        else:
            for block in blocks:
                block.translation = block.text

        if not translation_config.skip_scanned_detection:
            from babeldoc.format.pdf.document_il.midend.detect_scanned_file import (
                DetectScannedFile,
            )

            DetectScannedFile(translation_config).process(
                docs, temp_pdf_path, mediabox_data
            )

        from babeldoc.format.pdf.document_il.midend.layout_parser import LayoutParser

        docs = LayoutParser(translation_config).process(docs, doc_pdf2zh)
        close_process_pool()

        if translation_config.table_model:
            from babeldoc.format.pdf.document_il.midend.table_parser import TableParser

            docs = TableParser(translation_config).process(docs, doc_pdf2zh)

        from babeldoc.format.pdf.document_il.midend.paragraph_finder import (
            ParagraphFinder,
        )

        ParagraphFinder(translation_config).process(docs)

        from babeldoc.format.pdf.document_il.midend.styles_and_formulas import (
            StylesAndFormulas,
        )

        StylesAndFormulas(translation_config).process(docs)

        orderer = ParagraphOrderer()
        allowed_pages = set(selected_indices) if selected_indices else None
        paragraph_metas = orderer.assign_read_order(
            docs, blocks, allowed_pages=allowed_pages
        )
        ordered_metas = sorted(
            paragraph_metas,
            key=lambda m: (
                0 if m.read_order is not None else 1,
                m.read_order if m.read_order is not None else m.original_sort_key,
            ),
        )

        base_name = output_path.stem
        export_path, index_path = TranslationExporter(
            output_path.parent, base_name
        ).export(ordered_metas)
        translation_path = Path(
            translation_input if translation_input else export_path
        )

        TranslationImporter().import_translations(ordered_metas, translation_path)
        Typesetting(translation_config).typesetting_document(docs)

        pdf_creater = PDFCreater(temp_pdf_path, docs, translation_config, mediabox_data)
        result = pdf_creater.write(translation_config)
        add_metadata(result, translation_config)
        fix_cmap(result, translation_config)
        final_path = _move_output_to_target(
            result, output_path, prefer_dual_output=enable_dual
        )
        logger.info("Hybrid pipeline finished: %s", final_path)
        return final_path
    finally:
        try:
            if doc_pdf2zh:
                doc_pdf2zh.close()
        except Exception:
            logger.debug("Failed to close PyMuPDF document", exc_info=True)
        try:
            translation_config.cleanup_temp_files()
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("Failed to clean up temp files", exc_info=True)


def _parse_pdf_to_il(
    translation_config: TranslationConfig,
) -> Tuple[il_version_1.Document, dict, str, pymupdf.Document]:
    logger.info("Parsing PDF into BabelDOC IL ...")
    temp_pdf_path = translation_config.get_working_file_path("input.pdf")
    mupdf_doc = pymupdf.open(translation_config.input_file)
    safe_save(mupdf_doc, temp_pdf_path)
    fix_null_page_content(mupdf_doc)
    fix_filter(mupdf_doc)
    fix_null_xref(mupdf_doc)
    mediabox_data = fix_media_box(mupdf_doc)
    safe_save(mupdf_doc, temp_pdf_path)

    il_creater = ILCreater(translation_config)
    il_creater.mupdf = mupdf_doc
    with temp_pdf_path.open("rb") as handle:
        start_parse_il(
            handle,
            doc_zh=mupdf_doc,
            il_creater=il_creater,
            translation_config=translation_config,
            lang_in=translation_config.lang_in,
            lang_out=translation_config.lang_out,
        )
    document = il_creater.create_il()
    return document, mediabox_data, str(temp_pdf_path), mupdf_doc


def _move_output_to_target(
    result, target_path: Path, prefer_dual_output: bool = False
) -> Path:
    """Move BabelDOC outputs to the desired CLI path."""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if prefer_dual_output and result.dual_pdf_path:
        shutil.move(result.dual_pdf_path, target_path)
        result.dual_pdf_path = target_path
        result.no_watermark_dual_pdf_path = target_path
        if result.mono_pdf_path and Path(result.mono_pdf_path).exists():
            mono_target = target_path.with_name(f"{target_path.stem}.mono.pdf")
            shutil.move(result.mono_pdf_path, mono_target)
            result.mono_pdf_path = mono_target
            result.no_watermark_mono_pdf_path = mono_target
        return target_path

    if not result.mono_pdf_path:
        raise RuntimeError("PDFCreater did not produce a monolingual output.")

    shutil.move(result.mono_pdf_path, target_path)
    result.mono_pdf_path = target_path
    result.no_watermark_mono_pdf_path = target_path
    return target_path


def _should_run_translation() -> bool:
    flag = os.environ.get("DEEPBRIDGE_ENABLE_TRANSLATION")
    if flag is None:
        return False
    return flag.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_zero_based_page_list(
    translation_config: TranslationConfig,
) -> list[int] | None:
    """Convert the user-provided page ranges into zero-based indices for Marker."""
    if not translation_config.page_ranges:
        return None

    selected_pages: set[int] = set()
    for start, end in translation_config.page_ranges:
        if start <= 0:
            continue
        if end == -1:
            # Open-ended ranges require processing full document.
            return None
        if end < start:
            continue
        for page in range(start, end + 1):
            selected_pages.add(page - 1)

    return sorted(selected_pages) if selected_pages else None

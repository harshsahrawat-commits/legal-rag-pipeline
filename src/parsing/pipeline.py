"""Parsing pipeline orchestrator.

Scans Phase 1 output (``data/raw/{source}/``), loads each document's
``.meta.json``, routes to the appropriate parser, validates quality,
and saves ``ParsedDocument`` JSON to ``data/parsed/{source}/``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.acquisition._models import ContentFormat, RawDocument, SourceType
from src.parsing._config import load_parsing_config
from src.parsing._downloader import PdfDownloader
from src.parsing._exceptions import ParserNotAvailableError, ParsingError, UnsupportedFormatError
from src.parsing._models import ParsingConfig, ParsingResult
from src.parsing._router import ParserRouter
from src.parsing._validation import QualityValidator
from src.parsing.parsers._docling_pdf import DoclingPdfParser
from src.parsing.parsers._html_india_code import IndiaCodeHtmlParser
from src.parsing.parsers._html_indian_kanoon import IndianKanoonHtmlParser
from src.utils._logging import get_logger

if TYPE_CHECKING:
    from src.parsing._models import ParsedDocument

_log = get_logger(__name__)

_SOURCE_NAME_MAP: dict[str, SourceType] = {
    "indian kanoon": SourceType.INDIAN_KANOON,
    "india code": SourceType.INDIA_CODE,
}


class ParsingPipeline:
    """Orchestrates Phase 2: raw documents -> parsed documents.

    Scans the Phase 1 output directory for ``.meta.json`` files, loads
    each ``RawDocument``, routes to the appropriate parser, validates
    quality, and saves ``ParsedDocument`` JSON to the output directory.
    """

    def __init__(
        self,
        config: ParsingConfig | None = None,
        config_path: Path | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = load_parsing_config(config_path)
        self._settings = self._config.settings
        self._router = self._build_router()
        self._validator = QualityValidator(self._settings)
        self._downloader = PdfDownloader(self._settings)

    def _build_router(self) -> ParserRouter:
        """Register all available parsers in priority order."""
        router = ParserRouter(self._settings)
        router.register(IndianKanoonHtmlParser())
        router.register(IndiaCodeHtmlParser())
        router.register(DoclingPdfParser())
        return router

    async def run(
        self,
        *,
        source_name: str | None = None,
        dry_run: bool = False,
    ) -> ParsingResult:
        """Run the parsing pipeline.

        Args:
            source_name: Filter to a single source (e.g. "India Code").
                         Omit to parse all sources.
            dry_run: If True, discover documents but don't parse or write.

        Returns:
            ParsingResult summarizing the run.
        """
        started = datetime.now(UTC)
        result = ParsingResult(started_at=started)

        source_filter = _resolve_source_filter(source_name)
        if source_name is not None and source_filter is None:
            _log.error("unknown_source_name", name=source_name)
            result.errors.append(f"Unknown source: {source_name}")
            result.finished_at = datetime.now(UTC)
            return result

        meta_files = self._discover_meta_files(source_filter)
        result.documents_found = len(meta_files)

        _log.info(
            "pipeline_starting",
            documents_found=len(meta_files),
            source_filter=source_name,
            dry_run=dry_run,
        )

        if dry_run:
            result.finished_at = datetime.now(UTC)
            return result

        for meta_path in meta_files:
            await self._process_document(meta_path, result)

        result.finished_at = datetime.now(UTC)
        _log.info(
            "pipeline_complete",
            parsed=result.documents_parsed,
            skipped=result.documents_skipped,
            failed=result.documents_failed,
        )
        return result

    def _discover_meta_files(self, source_filter: SourceType | None) -> list[Path]:
        """Find all ``.meta.json`` files in the input directory."""
        input_dir = Path(self._settings.input_dir)
        if not input_dir.exists():
            _log.warning("input_dir_missing", path=str(input_dir))
            return []

        if source_filter is not None:
            source_dir = input_dir / source_filter.value
            if not source_dir.exists():
                return []
            return sorted(source_dir.glob("*.meta.json"))

        return sorted(input_dir.glob("**/*.meta.json"))

    async def _process_document(
        self,
        meta_path: Path,
        result: ParsingResult,
    ) -> None:
        """Process a single document: load, route, parse, validate, save."""
        try:
            raw_doc = _load_raw_document(meta_path)
        except Exception as exc:
            _log.error("meta_load_failed", path=str(meta_path), error=str(exc))
            result.documents_failed += 1
            result.errors.append(f"Failed to load {meta_path.name}: {exc}")
            return

        doc_id = str(raw_doc.document_id)

        # Idempotency: skip if output file already exists
        output_path = self._output_path_for(raw_doc)
        if output_path.exists():
            _log.debug("already_parsed", doc_id=doc_id, path=str(output_path))
            result.documents_skipped += 1
            return

        try:
            parsed_doc = await self._parse_single(raw_doc)
        except (ParsingError, UnsupportedFormatError, ParserNotAvailableError) as exc:
            _log.error("parse_failed", doc_id=doc_id, error=str(exc))
            result.documents_failed += 1
            result.errors.append(f"Failed to parse {doc_id}: {exc}")
            return
        except Exception as exc:
            _log.error("parse_unexpected_error", doc_id=doc_id, error=str(exc))
            result.documents_failed += 1
            result.errors.append(f"Unexpected error parsing {doc_id}: {exc}")
            return

        # Validate quality (replaces placeholder report from parser)
        parsed_doc.quality = self._validator.validate(parsed_doc)

        # Save
        self._save_parsed_document(parsed_doc, output_path)
        result.documents_parsed += 1
        _log.info(
            "document_parsed",
            doc_id=doc_id,
            parser=parsed_doc.parser_used,
            quality_passed=parsed_doc.quality.passed,
            duration=parsed_doc.parsing_duration_seconds,
        )

    async def _parse_single(self, raw_doc: RawDocument) -> ParsedDocument:
        """Route and parse a single document.

        For India Code documents with a ``download_url``, triggers the
        two-step flow: download PDF, parse with Docling, merge HTML metadata.
        """
        content_path = Path(raw_doc.raw_content_path)

        # India Code special flow: HTML metadata + PDF content
        if (
            raw_doc.source_type == SourceType.INDIA_CODE
            and raw_doc.preliminary_metadata.download_url
        ):
            return await self._parse_india_code(raw_doc, content_path)

        # Standard flow
        parser = self._router.select_parser(raw_doc)
        return parser.parse(content_path, raw_doc)

    async def _parse_india_code(
        self,
        raw_doc: RawDocument,
        html_path: Path,
    ) -> ParsedDocument:
        """India Code two-step parse: PDF content + HTML metadata merge."""
        download_url = raw_doc.preliminary_metadata.download_url
        doc_id = str(raw_doc.document_id)

        # Step 1: Download PDF
        pdf_path = await self._downloader.download(download_url, doc_id)

        # Step 2: Extract metadata from HTML detail page
        ic_html_parser = IndiaCodeHtmlParser()
        html_meta_doc = ic_html_parser.parse(html_path, raw_doc)

        # Step 3: Parse PDF for full content
        pdf_raw_doc = raw_doc.model_copy(update={"content_format": ContentFormat.PDF})
        pdf_parser = self._router.select_parser(pdf_raw_doc)
        parsed_doc = pdf_parser.parse(pdf_path, pdf_raw_doc)

        # Step 4: Merge HTML metadata onto PDF-parsed document
        return _merge_metadata(parsed_doc, html_meta_doc)

    def _output_path_for(self, raw_doc: RawDocument) -> Path:
        """Compute the output path for a parsed document."""
        output_dir = Path(self._settings.output_dir) / raw_doc.source_type.value
        return output_dir / f"{raw_doc.document_id}.json"

    @staticmethod
    def _save_parsed_document(doc: ParsedDocument, output_path: Path) -> None:
        """Serialize and save a ParsedDocument to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        _log.debug("document_saved", path=str(output_path))


def _resolve_source_filter(source_name: str | None) -> SourceType | None:
    """Map human-readable source name to SourceType enum."""
    if source_name is None:
        return None
    return _SOURCE_NAME_MAP.get(source_name.lower().strip())


def _load_raw_document(meta_path: Path) -> RawDocument:
    """Load and validate a RawDocument from a ``.meta.json`` file."""
    raw_json = meta_path.read_text(encoding="utf-8")
    return RawDocument.model_validate_json(raw_json)


def _merge_metadata(
    pdf_doc: ParsedDocument,
    html_doc: ParsedDocument,
) -> ParsedDocument:
    """Merge metadata from HTML parser onto PDF-parsed document.

    HTML parser has authoritative metadata (act_name, act_number,
    year, date, title).  PDF parser has authoritative content
    (raw_text, sections, tables, page_count).
    """
    updates: dict[str, object] = {}
    for field in ("title", "act_name", "act_number", "year", "date"):
        html_val = getattr(html_doc, field)
        if html_val is not None:
            updates[field] = html_val

    if updates:
        return pdf_doc.model_copy(update=updates)
    return pdf_doc

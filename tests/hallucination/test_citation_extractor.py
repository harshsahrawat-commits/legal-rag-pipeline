"""Tests for citation extractor — pure regex, no external deps."""

from __future__ import annotations

from src.hallucination._citation_extractor import (
    extract_citations,
    resolve_act_alias,
)
from src.hallucination._models import CitationType


class TestResolveActAlias:
    def test_ipc(self) -> None:
        assert resolve_act_alias("IPC") == "Indian Penal Code"

    def test_crpc(self) -> None:
        assert resolve_act_alias("CrPC") == "Code of Criminal Procedure"

    def test_bns(self) -> None:
        assert resolve_act_alias("BNS") == "Bharatiya Nyaya Sanhita"

    def test_unknown_passthrough(self) -> None:
        assert resolve_act_alias("Some Random Act") == "Some Random Act"

    def test_strips_trailing_punctuation(self) -> None:
        assert resolve_act_alias("IPC,.") == "Indian Penal Code"


class TestSectionExtraction:
    def test_section_of_act(self) -> None:
        text = "Section 420 of the Indian Penal Code provides for punishment."
        citations = extract_citations(text)
        section_refs = [c for c in citations if c.citation_type == CitationType.SECTION_REF]
        assert len(section_refs) >= 1
        assert section_refs[0].section == "420"
        assert section_refs[0].act == "Indian Penal Code"

    def test_s_dot_abbreviation(self) -> None:
        text = "S. 302 IPC is about murder."
        citations = extract_citations(text)
        section_refs = [c for c in citations if c.citation_type == CitationType.SECTION_REF]
        assert len(section_refs) >= 1
        assert section_refs[0].section == "302"

    def test_sec_abbreviation(self) -> None:
        text = "Sec 10 Contract Act deals with consideration."
        citations = extract_citations(text)
        section_refs = [c for c in citations if c.citation_type == CitationType.SECTION_REF]
        assert len(section_refs) >= 1
        assert section_refs[0].section == "10"

    def test_section_with_alpha_suffix(self) -> None:
        text = "Section 498A of the Indian Penal Code."
        citations = extract_citations(text)
        section_refs = [c for c in citations if c.citation_type == CitationType.SECTION_REF]
        assert len(section_refs) >= 1
        assert section_refs[0].section == "498A"

    def test_alias_resolution_in_section(self) -> None:
        text = "S. 302 IPC was invoked."
        citations = extract_citations(text)
        section_refs = [c for c in citations if c.citation_type == CitationType.SECTION_REF]
        assert len(section_refs) >= 1
        assert section_refs[0].act == "Indian Penal Code"

    def test_multiple_sections(self) -> None:
        text = "Section 420 of the Indian Penal Code and Section 34 of the Indian Penal Code."
        citations = extract_citations(text)
        section_refs = [c for c in citations if c.citation_type == CitationType.SECTION_REF]
        assert len(section_refs) >= 2


class TestArticleExtraction:
    def test_article_number(self) -> None:
        text = "Article 21 guarantees the right to life."
        citations = extract_citations(text)
        article_refs = [c for c in citations if c.citation_type == CitationType.ARTICLE_REF]
        assert len(article_refs) >= 1
        assert article_refs[0].article == "21"
        assert article_refs[0].act == "Constitution of India"

    def test_art_dot_abbreviation(self) -> None:
        text = "Art. 14 ensures equality before law."
        citations = extract_citations(text)
        article_refs = [c for c in citations if c.citation_type == CitationType.ARTICLE_REF]
        assert len(article_refs) >= 1
        assert article_refs[0].article == "14"

    def test_article_with_alpha(self) -> None:
        text = "Article 21A provides for right to education."
        citations = extract_citations(text)
        article_refs = [c for c in citations if c.citation_type == CitationType.ARTICLE_REF]
        assert len(article_refs) >= 1
        assert article_refs[0].article == "21A"


class TestCaseCitationExtraction:
    def test_air_citation(self) -> None:
        text = "In AIR 2023 SC 1234, the court held..."
        citations = extract_citations(text)
        case_refs = [c for c in citations if c.citation_type == CitationType.CASE_CITATION]
        assert len(case_refs) == 1
        assert case_refs[0].case_citation == "AIR 2023 SC 1234"

    def test_scc_citation(self) -> None:
        text = "Reported in (2023) 5 SCC 678."
        citations = extract_citations(text)
        case_refs = [c for c in citations if c.citation_type == CitationType.CASE_CITATION]
        assert len(case_refs) == 1
        assert case_refs[0].case_citation == "(2023) 5 SCC 678"

    def test_scc_online_citation(self) -> None:
        text = "See 2023 SCC OnLine SC 890."
        citations = extract_citations(text)
        case_refs = [c for c in citations if c.citation_type == CitationType.CASE_CITATION]
        assert len(case_refs) == 1
        assert case_refs[0].case_citation == "2023 SCC OnLine SC 890"

    def test_multiple_case_citations(self) -> None:
        text = "AIR 2022 SC 999 and (2023) 3 SCC 456 were both relied upon."
        citations = extract_citations(text)
        case_refs = [c for c in citations if c.citation_type == CitationType.CASE_CITATION]
        assert len(case_refs) == 2


class TestNotificationExtraction:
    def test_gsr(self) -> None:
        text = "Notified via GSR 1234(E) dated 01.01.2023."
        citations = extract_citations(text)
        notif_refs = [c for c in citations if c.citation_type == CitationType.NOTIFICATION_REF]
        assert len(notif_refs) == 1
        assert "GSR" in notif_refs[0].notification_ref

    def test_so_notification(self) -> None:
        text = "Published as S.O. 5678(E)."
        citations = extract_citations(text)
        notif_refs = [c for c in citations if c.citation_type == CitationType.NOTIFICATION_REF]
        assert len(notif_refs) == 1


class TestCircularExtraction:
    def test_rbi_circular(self) -> None:
        text = "RBI/2023-24/45 issued directions on lending."
        citations = extract_citations(text)
        circ_refs = [c for c in citations if c.citation_type == CitationType.CIRCULAR_REF]
        assert len(circ_refs) == 1

    def test_sebi_circular(self) -> None:
        text = "SEBI/HO/2023/123 mandated disclosure norms."
        citations = extract_citations(text)
        circ_refs = [c for c in citations if c.citation_type == CitationType.CIRCULAR_REF]
        assert len(circ_refs) == 1


class TestEdgeCases:
    def test_empty_text(self) -> None:
        assert extract_citations("") == []

    def test_no_citations(self) -> None:
        text = "The weather is nice today."
        assert extract_citations(text) == []

    def test_span_positions(self) -> None:
        text = "Section 420 of the Indian Penal Code provides..."
        citations = extract_citations(text)
        assert len(citations) >= 1
        c = citations[0]
        assert c.span_start >= 0
        assert c.span_end > c.span_start
        # The matched text should be at the expected position
        assert text[c.span_start : c.span_end].startswith("Section 420")

    def test_sorted_by_position(self) -> None:
        text = "Article 21 and Section 302 of the Indian Penal Code."
        citations = extract_citations(text)
        for i in range(len(citations) - 1):
            assert citations[i].span_start <= citations[i + 1].span_start

    def test_deduplication(self) -> None:
        # Same citation should not appear twice from overlapping patterns
        text = "Section 420 of the Indian Penal Code is important."
        citations = extract_citations(text)
        section_refs = [c for c in citations if c.citation_type == CitationType.SECTION_REF]
        # Should be deduplicated by span
        spans = [(c.span_start, c.span_end) for c in section_refs]
        assert len(spans) == len(set(spans))

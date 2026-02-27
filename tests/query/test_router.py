"""Tests for AdaptiveQueryRouter and extract_act_references."""

from __future__ import annotations

import pytest

from src.query._models import QuerySettings, RouterResult
from src.query._router import AdaptiveQueryRouter, extract_act_references
from src.retrieval._models import QueryRoute

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def router() -> AdaptiveQueryRouter:
    """Router with default settings."""
    return AdaptiveQueryRouter(QuerySettings())


# ---------------------------------------------------------------------------
# SIMPLE queries
# ---------------------------------------------------------------------------


class TestSimpleRouting:
    """SIMPLE route — direct section/definition lookups."""

    def test_what_does_section(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("What does Section 302 IPC say?")
        assert result.route == QueryRoute.SIMPLE
        assert result.confidence == 1.0
        assert "pattern:what_does_section" in result.signals

    def test_what_is_section(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("What is Section 498A of the Indian Penal Code?")
        assert result.route == QueryRoute.SIMPLE
        assert result.confidence == 1.0
        assert "pattern:what_is_section" in result.signals

    def test_define_quoted(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Define 'cognizable offence'")
        assert result.route == QueryRoute.SIMPLE
        assert "pattern:define" in result.signals

    def test_define_unquoted(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Define murder under IPC")
        assert result.route == QueryRoute.SIMPLE
        assert "pattern:define" in result.signals

    def test_text_of_section(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Text of Section 138 NI Act")
        assert result.route == QueryRoute.SIMPLE
        assert "pattern:text_of_section" in result.signals

    def test_show_me_section(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Show me Section 10 of the Contract Act")
        assert result.route == QueryRoute.SIMPLE
        assert "pattern:show_me_section" in result.signals

    def test_read_section(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Read Section 375 IPC")
        assert result.route == QueryRoute.SIMPLE
        assert "pattern:read_section" in result.signals

    def test_what_is_article(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("What is Article 21 of the Constitution?")
        assert result.route == QueryRoute.SIMPLE
        assert "pattern:what_is_article" in result.signals

    def test_what_does_article(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("What does Article 14 say?")
        assert result.route == QueryRoute.SIMPLE
        assert "pattern:what_does_article" in result.signals

    def test_meaning_of_section(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Meaning of Section 2(d) of Contract Act")
        assert result.route == QueryRoute.SIMPLE
        assert "pattern:meaning_of_section" in result.signals

    def test_bare_text_of(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Bare text of Section 420 IPC")
        assert result.route == QueryRoute.SIMPLE
        assert "pattern:bare_text_of" in result.signals

    def test_simple_case_insensitive(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("WHAT DOES SECTION 302 IPC SAY?")
        assert result.route == QueryRoute.SIMPLE

    def test_define_double_quoted(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify('Define "abetment"')
        assert result.route == QueryRoute.SIMPLE
        assert "pattern:define" in result.signals


# ---------------------------------------------------------------------------
# ANALYTICAL queries
# ---------------------------------------------------------------------------


class TestAnalyticalRouting:
    """ANALYTICAL route — comparative, historical, exhaustive queries."""

    def test_compare(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify(
            "Compare the eviction grounds under Delhi and Mumbai rent control laws"
        )
        assert result.route == QueryRoute.ANALYTICAL
        assert result.confidence == 0.9
        assert "signal:compare" in result.signals

    def test_trace_evolution(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify(
            "Trace the evolution of Section 377 jurisprudence from Naz Foundation to Navtej Johar"
        )
        assert result.route == QueryRoute.ANALYTICAL
        assert any("trace" in s for s in result.signals)

    def test_all_grounds(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("What are ALL grounds for eviction under Delhi Rent Control Act?")
        assert result.route == QueryRoute.ANALYTICAL
        assert "signal:all_grounds" in result.signals

    def test_interpreted_over_years(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("How has Section 498A been interpreted over the years?")
        assert result.route == QueryRoute.ANALYTICAL
        assert "signal:interpreted_over_time" in result.signals

    def test_contrast(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Contrast the bail provisions under CrPC and BNSS")
        assert result.route == QueryRoute.ANALYTICAL
        assert "signal:contrast" in result.signals

    def test_comprehensive(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Comprehensive analysis of anticipatory bail jurisprudence")
        assert result.route == QueryRoute.ANALYTICAL
        assert "signal:comprehensive" in result.signals

    def test_exhaustive(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Exhaustive list of fundamental rights under Part III")
        assert result.route == QueryRoute.ANALYTICAL
        assert "signal:exhaustive" in result.signals

    def test_every_provision_under(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Every remedy under the Consumer Protection Act")
        assert result.route == QueryRoute.ANALYTICAL
        assert "signal:every_under" in result.signals

    def test_interplay_between(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("What is the interplay between SARFAESI and the DV Act?")
        assert result.route == QueryRoute.ANALYTICAL
        assert "signal:interplay_between" in result.signals

    def test_relationship_between(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Relationship between Article 14 and Article 19")
        assert result.route == QueryRoute.ANALYTICAL
        assert "signal:relationship_between" in result.signals

    def test_history_of(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("History of death penalty jurisprudence in India")
        assert result.route == QueryRoute.ANALYTICAL
        assert "signal:history_of" in result.signals

    def test_development_of(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Development of the doctrine of basic structure")
        assert result.route == QueryRoute.ANALYTICAL
        assert "signal:development_of" in result.signals

    def test_trace_jurisprudence(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify(
            "Trace the Section 498A jurisprudence from Sushil Kumar Sharma onwards"
        )
        assert result.route == QueryRoute.ANALYTICAL
        assert "signal:trace_jurisprudence" in result.signals

    def test_multiple_analytical_signals(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify(
            "Compare and trace the comprehensive evolution of bail jurisprudence"
        )
        assert result.route == QueryRoute.ANALYTICAL
        assert len(result.signals) > 1

    def test_all_provisions(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("List all provisions under the Motor Vehicles Act for road safety")
        assert result.route == QueryRoute.ANALYTICAL
        assert "signal:all_provisions" in result.signals


# ---------------------------------------------------------------------------
# COMPLEX queries
# ---------------------------------------------------------------------------


class TestComplexRouting:
    """COMPLEX route — multi-act, multi-section, cross-jurisdictional."""

    def test_multi_act_ipc_dv_act(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("How does Section 498A IPC relate to the DV Act provisions?")
        assert result.route == QueryRoute.COMPLEX
        assert result.confidence == 0.8
        assert any("multi_act" in s for s in result.signals)

    def test_read_with(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify(
            "What are the eviction provisions under Delhi Rent Control Act read with Transfer of Property Act?"
        )
        assert result.route == QueryRoute.COMPLEX
        assert any("multi_hop" in s or "multi_act" in s for s in result.signals)

    def test_in_light_of(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Explain Section 138 NI Act in light of the Supreme Court ruling")
        assert result.route == QueryRoute.COMPLEX
        assert "heuristic:multi_hop" in result.signals

    def test_read_together(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Section 34 read together with Section 302 IPC")
        assert result.route == QueryRoute.COMPLEX
        assert "heuristic:multi_hop" in result.signals

    def test_multi_act_crpc_bnss(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Differences between CrPC and BNSS bail provisions")
        assert result.route in (QueryRoute.COMPLEX, QueryRoute.ANALYTICAL)

    def test_cross_jurisdictional_delhi_mumbai(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Rent control in Delhi and Mumbai differ significantly")
        assert result.route in (QueryRoute.COMPLEX, QueryRoute.ANALYTICAL)

    def test_state_and_central(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Jurisdiction overlap between state and central GST officers")
        assert result.route == QueryRoute.COMPLEX
        assert "heuristic:cross_jurisdictional" in result.signals

    def test_supreme_court_and_high_court(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify(
            "Conflicting views of Supreme Court and High Court on Section 498A"
        )
        assert result.route == QueryRoute.COMPLEX
        assert "heuristic:cross_jurisdictional" in result.signals

    def test_multi_section_three_plus(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify(
            "Explain Section 302 of IPC, Section 34 of IPC, and Section 120B of IPC together"
        )
        assert result.route == QueryRoute.COMPLEX
        assert any("multi_section" in s for s in result.signals)

    def test_multi_act_ipc_ni_act(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("How does IPC fraud relate to the NI Act dishonour provisions?")
        assert result.route == QueryRoute.COMPLEX
        assert any("multi_act" in s for s in result.signals)

    def test_central_and_state(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Powers of central and state governments under GST")
        assert result.route == QueryRoute.COMPLEX
        assert "heuristic:cross_jurisdictional" in result.signals


# ---------------------------------------------------------------------------
# STANDARD queries (default)
# ---------------------------------------------------------------------------


class TestStandardRouting:
    """STANDARD route — default classification."""

    def test_general_punishment_query(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("What is the punishment for cheating?")
        assert result.route == QueryRoute.STANDARD
        assert result.confidence == 0.5

    def test_tenant_eviction(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Can a landlord evict a tenant for personal use?")
        assert result.route == QueryRoute.STANDARD

    def test_divorce_grounds(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("What are the grounds for divorce under Hindu Marriage Act?")
        assert result.route == QueryRoute.STANDARD

    def test_anticipatory_bail(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Is anticipatory bail available in cases under Section 498A?")
        assert result.route == QueryRoute.STANDARD

    def test_simple_legal_question(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("How to file an FIR in India?")
        assert result.route == QueryRoute.STANDARD

    def test_bail_conditions(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("What are the conditions for getting bail?")
        assert result.route == QueryRoute.STANDARD

    def test_maintenance_rights(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Can a wife claim maintenance after divorce?")
        assert result.route == QueryRoute.STANDARD

    def test_property_dispute(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("What is the procedure for property partition suit?")
        assert result.route == QueryRoute.STANDARD

    def test_limitation_period(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("What is the limitation period for civil suits?")
        assert result.route == QueryRoute.STANDARD

    def test_standard_has_default_signal(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("General legal question about contracts")
        assert result.route == QueryRoute.STANDARD
        assert "default:standard" in result.signals

    def test_cheque_bounce(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("What happens in a cheque bounce case?")
        assert result.route == QueryRoute.STANDARD

    def test_cyber_crime(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("How to report cyber crime in India?")
        assert result.route == QueryRoute.STANDARD


# ---------------------------------------------------------------------------
# extract_act_references
# ---------------------------------------------------------------------------


class TestExtractActReferences:
    """Tests for the extract_act_references helper function."""

    def test_single_abbreviation_ipc(self) -> None:
        acts = extract_act_references("Section 302 IPC")
        assert "Indian Penal Code" in acts

    def test_single_abbreviation_crpc(self) -> None:
        acts = extract_act_references("bail under CrPC")
        assert "Code of Criminal Procedure" in acts

    def test_single_abbreviation_ni_act(self) -> None:
        acts = extract_act_references("Section 138 NI Act")
        assert "Negotiable Instruments Act" in acts

    def test_multiple_abbreviations(self) -> None:
        acts = extract_act_references("IPC and CrPC provisions for bail")
        assert len(acts) == 2
        assert "Indian Penal Code" in acts
        assert "Code of Criminal Procedure" in acts

    def test_full_name_with_year(self) -> None:
        acts = extract_act_references("under the Companies Act, 2013")
        assert any("Companies Act" in a for a in acts)

    def test_full_name_without_year(self) -> None:
        acts = extract_act_references("under the Limitation Act")
        assert any("Limitation Act" in a for a in acts)

    def test_abbreviation_bns(self) -> None:
        acts = extract_act_references("offences under BNS")
        assert "Bharatiya Nyaya Sanhita" in acts

    def test_abbreviation_bnss(self) -> None:
        acts = extract_act_references("arrest procedure under BNSS")
        assert "Bharatiya Nagarik Suraksha Sanhita" in acts

    def test_abbreviation_bsa(self) -> None:
        acts = extract_act_references("evidence rules under BSA")
        assert "Bharatiya Sakshya Adhiniyam" in acts

    def test_no_act_references(self) -> None:
        acts = extract_act_references("What is the punishment for theft?")
        assert acts == []

    def test_dv_act(self) -> None:
        acts = extract_act_references("DV Act provisions for protection orders")
        assert "Domestic Violence Act" in acts

    def test_pocso(self) -> None:
        acts = extract_act_references("offences under POCSO")
        assert "Protection of Children from Sexual Offences Act" in acts

    def test_it_act(self) -> None:
        acts = extract_act_references("income exemptions under IT Act")
        assert "Income Tax Act" in acts

    def test_full_name_indian_penal_code(self) -> None:
        acts = extract_act_references("Section 302 of the Indian Penal Code")
        # Should resolve from the full act name pattern
        assert len(acts) >= 1

    def test_gst(self) -> None:
        acts = extract_act_references("GST registration requirements")
        assert "Goods and Services Tax Act" in acts

    def test_sarfaesi(self) -> None:
        acts = extract_act_references("recovery under SARFAESI")
        assert "SARFAESI Act" in acts

    def test_rera(self) -> None:
        acts = extract_act_references("home buyer rights under RERA")
        assert "Real Estate Regulation Act" in acts

    def test_posh(self) -> None:
        acts = extract_act_references("complaints under POSH")
        assert "Prevention of Sexual Harassment Act" in acts

    def test_mv_act(self) -> None:
        acts = extract_act_references("penalties under MV Act")
        assert "Motor Vehicles Act" in acts

    def test_deduplication(self) -> None:
        acts = extract_act_references("IPC and IPC provisions")
        assert acts.count("Indian Penal Code") == 1

    def test_case_insensitive(self) -> None:
        acts = extract_act_references("section under ipc and CRPC")
        assert "Indian Penal Code" in acts
        assert "Code of Criminal Procedure" in acts


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_query(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("")
        assert result.route == QueryRoute.STANDARD
        assert result.confidence == 0.5

    def test_very_long_query(self, router: AdaptiveQueryRouter) -> None:
        long_query = "What is the law regarding " + "property " * 500 + "disputes?"
        result = router.classify(long_query)
        assert isinstance(result, RouterResult)
        assert result.route in QueryRoute

    def test_hindi_text(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("धारा 302 आईपीसी क्या कहती है?")
        # Hindi text won't match English patterns, should default to STANDARD
        assert result.route == QueryRoute.STANDARD

    def test_mixed_language(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("What is धारा 302 IPC?")
        # Has IPC but "What is" followed by Hindi won't match "what is section \d+"
        assert isinstance(result, RouterResult)

    def test_whitespace_only(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("   ")
        assert result.route == QueryRoute.STANDARD

    def test_result_has_signals(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("What does Section 302 IPC say?")
        assert len(result.signals) > 0

    def test_standard_result_has_signals(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Some general question")
        assert len(result.signals) > 0

    def test_router_result_is_pydantic(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Test query")
        assert isinstance(result, RouterResult)
        assert hasattr(result, "model_dump")

    def test_simple_priority_over_analytical(self, router: AdaptiveQueryRouter) -> None:
        """SIMPLE should win when a query matches both SIMPLE and ANALYTICAL patterns."""
        result = router.classify("What is Section 377 and trace its evolution?")
        # "What is Section 377" matches SIMPLE (higher priority)
        assert result.route == QueryRoute.SIMPLE

    def test_analytical_priority_over_complex(self, router: AdaptiveQueryRouter) -> None:
        """ANALYTICAL should win when a query matches both ANALYTICAL and COMPLEX."""
        result = router.classify("Compare IPC and BNS provisions on murder")
        # "compare" triggers ANALYTICAL (checked before COMPLEX)
        assert result.route == QueryRoute.ANALYTICAL

    def test_classification_error_returns_standard(self, router: AdaptiveQueryRouter) -> None:
        """If classification logic crashes, return STANDARD as safe default."""
        # Monkey-patch to force an error
        original = router._check_simple
        router._check_simple = lambda q: (_ for _ in ()).throw(  # type: ignore[assignment]
            RuntimeError("forced error")
        )
        try:
            result = router.classify("What does Section 302 say?")
            assert result.route == QueryRoute.STANDARD
            assert "error:fallback_standard" in result.signals
        finally:
            router._check_simple = original  # type: ignore[assignment]

    def test_special_characters(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("Section 2(d) & 2(e) of the Contract Act???")
        assert isinstance(result, RouterResult)

    def test_numbers_only(self, router: AdaptiveQueryRouter) -> None:
        result = router.classify("302 498A 420")
        assert result.route == QueryRoute.STANDARD


# ---------------------------------------------------------------------------
# Router initialization and settings
# ---------------------------------------------------------------------------


class TestRouterInit:
    """Test router initialization and configuration."""

    def test_default_settings(self) -> None:
        settings = QuerySettings()
        router = AdaptiveQueryRouter(settings)
        assert router._settings is settings

    def test_custom_settings(self) -> None:
        settings = QuerySettings(router_version="v2_custom")
        router = AdaptiveQueryRouter(settings)
        assert router._settings.router_version == "v2_custom"

    def test_patterns_compiled_once(self) -> None:
        router = AdaptiveQueryRouter(QuerySettings())
        # Patterns should be pre-compiled re.Pattern objects
        for pattern, _signal in router._simple_patterns:
            assert hasattr(pattern, "search")
        for pattern, _signal in router._analytical_patterns:
            assert hasattr(pattern, "search")

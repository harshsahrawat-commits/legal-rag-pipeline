from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

from src.knowledge_graph._models import (
    ActNode,
    AmendmentNode,
    CourtNode,
    ExtractedEntities,
    IntegrityCheck,
    IntegrityReport,
    JudgeNode,
    JudgmentNode,
    KGConfig,
    KGResult,
    KGSettings,
    LegalConceptNode,
    Relationship,
    SectionNode,
    SectionVersionNode,
)


class TestActNode:
    def test_defaults(self) -> None:
        act = ActNode(name="Indian Penal Code")
        assert act.name == "Indian Penal Code"
        assert act.jurisdiction == "India"
        assert act.status == "in_force"
        assert act.number is None
        assert act.year is None
        assert act.date_enacted is None

    def test_full_creation(self) -> None:
        act = ActNode(
            name="Bharatiya Nyaya Sanhita",
            number="45 of 2023",
            year=2023,
            date_enacted=date(2023, 12, 25),
            date_effective=date(2024, 7, 1),
            status="in_force",
        )
        assert act.year == 2023
        assert act.date_effective == date(2024, 7, 1)

    def test_repealed_act(self) -> None:
        act = ActNode(
            name="Indian Penal Code",
            status="repealed",
            date_repealed=date(2024, 7, 1),
        )
        assert act.status == "repealed"
        assert act.date_repealed == date(2024, 7, 1)


class TestSectionNode:
    def test_required_fields(self) -> None:
        section = SectionNode(number="302", parent_act="Indian Penal Code")
        assert section.number == "302"
        assert section.parent_act == "Indian Penal Code"
        assert section.is_in_force is True
        assert section.chunk_id is None

    def test_with_hierarchy(self) -> None:
        section = SectionNode(
            number="3",
            parent_act="Information Technology Act",
            chapter="II",
            part="Part A",
            chunk_id=uuid4(),
        )
        assert section.chapter == "II"
        assert section.part == "Part A"
        assert section.chunk_id is not None


class TestSectionVersionNode:
    def test_creation(self) -> None:
        sv = SectionVersionNode(
            version_id="IPC:302:v1",
            text_hash="abc123def456",
            effective_from=date(1860, 10, 6),
        )
        assert sv.version_id == "IPC:302:v1"
        assert sv.effective_until is None

    def test_bounded_version(self) -> None:
        sv = SectionVersionNode(
            version_id="IPC:302:v1",
            text_hash="abc123",
            effective_from=date(1860, 10, 6),
            effective_until=date(2024, 7, 1),
            amending_act="Bharatiya Nyaya Sanhita",
        )
        assert sv.effective_until == date(2024, 7, 1)
        assert sv.amending_act == "Bharatiya Nyaya Sanhita"


class TestJudgmentNode:
    def test_minimal(self) -> None:
        j = JudgmentNode(citation="(2024) 1 SCC 1", court="Supreme Court of India", court_level=1)
        assert j.citation == "(2024) 1 SCC 1"
        assert j.status == "good_law"
        assert j.alt_citations == []

    def test_full_judgment(self) -> None:
        j = JudgmentNode(
            citation="AIR 2023 SC 100",
            alt_citations=["(2023) 2 SCC 50"],
            court="Supreme Court of India",
            court_level=1,
            bench_type="Division Bench",
            bench_strength=2,
            date_decided=date(2023, 6, 15),
            case_type="Criminal Appeal",
            parties_petitioner="State of UP",
            parties_respondent="Ram Kumar",
            status="good_law",
            chunk_id=uuid4(),
        )
        assert j.bench_strength == 2
        assert len(j.alt_citations) == 1


class TestAmendmentNode:
    def test_creation(self) -> None:
        a = AmendmentNode(
            amending_act="Criminal Law Amendment Act, 2013",
            date=date(2013, 4, 2),
            nature="substitution",
            gazette_ref="GSR 123(E)",
        )
        assert a.nature == "substitution"
        assert a.gazette_ref == "GSR 123(E)"


class TestLegalConceptNode:
    def test_creation(self) -> None:
        lc = LegalConceptNode(
            name="mens rea", definition_source="Section 2, IPC", category="criminal"
        )
        assert lc.name == "mens rea"
        assert lc.category == "criminal"


class TestCourtNode:
    def test_creation(self) -> None:
        c = CourtNode(name="Supreme Court of India", hierarchy_level=1)
        assert c.hierarchy_level == 1
        assert c.state is None

    def test_high_court(self) -> None:
        c = CourtNode(
            name="Bombay High Court",
            hierarchy_level=2,
            state="Maharashtra",
            jurisdiction_type="constitutional",
        )
        assert c.state == "Maharashtra"


class TestJudgeNode:
    def test_creation(self) -> None:
        j = JudgeNode(name="Justice D.Y. Chandrachud", courts_served=["Supreme Court of India"])
        assert j.name == "Justice D.Y. Chandrachud"
        assert len(j.courts_served) == 1

    def test_empty_courts(self) -> None:
        j = JudgeNode(name="Justice A.B.")
        assert j.courts_served == []


class TestRelationship:
    def test_creation(self) -> None:
        r = Relationship(
            from_label="Act",
            from_key={"name": "IPC"},
            to_label="Section",
            to_key={"parent_act": "IPC", "number": "302"},
            rel_type="CONTAINS",
        )
        assert r.rel_type == "CONTAINS"
        assert r.properties == {}

    def test_with_properties(self) -> None:
        r = Relationship(
            from_label="Amendment",
            from_key={"amending_act": "CLA 2013", "date": "2013-04-02"},
            to_label="Section",
            to_key={"parent_act": "IPC", "number": "376"},
            rel_type="AMENDS",
            properties={"before_text": "old", "after_text": "new"},
        )
        assert r.properties["before_text"] == "old"


class TestExtractedEntities:
    def test_defaults_are_empty(self) -> None:
        ee = ExtractedEntities()
        assert ee.acts == []
        assert ee.sections == []
        assert ee.section_versions == []
        assert ee.judgments == []
        assert ee.amendments == []
        assert ee.legal_concepts == []
        assert ee.courts == []
        assert ee.judges == []

    def test_populated(self) -> None:
        ee = ExtractedEntities(
            acts=[ActNode(name="IPC")],
            sections=[SectionNode(number="302", parent_act="IPC")],
        )
        assert len(ee.acts) == 1
        assert len(ee.sections) == 1


class TestIntegrityModels:
    def test_check_passing(self) -> None:
        check = IntegrityCheck(name="section_versions", passed=True)
        assert check.violations == []

    def test_check_failing(self) -> None:
        check = IntegrityCheck(
            name="repealed_consistency",
            passed=False,
            violations=["IPC:302 is_in_force=true but IPC is repealed"],
        )
        assert not check.passed
        assert len(check.violations) == 1

    def test_report_passing(self) -> None:
        report = IntegrityReport(
            passed=True,
            checks=[IntegrityCheck(name="test", passed=True)],
        )
        assert report.passed

    def test_report_failing(self) -> None:
        report = IntegrityReport(
            passed=False,
            checks=[
                IntegrityCheck(name="ok", passed=True),
                IntegrityCheck(name="bad", passed=False, violations=["v1"]),
            ],
        )
        assert not report.passed


class TestKGSettings:
    def test_defaults(self) -> None:
        s = KGSettings()
        assert s.input_dir == Path("data/enriched")
        assert s.neo4j_uri == "bolt://localhost:7687"
        assert s.neo4j_user == "neo4j"
        assert s.neo4j_password == "password"
        assert s.neo4j_database == "neo4j"
        assert s.batch_size == 100

    def test_custom_values(self) -> None:
        s = KGSettings(neo4j_uri="bolt://db:7687", batch_size=50)
        assert s.neo4j_uri == "bolt://db:7687"
        assert s.batch_size == 50

    def test_path_coercion(self) -> None:
        s = KGSettings(input_dir="my/input")
        assert isinstance(s.input_dir, Path)


class TestKGConfig:
    def test_default_settings(self) -> None:
        config = KGConfig()
        assert isinstance(config.settings, KGSettings)
        assert config.settings.batch_size == 100

    def test_from_dict(self) -> None:
        data = {"settings": {"neo4j_uri": "bolt://remote:7687", "batch_size": 200}}
        config = KGConfig.model_validate(data)
        assert config.settings.neo4j_uri == "bolt://remote:7687"
        assert config.settings.batch_size == 200

    def test_round_trip(self) -> None:
        config = KGConfig(settings=KGSettings(batch_size=64))
        data = config.model_dump(mode="json")
        restored = KGConfig.model_validate(data)
        assert restored.settings.batch_size == 64


class TestKGResult:
    def test_defaults(self) -> None:
        r = KGResult()
        assert r.documents_found == 0
        assert r.documents_ingested == 0
        assert r.documents_skipped == 0
        assert r.documents_failed == 0
        assert r.nodes_created == 0
        assert r.relationships_created == 0
        assert r.integrity_passed is None
        assert r.errors == []
        assert r.source_type is None
        assert r.finished_at is None
        assert isinstance(r.started_at, datetime)

    def test_counts_are_mutable(self) -> None:
        r = KGResult()
        r.documents_ingested += 5
        r.nodes_created += 100
        r.relationships_created += 50
        assert r.documents_ingested == 5
        assert r.nodes_created == 100

    def test_errors_accumulate(self) -> None:
        r = KGResult()
        r.errors.append("doc1 failed")
        r.errors.append("doc2 failed")
        assert len(r.errors) == 2

    def test_finished_at_settable(self) -> None:
        r = KGResult()
        now = datetime.now(UTC)
        r.finished_at = now
        assert r.finished_at == now

# Knowledge Graph Schema — Neo4j

## Node Types

| Node | Key Properties | Notes |
|---|---|---|
| `Act` | name, number, year, date_enacted, date_effective, date_repealed, jurisdiction, status | status: in_force / repealed / partially_repealed |
| `Section` | number, parent_act, chapter, part, is_in_force | Unique constraint: (parent_act, number) |
| `SectionVersion` | version_id, text_hash, effective_from, effective_until, amending_act | Tracks text changes over time |
| `Judgment` | citation, court, date_decided, bench_type, bench_strength, status | status: good_law / overruled / distinguished |
| `Amendment` | amending_act, date, gazette_ref, nature | nature: substitution / insertion / omission |
| `LegalConcept` | name, definition_source, category | NER-extracted concepts |
| `Court` | name, hierarchy_level, state, jurisdiction_type | hierarchy: 1=SC, 2=HC, 3=District, 4=Tribunal |
| `Judge` | name, courts_served | For bench composition tracking |

## Relationship Types

```cypher
(:Act)-[:CONTAINS]->(:Section)
(:Section)-[:HAS_VERSION]->(:SectionVersion)
(:Amendment)-[:AMENDS {before_text, after_text}]->(:Section)
(:Amendment)-[:INSERTS]->(:Section)
(:Amendment)-[:OMITS]->(:Section)
(:Act)-[:REPEALS]->(:Act)
(:Act)-[:REPLACES]->(:Act)
(:Judgment)-[:INTERPRETS]->(:Section)
(:Judgment)-[:CITES_SECTION]->(:Section)
(:Judgment)-[:CITES_CASE]->(:Judgment)
(:Judgment)-[:OVERRULES]->(:Judgment)
(:Judgment)-[:DISTINGUISHES]->(:Judgment)
(:Judgment)-[:FOLLOWS]->(:Judgment)
(:Judgment)-[:DECIDED_BY]->(:Judge)
(:Judgment)-[:FILED_IN]->(:Court)
(:Section)-[:REFERENCES]->(:Section)
(:Section)-[:DEFINES]->(:LegalConcept)
```

## Critical Queries

```cypher
-- Point-in-time retrieval
MATCH (s:Section {number: $section, parent_act: $act})-[:HAS_VERSION]->(v:SectionVersion)
WHERE v.effective_from <= date($query_date)
  AND (v.effective_until IS NULL OR v.effective_until > date($query_date))
RETURN v.text

-- Find replacement for repealed section
MATCH (old:Act {name: $old_act})-[:CONTAINS]->(s:Section {number: $section}),
      (new:Act)-[:REPLACES]->(old)
RETURN new.name, s.replaced_by_section

-- All SC judgments interpreting a section
MATCH (j:Judgment)-[:INTERPRETS]->(s:Section {number: $section, parent_act: $act})
WHERE j.court = "Supreme Court of India"
RETURN j ORDER BY j.date_decided DESC

-- Amendment cascade: find all chunks affected by an amendment
MATCH (a:Amendment {amending_act: $amendment})-[:AMENDS|INSERTS|OMITS]->(s:Section)
RETURN s.parent_act, s.number, type(r) as change_type
```

## Indexing

Create these Neo4j indexes for query performance:
```cypher
CREATE CONSTRAINT act_name IF NOT EXISTS FOR (a:Act) REQUIRE a.name IS UNIQUE;
CREATE CONSTRAINT section_unique IF NOT EXISTS FOR (s:Section) REQUIRE (s.parent_act, s.number) IS UNIQUE;
CREATE CONSTRAINT judgment_citation IF NOT EXISTS FOR (j:Judgment) REQUIRE j.citation IS UNIQUE;
CREATE INDEX section_in_force IF NOT EXISTS FOR (s:Section) ON (s.is_in_force);
CREATE INDEX judgment_date IF NOT EXISTS FOR (j:Judgment) ON (j.date_decided);
CREATE INDEX judgment_court IF NOT EXISTS FOR (j:Judgment) ON (j.court);
```

## Data Integrity Rules

1. Every `Section` must have at least one `SectionVersion`.
2. If an `Act` has status "repealed", ALL its sections must have `is_in_force = false`.
3. An `OVERRULES` relationship requires the overruling judgment to be from a court of equal or higher hierarchy.
4. `SectionVersion.effective_from` ranges must not overlap for the same section.
5. Run integrity checks after every bulk ingestion: `scripts/kg_integrity_check.py`.

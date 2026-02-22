# Indian Legal Document Structure — For Non-Lawyers

Read this before working on parsing or chunking code.

## Statutes (Acts of Parliament)

```
Act
├── Preamble (optional, states purpose)
├── Part I, II, III... (major divisions, not all Acts have these)
│   ├── Chapter I, II, III... (subdivisions)
│   │   ├── Section 1, 2, 3... (the fundamental unit lawyers cite)
│   │   │   ├── Sub-section (1), (2), (3)...
│   │   │   │   ├── Clause (a), (b), (c)...
│   │   │   │   │   └── Sub-clause (i), (ii), (iii)...
│   │   │   ├── Proviso ("Provided that...") — exceptions/conditions
│   │   │   └── Explanation — clarifies meaning
│   │   └── ...more sections
│   └── ...more chapters
└── Schedule(s) — tables, forms, lists appended to the Act
```

**Key rule for chunking:** A Section is the atomic unit. Never split a section's sub-sections, provisos, or explanations into separate chunks. They are meaningless without their parent.

**Watch for:** "Notwithstanding anything contained in..." — these override provisions are critical context. "Subject to the provisions of Section X..." — creates dependency on another section.

## Indian Court Judgments

```
Judgment
├── Case number and court header
├── Bench composition (judges)
├── Parties (Petitioner/Appellant vs Respondent)
├── "Facts of the case" or "Background"
├── "Issues for consideration" or "Questions of law"
├── Per-issue analysis (this is the meat):
│   ├── Statutory provisions discussed
│   ├── Precedents cited (other judgments)
│   ├── Arguments by both sides
│   └── Court's reasoning and conclusion on that issue
├── "Ratio decidendi" — the legal principle established (most important part)
├── "Obiter dicta" — observations not central to the decision (less binding)
├── Final order: "Appeal allowed/dismissed", "Writ issued/refused", etc.
└── Costs (if any)
```

**Key rule for chunking:** The reasoning sections should keep the cited provisions and the court's analysis together. Separating "Section 302 IPC provides..." from "...we hold that the accused is guilty under this section" destroys meaning.

## Court Hierarchy (Binding Precedent)

```
Supreme Court of India              ← Binds ALL courts below
    ├── 25 High Courts              ← Binds courts in their state
    │   ├── District Courts
    │   │   ├── Sessions Courts
    │   │   └── Civil Courts
    │   └── Tribunals (NCLT, ITAT, NGT, etc.)
    └── Specialized courts (Family, Labour, Consumer)
```

**Why hierarchy matters for RAG:** A Supreme Court judgment overrides a conflicting High Court judgment. If retrieval returns both, the system must know which carries more weight. This is tracked in `CourtHierarchy` enum.

## The 2024 Criminal Code Overhaul

Effective July 1, 2024, India replaced three foundational criminal laws:

| Old Law | New Law | Key Difference |
|---|---|---|
| Indian Penal Code, 1860 (IPC) | Bharatiya Nyaya Sanhita, 2023 (BNS) | Section numbers completely changed |
| Code of Criminal Procedure, 1973 (CrPC) | Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS) | New procedures added |
| Indian Evidence Act, 1872 | Bharatiya Sakshya Adhiniyam, 2023 (BSA) | Digital evidence rules updated |

**Critical for the system:** Old section numbers (like "Section 420 IPC" for cheating) are still used in ALL pre-2024 judgments. The system must map old → new and track both.

## Citation Formats

| Format | Example | When Used |
|---|---|---|
| AIR | AIR 2023 SC 1234 | All India Reporter (most common) |
| SCC | (2023) 5 SCC 678 | Supreme Court Cases |
| SCC OnLine | 2023 SCC OnLine SC 890 | Online-first publication |
| SCR | [2023] 3 SCR 456 | Supreme Court Reports |
| Cri LJ | 2023 Cri LJ 789 | Criminal Law Journal |

Your citation extraction regex must handle ALL of these formats.

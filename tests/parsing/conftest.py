from __future__ import annotations

from pathlib import Path  # noqa: TC003

import pytest

from src.parsing._models import (
    ParsedSection,
    ParsingConfig,
    ParsingSettings,
    QualityCheckResult,
    QualityReport,
    SectionLevel,
)

# --- Sample data for Indian Kanoon judgment HTML ---

SAMPLE_JUDGMENT_HTML = """\
<html>
<head><title>State Of Maharashtra vs Rajesh Kumar on 15 March, 2024</title></head>
<body>
<div class="doc_title">State Of Maharashtra vs Rajesh Kumar</div>
<div class="doc_bench">Justice A.B. Sharma, Justice C.D. Verma</div>
<div class="doc_citations">AIR 2024 SC 1500, (2024) 3 SCC 456</div>
<div class="judgments">
<p><b>IN THE SUPREME COURT OF INDIA</b></p>
<p>Criminal Appeal No. 1234 of 2023</p>
<p>State of Maharashtra ... Appellant</p>
<p>Versus</p>
<p>Rajesh Kumar ... Respondent</p>

<p><b>FACTS OF THE CASE</b></p>
<p>The appellant State of Maharashtra filed this appeal challenging the order
of the High Court of Bombay dated 10 January 2023. The respondent was charged
under Section 302 of the Indian Penal Code, 1860 for the alleged murder of
one Suresh Sharma on 5 May 2022. The prosecution case was that the respondent
had a longstanding property dispute with the deceased.</p>

<p><b>ISSUES FOR CONSIDERATION</b></p>
<p>1. Whether the High Court was correct in acquitting the respondent?</p>
<p>2. Whether the circumstantial evidence was sufficient to prove guilt
beyond reasonable doubt?</p>

<p><b>ANALYSIS AND REASONING</b></p>
<p>We have carefully considered the submissions made by learned counsel for
both parties. Section 302 IPC provides that whoever commits murder shall be
punished with death or imprisonment for life. The burden of proof lies on
the prosecution to establish guilt beyond reasonable doubt.</p>
<p>In K.M. Nanavati v. State of Maharashtra, AIR 1962 SC 605, this Court
held that circumstantial evidence must form a complete chain.</p>

<p><b>HOLDING</b></p>
<p>For the foregoing reasons, we hold that the High Court was correct in
its assessment. The circumstantial evidence does not form a complete chain.
The appeal is dismissed.</p>

<p><b>ORDER</b></p>
<p>The appeal is dismissed. No order as to costs.</p>
</div>
</body>
</html>
"""

# --- Sample data for Indian Kanoon statute HTML ---

SAMPLE_STATUTE_HTML = """\
<html>
<head><title>The Indian Contract Act, 1872</title></head>
<body>
<div class="doc_title">The Indian Contract Act, 1872</div>
<div class="judgments">
<p><b>THE INDIAN CONTRACT ACT, 1872</b></p>
<p>(Act No. 9 of 1872)</p>
<p>[25th April, 1872]</p>

<p>An Act to define and amend certain parts of the law relating to contracts.</p>

<p><b>CHAPTER I - PRELIMINARY</b></p>

<p><b>Section 1. Short title.</b></p>
<p>This Act may be called the Indian Contract Act, 1872.</p>

<p><b>Section 2. Interpretation clause.</b></p>
<p>In this Act the following words and expressions are used in the following
senses, unless a contrary intention appears from the context:—</p>
<p>(a) When one person signifies to another his willingness to do or to
abstain from doing anything, with a view to obtaining the assent of that
other to such act or abstinence, he is said to make a proposal;</p>
<p>(b) When the person to whom the proposal is made signifies his assent
thereto, the proposal is said to be accepted;</p>

<p><b>Section 10. What agreements are contracts.</b></p>
<p>All agreements are contracts if they are made by the free consent of
parties competent to contract, for a lawful consideration and with a
lawful object, and are not hereby expressly declared to be void.</p>
<p>Explanation.—Nothing herein contained shall be deemed to affect any law
in force in India, and not hereby expressly repealed, by which any contract
is required to be made in writing or in the presence of witnesses.</p>
<p>Provided that nothing in this section shall apply to contracts made
before the commencement of this Act.</p>

<p><b>CHAPTER II - OF CONTRACTS, VOIDABLE CONTRACTS AND VOID AGREEMENTS</b></p>

<p><b>Section 23. What considerations and objects are lawful, and what not.</b></p>
<p>The consideration or object of an agreement is lawful, unless—</p>
<p>it is forbidden by law; or</p>
<p>is of such a nature that, if permitted, it would defeat the provisions
of any law; or</p>
<p>is fraudulent; or</p>
<p>involves or implies injury to the person or property of another; or</p>
<p>the Court regards it as immoral, or opposed to public policy.</p>
</div>
</body>
</html>
"""

# --- Sample India Code detail page HTML ---

SAMPLE_INDIA_CODE_HTML = """\
<html>
<head>
<title>India Code: Academy of Scientific and Innovative Research Act, 2011</title>
</head>
<body>
<div class="item-page-field-wrapper">
<h2>The Academy of Scientific and Innovative Research Act, 2011</h2>
</div>
<table class="table itemDisplayTable">
<tr><td class="metadataFieldLabel">Act Number:</td>
    <td class="metadataFieldValue">13</td></tr>
<tr><td class="metadataFieldLabel">Enactment Date:</td>
    <td class="metadataFieldValue">6-Feb-2012</td></tr>
<tr><td class="metadataFieldLabel">Act Year:</td>
    <td class="metadataFieldValue">2011</td></tr>
</table>
<div class="file-list">
<a href="/bitstream/123456789/2110/1/a2012-13.pdf">
The Academy of Scientific and Innovative Research Act, 2011
</a>
</div>
</body>
</html>
"""


# --- Fixtures ---


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for parsed output."""
    out = tmp_path / "parsed"
    out.mkdir()
    return out


@pytest.fixture()
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Temporary directory for PDF cache."""
    cache = tmp_path / "cache" / "pdf"
    cache.mkdir(parents=True)
    return cache


@pytest.fixture()
def parsing_settings(tmp_path: Path) -> ParsingSettings:
    """ParsingSettings pointing at temp directories."""
    return ParsingSettings(
        input_dir=tmp_path / "raw",
        output_dir=tmp_path / "parsed",
        pdf_cache_dir=tmp_path / "cache" / "pdf",
    )


@pytest.fixture()
def parsing_config(parsing_settings: ParsingSettings) -> ParsingConfig:
    """ParsingConfig wrapping temp settings."""
    return ParsingConfig(settings=parsing_settings)


@pytest.fixture()
def sample_quality_report() -> QualityReport:
    """A passing quality report."""
    return QualityReport(
        overall_score=0.95,
        passed=True,
        checks=[
            QualityCheckResult(
                check_name="text_completeness",
                passed=True,
                score=0.95,
                details="Adequate text extracted",
            ),
        ],
    )


@pytest.fixture()
def sample_parsed_section() -> ParsedSection:
    """A sample statute section with children."""
    return ParsedSection(
        id="s10",
        level=SectionLevel.SECTION,
        number="10",
        title="What agreements are contracts",
        text="All agreements are contracts if they are made by the free consent...",
        token_count=15,
        children=[
            ParsedSection(
                id="s10_exp",
                level=SectionLevel.EXPLANATION,
                title=None,
                text="Nothing herein contained shall be deemed to affect any law...",
                parent_id="s10",
                token_count=12,
            ),
            ParsedSection(
                id="s10_prov",
                level=SectionLevel.PROVISO,
                title=None,
                text="Provided that nothing in this section shall apply...",
                parent_id="s10",
                token_count=10,
            ),
        ],
    )

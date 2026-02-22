from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

SAMPLE_INDIAN_KANOON_JUDGMENT = """
<html>
<head><title>State Of Maharashtra vs Bandishala on 15 March, 2024</title></head>
<body>
<div class="doc_title">State Of Maharashtra vs Bandishala on 15 March, 2024</div>
<div class="doc_bench">Supreme Court of India</div>
<div class="doc_citations">AIR 2024 SC 1500</div>
<div class="doc_author">Justice A.B. Patel</div>
<div class="judgments">
<p>This appeal is filed against the judgment of the High Court...</p>
<p>Section 302 of the Indian Penal Code is applicable in this case.</p>
<p>The appeal is dismissed. The conviction is upheld.</p>
</div>
</body>
</html>
"""

SAMPLE_INDIAN_KANOON_STATUTE = """
<html>
<head><title>Section 302 in The Indian Penal Code</title></head>
<body>
<div class="doc_title">Section 302 in The Indian Penal Code</div>
<div class="judgments">
<p><b>302. Punishment for murder.—</b>Whoever commits murder shall be punished with death,
or imprisonment for life, and shall also be liable to fine.</p>
</div>
</body>
</html>
"""

SAMPLE_INDIAN_KANOON_SEARCH = """
<html>
<head><title>Indian Kanoon - Search results</title></head>
<body>
<div class="results_middle_col">
<div class="result">
<div class="result_title"><a href="/doc/1001/">State vs Ram on 1 January 2024</a></div>
</div>
<div class="result">
<div class="result_title"><a href="/doc/1002/">Section 420 in The Indian Penal Code</a></div>
</div>
<div class="result">
<div class="result_title"><a href="/doc/1003/">Union of India vs Sharma on 5 February 2024</a></div>
</div>
</div>
</body>
</html>
"""

SAMPLE_INDIA_CODE_BROWSE = """
<html>
<head><title>India Code: Browsing DSpace</title></head>
<body>
<div id="content">
<h4>Browsing "Central Act Legislation" by Short Title</h4>
<div>Showing items 1 to 3 of 848</div>
<table class="table" summary="This table browses all dspace content">
<thead>
<tr>
<th>Enactment Date</th>
<th>Act Number</th>
<th>Short Title</th>
<th>View</th>
</tr>
</thead>
<tbody>
<tr>
<td>6-Oct-1860</td>
<td><em>45</em></td>
<td><strong>The Indian Penal Code, 1860</strong></td>
<td><a href="/handle/123456789/1505?view_type=browse">View...</a></td>
</tr>
<tr>
<td>9-Jun-2000</td>
<td><em>21</em></td>
<td><strong>The Information Technology Act, 2000</strong></td>
<td><a href="/handle/123456789/1999?view_type=browse">View...</a></td>
</tr>
<tr>
<td>25-Dec-2023</td>
<td><em>45</em></td>
<td><strong>The Bharatiya Nyaya Sanhita, 2023</strong></td>
<td><a href="/handle/123456789/22501?view_type=browse">View...</a></td>
</tr>
</tbody>
</table>
<div>Showing items 1 to 3 of 848</div>
</div>
</body>
</html>
"""

SAMPLE_INDIA_CODE_BROWSE_EMPTY = """
<html>
<head><title>India Code: Browsing DSpace</title></head>
<body>
<div id="content">
<h4>Browsing "Central Act Legislation" by Short Title</h4>
<table class="table" summary="This table browses all dspace content">
<thead>
<tr>
<th>Enactment Date</th>
<th>Act Number</th>
<th>Short Title</th>
<th>View</th>
</tr>
</thead>
<tbody>
</tbody>
</table>
</div>
</body>
</html>
"""

SAMPLE_INDIA_CODE_DETAIL = """
<html>
<head><title>India Code: Information Technology Act, 2000</title></head>
<body>
<div id="content">
<div class="item-summary-view-metadata">
<a href="/bitstream/123456789/1999/1/A2000-21%20%281%29.pdf">
<p>The Information Technology Act, 2000<img src="/image/pdf-icon.png"/></p>
</a>
<a href="/bitstream/123456789/1999/2/H2000-21.pdf">
<p>सूचना प्रौद्योगिकी अधिनियम, 2000<img src="/image/pdf-icon.png"/></p>
</a>
</div>
<ul class="nav nav-tabs">
<li><a href="#tb1">Sections</a></li>
<li><a href="#tb10">Schedule</a></li>
<li><a href="#tb2">Actdetails</a></li>
</ul>
</div>
</body>
</html>
"""

SAMPLE_INDIA_CODE_DETAIL_NO_PDF = """
<html>
<head><title>India Code: Some Act</title></head>
<body>
<div id="content">
<div class="item-summary-view-metadata">
<p>No PDF available for this act.</p>
</div>
</div>
</body>
</html>
"""


@pytest.fixture()
def tmp_state_dir(tmp_path: Path) -> Path:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return state_dir


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "raw"
    output_dir.mkdir()
    return output_dir


@pytest.fixture()
def sample_search_html() -> str:
    return SAMPLE_INDIAN_KANOON_SEARCH


@pytest.fixture()
def sample_judgment_html() -> str:
    return SAMPLE_INDIAN_KANOON_JUDGMENT


@pytest.fixture()
def sample_statute_html() -> str:
    return SAMPLE_INDIAN_KANOON_STATUTE


@pytest.fixture()
def sample_india_code_browse_html() -> str:
    return SAMPLE_INDIA_CODE_BROWSE


@pytest.fixture()
def sample_india_code_browse_empty_html() -> str:
    return SAMPLE_INDIA_CODE_BROWSE_EMPTY


@pytest.fixture()
def sample_india_code_detail_html() -> str:
    return SAMPLE_INDIA_CODE_DETAIL


@pytest.fixture()
def sample_india_code_detail_no_pdf_html() -> str:
    return SAMPLE_INDIA_CODE_DETAIL_NO_PDF

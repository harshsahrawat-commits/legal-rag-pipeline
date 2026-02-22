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

SAMPLE_INDIA_CODE_ACT = """
<html>
<head><title>The Indian Penal Code, 1860</title></head>
<body>
<div class="actTitle">THE INDIAN PENAL CODE, 1860</div>
<div class="actNumber">Act No. 45 of 1860</div>
<div class="enactmentDate">[6th October, 1860]</div>
<div class="sections">
<div class="section">
<h3>Section 1. Title and extent of operation of the Code.</h3>
<p>This Act shall be called the Indian Penal Code, and shall extend to the whole of India
except the State of Jammu and Kashmir.</p>
</div>
<div class="section">
<h3>Section 2. Punishment of offences committed within India.</h3>
<p>Every person shall be liable to punishment under this Code and not otherwise for every act
or omission contrary to the provisions thereof.</p>
</div>
</div>
</body>
</html>
"""

SAMPLE_INDIA_CODE_LISTING = """
<html>
<head><title>India Code - Act Listing</title></head>
<body>
<div class="actListing">
<table>
<tr>
<td><a href="/handle/123456789/1362">The Indian Penal Code, 1860</a></td>
<td>Act No. 45 of 1860</td>
</tr>
<tr>
<td><a href="/handle/123456789/2263">The Code of Criminal Procedure, 1973</a></td>
<td>Act No. 2 of 1974</td>
</tr>
</table>
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
def sample_india_code_act_html() -> str:
    return SAMPLE_INDIA_CODE_ACT


@pytest.fixture()
def sample_india_code_listing_html() -> str:
    return SAMPLE_INDIA_CODE_LISTING

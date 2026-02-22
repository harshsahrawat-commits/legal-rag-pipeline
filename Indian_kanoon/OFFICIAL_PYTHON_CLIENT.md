# Indian Kanoon Official Python Client (IKAPI)

> Source: https://github.com/sushant354/IKAPI
> License: MIT
> Latest Release: 1.0 (February 2026)

## Setup

```bash
git clone https://github.com/sushant354/IKAPI.git
cd IKAPI/python
pip install -r requirements.txt
```

### Requirements

```
beautifulsoup4
```

## CLI Usage

```bash
python ikapi.py -D <datadir> -s <token> [options]
```

### Required Arguments

| Flag | Description |
|------|-------------|
| `-D DATADIR` | Directory to store downloaded files |
| `-s TOKEN` | API shared token |

### Query Options

| Flag | Description |
|------|-------------|
| `-q QUERY` | Search query string |
| `-Q QFILE` | File containing queries (one per line) |
| `-d DOCID` | Specific document ID to download |
| `-c DOCTYPE` | Download all docs of a type (e.g., `supremecourt`) |

### Filtering

| Flag | Description |
|------|-------------|
| `-f FROMDATE` | From date (DD-MM-YYYY) |
| `-t TODATE` | To date (DD-MM-YYYY) |
| `-a` | Only documents added today |
| `-S SORTBY` | Sort by `mostrecent` or `leastrecent` |

### Citation Options

| Flag | Description |
|------|-------------|
| `-m MAXCITES` | Max citations to fetch per doc (up to 50) |
| `-M MAXCITEDBY` | Max cited-by refs to fetch per doc (up to 50) |
| `-C DOCID [DOCID ...]` | Fetch cited-by for list of doc IDs |
| `-r` | Process next level of cited-by chain |

### Download Options

| Flag | Description |
|------|-------------|
| `-o` | Download original court copies (PDF/HTML) |
| `-p MAXPAGES` | Max search result pages (default: 1, max: 100) |
| `-P` | Organize files by source court |
| `-N NUMWORKERS` | Parallel download workers (default: 5) |

### Output Options

| Flag | Description |
|------|-------------|
| `-x` | Disable CSV output |
| `-n` | Count results only (don't download) |
| `-l LOGLEVEL` | Log level: error, warning, info, debug |
| `-g LOGFILE` | Log to file |

## Examples

```bash
# Search and download Supreme Court judgments about murder
python ikapi.py -D ./data -s YOUR_TOKEN -q "murder" -c supremecourt -p 5

# Download a specific document with citations
python ikapi.py -D ./data -s YOUR_TOKEN -d 12345 -m 50 -M 50 -o

# Batch download from query file
python ikapi.py -D ./data -s YOUR_TOKEN -Q queries.txt -N 10

# Count results for a query without downloading
python ikapi.py -D ./data -s YOUR_TOKEN -q "Section 302 IPC" -n

# Get documents added today from Delhi High Court
python ikapi.py -D ./data -s YOUR_TOKEN -c delhi -a
```

## Key Classes

### `IKApi`

Main API client. Authenticates via token header.

```python
class IKApi:
    def __init__(self, args, storage):
        self.headers = {'Authorization': 'Token %s' % args.token,
                        'Accept': 'application/json'}
        self.basehost = 'api.indiankanoon.org'
```

**Core methods:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `search(q, pagenum, maxpages)` | `/search/` | Search with pagination |
| `fetch_doc(docid)` | `/doc/<id>/` | Full document + citations |
| `fetch_docmeta(docid)` | `/docmeta/<id>/` | Metadata only |
| `fetch_orig_doc(docid)` | `/origdoc/<id>/` | Original court copy |
| `fetch_doc_fragment(docid, q)` | `/docfragment/<id>/` | Search within doc |
| `download_doc(docid, docpath)` | — | Fetch + save to disk |
| `save_search_results(q)` | — | Paginate search + save all |
| `fetch_citedby_docs(docid)` | — | Get all citing documents |
| `execute_tasks(queries)` | — | Parallel multi-query download |

**Retry logic:** `call_api()` retries up to 3 times with exponential backoff (`sleep(count * 10)`).

### `FileStorage`

Handles saving documents to disk, organized by source court and date.

```python
class FileStorage:
    def get_docpath(self, docsource, publishdate):
        # Creates: datadir/court/year/date/
        ...
    def get_json_orig_path(self, docpath, docid):
        # Returns: docpath/docid.json, docpath/docid_original
        ...
    def save_original(self, orig, origpath):
        # Decodes base64, determines extension from Content-Type
        ...
```

## File Organization

When using `-P` (path by source), documents are organized as:

```
datadir/
├── Supreme Court of India/
│   ├── 2024/
│   │   ├── 2024-01-15/
│   │   │   ├── 12345.json
│   │   │   └── 12345_original.pdf
│   │   └── 2024-02-20/
│   │       └── 67890.json
│   └── 2025/
│       └── ...
├── Delhi High Court/
│   └── ...
└── toc.csv   ← table of contents with docid, title, position, cites, date, court
```

## Integration Notes for Our Pipeline

The official client gives us a reference for:
1. **Authentication** — simple token header approach
2. **Retry logic** — 3 retries with `sleep(count * 10)` backoff
3. **Pagination** — `maxpages` for bulk fetching, handle empty `docs` array to stop
4. **Citation traversal** — `fetch_citedby_docs` + recursive `process_level` for citation graphs
5. **Original documents** — base64 decode + Content-Type → file extension mapping
6. **File organization** — by source court and date

When we refactor `IndianKanoonScraper` to use the API, we'll adopt the token auth and retry patterns but integrate with our existing `BaseScraper` / `HttpClient` / `CrawlStateStore` architecture rather than using `FileStorage` directly.

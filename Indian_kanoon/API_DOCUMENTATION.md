# Indian Kanoon API — Complete Documentation

> Source: https://api.indiankanoon.org/documentation/

## Base URL

```
https://api.indiankanoon.org
```

All requests use **POST** method. Results in JSON or XML via `Accept` header.
Page numbering starts at **0**.

---

## Authentication

### Option 1: API Token (simpler)

```
Authorization: Token <your_token>
```

### Option 2: Public-Private Key (HMAC-SHA256)

Three headers required:

| Header | Value |
|--------|-------|
| `X-Customer` | Registered email address |
| `X-Message` | Base64-encoded unique message (UUID or concatenated client IP + timestamp + URL) |
| `Authorization` | `HMAC <base64-encoded-signature>` |

The API drops duplicate messages to prevent replay attacks.

**Key Generation (OpenSSL):**
```bash
openssl genrsa -out private.pem 2048
openssl rsa -in private.pem -outform PEM -pubout -out public.pem
```

**Signing (Python):**
```python
from Crypto.PublicKey import RSA
from Crypto.Hash import SHA256
from Crypto.Signature import PKCS1_v1_5

def sign_message(message):
    key = open('private.pem', 'r').read()
    private_key = RSA.importKey(key)
    signer = PKCS1_v1_5.new(private_key)
    digest = SHA256.new(message)
    return signer.sign(digest)
```

---

## Endpoints

### 1. Search

```
POST /search/?formInput=<query>&pagenum=<pagenum>
```

**Parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `formInput` | Query string. Supports phrases (`"text"`), operators (`ANDD`, `ORR`, `NOTT` — case-sensitive, spaces required) | `"freedom of speech"`, `murder ORR kidnapping` |
| `pagenum` | Page number, starts at 0 | `0` |
| `maxpages` | Fetch multiple pages in one call (max 1000). Billed only for pages returned | `50` |
| `doctypes` | Filter by court/tribunal (comma-separated) | `supremecourt,delhi` |
| `fromdate` | Min date, DD-MM-YYYY | `1-10-2016` |
| `todate` | Max date, DD-MM-YYYY | `1-10-2016` |
| `title` | Words that must appear in document title | `murder` |
| `cite` | Filter by citation | `1993 AIR` |
| `author` | Filter by judge name | `arijit pasayat` |
| `bench` | Filter by bench judges | `arijit pasayat` |
| `maxcites` | Include up to 50 citations per result | `20` |
| `docsize` | Returns character count in results | — |

**Response fields:**
- `found` — total results count
- `docs` — array of result objects, each with:
  - `tid` — document ID
  - `title` — document title
  - `headline` — matching text snippet
  - `docsource` — court/source name
  - `publishdate` — publication date
  - `numcites` — number of citations
  - `numcitedby` — number of citing documents
- `categories` — faceted filters
- `encodedformInput` — URL-encoded query

**Available `doctypes` values:**

Courts: `supremecourt`, `delhi`, `bombay`, `kolkata`, `chennai`, `allahabad`, `andhra`, `chattisgarh`, `gauhati`, `jammu`, `srinagar`, `kerala`, `lucknow`, `orissa`, `uttaranchal`, `gujarat`, `himachal_pradesh`, `jharkhand`, `karnataka`, `madhyapradesh`, `patna`, `punjab`, `rajasthan`, `sikkim`, `kolkata_app`, `jodhpur`, `patna_orders`, `meghalaya`, `delhidc`

Tribunals: `aptel`, `drat`, `cat`, `cegat`, `stt`, `itat`, `consumer`, `cerc`, `cic`, `clb`, `copyrightboard`, `ipab`, `mrtp`, `sebisat`, `tdsat`, `trademark`, `greentribunal`, `cci`

Aggregators: `tribunals`, `judgments`, `laws`, `highcourts`

---

### 2. Document

```
POST /doc/<docid>/
```

Returns full document HTML and metadata.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `maxcites` | Up to 50 cited documents in `citeList` field |
| `maxcitedby` | Up to 50 citing documents in `citedbyList` field |

**Example:**
```
/doc/12345/?maxcites=10&maxcitedby=20
```

Default returns top 5 in each list without parameters.

**Response includes:**
- `doc` — full HTML of the document
- `title` — document title
- `courtcopy` — whether original court copy is available
- `citeList` — list of cited documents
- `citedbyList` — list of documents citing this one

---

### 3. Document Metadata

```
POST /docmeta/<docid>/
```

Returns document information only (no full text). Cheapest endpoint (₹0.02/request).

**Parameters:** Same as Document (`maxcites`, `maxcitedby`).

---

### 4. Original Court Copy

```
POST /origdoc/<docid>/
```

Returns the original court document (PDF/HTML/etc.) as base64-encoded content.

**Response:**
- `doc` — base64-encoded document content
- `Content-Type` — MIME type (`application/pdf`, `text/html`, etc.)

---

### 5. Document Fragments

```
POST /docfragment/<docid>/?formInput=<query>
```

Returns sections of a document matching the search query. Cheapest per-request cost (₹0.05).

**Response:**
- `formInput` — the query
- `title` — document title
- `tid` — document ID
- `headline` — matching text fragments

---

## Structural Analysis

The Document endpoint returns paragraphs in court judgments classified into **8 categories**:
1. Facts
2. Issues
3. Argument by petitioner
4. Argument by respondent
5. Precedent Analysis
6. Analysis of Law
7. Court's Discourse
8. Conclusion

This is embedded in the HTML returned by `/doc/` endpoint.

---

## Pricing

| Endpoint | Cost (INR) | Cost (approx USD) |
|----------|------------|-------------------|
| Search | ₹0.50 | $0.006 |
| Document | ₹0.20 | $0.0024 |
| Original Document | ₹0.50 | $0.006 |
| Document Fragment | ₹0.05 | $0.0006 |
| Document Metainfo | ₹0.02 | $0.00024 |

**Free tier:**
- New users: ₹500 complimentary credit
- Non-commercial use: ₹10,000/month (requires admin verification)

**Billing:** Prepaid, per-request deduction. `maxpages` only bills for pages actually returned.

---

## Rate Limits

No explicit rate limit documented. The API is designed for programmatic use. The official Python client uses retry logic (up to 3 attempts with exponential backoff).

---

## Key Advantages Over HTML Scraping

1. **Structured JSON** — no HTML parsing needed
2. **Structural analysis** — judgment sections pre-classified (Facts, Issues, Reasoning, Conclusion)
3. **Citation graph** — `maxcites`/`maxcitedby` give linked documents
4. **Court filtering** — `doctypes` parameter for precise source targeting
5. **Date filtering** — `fromdate`/`todate` for temporal queries
6. **Original court copies** — PDF originals via `/origdoc/`
7. **Stable contract** — API vs fragile HTML scraping
8. **Legal clarity** — official API with clear ToS vs scraping gray area

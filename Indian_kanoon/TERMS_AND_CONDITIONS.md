# Indian Kanoon API — Terms and Conditions

> Source: https://api.indiankanoon.org/terms/
> Retrieved: 2026-02-22

---

## Service Description

Indiankanoon.org provides access to Indian legal documents and search functionality over these materials. The site displays "document fragments" containing query matches. The IKanoon Application Programming Interface enables programmatic access to search functionality, documents, and document fragments for display on third-party websites or products.

## Search Features

Users may request new search filters or ranking features. IKanoon reserves discretionary authority regarding implementation and bears no liability for refusals.

## Non-Exclusive Features

IKanoon provides no exclusive search features to any user. New filters and ranking features are universally available across the API and indiankanoon.org website, with updates posted on the search tips page.

## Authentication

All API requests require public-private key authentication (detailed in documentation). Users are responsible for securing their private key and updating it if compromised. IKanoon provides no refunds for misused keys.

## Usage Evidence

IKanoon supplies 30-day usage evidence including IP addresses, queries, and signed certificates.

## Consideration — Attribution and Logos

Users must provide **"clear and conspicuous attribution"** when displaying search results, documents, or classifiers — whether for:
- Direct display to users
- Retrieval-Augmented Generation (RAG) systems
- Language model fine-tuning

The **"powered by IKanoon" logo** must appear prominently:
- **Direct display:** Top of search results
- **Integrated use (RAG/LLM):** About section or product documentation

Logo requirements:
- Must be full and clearly visible
- Never altered, resized, or partially covered

**Prohibited logo uses:**
- Implying endorsement of other products/services
- Suggesting IKanoon authored editorial content
- Representing IKanoon personnel's views
- Misleading, unfair, defamatory, infringing, libelous, disparaging, or obscene applications

## Pricing

Users agree to charges per the pricing page. Balances deduct automatically. IKanoon may revise pricing with one-week advance notice; users may renew at revised rates or terminate.

**Pricing (as of Feb 2026):**

| Endpoint | Cost (INR) |
|----------|------------|
| Search | ₹0.50 |
| Document | ₹0.20 |
| Original Document | ₹0.50 |
| Document Fragment | ₹0.05 |
| Document Metainfo | ₹0.02 |

## Refusal on Non-Availability of Funds

The service operates on pre-paid mode; exhausted balances prevent API result returns.

## Representations and Warranties — Service Availability

IKanoon maintains best efforts for highly available services but assumes no liability for unavailability due to unforeseen circumstances.

## Quality

All API and indiankanoon.org users receive identical search results for identical queries (excluding inadvertent caching differences). No systematic user bias occurs.

## Liability Limitations

Data sourced from public government websites (decisions, statutes, orders, notifications) are provided **"AS IS"** without accuracy, reliability, or fitness warranties. IKanoon assumes no liability for claims, damages, or liability arising from data or service use.

## Termination

Either party may terminate by registered email or phone with **one-month advance notice**.

## Governing Law and Dispute Resolution

- Governed by **Indian law**
- Subject to **Bangalore court jurisdiction**
- Disputes attempted amicably first
- Unresolved disputes proceed to **arbitration** under the Arbitration and Conciliation Act, 1996
- Joint arbitrator appointment, Bangalore venue

Neither party may assign the agreement (including to affiliates) without written consent, reasonably withheld. Force majeure events exempt parties from performance claims.

---

## Compliance Checklist for Our Project

- [x] Attribution: Add "powered by IKanoon" logo in project documentation / About section
- [x] Non-commercial: Applied for non-commercial use (₹10,000/month free tier)
- [x] No redistribution: Raw API data not served to end users directly
- [x] Key security: Store API token in `.env` (gitignored), never committed
- [x] Prepaid monitoring: Track credit balance, alert before exhaustion

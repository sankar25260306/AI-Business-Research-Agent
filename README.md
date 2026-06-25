# 🔍 AI Business Research Agent

> **Multi-source business intelligence pipeline** — natural language query → verified, deduplicated, AI-enriched business records, streamed live to your browser.

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [16-Phase Pipeline](#16-phase-pipeline)
- [Agent Breakdown](#agent-breakdown)
- [Data Model](#data-model)
- [WebSocket Streaming Protocol](#websocket-streaming-protocol)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Configuration & Secrets](#configuration--secrets)
- [Local Development](#local-development)
- [Deployment (Hugging Face Spaces)](#deployment-hugging-face-spaces)
- [Key Design Decisions](#key-design-decisions)
- [Advantages & Differentiators](#advantages--differentiators)

---

## Overview

The AI Business Research Agent takes a plain-English query like **"Cardiologists in Chennai"** and returns a verified, deduplicated list of businesses with names, addresses, phones, emails, websites, hours, ratings, services, certifications, and social profiles — all sourced from 14+ directories and search engines in a single pipeline run, streamed to the user in real time.

The system is a **FastAPI backend + Streamlit frontend** packaged in one Docker container, making it one-click deployable to Hugging Face Spaces.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Groq (`llama-3.1-8b-instant`) | Query parsing, data extraction, report summarisation |
| **Agent orchestration** | LangChain + LangGraph | Prompt chaining, typed state machine for report generation |
| **Web scraping** | Scrapy (child process) + BeautifulSoup 4 | Concurrent page fetch, HTML parsing |
| **Search engines** | DuckDuckGo (`ddgs`), Bing (aiohttp scrape) | Free, no-key, ToS-safe web search |
| **Directories** | Yelp, Yellow Pages, Justdial, Sulekha, LinkedIn, Facebook | Business listing sources |
| **Fuzzy dedup** | RapidFuzz | Name + address similarity matching |
| **Vector dedup** | ChromaDB + `sentence-transformers` | Semantic similarity deduplication (optional) |
| **Database** | SQLite via `DatabaseStorage` | Search history, business cache |
| **Vector store** | ChromaDB `VectorStore` | Semantic search over stored records |
| **Real-time streaming** | FastAPI WebSockets | Live pipeline events → Streamlit UI |
| **Frontend** | Streamlit + custom HTML/CSS/JS dashboard | Search bar, live progress, results viewer |
| **Container** | Docker | Single-container deploy (FastAPI on :8000, Streamlit on :7860) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Docker Container                        │
│                                                                 │
│  ┌─────────────────────┐        ┌──────────────────────────┐   │
│  │   Streamlit (7860)  │◄──────►│   FastAPI (8000)         │   │
│  │                     │  WS +  │                          │   │
│  │  • Search bar       │  REST  │  • WebSocket /ws/research│   │
│  │  • Live progress    │        │  • REST /api/export      │   │
│  │  • HTML dashboard   │        │  • REST /api/history     │   │
│  └─────────────────────┘        └────────────┬─────────────┘   │
│                                              │                  │
│                                   ┌──────────▼──────────┐      │
│                                   │   16-Phase Pipeline  │      │
│                                   │  (run_pipeline())    │      │
│                                   └──────────┬──────────┘      │
│                                              │                  │
│         ┌────────────────────────────────────▼──────────────┐  │
│         │                   Agents                           │  │
│         │  QueryAgent → SearchAgent → ScraperAgent          │  │
│         │  → ExtractionAgent → VerificationAgent            │  │
│         │  → ReliabilityScoringAgent → DeduplicationAgent   │  │
│         │  → SummaryAgent                                    │  │
│         └───────────────────────────────────────────────────┘  │
│                                                                 │
│         ┌──────────────────────────────────┐                   │
│         │  Core Storage                    │                   │
│         │  DatabaseStorage (SQLite)        │                   │
│         │  VectorStore (ChromaDB)          │                   │
│         └──────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

External data sources fetched at runtime:
- **Search engines:** Bing, DuckDuckGo
- **Directories:** Yelp, Yellow Pages, Justdial, Sulekha, LinkedIn, Facebook
- **Specialty:** Healthcare dirs (Practo, Healthgrades, Zocdoc), Legal dirs (Avvo, FindLaw), Gov licensing (`.gov`), Industry dirs, Professional associations, Review platforms (Trustpilot, BBB)

---

## 16-Phase Pipeline

Each phase emits a `PipelineEvent` over WebSocket so the frontend progress bar and status log update in real time.

| Phase | Name | Agent / Component | What Happens |
|---|---|---|---|
| **1** | Query Agent | `QueryAgent` | LLM parses query → `{category, location, search_variations}` |
| **2** | Search Agent | `SearchAgent` | Fires concurrent searches across 7+ sources, collects candidate URLs |
| **3** | Concurrent Search | `SearchAgent` | Aggregates, deduplicates, and filters junk/blocked domains from URLs |
| **4** | URL Collection | pipeline | Final URL set assembled and deduplicated |
| **5** | Web Scraper | `ScraperAgent` | Scrapy spider (child process) concurrently fetches up to 20 pages |
| **6** | Extraction Agent | `ExtractionAgent` | LLM reads each page's HTML and fills structured business JSON |
| **7** | Verification Agent | `VerificationAgent` | Groups records for the same business, detects field conflicts |
| **8** | Reliability Scoring | `ReliabilityScoringAgent` | Weights each source (gov = 0.95, official = 1.0, DDG = 0.60…) and computes `reliability_score` |
| **9** | Deduplication | `DeduplicationAgent` | 5-strategy dedup: exact phone, fuzzy name, fuzzy address, website domain, email, (+ optional vector) |
| **10** | Save Search | `DatabaseStorage` | Persists query metadata to SQLite |
| **11** | Save Businesses | `DatabaseStorage` | Stores verified + deduped business records |
| **12** | Summary Agent | `SummaryAgent` | LangGraph state machine: stats → quality → AI summary |
| **13** | Report Generator | `SummaryAgent` | Assembles final `ResearchReport` dict |
| **14** | FastAPI Server | `server.py` | WebSocket streams `summary` event with full report to frontend |
| **15** | Export Ready | REST `/api/export` | Report available as JSON or CSV download |
| **16** | Complete | frontend | Streamlit renders HTML dashboard from report dict, no page reload |

---

## Agent Breakdown

### 1 · QueryAgent (`agents/query_agent.py`)

Turns a free-text query into a structured search spec.

- **LLM:** `llama-3.1-8b-instant` via Groq, `temperature=0` for determinism
- **Prompt strategy:** strict JSON-only system prompt; zero markdown, no code fences
- **Output:** `{category, location, location_type, search_variations[3], intent}`
- **Fallback:** regex `"X in Y"` pattern when LLM output is unparsable
- **Filters junk domains** (Wikipedia, Amazon, Reddit, news sites) before any URL is used

---

### 2 · SearchAgent (`agents/search_agent.py`)

Multi-source concurrent URL harvesting.

- **Sources:** DuckDuckGo (`ddgs`), Bing (aiohttp), Yelp, Yellow Pages, Justdial, Sulekha, LinkedIn, Facebook (via site: searches), Healthcare dirs, Legal dirs, Gov licensing, Industry dirs, Review platforms
- **Note on Google:** Google's HTML is aggressively anti-scraping; the agent surfaces Google-indexed pages through Bing/DDG results instead
- **URL filtering:** blocks `BLOCKED_DOMAINS` (200+ noise domains), blocked URL patterns (ad redirects), junk titles (dictionary definitions, news articles), and enforces `POSITIVE_URL_PATTERNS` / `POSITIVE_RESULT_KEYWORDS` scoring
- **Trusted domains** bypass keyword matching entirely

---

### 3 · ScraperAgent (`agents/scraper_agent.py`)

Fetches raw HTML from candidate URLs.

- **Engine:** Scrapy spider run in a **child process** via `multiprocessing` — isolates Twisted reactor from FastAPI's asyncio event loop entirely
- **Directory pages** (Yelp, Justdial, etc.) return seed data directly without a full fetch to avoid bot detection
- **Config:** AutoThrottle, concurrent requests, retry middleware, timeouts — all Scrapy-native
- **Output:** `{url, html, seed_data}` per page

---

### 4 · ExtractionAgent (`agents/extraction_agent.py`)

LLM-powered field extraction from raw HTML.

- **LLM:** `llama-3.1-8b-instant`, `temperature=0`, 1024 max tokens
- **Input:** first 3,000 chars of HTML + scraper seed data (stays within token budget)
- **Output schema:** 16 fields — `business_name, address, phone, email, website, working_hours, rating, review_count, services[], specialties[], license_information, certifications[], awards[], social_profiles[], images_urls[]`
- **Merge logic:** LLM fills fields that BeautifulSoup missed; scraper seed values are never overwritten by empty LLM output
- **Rules enforced in prompt:** no invention, validate email format, normalise phone, `""` / `[]` for missing fields

---

### 5 · VerificationAgent (`agents/verification_agent.py`)

Cross-source validation and conflict detection.

- Groups records that refer to the same business (by phone, name, address proximity)
- Merges conflicting field values (e.g., two different phone numbers from two sources → `conflicts[]` list + most-trusted-source wins)
- Produces `verification_score` (0–100): higher score = more sources agree

**Source reliability weights:**

| Source | Weight |
|---|---|
| Business official website | 1.00 |
| Government licensing DB | 0.95 |
| Professional association | 0.80 |
| Healthcare / legal directory | 0.80 |
| Yellow Pages | 0.85 |
| Yelp | 0.80 |
| LinkedIn | 0.75 |
| Justdial | 0.70 |
| Bing | 0.65 |
| DuckDuckGo | 0.60 |
| Facebook | 0.60 |
| Unknown | 0.50 |

---

### 6 · DeduplicationAgent (`agents/dedup_agent.py`)

Five-strategy deduplication pipeline (runs in order, short-circuits on first match):

1. **Exact phone match** — normalised digits compared
2. **RapidFuzz name similarity** — threshold 88%
3. **RapidFuzz address similarity** — threshold 85%
4. **Website domain match** — parsed from URL
5. **Email match** — normalised lowercase
6. **ChromaDB vector similarity** *(optional, if `sentence-transformers` installed)* — cosine distance < 0.18

Duplicates are **merged**, not discarded — the richest field values from all copies are combined into one record.

---

### 7 · SummaryAgent (`agents/summary_agent.py`)

LangGraph state-machine report generator.

**Graph nodes (run in sequence):**

```
compute_stats → compute_quality → generate_summary → build_report → END
```

- `compute_stats`: counts businesses, sources, coverage metrics
- `compute_quality`: scores data completeness per field across all records
- `generate_summary`: `llama-3.1-8b-instant` writes a 3–5 sentence executive summary
- `build_report`: assembles final `ResearchReport` dataclass → dict for WebSocket delivery

---

## Data Model

### BusinessRecord

```python
business_name       str
address             str
phone               str
email               str
website             str
working_hours       str
rating              str
review_count        str
services            List[str]
specialties         List[str]
license_information str
certifications      List[str]
awards              List[str]
social_profiles     List[str]
images_urls         List[str]
source_urls         Dict[str, str]   # source_name → url
verification_score  float            # 0–100
reliability_score   float            # 0–1
sources_count       int
conflicts           List[str]        # field-level conflict descriptions
```

### ResearchReport

```python
query               str
category            str
location            str
businesses_found    int
businesses_verified int
duplicates_removed  int
sources_searched    int
duration_seconds    float
businesses          List[BusinessRecord.to_dict()]
data_quality        Dict[str, str]   # field → coverage %
ai_summary          str
```

---

## WebSocket Streaming Protocol

Endpoint: `ws://<host>:8000/ws/research`

**Client → Server** (once, on connect):
```json
{ "query": "Cardiologists in Chennai" }
```

**Server → Client** (many, during pipeline):
```json
{
  "event_type": "status",     // "status" | "business" | "summary" | "error"
  "phase": 5,
  "phase_name": "Web Scraper",
  "message": "Scraping 18 pages (Scrapy + BS4)…",
  "data": null,
  "progress": 35.0            // 0–100
}
```

`event_type` values:

| Type | When | `data` field |
|---|---|---|
| `status` | Each pipeline phase | `null` |
| `business` | Each business found | `BusinessRecord dict` |
| `summary` | Pipeline complete | `ResearchReport dict` |
| `error` | Any failure | `null` |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `WS` | `/ws/research` | Full streaming pipeline |
| `GET` | `/api/history` | Recent search queries from SQLite |
| `GET` | `/api/export/{session_id}` | Download report as JSON |
| `GET` | `/api/export/{session_id}?format=csv` | Download report as CSV |

---

## Project Structure

```
business_research_agent/
│
├── app.py                    # Streamlit frontend (search bar + live pipeline + HTML dashboard)
├── server.py                 # FastAPI backend (WebSocket orchestrator, REST endpoints)
├── start.sh                  # Container entrypoint: starts FastAPI (bg) + Streamlit (fg)
├── Dockerfile                # Python 3.11-slim; exposes :7860
├── requirements.txt          # Pinned dependencies
│
├── agents/
│   ├── query_agent.py        # Phase 1  — LangChain query parser
│   ├── search_agent.py       # Phase 2+3 — Multi-source URL harvester
│   ├── scraper_agent.py      # Phase 4+5 — Scrapy child-process spider
│   ├── extraction_agent.py   # Phase 6  — LLM field extractor
│   ├── verification_agent.py # Phase 7+8 — Cross-source verifier + reliability scorer
│   ├── dedup_agent.py        # Phase 9  — 5-strategy deduplicator
│   ├── summary_agent.py      # Phase 12+13 — LangGraph report generator
│   └── __init__.py
│
└── core/
    ├── config.py             # Pydantic models, dataclasses, API key loader, source list
    ├── storage.py            # DatabaseStorage (SQLite) + VectorStore (ChromaDB)
    └── __init__.py
```

---

## Configuration & Secrets

| Variable | Where | Description |
|---|---|---|
| `GROQ_API_KEY` | HF Space → Settings → Secrets | Required. Powers all LLM calls (QueryAgent, ExtractionAgent, SummaryAgent) |

No other API keys are required. All search sources are scraped without authentication.

---

## Local Development

```bash
# Build
docker build -t research-agent .

# Run (set your Groq key)
docker run -p 7860:7860 -e GROQ_API_KEY=gsk_... research-agent

# Open
open http://localhost:7860
```

Without Docker:
```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...

# Terminal 1 — backend
uvicorn server:app --port 8000 --reload

# Terminal 2 — frontend
streamlit run app.py --server.port 7860
```

---

## Deployment (Hugging Face Spaces)

1. Create a new Space → **Docker** SDK → port `7860`
2. Push this repository to the Space's Git remote
3. Add `GROQ_API_KEY` under **Settings → Variables and secrets**
4. HF Spaces builds the Docker image and starts the container automatically

The `start.sh` script launches FastAPI on port 8000 in the background and Streamlit on port 7860 in the foreground. Only port 7860 is exposed externally; FastAPI is internal-only.

---

## Key Design Decisions

**Single container, two processes** — Streamlit (user-facing) and FastAPI (pipeline engine) run in the same Docker container and communicate over `localhost`. This avoids cross-origin issues and simplifies HF Spaces deployment (one port, one container).

**Scrapy in a child process** — Scrapy's Twisted event loop cannot share a thread with FastAPI's asyncio loop. Spawning a fresh child process for each scrape batch sidesteps the conflict completely and allows clean Twisted teardown.

**LLM for extraction, not search** — Web search is done with free, no-key engines (DDG, Bing scrape) to avoid API costs. The LLM is used only for the steps where structured reasoning adds value: query parsing and field extraction from noisy HTML.

**5-strategy dedup in order** — Exact signals (phone, email, domain) run first and short-circuit — they are cheap and unambiguous. Fuzzy and vector strategies run only on records that exact matching didn't resolve.

**WebSocket streaming over polling** — The Streamlit UI subscribes to a single long-lived WebSocket connection and renders each event as it arrives. No polling, no session ID round-trips, no page reloads for the results dashboard.

**Source reliability weighting** — Rather than treating all sources equally, each source is assigned a `reliability_score` weight (0.5–1.0). Government databases and official business websites rank highest; generic search engine snippets rank lowest. This surfaces the most trustworthy contact data at the top.

---

## Advantages & Differentiators

- **Zero paid search API keys required** — uses DuckDuckGo and Bing scraping
- **14+ sources in a single query** — general engines, India directories, healthcare/legal/gov specialty sources
- **Real-time streaming UI** — users see businesses appearing as they are found, not after a 60-second wait
- **LLM-powered gap filling** — BeautifulSoup extracts what it can structurally; the LLM fills the rest from unstructured HTML text
- **Conflict-aware merging** — contradictory data between sources is flagged in `conflicts[]`, not silently dropped
- **Source-weighted reliability scores** — government and official sources always outrank low-trust aggregators
- **Multi-strategy deduplication** — five independent signals prevent both false merges and missed duplicates
- **Exportable results** — full JSON or CSV download via REST API
- **Search history** — all queries persisted to SQLite for reference and re-run
- **Optional vector dedup** — ChromaDB + `sentence-transformers` adds semantic similarity dedup when installed, with graceful fallback if not
- **One-command Docker deploy** — runs identically locally and on Hugging Face Spaces

# 👨‍💻 Developer

**Sankar Pandi**

B.Tech Information Technology

SSM Institute of Engineering and Technology

Tamil Nadu, India

---

# 📄 License

This project is developed for educational, research, and demonstration purposes.

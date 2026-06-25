"""
Phase 14 — FastAPI + WebSocket (SPEED OPTIMIZED)
Target: 30-50 seconds total
Key changes:
  - Web search + OpenData run in PARALLEL
  - scrape.do handles blocked sites (no slow Playwright)
  - LLM extractions run in PARALLEL with 8s timeout
  - Skip LLM if scraper already got enough data
"""
import asyncio, json, os, sys, time
from typing import Dict, List
from fastapi.responses import FileResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import PipelineEvent
from core.storage import DatabaseStorage, VectorStore
from agents.query_agent         import QueryAgent
from agents.search_agent        import SearchAgent
from agents.scraper_agent       import ScraperAgent
from agents.extraction_agent    import ExtractionAgent
from agents.verification_agent  import VerificationAgent, ReliabilityScoringAgent
from agents.dedup_agent         import DeduplicationAgent
from agents.summary_agent       import SummaryAgent, report_to_csv
from agents.open_data_agent     import OpenDataAgent

app = FastAPI(title="AI Business Research Agent", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_db       = DatabaseStorage()
_vecstore = VectorStore()
_reports: Dict[str, dict] = {}


async def _emit(ws, event: PipelineEvent):
    await ws.send_text(event.model_dump_json())

async def _status(ws, phase, name, msg, pct):
    await _emit(ws, PipelineEvent(
        event_type="status", phase=phase,
        phase_name=name, message=msg, progress=pct))

async def _biz(ws, biz, phase, label, pct):
    safe = {k:v for k,v in biz.items()
            if k not in ("osm_tags","_html") and not isinstance(v,bytes)}
    await _emit(ws, PipelineEvent(
        event_type="business", phase=phase, phase_name=label,
        message=safe.get("business_name","")[:60], data=safe, progress=pct))


@app.websocket("/ws/research")
async def ws_research(ws: WebSocket):
    await ws.accept()
    try:
        raw     = await ws.receive_text()
        payload = json.loads(raw)
        query   = payload.get("query","").strip()
        if not query:
            await _emit(ws, PipelineEvent(event_type="error", message="Empty query"))
            return
        session_id = str(time.time())
        report     = await _pipeline(query, ws, session_id)
        _reports[session_id] = report
        await _emit(ws, PipelineEvent(
            event_type="summary", phase=16, phase_name="Complete",
            message="Research complete", data=report, progress=100.0))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try: await _emit(ws, PipelineEvent(event_type="error", message=f"Error: {e}"))
        except: pass


async def _pipeline(query: str, ws: WebSocket, session_id: str) -> dict:
    t0 = time.time()
    raw_businesses: List[dict] = []

    # ── Phase 1: Query Parse ─────────────────────────────────────────
    await _status(ws,1,"Query Agent","🧠 Parsing query…",3)
    try:
        parsed = QueryAgent().parse(query)
    except Exception:
        parsed = {"category":query,"location":"","search_variations":[]}
    category   = parsed.get("category", query)
    location   = parsed.get("location","")
    variations = parsed.get("search_variations",[])
    await _status(ws,1,"Query Agent",
                  f"✅ Category: **{category}** | Location: **{location}**",7)
    search_id = _db.save_search(query, category, location)

    # ── Phase 2: ALL sources PARALLEL ────────────────────────────────
    await _status(ws,2,"Search + OpenData",
                  "🔍 DuckDuckGo · Yelp · YP · Justdial · Sulekha  +  🗺️ OSM · GovData  [PARALLEL]",10)

    web_task  = SearchAgent().search_all(category, location, variations)
    osm_task  = OpenDataAgent().fetch_all(category, location)

    (web_results, osm_results) = await asyncio.gather(
        web_task, osm_task, return_exceptions=True)

    candidates   = web_results  if isinstance(web_results, list)  else []
    osm_records  = osm_results  if isinstance(osm_results, list)  else []

    await _status(ws,3,"Concurrent Search",
                  f"✅ Web: **{len(candidates)}** URLs | OSM+Gov: **{len(osm_records)}** records",18)

    # ── Stream OSM records immediately (already structured) ──────────
    await _status(ws,3,"OpenData",
                  f"📦 Injecting {len(osm_records)} OpenStreetMap + GovData records…",20)
    for od in osm_records:
        safe = {k:v for k,v in od.items() if k!="osm_tags"}
        raw_businesses.append(safe)
        if safe.get("business_name"):
            await _biz(ws, safe, 3, "OSM/GovData", 21)

    # ── Phase 4: Seed from snippets ─────────────────────────────────
    await _status(ws,4,"URL Collection",
                  f"📋 {len(candidates)} URLs + {len(osm_records)} OSM records",22)
    seed_count = 0
    for c in candidates:
        if c.get("phone") or c.get("address") or c.get("business_name"):
            seed = {
                "business_name": c.get("business_name","") or c.get("title",""),
                "phone":         c.get("phone",""),
                "address":       c.get("address",""),
                "email":         c.get("email",""),
                "rating":        c.get("rating",""),
                "working_hours": c.get("working_hours",""),
                "website":       c.get("url",""),
                "source_url":    c.get("url",""),
                "source":        c.get("source",""),
                "services":[],"specialties":[],"license_information":"",
                "certifications":[],"awards":[],"social_profiles":[],"images_urls":[],
            }
            raw_businesses.append(seed)
            seed_count += 1
            if seed.get("business_name"):
                await _biz(ws, seed, 4, "Snippet", 23)

    # ── Phase 5: Scrape ALL pages (scrape.do + aiohttp parallel) ────
    await _status(ws,5,"Web Scraper",
                  f"🌐 Scraping {min(len(candidates),12)} pages via scrape.do + aiohttp…",25)
    t_scrape = time.time()
    raw_pages = await ScraperAgent().scrape_batch(candidates)
    await _status(ws,5,"Web Scraper",
                  f"✅ Scraped in {time.time()-t_scrape:.1f}s — "
                  f"{len(raw_pages)} pages processed",38)

    # ── Phase 6: LLM Extraction — ALL IN PARALLEL ───────────────────
    await _status(ws,6,"Extraction Agent",
                  "🤖 LLM extractions running in PARALLEL (8s timeout each)…",40)
    t_extract = time.time()

    # Flatten list pages
    flat_pages: List[dict] = []
    flat_cands: List[dict] = []
    for page, cand in zip(raw_pages, candidates[:len(raw_pages)]):
        if isinstance(page, list):
            for p in page:
                flat_pages.append(p)
                flat_cands.append(cand)
        elif isinstance(page, dict) and page:
            flat_pages.append(page)
            flat_cands.append(cand)

    extractor = ExtractionAgent()
    enriched_list = await extractor.extract_batch_parallel(flat_pages, flat_cands)

    for biz in enriched_list:
        raw_businesses.append(biz)
        if biz.get("business_name"):
            await _biz(ws, biz, 6, "LLM", 50)

    await _status(ws,6,"Extraction Agent",
                  f"✅ Extraction done in {time.time()-t_extract:.1f}s | "
                  f"Total: **{len(raw_businesses)}** records",55)

    # ── Phase 7: Verify ──────────────────────────────────────────────
    await _status(ws,7,"Verification","🔒 Cross-source validation…",58)
    verified = VerificationAgent().verify(raw_businesses)
    await _status(ws,7,"Verification",
                  f"✅ {len(verified)} verified | "
                  f"{sum(1 for b in verified if b.get('conflicts'))} conflicts",63)

    # ── Phase 8: Reliability Score ───────────────────────────────────
    await _status(ws,8,"Reliability","📊 Scoring reliability…",66)
    scored = ReliabilityScoringAgent().score(verified)
    avg = sum(b.get("verification_score",0) for b in scored)/max(len(scored),1)
    await _status(ws,8,"Reliability",f"✅ Avg score: {avg:.1f}/100",69)

    # ── Phase 9: Dedup ───────────────────────────────────────────────
    await _status(ws,9,"Deduplication","🔄 RapidFuzz + ChromaDB dedup…",71)
    final, n_removed = DeduplicationAgent().deduplicate(scored)
    await _status(ws,9,"Deduplication",
                  f"✅ {n_removed} dupes removed → **{len(final)}** unique",76)

    clean = [{k:v for k,v in b.items()
              if k not in ("_phone_sources","_email_sources",
                           "_address_sources","source_list","osm_tags","_html")}
             for b in final]

    # ── Phase 10+11: Storage ─────────────────────────────────────────
    await _status(ws,10,"Database","💾 Saving…",79)
    _db.save_businesses(search_id, clean)
    await _status(ws,11,"ChromaDB","🧠 Indexing…",82)
    _vecstore.store_businesses(clean, search_id)

    # ── Phase 12+13: Summary ─────────────────────────────────────────
    await _status(ws,12,"Summary Agent","✍️ Generating AI summary…",85)
    report = SummaryAgent().generate(
        query=query, category=category, location=location,
        businesses=clean, duration=round(time.time()-t0,1),
        duplicates_removed=n_removed)
    report["session_id"]       = session_id
    report["businesses_found"] = len(raw_businesses)
    report["sources_searched"] = 12

    await _status(ws,13,"Report","📋 Building report…",91)
    _db.save_report(search_id, report, report.get("ai_summary",""))

    total_time = round(time.time()-t0, 1)
    await _status(ws,14,"Complete",f"🚀 Done in **{total_time}s**",95)
    await _status(ws,15,"Dashboard","📊 Rendering…",98)
    await _status(ws,16,"Export","✅ JSON · CSV · Report ready",100)
    return report


@app.get("/api/history")
async def history(): return JSONResponse(_db.get_recent_searches())

@app.get("/api/export/json/{sid}")
async def exp_json(sid): return JSONResponse(_reports.get(sid,{}))

@app.get("/api/export/csv/{sid}")
async def exp_csv(sid):
    biz = _reports.get(sid,{}).get("businesses",[])
    return PlainTextResponse(report_to_csv(biz), media_type="text/csv",
        headers={"Content-Disposition":"attachment; filename=businesses.csv"})

@app.get("/health")
async def health(): return {"status":"ok","sources":12,"vector":_vecstore.available}

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")
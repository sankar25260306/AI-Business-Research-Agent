"""
Phase 6 — Extraction Agent (SPEED OPTIMIZED)
FAST MODE: Run ALL LLM extractions in parallel with 8s timeout.
Only send pages that actually have missing data.
"""
import asyncio
import json
import re
from typing import Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from core.config import get_groq_key

_FENCE = re.compile(r"```(?:json)?|```")

def _clean(t): return _FENCE.sub("", t).strip()

SCHEMA = '{"business_name":"","address":"","phone":"","email":"","website":"","working_hours":"","rating":"","services":[],"specialties":[],"license_information":""}'


class ExtractionAgent:
    def __init__(self):
        self._llm = ChatGroq(
            api_key=get_groq_key(),
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=512,      # reduced — we only need key fields
        )
        self._prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Extract business info from HTML. Return ONLY valid JSON. "
             "Never invent data. Use '' for missing fields."),
            ("human",
             "URL: {url}\nHTML: {html}\nReturn JSON: {schema}"),
        ])
        self._chain = self._prompt | self._llm | StrOutputParser()

    def extract(self, html: str, url: str, seed: Dict) -> Dict:
        """Sync extract — called from async context via gather."""
        result = {k: v for k, v in seed.items()}

        # Skip LLM if we already have enough data from scraper/snippet
        filled = sum(1 for k in ["business_name","phone","address","website"]
                     if result.get(k))
        if filled >= 3:
            result["source_url"] = url
            return result  # already good enough — skip LLM call

        if not html or len(html.strip()) < 200:
            result["source_url"] = url
            return result

        try:
            raw = self._chain.invoke({
                "url":    url,
                "html":   html[:2000],   # reduced from 3000 → faster
                "schema": SCHEMA,
            })
            data = json.loads(_clean(raw))
            STRING_FIELDS = ["business_name","address","phone","email",
                             "website","working_hours","rating","license_information"]
            LIST_FIELDS   = ["services","specialties","certifications","awards"]
            for f in STRING_FIELDS:
                if data.get(f) and not result.get(f):
                    result[f] = str(data[f]).strip()
            for f in LIST_FIELDS:
                existing = result.get(f) or []
                new_vals = data.get(f) or []
                if isinstance(new_vals, list):
                    result[f] = list(dict.fromkeys(existing + new_vals))
        except Exception:
            pass

        result["source_url"] = url
        return result

    async def extract_batch_parallel(
        self, pages: List[Dict], candidates: List[Dict]
    ) -> List[Dict]:
        """
        Run ALL LLM extractions in parallel with timeout.
        Much faster than sequential.
        """
        async def _one(page, candidate):
            html = page.pop("_html","") if isinstance(page,dict) else ""
            url  = page.get("source_url","") or candidate.get("url","")
            loop = asyncio.get_event_loop()
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, self.extract, html, url, page),
                    timeout=8.0  # max 8s per LLM call
                )
                return result
            except asyncio.TimeoutError:
                page["source_url"] = url
                return page  # return what we have
            except Exception:
                page["source_url"] = url
                return page

        tasks = [_one(p, c) for p, c in zip(pages, candidates)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict) and r.get("business_name")]
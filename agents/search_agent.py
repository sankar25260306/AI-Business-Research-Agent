"""
Phase 2+3 — Search Agent
FIXED:
  - SPECIFIC query ("Apollo Hospital Chennai"):
      → Searches exact business name, returns only that business's pages
  - GENERIC query ("Cardiologists in Chennai"):
      → Category search across all 7 sources as before
  - Junk URL filter built-in
  - Speed optimized (parallel DDG queries)
"""
import asyncio
import re
from typing import Dict, List
from urllib.parse import quote, urlparse

import aiohttp
from bs4 import BeautifulSoup
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from core.config import SCRAPER_HEADERS
try:
    from agents.google_scraper import GoogleSearchScraper
    _GOOGLE_SCRAPER = GoogleSearchScraper()
except Exception:
    _GOOGLE_SCRAPER = None

FAST_TIMEOUT = aiohttp.ClientTimeout(total=8)
DDG_TIMEOUT  = 10

BLOCKED_DOMAINS = {
    "merriam-webster.com","dictionary.com","collinsdictionary.com",
    "wordreference.com","britannica.com","wikipedia.org","wikimedia.org",
    "wiktionary.org","amazon.com","amazon.in","ebay.com","flipkart.com",
    "reddit.com","quora.com","stackoverflow.com","youtube.com",
    "twitter.com","x.com","instagram.com","pinterest.com",
    "bbc.com","cnn.com","ndtv.com","timesofindia.com","thehindu.com",
    "hindustantimes.com","indiatimes.com","indianexpress.com",
    "maps.google.com","google.com","bing.com",
}

JUNK_TITLE_KEYWORDS = [
    "definition","meaning","dictionary","encyclopedia","wikipedia",
    "what is","how to","reddit","quora","youtube",
    "top 10 places","travel guide","tourism",
]


def _is_valid_url(url: str) -> bool:
    if not url or not url.startswith("http"):
        return False
    try:
        domain = urlparse(url).netloc.lower().replace("www.","")
        return not any(domain == b or domain.endswith("."+b) for b in BLOCKED_DOMAINS)
    except Exception:
        return False


def _is_valid_result(item: Dict) -> bool:
    combined = ((item.get("title","") or "") + " " + (item.get("snippet","") or "")).lower()
    return not any(kw in combined for kw in JUNK_TITLE_KEYWORDS)


def _clean_name(title: str) -> str:
    for noise in [
        " - Google Maps"," | Yelp"," | Yellow Pages"," - Healthgrades",
        " - Practo"," - Lybrate","| Book Appointment","- Book",
        " - Facebook"," | LinkedIn"," - Justdial",
    ]:
        title = title.replace(noise, "")
    return title.strip()


def _extract_snippet(title: str, snippet: str, url: str, source: str) -> Dict:
    text = f"{title} {snippet}"
    seed = {
        "url": url, "title": title, "source": source,
        "business_name": _clean_name(title),
        "phone": "", "address": "", "email": "", "rating": "", "working_hours": "",
    }
    m = re.search(r"\+?1?\s?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}", text)
    if m: seed["phone"] = m.group(0).strip()
    m = re.search(r"\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|Ave|Rd|Blvd|Dr|Street|Avenue|Road)", text)
    if m: seed["address"] = m.group(0).strip()
    m = re.search(r"(\d\.?\d?)\s*(?:/5|stars?|out of 5)", text, re.IGNORECASE)
    if m: seed["rating"] = m.group(1)
    return seed


class SearchAgent:
    TIMEOUT = FAST_TIMEOUT

    async def search_all(
        self,
        category: str,
        location: str,
        variations: List[str],
        max_ddg: int = 10,
        query_type: str = "generic",
        business_name: str = "",
    ) -> List[Dict]:

        # ── SPECIFIC: search for exact business name ──────────────────
        if query_type == "specific" and business_name:
            tasks = [
                self._ddg_specific(business_name, location),
                self._justdial_specific(business_name, location),
                self._sulekha_specific(business_name, location),
                self._ddg_one(f'site:linkedin.com/company "{business_name}" "{location}"', 3, "linkedin"),
                self._ddg_one(f'site:facebook.com "{business_name}" "{location}"', 3, "facebook"),
            ]
        # ── GENERIC: category search across all sources ───────────────
        else:
            tasks = [
                self._ddg_fast(category, location, max_ddg),
                self._yelp(category, location),
                self._yellowpages(category, location),
                self._justdial(category, location),
                self._sulekha(category, location),
                self._ddg_one(f'site:linkedin.com/company "{category}" "{location}"', 5, "linkedin"),
                self._ddg_one(f'site:facebook.com "{category}" "{location}"', 5, "facebook"),
            ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        flat: List[Dict] = []
        for r in results:
            if isinstance(r, list):
                flat.extend(r)

        # Filter junk
        flat = [
            item for item in flat
            if _is_valid_url(item.get("url","")) and _is_valid_result(item)
        ]
        return self._dedup(flat)

    # ── SPECIFIC search helpers ───────────────────────────────────────
    async def _ddg_specific(self, biz_name: str, location: str) -> List[Dict]:
        out = []
        loc = location or ""
        queries = [
            f'"{biz_name}" {loc} official website contact phone',
            f'"{biz_name}" {loc} address phone number',
            f'{biz_name} {loc}',
        ]
        tasks = [self._ddg_one(q.strip(), 8, "duckduckgo") for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, list):
                out.extend(r)
        return out

    async def _justdial_specific(self, biz_name: str, location: str) -> List[Dict]:
        if not location:
            return []
        url = f"https://www.justdial.com/{quote(location)}/{quote(biz_name.replace(' ','-'))}"
        html = await self._get(url)
        if not html:
            return [{"url": url, "source": "justdial", "title": biz_name,
                     "business_name": biz_name, "phone": "", "address": "", "email": ""}]
        return self._parse_justdial(html, url, filter_name=biz_name)

    async def _sulekha_specific(self, biz_name: str, location: str) -> List[Dict]:
        if not location:
            return []
        url = f"https://www.sulekha.com/{quote(biz_name.lower().replace(' ','-'))}/{quote(location.lower())}"
        html = await self._get(url)
        if not html:
            return [{"url": url, "source": "sulekha", "title": biz_name,
                     "business_name": biz_name, "phone": "", "address": "", "email": ""}]
        return self._parse_sulekha(html, url, filter_name=biz_name)

    # ── GENERIC search helpers ────────────────────────────────────────
    async def _ddg_fast(self, cat: str, loc: str, n: int) -> List[Dict]:
        queries = [
            f'"{cat}" "{loc}" phone address contact',
            f'{cat} {loc} clinic office -"top 10"',
        ]
        tasks = [self._ddg_one(q, n, "duckduckgo") for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = []
        for r in results:
            if isinstance(r, list):
                out.extend(r)
        return out

    async def _ddg_one(self, query: str, n: int, source: str) -> List[Dict]:
        out: List[Dict] = []
        try:
            loop = asyncio.get_running_loop()
            def search():
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=n))
            results = await asyncio.wait_for(
                loop.run_in_executor(None, search), timeout=DDG_TIMEOUT
            )
            for r in results:
                seed = _extract_snippet(
                    r.get("title",""), r.get("body",""), r.get("href",""), source
                )
                if seed["url"]:
                    out.append(seed)
        except Exception:
            pass
        return out

    async def _yelp(self, cat: str, loc: str) -> List[Dict]:
        if not loc:
            return []
        url = f"https://www.yelp.com/search?find_desc={quote(cat)}&find_loc={quote(loc)}"
        html = await self._get(url)
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        items = []
        for a in soup.select('a[href*="/biz/"]'):
            href = a.get("href","")
            name = a.get_text(strip=True)
            if href.startswith("/biz/") and name and len(name) > 2:
                items.append({
                    "url": "https://www.yelp.com" + href.split("?")[0],
                    "business_name": name, "title": name,
                    "source": "yelp", "phone": "", "address": "", "email": "", "rating": "",
                })
        return items[:12]

    async def _yellowpages(self, cat: str, loc: str) -> List[Dict]:
        if not loc:
            return []
        url = f"https://www.yellowpages.com/search?search_terms={quote(cat)}&geo_location_terms={quote(loc)}"
        html = await self._get(url)
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        items = []
        for div in soup.select(".result"):
            a = div.select_one("a.business-name")
            if not a or not a.get("href"):
                continue
            href = a["href"]
            name = a.get_text(strip=True)
            phone_el   = div.select_one(".phones")
            address_el = div.select_one(".street-address")
            items.append({
                "url":    "https://www.yellowpages.com"+href if href.startswith("/") else href,
                "business_name": name, "title": name, "source": "yellowpages",
                "phone":   phone_el.get_text(strip=True)   if phone_el   else "",
                "address": address_el.get_text(strip=True) if address_el else "",
                "email": "", "rating": "",
            })
        return items[:12]

    async def _justdial(self, cat: str, loc: str) -> List[Dict]:
        if not loc:
            return []
        url = f"https://www.justdial.com/{quote(loc)}/{quote(cat.replace(' ','-'))}"
        html = await self._get(url)
        if not html:
            return []
        return self._parse_justdial(html, url)

    async def _sulekha(self, cat: str, loc: str) -> List[Dict]:
        if not loc:
            return []
        url = f"https://www.sulekha.com/{cat.lower().replace(' ','-')}/{loc.lower().replace(' ','-')}"
        html = await self._get(url)
        if not html:
            return []
        return self._parse_sulekha(html, url)

    # ── HTML parsers ──────────────────────────────────────────────────
    def _parse_justdial(self, html: str, url: str, filter_name: str = "") -> List[Dict]:
        soup = BeautifulSoup(html, "html.parser")
        items = []
        for div in soup.select("[class*='resultbox'],[class*='cntanr'],[class*='store']"):
            name_el    = div.select_one("[class*='title'],[class*='storename'],h2,h3")
            phone_el   = div.select_one("[class*='contact'],[class*='phone'],[class*='number']")
            address_el = div.select_one("[class*='address'],[class*='jdgrey']")
            rating_el  = div.select_one("[class*='rating'],[class*='rtngscore']")
            if not name_el:
                continue
            name = name_el.get_text(strip=True)
            if not name or len(name) < 2:
                continue
            if filter_name and filter_name.lower() not in name.lower():
                continue
            items.append({
                "url": url, "business_name": name, "title": name, "source": "justdial",
                "phone":   phone_el.get_text(strip=True)   if phone_el   else "",
                "address": address_el.get_text(strip=True) if address_el else "",
                "rating":  rating_el.get_text(strip=True)  if rating_el  else "",
                "email": "", "working_hours": "",
            })
        return items[:10]

    def _parse_sulekha(self, html: str, url: str, filter_name: str = "") -> List[Dict]:
        soup = BeautifulSoup(html, "html.parser")
        items = []
        for el in soup.select(".comp-name,.companyname,[class*='provider-name']"):
            name = el.get_text(strip=True)
            if not name or len(name) < 2:
                continue
            if filter_name and filter_name.lower() not in name.lower():
                continue
            items.append({
                "url": url, "business_name": name, "title": name, "source": "sulekha",
                "phone": "", "address": "", "email": "", "rating": "",
            })
        return items[:8]

    # ── Helpers ───────────────────────────────────────────────────────
    async def _google_search(self, query: str) -> list:
        """Google Search via Selenium (runs in thread to avoid blocking event loop)."""
        if not _GOOGLE_SCRAPER or not _GOOGLE_SCRAPER.available:
            return []
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            results = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: _GOOGLE_SCRAPER.search(query, max_pages=2)),
                timeout=25,
            )
            return results if isinstance(results, list) else []
        except Exception:
            return []

    async def _get(self, url: str) -> str:
        try:
            async with aiohttp.ClientSession(headers=SCRAPER_HEADERS) as sess:
                async with sess.get(url, timeout=FAST_TIMEOUT, allow_redirects=True) as resp:
                    if resp.status == 200:
                        return await resp.text(errors="replace")
        except Exception:
            pass
        return ""

    @staticmethod
    def _dedup(items: List[Dict]) -> List[Dict]:
        seen, out = set(), []
        for item in items:
            url = item.get("url","").split("?")[0].rstrip("/")
            if url and url not in seen:
                seen.add(url)
                out.append(item)
            elif not url:
                out.append(item)
        return out
"""
Phase 4+5 — Web Scraper
KEY FIX: Detects list pages and extracts individual businesses FROM them,
instead of treating the whole page as one business.
"""
import asyncio
import re
from typing import Dict, List, Optional, Tuple

import aiohttp
from bs4 import BeautifulSoup

from core.config import SCRAPER_HEADERS

# Signals that a page is a list/directory, not a single business
LIST_PAGE_SIGNALS = [
    "top 10", "top 5", "best ", "list of", "near me",
    "directory", "find a", "search results", "doctors in",
    "cardiologists in", "dentists in", "lawyers in",
]

# Site-specific selectors for extracting businesses FROM list pages
LIST_PAGE_EXTRACTORS = {
    "yelp.com": {
        "item":   'a[href*="/biz/"]',
        "name":   None,       # use link text
        "phone":  ".biz-phone",
        "addr":   ".secondary-attributes",
        "rating": "[aria-label*='star']",
    },
    "yellowpages.com": {
        "item":   ".result",
        "name":   "a.business-name",
        "phone":  ".phones",
        "addr":   ".street-address",
        "rating": ".rating",
    },
    "healthgrades.com": {
        "item":   "[class*='provider-'], [class*='result-']",
        "name":   "h3, h2, [class*='name']",
        "phone":  "[class*='phone'], [class*='contact']",
        "addr":   "[class*='address'], [class*='location']",
        "rating": "[class*='rating'], [class*='star']",
    },
    "practo.com": {
        "item":   "[class*='doctor-card'], [class*='listing']",
        "name":   "[class*='name'], h2, h3",
        "phone":  None,
        "addr":   "[class*='clinic'], [class*='location']",
        "rating": "[class*='rating']",
    },
    "justdial.com": {
        "item":   "[class*='resultbox'], [class*='cntanr'], .jsx-result",
        "name":   "[class*='title'], [class*='storename'], h2",
        "phone":  "[class*='contact'], [class*='mobilesv']",
        "addr":   "[class*='address'], [class*='jdgrey']",
        "rating": "[class*='rating'], [class*='rtngscore']",
    },
    "lybrate.com": {
        "item":   "[class*='doctor-card'], [class*='provider']",
        "name":   "[class*='name'], h2",
        "phone":  None,
        "addr":   "[class*='location'], [class*='clinic']",
        "rating": "[class*='rating']",
    },
    "default": {
        "item":   "h2 a, h3 a, .result a, .listing a",
        "name":   None,
        "phone":  None,
        "addr":   None,
        "rating": None,
    },
}


def _get_domain(url: str) -> str:
    m = re.search(r"(?:https?://)?(?:www\.)?([^/]+)", url)
    return m.group(1).lower() if m else ""


def _is_list_page(url: str, title: str, html: str) -> bool:
    """Decide if this is a list/directory page."""
    title_lower = (title or "").lower()
    url_lower = url.lower()
    # Title signals
    if any(s in title_lower for s in LIST_PAGE_SIGNALS):
        return True
    # URL signals
    if re.search(r"/(search|find|list|directory|top-\d|best-)", url_lower):
        return True
    # Known list domains with search paths
    list_domains = ["healthgrades.com", "vitals.com", "zocdoc.com",
                    "practo.com", "lybrate.com", "justdial.com",
                    "sulekha.com", "1mg.com", "apollo247.com",
                    "yelp.com/search", "yellowpages.com/search"]
    if any(d in url_lower for d in list_domains):
        return True
    return False


class ScraperAgent:
    TIMEOUT        = aiohttp.ClientTimeout(total=15)
    MAX_CONCURRENT = 5

    async def scrape_batch(self, candidates: List[Dict]) -> List[Dict]:
        sem     = asyncio.Semaphore(self.MAX_CONCURRENT)
        tasks   = [self._scrape_one(c, sem) for c in candidates[:8]]
        raw     = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten: list pages return multiple businesses
        out = []
        for r in raw:
            if isinstance(r, list):
                out.extend(r)
            elif isinstance(r, dict) and r:
                out.append(r)
        return out

    async def _scrape_one(self, candidate: Dict, sem: asyncio.Semaphore):
        async with sem:
            url  = candidate.get("url","")
            if not url or not url.startswith("http"):
                return {}

            html = await self._fetch(url)
            # Detect list page
            title = candidate.get("title","") or candidate.get("business_name","")
            if _is_list_page(url, title, html or ""):
                # Extract multiple individual businesses from this page
                businesses = self._extract_from_list_page(html or "", url, candidate)
                return businesses if businesses else {}

            # Normal single-business page
            parsed = self._parse_business_page(html or "", url)
            # Carry seed data (phone/address from directory listing)
            for src_k, dst_k in [("business_name","business_name"),
                                   ("phone","phone"),("address","address"),
                                   ("rating","rating"),("working_hours","working_hours")]:
                if candidate.get(src_k) and not parsed.get(dst_k):
                    parsed[dst_k] = candidate[src_k]
            # If name is still the page title (list-like), use seed name
            name = parsed.get("business_name","")
            if name and any(s in name.lower() for s in LIST_PAGE_SIGNALS[:5]):
                parsed["business_name"] = candidate.get("business_name","") or name
            parsed["source"] = candidate.get("source","")
            parsed["_html"]  = (html or "")[:4000]
            return parsed

    def _extract_from_list_page(self, html: str, url: str, candidate: Dict) -> List[Dict]:
        """Extract multiple individual business records from a list/directory page."""
        if not html:
            # Fallback: return the candidate's seed data as a partial record
            if candidate.get("business_name"):
                return [self._seed_record(candidate, url)]
            return []

        soup   = BeautifulSoup(html, "html.parser")
        domain = _get_domain(url)
        cfg    = next((v for k,v in LIST_PAGE_EXTRACTORS.items() if k in domain),
                      LIST_PAGE_EXTRACTORS["default"])

        businesses = []
        items = soup.select(cfg["item"])

        for el in items[:15]:
            biz = self._extract_item(el, cfg, url, domain)
            if biz.get("business_name") and len(biz["business_name"]) > 2:
                biz["source"] = candidate.get("source","list_page")
                biz["_html"]  = ""
                businesses.append(biz)

        # If site-specific extraction failed, try generic heuristics
        if not businesses:
            businesses = self._generic_list_extract(soup, url, candidate)

        return businesses

    def _extract_item(self, el, cfg: Dict, base_url: str, domain: str) -> Dict:
        biz = {
            "business_name":"","address":"","phone":"","email":"",
            "website":"","working_hours":"","rating":"","review_count":"",
            "services":[],"specialties":[],"license_information":"",
            "certifications":[],"awards":[],"social_profiles":[],
            "images_urls":[],"source_url": base_url,
        }

        # Name
        if cfg.get("name"):
            name_el = el.select_one(cfg["name"])
            if name_el:
                biz["business_name"] = name_el.get_text(strip=True)
        else:
            # Use element's own text if it's a link
            biz["business_name"] = el.get_text(strip=True)

        # Link URL (try to get individual page)
        link = el if el.name == "a" else el.select_one("a[href]")
        if link and link.get("href"):
            href = link["href"]
            if href.startswith("/"):
                biz["website"] = f"https://{domain}{href}"
            elif href.startswith("http"):
                biz["website"] = href

        # Phone
        if cfg.get("phone"):
            ph_el = el.select_one(cfg["phone"])
            if ph_el:
                biz["phone"] = ph_el.get_text(strip=True)
        # Fallback phone regex
        if not biz["phone"]:
            text = el.get_text(" ")
            m = re.search(r"\+?1?\s?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}", text)
            if m: biz["phone"] = m.group(0).strip()

        # Address
        if cfg.get("addr"):
            addr_el = el.select_one(cfg["addr"])
            if addr_el:
                biz["address"] = addr_el.get_text(separator=", ", strip=True)

        # Rating
        if cfg.get("rating"):
            rating_el = el.select_one(cfg["rating"])
            if rating_el:
                aria = rating_el.get("aria-label","")
                text = aria or rating_el.get_text(strip=True)
                m = re.search(r"(\d\.?\d?)", text)
                if m:
                    v = float(m.group(1))
                    if 0 < v <= 10:
                        biz["rating"] = str(v)

        return biz

    def _generic_list_extract(self, soup: BeautifulSoup, url: str, candidate: Dict) -> List[Dict]:
        """Generic fallback: extract business names + phones from any list page."""
        businesses = []
        # Look for any h2/h3 that could be business names
        for heading in soup.select("h2, h3, h4"):
            text = heading.get_text(strip=True)
            if 3 < len(text) < 80 and not any(
                s in text.lower() for s in ["top ", "best ", "list", "find", "search"]):
                parent = heading.find_parent(["div","li","article"])
                phone  = ""
                addr   = ""
                rating = ""
                if parent:
                    t = parent.get_text(" ")
                    pm = re.search(r"\+?1?\s?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}", t)
                    if pm: phone = pm.group(0).strip()
                    am = re.search(r"\d+\s+\w+\s+(?:St|Ave|Rd|Blvd|Dr)\b[^,]{0,30}", t)
                    if am: addr = am.group(0).strip()
                    rm = re.search(r"(\d\.?\d?)\s*(?:stars?|\/5|out of)", t, re.IGNORECASE)
                    if rm: rating = rm.group(1)
                businesses.append({
                    "business_name": text,
                    "phone":   phone,
                    "address": addr,
                    "rating":  rating,
                    "source_url": url,
                    "website": "", "email":"","working_hours":"",
                    "services":[],"specialties":[],"license_information":"",
                    "certifications":[],"awards":[],"social_profiles":[],
                    "images_urls":[],"_html":"",
                })
                if len(businesses) >= 10:
                    break

        # If still nothing useful, return candidate seed as fallback
        if not businesses and candidate.get("business_name"):
            businesses = [self._seed_record(candidate, url)]

        return businesses

    @staticmethod
    def _seed_record(candidate: Dict, url: str) -> Dict:
        return {
            "business_name": candidate.get("business_name","") or candidate.get("title",""),
            "phone":   candidate.get("phone",""),
            "address": candidate.get("address",""),
            "email":   candidate.get("email",""),
            "rating":  candidate.get("rating",""),
            "working_hours": candidate.get("working_hours",""),
            "website": url,
            "source_url": url,
            "source": candidate.get("source",""),
            "services":[],"specialties":[],"license_information":"",
            "certifications":[],"awards":[],"social_profiles":[],
            "images_urls":[],"_html":"",
        }

    async def _fetch(self, url: str) -> Optional[str]:
        skip = ["facebook.com","instagram.com"]
        if any(s in url for s in skip):
            return ""
        try:
            async with aiohttp.ClientSession(headers=SCRAPER_HEADERS) as sess:
                async with sess.get(url, timeout=self.TIMEOUT, allow_redirects=True) as resp:
                    if resp.status == 200:
                        ct = resp.headers.get("content-type","")
                        if "text" in ct:
                            return await resp.text(errors="replace")
        except Exception:
            pass
        # Playwright fallback
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True, args=["--no-sandbox","--disable-gpu"])
                page = await (await browser.new_context(
                    user_agent=SCRAPER_HEADERS["User-Agent"])).new_page()
                try:
                    await page.goto(url, timeout=12000, wait_until="domcontentloaded")
                    return await page.content()
                finally:
                    await browser.close()
        except Exception:
            return ""

    def _parse_business_page(self, html: str, url: str) -> Dict:
        base = {
            "business_name":"","address":"","phone":"","email":"",
            "website":url,"working_hours":"","rating":"","review_count":"",
            "services":[],"specialties":[],"license_information":"",
            "certifications":[],"awards":[],"social_profiles":[],
            "images_urls":[],"source_url":url,
        }
        if not html:
            return base
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script","style","noscript"]):
            tag.decompose()
        base["business_name"] = self._name(soup)
        base["address"]       = self._address(soup)
        base["phone"]         = self._phone(soup)
        base["email"]         = self._email(soup)
        base["website"]       = self._website(soup) or url
        base["working_hours"] = self._hours(soup)
        base["rating"]        = self._rating(soup)
        base["review_count"]  = self._review_count(soup)
        base["services"]      = self._list_items(soup,[".services li",".service-list li"])
        base["specialties"]   = self._list_items(soup,[".specialties li","[itemprop='medicalSpecialty']"])
        base["social_profiles"]= self._socials(soup)
        base["images_urls"]   = self._images(soup)
        return base

    @staticmethod
    def _name(soup) -> str:
        for sel in ["[itemprop='name']","h1",".business-name",".company-name",".biz-name",
                    "[class*='doctor-name']","[class*='provider-name']"]:
            el = soup.select_one(sel)
            if el:
                t = el.get_text(strip=True)
                # Skip if this looks like a list title
                if t and 2 < len(t) < 120 and not any(
                    s in t.lower() for s in ["top 10","best ","list of"]):
                    return t
        t = soup.find("title")
        if t:
            raw = t.get_text(strip=True)
            # Clean pipe-separated titles
            parts = [p.strip() for p in raw.split("|")]
            return parts[0][:80] if parts else raw[:80]
        return ""

    @staticmethod
    def _address(soup) -> str:
        for sel in ["[itemprop='address']","[itemprop='streetAddress']",
                    ".address",".biz-location","#address",".street-address",
                    "[class*='address']","[class*='location']"]:
            el = soup.select_one(sel)
            if el:
                t = el.get_text(separator=", ",strip=True)
                if t and 5 < len(t) < 250:
                    return t
        return ""

    @staticmethod
    def _phone(soup) -> str:
        el = soup.select_one("[itemprop='telephone'], a[href^='tel:']")
        if el:
            href = el.get("href","")
            if href.startswith("tel:"):
                return href[4:]
            return el.get_text(strip=True)
        text = soup.get_text(" ")
        m = re.search(r"\+?1?\s?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}", text)
        return m.group(0).strip() if m else ""

    @staticmethod
    def _email(soup) -> str:
        for a in soup.find_all("a", href=True):
            if a["href"].startswith("mailto:"):
                return a["href"][7:].split("?")[0]
        m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
                      soup.get_text(" "))
        if m:
            e = m.group(0)
            if not any(x in e for x in ["example","test.","domain.","sentry","pixel","noreply"]):
                return e
        return ""

    @staticmethod
    def _website(soup) -> str:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http") and not any(
                x in href.lower() for x in
                ["facebook","twitter","instagram","linkedin","yelp",
                 "yellowpages","google","#","javascript","mailto"]):
                return href.split("?")[0]
        return ""

    @staticmethod
    def _hours(soup) -> str:
        for sel in ["[itemprop='openingHours']",".hours",".business-hours",
                    "#hours",".opening-hours","[class*='hours']"]:
            el = soup.select_one(sel)
            if el:
                t = el.get_text(separator=" | ",strip=True)
                if t and len(t) > 3:
                    return t[:300]
        m = re.search(
            r"(Mon|Open|Hours?)[^\.]{0,80}(?:AM|PM|am|pm)",
            soup.get_text(" "), re.IGNORECASE)
        return m.group(0)[:120] if m else ""

    @staticmethod
    def _rating(soup) -> str:
        for sel in ["[itemprop='ratingValue']",".rating","[class*='rating']",".stars"]:
            el = soup.select_one(sel)
            if el:
                aria = el.get("aria-label","")
                text = aria or el.get_text(strip=True)
                m = re.search(r"(\d+\.?\d*)", text)
                if m:
                    v = float(m.group(1))
                    if 0 < v <= 10: return str(v)
        return ""

    @staticmethod
    def _review_count(soup) -> str:
        for sel in ["[itemprop='reviewCount']","[class*='review-count']"]:
            el = soup.select_one(sel)
            if el:
                m = re.search(r"\d+", el.get_text())
                if m: return m.group(0)
        m = re.search(r"(\d[\d,]*)\s*(reviews?|ratings?)",
                      soup.get_text(" "), re.IGNORECASE)
        return m.group(1).replace(",","") if m else ""

    @staticmethod
    def _list_items(soup, selectors: List[str]) -> List[str]:
        for sel in selectors:
            items = [e.get_text(strip=True) for e in soup.select(sel)
                     if e.get_text(strip=True)]
            if items: return items[:10]
        return []

    @staticmethod
    def _socials(soup) -> List[str]:
        platforms = ["facebook.com","linkedin.com","twitter.com","instagram.com","youtube.com"]
        seen, out = set(), []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if any(p in href.lower() for p in platforms) and href not in seen:
                seen.add(href); out.append(href)
        return out

    @staticmethod
    def _images(soup) -> List[str]:
        out = []
        for img in soup.find_all("img",src=True):
            src = img["src"]
            if src.startswith(("http","//")) and any(
                src.lower().endswith(e) for e in [".jpg",".jpeg",".png",".webp"]):
                out.append(src)
        return out[:5]
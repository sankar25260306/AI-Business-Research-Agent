"""
Phase 1 — Query Agent
FIXED:
  "Dentists in Austin"       → category="Dentist",        location="Austin"   (generic)
  "Cardiologists in Chennai" → category="Cardiologist",   location="Chennai"  (generic)
  "Apollo Hospital Chennai"  → category="Hospital",       location="Chennai"  (specific)
  "Plumbers Houston"         → category="Plumber",        location="Houston"  (generic)
  "best dentists Austin TX"  → category="Dentist",        location="Austin TX"(generic)
"""
import json
import re
from typing import Dict, Tuple

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from core.config import get_groq_key


# ── Helpers ────────────────────────────────────────────────────────────
GENERIC_ENDINGS = (
    "ists","ians","ents","ers","ors",
    "doctors","dentists","lawyers","nurses",
    "contractors","plumbers","agents","clinics",
    "hospitals","services","shops","stores",
    "centres","centers","specialists",
)


def _smart_split(query: str) -> Tuple[str, str]:
    """
    Robustly split any query into (category, location).
    Handles: 'X in Y', 'X near Y', 'best X in Y', 'X City', 'X near me'.
    """
    query = query.strip()

    # Pattern 1: "X in Y"
    m = re.match(r"^(.+?)\s+in\s+(.+)$", query, re.IGNORECASE)
    if m:
        cat = re.sub(r"^(best|top|find|list of|all)\s+", "", m.group(1).strip(), flags=re.IGNORECASE).strip()
        return cat, m.group(2).strip()

    # Pattern 2: "X near Y" (ignore "near me")
    m = re.match(r"^(.+?)\s+near\s+(.+)$", query, re.IGNORECASE)
    if m:
        cat = re.sub(r"^(best|top)\s+", "", m.group(1).strip(), flags=re.IGNORECASE).strip()
        loc = m.group(2).strip()
        return cat, "" if loc.lower() == "me" else loc

    # Pattern 3: "best/top X in Y" or "best X CityName"
    m2 = re.match(r"^(best|top|find|cheap|good|affordable)\s+(.+)$", query, re.IGNORECASE)
    if m2:
        rest = m2.group(2).strip()
        m3 = re.match(r"^(.+?)\s+in\s+(.+)$", rest, re.IGNORECASE)
        if m3:
            return m3.group(1).strip(), m3.group(2).strip()
        # Last capitalized word = city
        words = rest.split()
        if len(words) >= 2 and words[-1][0].isupper():
            return " ".join(words[:-1]).strip(), words[-1].strip()
        return rest, ""

    # Pattern 4: Last word looks like a city name
    words = query.split()
    if len(words) >= 2:
        last  = words[-1]
        rest  = " ".join(words[:-1])
        skip  = {"best","top","near","find","list","all","cheap","good","the","me"}
        if last[0].isupper() and last.lower() not in skip and len(last) > 2:
            return rest.strip(), last.strip()

    return query.strip(), ""


def _normalize_category(cat: str) -> str:
    """Convert plural to singular for cleaner search: 'Dentists' → 'Dentist'."""
    replacements = [
        ("ologists","ologist"), ("iatrists","iatrist"),
        ("icians","ician"),     ("urgeons","urgeon"),
        ("entists","entist"),   ("awyers","awyer"),
        ("lumbers","lumber"),   ("ontractors","ontractor"),
        ("pecialists","pecialist"),
    ]
    for plural, singular in replacements:
        if cat.lower().endswith(plural):
            return cat[:-len(plural)] + singular
    return cat


def _is_generic(category: str) -> bool:
    cat_lower = category.lower()
    return any(cat_lower.endswith(e) for e in GENERIC_ENDINGS) or len(category.split()) == 1


# ── Query Agent ────────────────────────────────────────────────────────
class QueryAgent:

    SYSTEM = (
        "You are a query parser for a BUSINESS DIRECTORY search engine. "
        "Return ONLY valid JSON. No explanation, no markdown, no code fences."
    )

    HUMAN = """Parse this business search query and return JSON.

Query: {query}

Rules:
1. "Dentists in Austin"       → generic, category="Dentist" (singular!), location="Austin"
2. "Cardiologists in Chennai" → generic, category="Cardiologist" (singular!), location="Chennai"
3. "Apollo Hospital Chennai"  → specific, business_name="Apollo Hospital", location="Chennai"
4. ALWAYS extract location if a city name is present
5. category must be SINGULAR (Dentist NOT Dentists, Cardiologist NOT Cardiologists)

Return ONLY this JSON:
{{
  "query_type": "specific or generic",
  "business_name": "brand name if specific, else empty string",
  "category": "SINGULAR business type (Dentist / Cardiologist / Hospital / Restaurant)",
  "location": "city name — extract if present, else empty string",
  "location_type": "city",
  "search_variations": ["3 alternative search phrases"],
  "intent": "search"
}}

Examples:
"Dentists in Austin"       → {{"query_type":"generic","business_name":"","category":"Dentist","location":"Austin","location_type":"city","search_variations":["dental clinic Austin","teeth doctor Austin","dentist near Austin TX"],"intent":"search"}}
"Apollo Hospital Chennai"  → {{"query_type":"specific","business_name":"Apollo Hospital","category":"Hospital","location":"Chennai","location_type":"city","search_variations":["Apollo Hospitals Chennai contact","Apollo Hospital Chennai phone","apollohospitals.com Chennai"],"intent":"search"}}
"Plumbers Houston"         → {{"query_type":"generic","business_name":"","category":"Plumber","location":"Houston","location_type":"city","search_variations":["plumbing service Houston","plumber near Houston","pipe repair Houston TX"],"intent":"search"}}
"""

    def __init__(self):
        self.llm = ChatGroq(
            api_key=get_groq_key(),
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=512,
        )
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM),
            ("human",  self.HUMAN),
        ])
        self._chain = self._prompt | self.llm

    def parse(self, query: str) -> Dict:
        try:
            response = self._chain.invoke({"query": query})
            raw = response.content if hasattr(response, "content") else str(response)
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            result = json.loads(raw)

            # ── Post-validate & fix LLM output ───────────────────────
            category = (result.get("category") or "").strip()
            location = (result.get("location") or "").strip()

            # Fix: location empty but query clearly has a city
            if not location:
                _, loc_fix = _smart_split(query)
                if loc_fix:
                    result["location"] = loc_fix

            # Fix: category is wrong (same as full query, or missing)
            if not category or category.lower() == query.lower() or len(category) > 60:
                cat_fix, _ = _smart_split(query)
                result["category"] = _normalize_category(cat_fix)
            else:
                result["category"] = _normalize_category(category)

            # Fix: query_type missing
            if not result.get("query_type"):
                result["query_type"] = (
                    "generic" if _is_generic(result["category"]) else "specific"
                )

            # Fix: search_variations must be list
            sv = result.get("search_variations", [])
            if isinstance(sv, str):
                try:    sv = json.loads(sv)
                except: sv = [sv]
            result["search_variations"] = sv if isinstance(sv, list) else []

            return result

        except Exception as e:
            return self._fallback(query, str(e))

    @staticmethod
    def _fallback(query: str, error: str = "") -> Dict:
        """Pure regex fallback — always works without LLM."""
        cat, loc = _smart_split(query)
        cat = _normalize_category(
            re.sub(r"^(best|top|find|list of|all|cheap|good)\s+",
                   "", cat, flags=re.IGNORECASE).strip()
        )
        is_gen     = _is_generic(cat)
        query_type = "generic" if is_gen else "specific"
        biz_name   = "" if is_gen else cat

        return {
            "query_type":    query_type,
            "business_name": biz_name,
            "category":      cat,
            "location":      loc,
            "location_type": "city",
            "search_variations": [
                f"{cat} {loc} contact phone".strip(),
                f"{cat} {loc} address".strip(),
                f"best {cat} in {loc}".strip() if loc else f"top {cat} near me",
            ],
            "intent":    "search",
            "_fallback": True,
            "_error":    error,
        }
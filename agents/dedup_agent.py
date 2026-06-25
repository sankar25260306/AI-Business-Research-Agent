"""
Phase 9 — Deduplication Agent
Multi-strategy deduplication:
  1. Exact phone match
  2. RapidFuzz fuzzy name + address matching
  3. ChromaDB semantic similarity (if available)
  4. Website domain match
  5. Email match
Merges duplicate records, keeping the richest data.
"""
import re
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from rapidfuzz import fuzz, process

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    _CHROMA = True
except ImportError:
    _CHROMA = False


class DeduplicationAgent:
    """
    Runs multiple deduplication passes on verified business records.
    Returns a clean list with merged, unique businesses.
    """

    # Thresholds
    FUZZY_NAME_THRESHOLD    = 88   # % similarity for name match
    FUZZY_ADDRESS_THRESHOLD = 85
    SEMANTIC_THRESHOLD      = 0.18 # ChromaDB cosine distance (lower = more similar)

    def __init__(self):
        self.use_vector = False
        if _CHROMA:
            try:
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self._chroma = chromadb.Client()
                self._col = self._chroma.get_or_create_collection(
                    "dedup_businesses",
                    metadata={"hnsw:space": "cosine"},
                )
                self.use_vector = True
            except Exception:
                self.use_vector = False

    def deduplicate(self, businesses: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Returns (unique_businesses, duplicates_removed).
        """
        if not businesses:
            return [], 0

        n_before = len(businesses)

        # Pass 1: exact phone dedup
        businesses = self._exact_phone(businesses)
        # Pass 2: exact email dedup
        businesses = self._exact_email(businesses)
        # Pass 3: exact website domain dedup
        businesses = self._exact_website(businesses)
        # Pass 4: fuzzy name + address
        businesses = self._fuzzy_dedup(businesses)
        # Pass 5: semantic (ChromaDB)
        if self.use_vector:
            businesses = self._semantic_dedup(businesses)

        return businesses, n_before - len(businesses)

    # ── Pass 1: Phone ────────────────────────────────────────────────
    def _exact_phone(self, items: List[Dict]) -> List[Dict]:
        seen: Set[str] = set()
        out: List[Dict] = []
        for biz in items:
            phone = re.sub(r"\D", "", biz.get("phone", "") or "")[-10:]
            if phone and len(phone) >= 7:
                if phone in seen:
                    continue
                seen.add(phone)
            out.append(biz)
        return out

    # ── Pass 2: Email ────────────────────────────────────────────────
    def _exact_email(self, items: List[Dict]) -> List[Dict]:
        seen: Set[str] = set()
        out: List[Dict] = []
        for biz in items:
            email = (biz.get("email", "") or "").strip().lower()
            if email and "@" in email:
                if email in seen:
                    continue
                seen.add(email)
            out.append(biz)
        return out

    # ── Pass 3: Website domain ───────────────────────────────────────
    def _exact_website(self, items: List[Dict]) -> List[Dict]:
        seen: Set[str] = set()
        out: List[Dict] = []
        for biz in items:
            site = biz.get("website", "") or ""
            if site.startswith("http"):
                domain = urlparse(site).netloc.lower().replace("www.", "")
                if domain and domain not in ("yelp.com", "yellowpages.com",
                                              "justdial.com", "sulekha.com"):
                    if domain in seen:
                        continue
                    seen.add(domain)
            out.append(biz)
        return out

    # ── Pass 4: RapidFuzz ────────────────────────────────────────────
    def _fuzzy_dedup(self, items: List[Dict]) -> List[Dict]:
        if len(items) <= 1:
            return items

        names    = [self._norm(b.get("business_name", "")) for b in items]
        addresses = [self._norm(b.get("address", "")) for b in items]
        kept: List[bool] = [True] * len(items)

        for i in range(len(items)):
            if not kept[i]:
                continue
            for j in range(i + 1, len(items)):
                if not kept[j]:
                    continue
                name_sim = fuzz.token_sort_ratio(names[i], names[j])
                addr_sim = fuzz.token_sort_ratio(addresses[i], addresses[j])

                # Both name AND address similar → duplicate
                if name_sim >= self.FUZZY_NAME_THRESHOLD and addr_sim >= self.FUZZY_ADDRESS_THRESHOLD:
                    # Merge j into i, discard j
                    items[i] = self._merge(items[i], items[j])
                    kept[j] = False
                # Name very similar and one has no address → likely duplicate
                elif name_sim >= 94 and (not addresses[i] or not addresses[j]):
                    items[i] = self._merge(items[i], items[j])
                    kept[j] = False

        return [b for b, k in zip(items, kept) if k]

    # ── Pass 5: ChromaDB semantic ────────────────────────────────────
    def _semantic_dedup(self, items: List[Dict]) -> List[Dict]:
        out: List[Dict] = []
        for i, biz in enumerate(items):
            text = self._biz_text(biz)
            emb  = self._model.encode(text).tolist()
            bid  = f"biz_{i}"
            try:
                res = self._col.query(query_embeddings=[emb], n_results=3)
                distances = res["distances"][0] if res.get("distances") else []
                if any(d < self.SEMANTIC_THRESHOLD for d in distances):
                    continue  # semantic duplicate
            except Exception:
                pass
            self._col.add(ids=[bid], embeddings=[emb], documents=[text])
            out.append(biz)
        return out

    # ── Helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", (text or "").lower().strip())

    @staticmethod
    def _biz_text(biz: Dict) -> str:
        return " ".join(filter(None, [
            biz.get("business_name", ""),
            biz.get("address", ""),
            biz.get("phone", ""),
            biz.get("website", ""),
        ]))

    @staticmethod
    def _merge(primary: Dict, secondary: Dict) -> Dict:
        """Fill empty fields in primary from secondary."""
        result = primary.copy()
        STRING_FIELDS = [
            "address","phone","email","website","working_hours",
            "rating","review_count","license_information",
        ]
        LIST_FIELDS = ["services","specialties","certifications",
                       "awards","social_profiles","images_urls"]
        for f in STRING_FIELDS:
            if not result.get(f) and secondary.get(f):
                result[f] = secondary[f]
        for f in LIST_FIELDS:
            existing = result.get(f, [])
            extra    = secondary.get(f, [])
            if isinstance(extra, list):
                result[f] = list(dict.fromkeys(existing + extra))
        return result

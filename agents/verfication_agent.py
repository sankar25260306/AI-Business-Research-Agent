"""
Phase 7+8 — Verification Agent + Reliability Scoring Agent
FIXED:
  1. Filters out empty/junk records (no name, no phone, no address)
  2. Filters out location names being used as business names
  3. Minimum data threshold before including a record
  4. Score now 0-100 properly
"""
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


SOURCE_WEIGHTS: Dict[str, float] = {
    "yellowpages":    0.85,
    "yelp":           0.80,
    "duckduckgo":     0.60,
    "duckduckgo_var": 0.55,
    "justdial":       0.70,
    "sulekha":        0.65,
    "linkedin":       0.75,
    "facebook":       0.60,
    "official":       1.00,
    "unknown":        0.50,
}

# Common Indian city/area names that should NOT be business names
LOCATION_NAMES = {
    "koyambedu", "chennai", "mumbai", "delhi", "bangalore", "bengaluru",
    "hyderabad", "pune", "kolkata", "ahmedabad", "coimbatore", "madurai",
    "trichy", "salem", "erode", "vellore", "tirunelveli", "thoothukudi",
    "anna nagar", "t nagar", "adyar", "velachery", "tambaram", "porur",
    "nungambakkam", "egmore", "mylapore", "besant nagar", "perambur",
    "tondiarpet", "ambattur", "avadi", "chrompet", "pallavaram",
    "austin", "dallas", "houston", "chicago", "birmingham", "london",
    "new york", "los angeles", "san francisco", "boston", "seattle",
}


def _is_junk_record(biz: Dict) -> bool:
    """Return True if this record has no useful data."""
    name    = (biz.get("business_name", "") or "").strip()
    phone   = (biz.get("phone", "")         or "").strip()
    address = (biz.get("address", "")       or "").strip()
    email   = (biz.get("email", "")         or "").strip()
    website = (biz.get("website", "")       or "").strip()

    # No name at all
    if not name or len(name) < 2:
        return True

    # Name is just a location/area name
    if name.lower() in LOCATION_NAMES:
        return True

    # Name looks like a URL path fragment
    if name.startswith(("http", "www.", "/", "search?")):
        return True

    # Name is too short or just numbers
    if len(name) < 3 or name.isdigit():
        return True

    # No contact data at all (phone, email, address, website all empty)
    has_contact = any([
        phone, email, address,
        website and "yelp.com/search" not in website
               and "yellowpages.com/search" not in website
               and "justdial.com" not in website
               and "sulekha.com" not in website,
    ])
    if not has_contact:
        return True

    return False


class VerificationAgent:

    def verify(self, businesses: List[Dict]) -> List[Dict]:
        if not businesses:
            return []

        # ── Step 1: Filter junk records ───────────────────────────────
        clean = [b for b in businesses if not _is_junk_record(b)]

        if not clean:
            # Fallback: if everything filtered, return top records with names at least
            clean = [b for b in businesses
                     if (b.get("business_name") or "").strip()
                     and (b.get("business_name") or "").strip().lower() not in LOCATION_NAMES]

        # ── Step 2: Group by phone/address/name ───────────────────────
        groups = self._group(clean)

        # ── Step 3: Merge each group ──────────────────────────────────
        verified = [self._merge_group(g) for g in groups.values()]

        # ── Step 4: Remove merged results that are still junk ─────────
        verified = [v for v in verified if not _is_junk_record(v)]

        verified.sort(key=lambda x: x.get("verification_score", 0), reverse=True)
        return verified

    @staticmethod
    def _group(businesses: List[Dict]) -> Dict[str, List[Dict]]:
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for biz in businesses:
            phone   = re.sub(r"\D", "", biz.get("phone", "") or "")[-10:]
            address = (biz.get("address", "") or "").strip().lower()[:30]
            name    = (biz.get("business_name", "") or "").strip().lower()[:25]

            if phone and len(phone) >= 7:
                key = f"phone:{phone}"
            elif address and len(address) > 8:
                key = f"addr:{address}"
            elif name and len(name) > 3:
                key = f"name:{name}"
            else:
                key = f"uid:{id(biz)}"

            groups[key].append(biz)
        return groups

    def _merge_group(self, group: List[Dict]) -> Dict:
        merged: Dict = {
            "business_name": "", "address": "", "phone": "", "email": "",
            "website": "", "working_hours": "", "rating": "", "review_count": "",
            "services": [], "specialties": [], "license_information": "",
            "certifications": [], "awards": [], "social_profiles": [],
            "images_urls": [], "source_urls": {}, "conflicts": [],
        }

        all_phones:    List[Tuple[str, str]] = []
        all_emails:    List[Tuple[str, str]] = []
        all_addresses: List[Tuple[str, str]] = []

        STRING_FIELDS = [
            "business_name", "address", "email", "website",
            "working_hours", "rating", "review_count", "license_information",
        ]
        LIST_FIELDS = [
            "services", "specialties", "certifications",
            "awards", "social_profiles", "images_urls",
        ]

        for biz in group:
            src = biz.get("source_url", "")
            for f in STRING_FIELDS:
                val = (biz.get(f) or "").strip() if isinstance(biz.get(f), str) else biz.get(f)
                if val and not merged[f]:
                    merged[f] = str(val).strip()
                    if src:
                        merged["source_urls"][f] = src
            for f in LIST_FIELDS:
                if biz.get(f) and isinstance(biz[f], list):
                    merged[f] = list(dict.fromkeys(merged[f] + biz[f]))

            if biz.get("phone"):
                p = re.sub(r"\D", "", str(biz["phone"]))[-10:]
                if p and len(p) >= 7:
                    all_phones.append((p, src))
            if biz.get("email"):
                e = str(biz["email"]).strip().lower()
                if "@" in e:
                    all_emails.append((e, src))
            if biz.get("address"):
                all_addresses.append((str(biz["address"]).strip(), src))

        # Best phone
        if all_phones:
            cnt  = Counter(p for p, _ in all_phones)
            best = cnt.most_common(1)[0][0]
            merged["phone"] = best
            if len(set(cnt.keys())) > 1:
                merged["conflicts"].append(f"Phone conflict: {' vs '.join(set(cnt.keys()))}")

        # Best email
        if all_emails:
            cnt = Counter(e for e, _ in all_emails)
            merged["email"] = cnt.most_common(1)[0][0]
            if len(set(cnt.keys())) > 1:
                merged["conflicts"].append("Email conflict: " + " vs ".join(set(cnt.keys())))

        merged["_phone_sources"]   = all_phones
        merged["_email_sources"]   = all_emails
        merged["_address_sources"] = all_addresses
        merged["sources_count"]    = len(group)
        merged["source_list"]      = [b.get("source", "") for b in group]
        merged["verification_score"] = self._verification_score(
            merged, all_phones, all_emails, len(group)
        )
        return merged

    @staticmethod
    def _verification_score(m: Dict, phones, emails, n: int) -> float:
        score = 0.0

        # Data completeness (max 30)
        fields = ["business_name", "address", "phone", "email",
                  "website", "working_hours", "rating", "license_information"]
        filled = sum(1 for f in fields if m.get(f))
        score += (filled / len(fields)) * 30

        # Multi-source (max 25)
        score += min(n * 8, 25)

        # Phone (max 20)
        if   len(phones) >= 3: score += 20
        elif len(phones) == 2: score += 13
        elif len(phones) == 1: score += 7

        # Email (max 10)
        if emails: score += 10

        # No conflicts (max 10)
        if not m.get("conflicts"): score += 10

        # Website (max 5)
        if m.get("website"): score += 5

        return round(min(score, 100), 1)

    @staticmethod
    def confidence_label(score: float) -> str:
        if score >= 80: return "✅ Highly Verified"
        if score >= 60: return "🟡 Verified"
        if score >= 40: return "🟠 Partial"
        return "🔴 Low Confidence"


class ReliabilityScoringAgent:

    def score(self, businesses: List[Dict]) -> List[Dict]:
        for biz in businesses:
            biz["reliability_score"] = self._compute(biz)
            biz["reliability_label"] = self._label(biz["reliability_score"])
        return businesses

    @staticmethod
    def _compute(biz: Dict) -> float:
        score = 0.0

        # Source reliability (max 30)
        sources = biz.get("source_list", [])
        if sources:
            avg = sum(SOURCE_WEIGHTS.get(s, 0.5) for s in sources) / len(sources)
            score += avg * 30

        # Completeness (max 25)
        key_fields = ["business_name", "phone", "address", "website", "email", "working_hours"]
        filled = sum(1 for f in key_fields if biz.get(f))
        score += (filled / len(key_fields)) * 25

        # Verification score (max 30)
        score += (biz.get("verification_score", 0) / 100) * 30

        # No conflicts (max 10)
        if not biz.get("conflicts"):
            score += 10

        # Services (max 5)
        if biz.get("services") or biz.get("specialties"):
            score += 5

        return round(min(score, 100), 1)

    @staticmethod
    def _label(score: float) -> str:
        if score >= 80: return "🌟 High Reliability"
        if score >= 60: return "✅ Reliable"
        if score >= 40: return "⚠️ Moderate"
        return "❓ Uncertain"

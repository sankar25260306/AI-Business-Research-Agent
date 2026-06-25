"""
Open Data Source Agent — Phase 2 Extension
Adds FREE, unlimited data sources to the search pipeline:

  1. OpenStreetMap / Overpass API  — real business nodes with address, phone, hours
  2. Nominatim (OSM Geocoder)      — convert city name → bounding box for Overpass
  3. data.gov.in                   — India Open Government Data (hospitals, clinics)
  4. Wikidata SPARQL               — structured business/org data
  5. OpenCorporates API            — company registration data (free tier)

All sources are FREE with no API key required (except data.gov.in which has a free key).
"""
import asyncio
import json
import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import aiohttp

# ─── Category → OSM amenity/tag mapping ─────────────────────────────
OSM_AMENITY_MAP: Dict[str, List[str]] = {
    # Medical
    "cardiologist":    ["hospital", "clinic", "doctors"],
    "dentist":         ["dentist", "clinic"],
    "doctor":          ["doctors", "clinic", "hospital"],
    "hospital":        ["hospital"],
    "pharmacy":        ["pharmacy"],
    "physiotherapist": ["physiotherapist", "clinic"],
    "optician":        ["optician", "clinic"],
    "veterinarian":    ["veterinary", "clinic"],
    # Legal / Finance
    "lawyer":          ["lawyers"],
    "bank":            ["bank"],
    "insurance":       ["insurance"],
    "accountant":      ["accounting"],
    # Trades
    "plumber":         ["plumber"],
    "electrician":     ["electrician"],
    "contractor":      ["contractor"],
    # Hospitality
    "restaurant":      ["restaurant", "fast_food", "cafe", "food_court"],
    "hotel":           ["hotel", "guest_house", "motel"],
    "cafe":            ["cafe", "coffee_shop"],
    # Education
    "school":          ["school"],
    "college":         ["college", "university"],
    "coaching":        ["school", "college"],
    # Retail
    "supermarket":     ["supermarket", "grocery"],
    "pharmacy":        ["pharmacy", "chemist"],
    # IT / Tech
    "it company":      ["office", "coworking_space"],
    "software":        ["office"],
}

# ─── data.gov.in dataset IDs for common categories ───────────────────
GOV_DATASETS: Dict[str, str] = {
    "hospital":        "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69",
    "clinic":          "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69",
    "doctor":          "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69",
    "cardiologist":    "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69",
    "school":          "c1e8a0d0-8c6a-4f78-b5f5-d7f0ad6e2d4b",
    "college":         "c1e8a0d0-8c6a-4f78-b5f5-d7f0ad6e2d4b",
    "pharmacy":        "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69",
}

# ─── Overpass API endpoints (tried in order) ─────────────────────────
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

NOMINATIM_URL  = "https://nominatim.openstreetmap.org/search"
WIKIDATA_SPARQL= "https://query.wikidata.org/sparql"

HEADERS = {
    "User-Agent": "BusinessResearchAgent/2.0 (educational research tool; contact@research.com)",
    "Accept":     "application/json",
}


class OpenDataAgent:
    """
    Fetches business data from OpenStreetMap, Open Government sources,
    and Wikidata — all free, no paid API keys needed.
    """

    TIMEOUT = aiohttp.ClientTimeout(total=20)

    async def fetch_all(
        self, category: str, location: str
    ) -> List[Dict]:
        """
        Run all open data sources concurrently.
        Returns merged, deduplicated list of business records.
        """
        # Step 1: Geocode the location to get bounding box
        bbox = await self._geocode(location)

        tasks = [
            self._overpass(category, location, bbox),
            self._gov_data(category, location),
            self._wikidata(category, location),
            self._nominatim_places(category, location, bbox),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_records: List[Dict] = []
        source_counts = {}
        for r in results:
            if isinstance(r, list):
                for rec in r:
                    all_records.append(rec)
                    src = rec.get("source","")
                    source_counts[src] = source_counts.get(src,0) + 1

        print(f"[OpenData] Sources: {source_counts}")
        return self._dedup(all_records)

    # ════════════════════════════════════════════════════════════════
    # 1. Nominatim Geocoder
    # ════════════════════════════════════════════════════════════════
    async def _geocode(self, location: str) -> Optional[Tuple[float,float,float,float]]:
        """
        Convert location name → (south, west, north, east) bounding box.
        Used to query Overpass API for a specific area.
        """
        try:
            params = {
                "q":      location,
                "format": "json",
                "limit":  1,
                "addressdetails": 1,
            }
            async with aiohttp.ClientSession(headers=HEADERS) as sess:
                async with sess.get(NOMINATIM_URL, params=params,
                                    timeout=self.TIMEOUT) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                    if not data:
                        return None
                    bb = data[0].get("boundingbox")
                    if bb and len(bb) == 4:
                        s, n, w, e = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
                        return (s, w, n, e)
        except Exception as e:
            print(f"[Geocode] Error: {e}")
        return None

    # ════════════════════════════════════════════════════════════════
    # 2. Nominatim Place Search (alternative to Overpass)
    # ════════════════════════════════════════════════════════════════
    async def _nominatim_places(
        self, category: str, location: str,
        bbox: Optional[Tuple] = None
    ) -> List[Dict]:
        """
        Search OSM places by name+category using Nominatim's search API.
        Returns individual place records.
        """
        records = []
        amenities = self._get_amenities(category)
        search_terms = [category] + [f"{a} {location}" for a in amenities[:2]]

        try:
            async with aiohttp.ClientSession(headers=HEADERS) as sess:
                for term in search_terms[:3]:
                    # Rate limit: Nominatim requires 1 req/sec
                    await asyncio.sleep(1.1)
                    params = {
                        "q":              term if location.lower() in term.lower()
                                          else f"{term} {location}",
                        "format":         "json",
                        "limit":          20,
                        "addressdetails": 1,
                        "extratags":      1,
                        "namedetails":    1,
                    }
                    if bbox:
                        params["viewbox"] = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
                        params["bounded"]  = 1

                    try:
                        async with sess.get(NOMINATIM_URL, params=params,
                                            timeout=self.TIMEOUT) as resp:
                            if resp.status != 200:
                                continue
                            data = await resp.json()
                            for place in data:
                                rec = self._nominatim_to_record(place)
                                if rec:
                                    records.append(rec)
                    except Exception:
                        continue

        except Exception as e:
            print(f"[Nominatim] Error: {e}")

        return records

    def _nominatim_to_record(self, place: Dict) -> Optional[Dict]:
        """Convert Nominatim result → standard business record."""
        name = (place.get("namedetails",{}).get("name")
                or place.get("display_name","").split(",")[0])
        if not name or len(name) < 2:
            return None

        addr   = place.get("address",{})
        extratags = place.get("extratags",{})

        address_parts = [
            addr.get("house_number",""),
            addr.get("road",""),
            addr.get("suburb",""),
            addr.get("city","") or addr.get("town","") or addr.get("village",""),
            addr.get("state",""),
            addr.get("postcode",""),
        ]
        full_address = ", ".join(p for p in address_parts if p)

        return {
            "business_name":      name.strip(),
            "address":            full_address,
            "phone":              extratags.get("phone","") or extratags.get("contact:phone",""),
            "email":              extratags.get("email","") or extratags.get("contact:email",""),
            "website":            extratags.get("website","") or extratags.get("contact:website",""),
            "working_hours":      extratags.get("opening_hours",""),
            "rating":             "",
            "review_count":       "",
            "services":           [],
            "specialties":        [],
            "license_information":"",
            "certifications":     [],
            "awards":             [],
            "social_profiles":    [],
            "images_urls":        [],
            "source_url":         f"https://www.openstreetmap.org/{place.get('osm_type','node')}/{place.get('osm_id','')}",
            "source":             "openstreetmap_nominatim",
            "lat":                place.get("lat",""),
            "lon":                place.get("lon",""),
        }

    # ════════════════════════════════════════════════════════════════
    # 3. Overpass API (OSM full data)
    # ════════════════════════════════════════════════════════════════
    async def _overpass(
        self, category: str, location: str,
        bbox: Optional[Tuple] = None
    ) -> List[Dict]:
        """
        Query Overpass API for OSM nodes/ways matching the business category
        within the location's bounding box.

        Overpass QL supports rich filtering:
          - amenity type
          - name search
          - geographic bounding box
        """
        amenities = self._get_amenities(category)
        if not amenities:
            amenities = ["shop", "office", "clinic"]

        # Build bounding box string
        if bbox:
            s, w, n, e = bbox
            bbox_str = f"{s},{w},{n},{e}"
        else:
            # Fallback: use location name area search
            bbox_str = None

        records = []
        for amenity in amenities[:3]:
            query = self._build_overpass_query(amenity, category, bbox_str, location)
            result = await self._run_overpass(query)
            if result:
                for element in result.get("elements",[]):
                    rec = self._osm_element_to_record(element)
                    if rec:
                        records.append(rec)

        return records

    def _build_overpass_query(
        self, amenity: str, category: str,
        bbox_str: Optional[str], location: str
    ) -> str:
        """
        Build an Overpass QL query.
        Supports bbox (precise) and area name (fallback).
        """
        timeout = 25
        output  = "out body;"

        if bbox_str:
            # Bounding box query — most precise
            return f"""
[out:json][timeout:{timeout}];
(
  node["amenity"="{amenity}"]({bbox_str});
  way["amenity"="{amenity}"]({bbox_str});
  node["healthcare"="{amenity}"]({bbox_str});
  node["shop"="{amenity}"]({bbox_str});
  node["office"="{amenity}"]({bbox_str});
  node["name"~"{category}",i]({bbox_str});
);
{output}
"""
        else:
            # Area name query — fallback when geocoding fails
            safe_loc = location.replace('"', '')
            return f"""
[out:json][timeout:{timeout}];
area["name"~"{safe_loc}",i]->.searchArea;
(
  node["amenity"="{amenity}"](area.searchArea);
  way["amenity"="{amenity}"](area.searchArea);
  node["healthcare"="{amenity}"](area.searchArea);
  node["name"~"{category}",i](area.searchArea);
);
{output}
"""

    async def _run_overpass(self, query: str) -> Optional[Dict]:
        """Try each Overpass endpoint until one responds."""
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                async with aiohttp.ClientSession(headers=HEADERS) as sess:
                    async with sess.post(
                        endpoint,
                        data={"data": query},
                        timeout=self.TIMEOUT,
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            return data
                        elif resp.status == 429:
                            # Rate limited — wait and try next
                            await asyncio.sleep(2)
            except asyncio.TimeoutError:
                continue
            except Exception:
                continue
        return None

    def _osm_element_to_record(self, element: Dict) -> Optional[Dict]:
        """Convert an OSM node/way element → standard business record."""
        tags = element.get("tags", {})
        name = tags.get("name") or tags.get("name:en") or tags.get("brand")
        if not name or len(name) < 2:
            return None

        # Build address from OSM addr:* tags
        addr_parts = [
            tags.get("addr:housenumber",""),
            tags.get("addr:street",""),
            tags.get("addr:suburb",""),
            tags.get("addr:city",""),
            tags.get("addr:state",""),
            tags.get("addr:postcode",""),
            tags.get("addr:country",""),
        ]
        address = ", ".join(p for p in addr_parts if p)

        # Phone: try multiple tag formats
        phone = (tags.get("phone")
                 or tags.get("contact:phone")
                 or tags.get("telephone")
                 or "")

        # Website
        website = (tags.get("website")
                   or tags.get("contact:website")
                   or tags.get("url")
                   or "")

        # Email
        email = (tags.get("email")
                 or tags.get("contact:email")
                 or "")

        # Hours
        hours = tags.get("opening_hours","")

        # OSM link
        osm_type = element.get("type","node")
        osm_id   = element.get("id","")
        osm_url  = f"https://www.openstreetmap.org/{osm_type}/{osm_id}"

        # Specialties from healthcare tags
        specialties = []
        if tags.get("healthcare:speciality"):
            specialties = [s.strip() for s in tags["healthcare:speciality"].split(";")]
        if tags.get("medical_system:western"):
            specialties.append("Western Medicine")

        # Services
        services = []
        for key in ["healthcare","amenity","shop","office"]:
            if tags.get(key):
                services.append(tags[key].replace("_"," ").title())

        return {
            "business_name":       name.strip(),
            "address":             address,
            "phone":               phone,
            "email":               email,
            "website":             website,
            "working_hours":       hours,
            "rating":              "",
            "review_count":        "",
            "services":            services,
            "specialties":         specialties,
            "license_information": tags.get("licence","") or tags.get("ref",""),
            "certifications":      [],
            "awards":              [],
            "social_profiles":     [
                tags[k] for k in ["contact:facebook","contact:twitter","contact:instagram"]
                if tags.get(k)
            ],
            "images_urls":         [],
            "source_url":          osm_url,
            "source":              "openstreetmap_overpass",
            "lat":                 element.get("lat",""),
            "lon":                 element.get("lon",""),
            "osm_id":              str(osm_id),
            "osm_tags":            tags,   # full raw tags for reference
        }

    # ════════════════════════════════════════════════════════════════
    # 4. Open Government Data (data.gov.in)
    # ════════════════════════════════════════════════════════════════
    async def _gov_data(self, category: str, location: str) -> List[Dict]:
        """
        Fetch from India's Open Government Data portal.
        Hospital, clinic, school data by state/district.
        Free API — register at data.gov.in for a key.
        Falls back to demo key for testing.
        """
        import os
        api_key = os.getenv("DATA_GOV_IN_KEY", "demo")

        # Find relevant dataset
        cat_lower = category.lower()
        dataset_id = None
        for keyword, ds_id in GOV_DATASETS.items():
            if keyword in cat_lower:
                dataset_id = ds_id
                break

        if not dataset_id:
            return []

        # Extract state/district from location
        location_filter = self._extract_location_parts(location)

        records = []
        try:
            # Try direct API
            params = {
                "api-key": api_key,
                "format":  "json",
                "limit":   50,
                "offset":  0,
            }
            # Add location filter if we have state/district
            if location_filter.get("state"):
                params["filters[state]"] = location_filter["state"]
            if location_filter.get("district"):
                params["filters[district]"] = location_filter["district"]

            url = f"https://api.data.gov.in/resource/{dataset_id}"
            async with aiohttp.ClientSession(headers=HEADERS) as sess:
                async with sess.get(url, params=params, timeout=self.TIMEOUT) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        fields = data.get("records",[]) or data.get("data",[]) or []
                        for row in fields:
                            rec = self._gov_row_to_record(row, category)
                            if rec:
                                records.append(rec)
                    elif resp.status == 401:
                        print("[GovData] API key required — register at data.gov.in")
                    else:
                        print(f"[GovData] Status {resp.status}")

        except Exception as e:
            print(f"[GovData] Error: {e}")

        # Also try state-specific portals
        state_records = await self._state_gov_portals(category, location)
        records.extend(state_records)

        return records

    @staticmethod
    def _gov_row_to_record(row: Dict, category: str) -> Optional[Dict]:
        """Convert a data.gov.in row → standard business record."""
        # Try common field names used across different datasets
        name = (row.get("hospital_name") or row.get("institution_name")
                or row.get("name") or row.get("facility_name")
                or row.get("school_name") or row.get("college_name",""))
        if not name:
            return None

        address_parts = [
            row.get("address","") or row.get("street",""),
            row.get("village","") or row.get("ward",""),
            row.get("district","") or row.get("city",""),
            row.get("state",""),
            row.get("pin_code","") or row.get("pincode",""),
        ]
        address = ", ".join(p for p in address_parts if p)

        return {
            "business_name":       name.strip(),
            "address":             address,
            "phone":               row.get("phone","") or row.get("contact_number","") or row.get("mobile",""),
            "email":               row.get("email","") or row.get("email_id",""),
            "website":             row.get("website","") or row.get("url",""),
            "working_hours":       "",
            "rating":              "",
            "review_count":        "",
            "services":            [category],
            "specialties":         [row.get("speciality","") or row.get("type","")] if row.get("speciality","") else [],
            "license_information": row.get("registration_no","") or row.get("license_no",""),
            "certifications":      [],
            "awards":              [],
            "social_profiles":     [],
            "images_urls":         [],
            "source_url":          "https://data.gov.in",
            "source":              "data_gov_in",
        }

    async def _state_gov_portals(self, category: str, location: str) -> List[Dict]:
        """
        Fetch from state-level open data portals.
        Tamil Nadu, Maharashtra, Karnataka have good open data.
        """
        records = []
        location_lower = location.lower()

        state_apis = []

        # Tamil Nadu
        if any(kw in location_lower for kw in
               ["tamil", "tn", "chennai", "coimbatore", "madurai", "trichy",
                "salem", "tirunelveli", "tirupur", "vellore"]):
            state_apis.append({
                "url":    "https://tnmaps.tn.gov.in/wss/services",
                "source": "tn_gov",
                "note":   "Tamil Nadu e-Services",
            })
            # TN Health Department data
            state_apis.append({
                "url":    f"https://nrhm-mis.nic.in/Pages/NHMFacilitiesSearch.aspx",
                "source": "nrhm_nic",
                "note":   "National Health Facility Registry",
            })

        # Karnataka
        if any(kw in location_lower for kw in
               ["karnataka", "bangalore", "bengaluru", "mysore", "hubli"]):
            state_apis.append({
                "url":    "https://data.karnataka.gov.in",
                "source": "karnataka_gov",
            })

        # Try NHFR (National Health Facility Registry) — best for medical queries
        cat_lower = category.lower()
        if any(kw in cat_lower for kw in
               ["hospital","clinic","doctor","cardiologist","dentist","pharmacy",
                "health","medical","nurse","surgeon"]):
            nhfr = await self._nhfr_search(category, location)
            records.extend(nhfr)

        return records

    async def _nhfr_search(self, category: str, location: str) -> List[Dict]:
        """
        National Health Facility Registry (India) — free, comprehensive.
        URL: https://facility.ndhm.gov.in
        """
        records = []
        try:
            # NHFR public search API
            params = {
                "search":    category,
                "district":  location,
                "pageSize":  20,
                "pageIndex": 0,
            }
            url = "https://facility.ndhm.gov.in/search/facility"
            async with aiohttp.ClientSession(headers=HEADERS) as sess:
                async with sess.get(url, params=params, timeout=self.TIMEOUT) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        facilities = data.get("content",[]) or data.get("facilities",[]) or []
                        for f in facilities:
                            rec = {
                                "business_name":       f.get("facilityName",""),
                                "address":             ", ".join(filter(None,[
                                    f.get("address",""), f.get("village",""),
                                    f.get("district",""), f.get("state",""),
                                    f.get("pincode",""),
                                ])),
                                "phone":               f.get("telephone","") or f.get("mobile",""),
                                "email":               f.get("email",""),
                                "website":             f.get("website",""),
                                "working_hours":       "",
                                "rating":              "",
                                "review_count":        "",
                                "services":            [f.get("facilityType","")],
                                "specialties":         [s.get("name","") for s in f.get("specialities",[])],
                                "license_information": f.get("registrationNumber",""),
                                "certifications":      [f.get("nhfr_code","")],
                                "awards":              [],
                                "social_profiles":     [],
                                "images_urls":         [],
                                "source_url":          f"https://facility.ndhm.gov.in/facility/{f.get('facilityId','')}",
                                "source":              "nhfr_india",
                            }
                            if rec["business_name"]:
                                records.append(rec)
        except Exception as e:
            print(f"[NHFR] {e}")
        return records

    # ════════════════════════════════════════════════════════════════
    # 5. Wikidata SPARQL
    # ════════════════════════════════════════════════════════════════
    async def _wikidata(self, category: str, location: str) -> List[Dict]:
        """
        Query Wikidata for businesses/organizations matching category + location.
        Wikidata has structured data on hospitals, universities, companies, etc.
        """
        # Map category to Wikidata instance type (P31)
        wikidata_types = {
            "hospital":     "Q16917",    # hospital
            "clinic":       "Q1774898",  # medical clinic
            "cardiologist": "Q16917",    # hospital (cardiology dept)
            "university":   "Q3918",     # university
            "school":       "Q3914",     # school
            "bank":         "Q22687",    # bank
            "hotel":        "Q27686",    # hotel
            "restaurant":   "Q11707",    # restaurant
            "pharmacy":     "Q507619",   # pharmacy
            "company":      "Q4830453",  # business
        }

        cat_lower = category.lower()
        wikidata_type = None
        for key, qid in wikidata_types.items():
            if key in cat_lower:
                wikidata_type = qid
                break

        if not wikidata_type:
            return []

        # Extract city name for SPARQL filter
        city = location.split(",")[0].strip()

        sparql = f"""
SELECT DISTINCT ?item ?itemLabel ?address ?phone ?website ?email
WHERE {{
  ?item wdt:P31/wdt:P279* wd:{wikidata_type} .
  ?item wdt:P131 ?location .
  ?location rdfs:label ?locLabel .
  FILTER(CONTAINS(LCASE(?locLabel), LCASE("{city}")))
  OPTIONAL {{ ?item wdt:P6375 ?address }}
  OPTIONAL {{ ?item wdt:P1329 ?phone }}
  OPTIONAL {{ ?item wdt:P856  ?website }}
  OPTIONAL {{ ?item wdt:P968  ?email }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 15
"""
        records = []
        try:
            async with aiohttp.ClientSession(headers={**HEADERS,"Accept":"application/sparql-results+json"}) as sess:
                async with sess.get(
                    WIKIDATA_SPARQL,
                    params={"query": sparql, "format": "json"},
                    timeout=self.TIMEOUT,
                ) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json(content_type=None)
                    bindings = data.get("results",{}).get("bindings",[])
                    for b in bindings:
                        name = b.get("itemLabel",{}).get("value","")
                        if not name or name.startswith("Q"):  # skip unlabelled
                            continue
                        records.append({
                            "business_name":       name,
                            "address":             b.get("address",{}).get("value",""),
                            "phone":               b.get("phone",{}).get("value",""),
                            "email":               b.get("email",{}).get("value",""),
                            "website":             b.get("website",{}).get("value",""),
                            "working_hours":       "",
                            "rating":              "",
                            "review_count":        "",
                            "services":            [category],
                            "specialties":         [],
                            "license_information": "",
                            "certifications":      [],
                            "awards":              [],
                            "social_profiles":     [],
                            "images_urls":         [],
                            "source_url":          b.get("item",{}).get("value",""),
                            "source":              "wikidata",
                        })
        except Exception as e:
            print(f"[Wikidata] Error: {e}")

        return records

    # ════════════════════════════════════════════════════════════════
    # Helpers
    # ════════════════════════════════════════════════════════════════
    @staticmethod
    def _get_amenities(category: str) -> List[str]:
        cat_lower = category.lower()
        for key, amenities in OSM_AMENITY_MAP.items():
            if key in cat_lower or cat_lower in key:
                return amenities
        return ["clinic", "office", "shop"]

    @staticmethod
    def _extract_location_parts(location: str) -> Dict[str, str]:
        """Extract state and district from location string."""
        parts = [p.strip() for p in location.split(",")]
        result = {"city": "", "district": "", "state": ""}
        if parts:
            result["city"] = parts[0]
        if len(parts) >= 2:
            result["district"] = parts[-2] if len(parts) > 2 else parts[1]
        if len(parts) >= 2:
            result["state"] = parts[-1]

        # Tamil Nadu city → state mapping
        tn_cities = ["coimbatore","chennai","madurai","trichy","salem",
                     "tirunelveli","tirupur","vellore","erode","thanjavur"]
        if any(c in location.lower() for c in tn_cities):
            result["state"] = "Tamil Nadu"

        return result

    @staticmethod
    def _dedup(records: List[Dict]) -> List[Dict]:
        seen, out = set(), []
        for r in records:
            key = (r.get("business_name","").lower().strip()[:30] +
                   re.sub(r"\D","",r.get("phone",""))[-7:])
            if key and key not in seen:
                seen.add(key)
                out.append(r)
            elif not key:
                out.append(r)
        return out

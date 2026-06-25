"""
Phase 10+11 — Storage Layer
PostgreSQL for structured data + ChromaDB for semantic search.
Falls back gracefully to in-memory SQLite when PostgreSQL unavailable.
"""
import json
import os
import sqlite3
import time
from typing import Dict, List, Optional

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    _CHROMA_OK = True
except ImportError:
    _CHROMA_OK = False


# ─── PostgreSQL / SQLite ─────────────────────────────────────────────
class DatabaseStorage:
    """
    Stores business records, search history, and reports.
    Uses PostgreSQL when DATABASE_URL is set, else SQLite (local dev).
    """

    def __init__(self):
        self._pg_url = os.getenv("DATABASE_URL", "")
        self._sqlite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research_agent.db")
        self._use_pg = bool(self._pg_url)
        self._conn_sqlite: Optional[sqlite3.Connection] = None
        self._init_db()

    # ── Init ─────────────────────────────────────────────────────────
    def _init_db(self):
        if self._use_pg:
            self._init_pg()
        else:
            self._init_sqlite()

    def _init_sqlite(self):
        self._conn_sqlite = sqlite3.connect(self._sqlite_path, check_same_thread=False)
        cur = self._conn_sqlite.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS searches (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query       TEXT NOT NULL,
            category    TEXT,
            location    TEXT,
            created_at  REAL DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS businesses (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            search_id           INTEGER,
            business_name       TEXT,
            address             TEXT,
            phone               TEXT,
            email               TEXT,
            website             TEXT,
            working_hours       TEXT,
            rating              TEXT,
            review_count        TEXT,
            services            TEXT,
            specialties         TEXT,
            license_information TEXT,
            certifications      TEXT,
            awards              TEXT,
            social_profiles     TEXT,
            images_urls         TEXT,
            source_urls         TEXT,
            verification_score  REAL,
            reliability_score   REAL,
            conflicts           TEXT,
            source_url          TEXT,
            created_at          REAL DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS reports (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            search_id       INTEGER,
            summary_json    TEXT,
            ai_summary      TEXT,
            created_at      REAL DEFAULT (strftime('%s','now'))
        );
        """)
        self._conn_sqlite.commit()

    def _init_pg(self):
        try:
            import psycopg2
            conn = psycopg2.connect(self._pg_url)
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS searches (
                id SERIAL PRIMARY KEY, query TEXT, category TEXT,
                location TEXT, created_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS businesses (
                id SERIAL PRIMARY KEY, search_id INT,
                data JSONB, created_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS reports (
                id SERIAL PRIMARY KEY, search_id INT,
                summary_json JSONB, ai_summary TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️ PostgreSQL init failed, falling back to SQLite: {e}")
            self._use_pg = False
            self._init_sqlite()

    # ── Public API ───────────────────────────────────────────────────
    def save_search(self, query: str, category: str, location: str) -> int:
        if self._use_pg:
            return self._pg_save_search(query, category, location)
        return self._sqlite_save_search(query, category, location)

    def save_businesses(self, search_id: int, businesses: List[Dict]) -> None:
        if self._use_pg:
            self._pg_save_businesses(search_id, businesses)
        else:
            self._sqlite_save_businesses(search_id, businesses)

    def save_report(self, search_id: int, report: Dict, ai_summary: str) -> None:
        if self._use_pg:
            self._pg_save_report(search_id, report, ai_summary)
        else:
            self._sqlite_save_report(search_id, report, ai_summary)

    def get_recent_searches(self, limit: int = 10) -> List[Dict]:
        if self._use_pg:
            return self._pg_recent()
        return self._sqlite_recent(limit)

    # ── SQLite implementations ────────────────────────────────────────
    def _sqlite_save_search(self, query, category, location) -> int:
        cur = self._conn_sqlite.cursor()
        cur.execute(
            "INSERT INTO searches (query,category,location) VALUES (?,?,?)",
            (query, category, location)
        )
        self._conn_sqlite.commit()
        return cur.lastrowid

    def _sqlite_save_businesses(self, search_id, businesses):
        cur = self._conn_sqlite.cursor()
        for biz in businesses:
            cur.execute("""
            INSERT INTO businesses
            (search_id,business_name,address,phone,email,website,working_hours,
             rating,review_count,services,specialties,license_information,
             certifications,awards,social_profiles,images_urls,source_urls,
             verification_score,reliability_score,conflicts,source_url)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                search_id,
                biz.get("business_name",""),
                biz.get("address",""),
                biz.get("phone",""),
                biz.get("email",""),
                biz.get("website",""),
                biz.get("working_hours",""),
                biz.get("rating",""),
                biz.get("review_count",""),
                json.dumps(biz.get("services",[])),
                json.dumps(biz.get("specialties",[])),
                biz.get("license_information",""),
                json.dumps(biz.get("certifications",[])),
                json.dumps(biz.get("awards",[])),
                json.dumps(biz.get("social_profiles",[])),
                json.dumps(biz.get("images_urls",[])),
                json.dumps(biz.get("source_urls",{})),
                biz.get("verification_score",0),
                biz.get("reliability_score",0),
                json.dumps(biz.get("conflicts",[])),
                biz.get("source_url",""),
            ))
        self._conn_sqlite.commit()

    def _sqlite_save_report(self, search_id, report, ai_summary):
        cur = self._conn_sqlite.cursor()
        cur.execute(
            "INSERT INTO reports (search_id,summary_json,ai_summary) VALUES (?,?,?)",
            (search_id, json.dumps(report), ai_summary)
        )
        self._conn_sqlite.commit()

    def _sqlite_recent(self, limit) -> List[Dict]:
        cur = self._conn_sqlite.cursor()
        cur.execute(
            "SELECT id,query,category,location,created_at FROM searches ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = cur.fetchall()
        return [{"id":r[0],"query":r[1],"category":r[2],"location":r[3],"created_at":r[4]} for r in rows]

    # ── PostgreSQL implementations ────────────────────────────────────
    def _pg_save_search(self, query, category, location) -> int:
        import psycopg2
        conn = psycopg2.connect(self._pg_url)
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO searches (query,category,location) VALUES (%s,%s,%s) RETURNING id",
            (query, category, location)
        )
        row = cur.fetchone()
        conn.commit(); conn.close()
        return row[0] if row else 0

    def _pg_save_businesses(self, search_id, businesses):
        import psycopg2, psycopg2.extras
        conn = psycopg2.connect(self._pg_url)
        cur  = conn.cursor()
        for biz in businesses:
            cur.execute(
                "INSERT INTO businesses (search_id,data) VALUES (%s,%s)",
                (search_id, psycopg2.extras.Json(biz))
            )
        conn.commit(); conn.close()

    def _pg_save_report(self, search_id, report, ai_summary):
        import psycopg2, psycopg2.extras
        conn = psycopg2.connect(self._pg_url)
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO reports (search_id,summary_json,ai_summary) VALUES (%s,%s,%s)",
            (search_id, psycopg2.extras.Json(report), ai_summary)
        )
        conn.commit(); conn.close()

    def _pg_recent(self) -> List[Dict]:
        import psycopg2
        conn = psycopg2.connect(self._pg_url)
        cur  = conn.cursor()
        cur.execute("SELECT id,query,category,location FROM searches ORDER BY id DESC LIMIT 10")
        rows = cur.fetchall()
        conn.close()
        return [{"id":r[0],"query":r[1],"category":r[2],"location":r[3]} for r in rows]


# ─── ChromaDB Semantic Store ─────────────────────────────────────────
class VectorStore:
    """
    Phase 11 — ChromaDB semantic store for business embeddings.
    Enables semantic search: "find cardiology clinics near downtown".
    """

    def __init__(self):
        self.available = False
        if _CHROMA_OK:
            try:
                self._model  = SentenceTransformer("all-MiniLM-L6-v2")
                self._client = chromadb.Client()
                self._col    = self._client.get_or_create_collection(
                    "business_embeddings",
                    metadata={"hnsw:space": "cosine"},
                )
                self.available = True
            except Exception:
                pass

    def store_businesses(self, businesses: List[Dict], search_id: int) -> None:
        if not self.available:
            return
        for i, biz in enumerate(businesses):
            text = self._text(biz)
            emb  = self._model.encode(text).tolist()
            bid  = f"s{search_id}_b{i}"
            meta = {
                "name":     biz.get("business_name","")[:50],
                "phone":    biz.get("phone",""),
                "search_id": str(search_id),
            }
            try:
                self._col.upsert(ids=[bid], embeddings=[emb], documents=[text], metadatas=[meta])
            except Exception:
                pass

    def semantic_search(self, query: str, n: int = 5) -> List[Dict]:
        if not self.available:
            return []
        try:
            emb = self._model.encode(query).tolist()
            res = self._col.query(query_embeddings=[emb], n_results=n)
            out = []
            for i, mid in enumerate(res["ids"][0]):
                out.append({
                    "id": mid,
                    "distance": res["distances"][0][i],
                    "metadata": res["metadatas"][0][i],
                })
            return out
        except Exception:
            return []

    @staticmethod
    def _text(biz: Dict) -> str:
        return " ".join(filter(None, [
            biz.get("business_name",""),
            biz.get("address",""),
            biz.get("specialties","") if isinstance(biz.get("specialties"), str)
                else " ".join(biz.get("specialties",[])),
            biz.get("services","") if isinstance(biz.get("services"), str)
                else " ".join(biz.get("services",[])),
        ]))

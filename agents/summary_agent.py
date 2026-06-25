"""
Phase 12+13 — Summary Agent + Research Report Generator
Uses LangGraph state machine + LangChain + Groq to:
  - Generate AI research summary
  - Build structured research report
  - Compute data quality metrics
"""
import json
import time
from typing import Any, Dict, List, TypedDict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from core.config import get_groq_key, ResearchReport


# ─── LangGraph State ─────────────────────────────────────────────────
class ReportState(TypedDict):
    query: str
    category: str
    location: str
    businesses: List[Dict]
    duration: float
    stats: Dict
    data_quality: Dict
    ai_summary: str
    report: Dict


# ─── Summary Agent (LangGraph nodes) ────────────────────────────────
class SummaryAgent:
    """
    LangGraph-powered report generation pipeline:
      compute_stats → compute_quality → generate_summary → build_report
    """

    def __init__(self):
        self.llm = ChatGroq(
            api_key=get_groq_key(),
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=800,
        )
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        g = StateGraph(ReportState)

        g.add_node("compute_stats",   self._node_stats)
        g.add_node("compute_quality", self._node_quality)
        g.add_node("generate_summary",self._node_summary)
        g.add_node("build_report",    self._node_report)

        g.set_entry_point("compute_stats")
        g.add_edge("compute_stats",    "compute_quality")
        g.add_edge("compute_quality",  "generate_summary")
        g.add_edge("generate_summary", "build_report")
        g.add_edge("build_report",     END)

        return g.compile()

    # ── Node: compute stats ──────────────────────────────────────────
    @staticmethod
    def _node_stats(state: ReportState) -> ReportState:
        businesses = state["businesses"]
        n = len(businesses)
        verified = sum(1 for b in businesses if b.get("verification_score", 0) >= 40)
        high_conf = sum(1 for b in businesses if b.get("verification_score", 0) >= 80)
        avg_score = sum(b.get("verification_score",0) for b in businesses) / max(n,1)

        state["stats"] = {
            "total": n,
            "verified": verified,
            "high_confidence": high_conf,
            "avg_verification": round(avg_score, 1),
        }
        return state

    # ── Node: compute data quality ────────────────────────────────────
    @staticmethod
    def _node_quality(state: ReportState) -> ReportState:
        bz = state["businesses"]
        n  = max(len(bz), 1)
        pct = lambda f: f"{sum(1 for b in bz if b.get(f)) / n:.0%}"
        state["data_quality"] = {
            "has_website":      pct("website"),
            "has_phone":        pct("phone"),
            "has_address":      pct("address"),
            "has_email":        pct("email"),
            "has_hours":        pct("working_hours"),
            "has_rating":       pct("rating"),
            "has_license":      pct("license_information"),
        }
        return state

    # ── Node: AI summary ─────────────────────────────────────────────
    def _node_summary(self, state: ReportState) -> ReportState:
        bz    = state["businesses"]
        stats = state["stats"]
        dq    = state["data_quality"]

        # Build compact business list for prompt (top 5)
        top5  = bz[:5]
        biz_text = "\n".join(
            f"  {i+1}. {b.get('business_name','N/A')} | "
            f"Phone: {b.get('phone','N/A')} | "
            f"Score: {b.get('verification_score',0):.0f}/100"
            for i, b in enumerate(top5)
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert business research analyst. "
             "Write clear, professional research summaries."),
            ("human",
             "Generate a concise research summary for this query.\n\n"
             "Query: {query}\n"
             "Category: {category}\n"
             "Location: {location}\n"
             "Businesses found: {total}\n"
             "Verified: {verified}\n"
             "Avg verification score: {avg_score}/100\n"
             "Data quality: {dq}\n"
             "Top businesses:\n{biz_list}\n\n"
             "Write a 3-4 sentence professional summary covering:\n"
             "- What was found\n"
             "- Data quality highlights\n"
             "- Notable businesses or patterns\n"
             "- Recommendation for next steps"),
        ])
        chain = prompt | self.llm
        try:
            resp = chain.invoke({
                "query":    state["query"],
                "category": state["category"],
                "location": state["location"],
                "total":    stats["total"],
                "verified": stats["verified"],
                "avg_score":stats["avg_verification"],
                "dq":       json.dumps(dq, indent=2),
                "biz_list": biz_text,
            })
            state["ai_summary"] = resp.content if hasattr(resp,"content") else str(resp)
        except Exception as e:
            state["ai_summary"] = f"Research completed. Found {stats['total']} businesses."
        return state

    # ── Node: build final report ──────────────────────────────────────
    @staticmethod
    def _node_report(state: ReportState) -> ReportState:
        from core.config import ALL_SOURCES
        report = ResearchReport(
            query       = state["query"],
            category    = state["category"],
            location    = state["location"],
            businesses_found   = state["stats"]["total"],
            businesses_verified= state["stats"]["verified"],
            duplicates_removed = 0,          # set by caller
            sources_searched   = len(ALL_SOURCES),
            duration_seconds   = state["duration"],
            businesses  = state["businesses"],
            data_quality= state["data_quality"],
            ai_summary  = state["ai_summary"],
        )
        state["report"] = report.to_dict()
        return state

    # ── Public entrypoint ─────────────────────────────────────────────
    def generate(
        self,
        query: str,
        category: str,
        location: str,
        businesses: List[Dict],
        duration: float,
        duplicates_removed: int = 0,
    ) -> Dict:
        initial: ReportState = {
            "query": query, "category": category, "location": location,
            "businesses": businesses, "duration": duration,
            "stats": {}, "data_quality": {}, "ai_summary": "", "report": {},
        }
        final = self._graph.invoke(initial)
        report = final["report"]
        report["duplicates_removed"] = duplicates_removed
        return report


# ─── Export helpers ──────────────────────────────────────────────────
def report_to_csv(businesses: List[Dict]) -> str:
    """Convert business list to CSV string."""
    import csv, io
    fields = [
        "business_name","address","phone","email","website",
        "working_hours","rating","review_count","license_information",
        "verification_score","reliability_score",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for biz in businesses:
        row = {f: biz.get(f,"") for f in fields}
        # Flatten lists
        for lf in ["services","specialties","certifications"]:
            v = biz.get(lf, [])
            row[lf] = "; ".join(v) if isinstance(v, list) else str(v)
        w.writerow(row)
    return buf.getvalue()
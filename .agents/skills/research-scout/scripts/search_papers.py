#!/usr/bin/env python3
"""Search Semantic Scholar for claudeCFD research candidates.

The script intentionally uses the Python standard library so it can run in a
minimal Codex sandbox. Network access must be enabled by the environment.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path


QUERIES = [
    "compressible multiphase pressure equilibrium preserving diffuse interface",
    "all Mach compressible multiphase five equation model pressure oscillation",
    "APEC ACID energy flux compressible two phase flow",
    "path conservative alpha source Kapila five equation model",
    "pressure Helmholtz IMEX all Mach multiphase flow",
]


def _pipeline_dir() -> Path:
    return Path(os.environ.get("PIPELINE_DIR", ".agents/pipeline"))


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _fetch(query: str, limit: int) -> list[dict]:
    params = urllib.parse.urlencode({
        "query": query,
        "limit": str(limit),
        "fields": "title,abstract,year,citationCount,tldr,url,venue",
    })
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("data", [])


def _text(paper: dict) -> str:
    tldr = paper.get("tldr") or {}
    return " ".join([
        str(paper.get("title") or ""),
        str(paper.get("abstract") or ""),
        str(tldr.get("text") or ""),
    ]).lower()


def _score(paper: dict, implemented: list[str], blocklist: list[str]) -> tuple[float, str]:
    text = _text(paper)
    if any(term.lower() in text for term in blocklist):
        return -1.0, "blocklisted"
    novelty = 0.3 if any(term.lower() in text for term in implemented) else 1.0
    keywords = {
        "pressure equilibrium": 4.0,
        "all mach": 3.0,
        "five-equation": 3.0,
        "five equation": 3.0,
        "path-conservative": 2.5,
        "pressure oscillation": 2.5,
        "helmholtz": 2.0,
        "positivity": 1.5,
        "thinc": 1.0,
    }
    relevance = sum(weight for key, weight in keywords.items() if key in text)
    year = int(paper.get("year") or 0)
    current_year = _dt.date.today().year
    recency = 1.5 if year >= current_year - 2 else 1.0
    citations = float(paper.get("citationCount") or 0)
    score = (relevance + 0.05 * citations + 1.0) * novelty * recency
    return score, "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    out_dir = _pipeline_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    implemented = _read_json(out_dir / "implemented_methods.json", [])
    blocklist = _read_json(out_dir / "blocklist.json", [])

    papers: dict[str, dict] = {}
    errors = []
    for query in QUERIES:
        try:
            for paper in _fetch(query, args.limit):
                key = paper.get("paperId") or paper.get("url") or paper.get("title")
                if key:
                    papers[str(key)] = paper
        except Exception as exc:
            errors.append({"query": query, "error": f"{type(exc).__name__}: {exc}"})

    candidates = []
    for paper in papers.values():
        score, reason = _score(paper, implemented, blocklist)
        if score <= 0:
            continue
        tldr = paper.get("tldr") or {}
        candidates.append({
            "score": score,
            "title": paper.get("title"),
            "year": paper.get("year"),
            "venue": paper.get("venue"),
            "citationCount": paper.get("citationCount"),
            "url": paper.get("url"),
            "tldr": tldr.get("text"),
            "abstract": paper.get("abstract"),
            "status": reason,
        })
    candidates.sort(key=lambda item: item["score"], reverse=True)

    report = {
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "queries": QUERIES,
        "errors": errors,
        "candidates": candidates[: args.top],
    }
    (out_dir / "scout_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {out_dir / 'scout_report.json'} with {len(report['candidates'])} candidates")
    if errors and not candidates:
        print("No candidates found. Network access may be disabled.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

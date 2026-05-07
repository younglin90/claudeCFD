---
name: paper-research-hybrid
description: Hybrid paper discovery and PDF retrieval workflow using the repo-local paper-search MCP/CLI plus pdf-downloader fallback. Use when searching CFD/scientific papers, downloading open-access PDFs, building a local paper library, or preparing research-scout inputs.
---

# Paper Research Hybrid

Use this skill for literature workflows that need both search breadth and robust local PDF capture.

## Tools

- Primary search/download CLI:
  `UV_CACHE_DIR=/tmp/uv-cache uv run --directory /home/younglin90/work/claude_code/claudeCFD/.agents/tools/paper-search-mcp paper-search ...`
- PDF fallback/preservation:
  `/home/younglin90/work/claude_code/claudeCFD/.agents/skills/pdf-downloader/.venv/bin/python /home/younglin90/work/claude_code/claudeCFD/.agents/skills/pdf-downloader/scripts/download_pdf.py ...`

## Storage

Save retrieved artifacts under:

```text
papers/library/
├── pdf/
├── md/
└── index.jsonl
```

Use stable filenames:

```text
{year}_{first-author}_{short-title}.pdf
{year}_{first-author}_{short-title}.md
```

## Workflow

1. Search with `paper-search` using targeted sources first.
   Prefer `arxiv,semantic,crossref,openalex,hal,zenodo` for CFD/numerics.
2. Deduplicate by DOI, arXiv ID, normalized title, and year.
3. Present candidate papers with title, authors, year, source, DOI/arXiv, and why relevant.
4. For selected papers, try `paper-search download` first when a source/paper ID is available.
5. If no direct paper-search download succeeds, pass the DOI/arXiv/publisher/PDF URL to `pdf-downloader`.
6. If a web page blocks direct fetch, create a temporary Markdown capture and generate an offline PDF with `pdf-downloader --markdown`.
7. Record every result in `papers/library/index.jsonl`.
8. Never use paywall bypasses unless the user explicitly asks and accepts the legal/access responsibility. Prefer open-access sources.

## Commands

Search:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --directory /home/younglin90/work/claude_code/claudeCFD/.agents/tools/paper-search-mcp paper-search search "five equation all Mach diffuse interface multiphase" -n 5 -s arxiv,semantic,crossref,openalex
```

List sources:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --directory /home/younglin90/work/claude_code/claudeCFD/.agents/tools/paper-search-mcp paper-search sources
```

Fallback PDF capture:

```bash
/home/younglin90/work/claude_code/claudeCFD/.agents/skills/pdf-downloader/.venv/bin/python /home/younglin90/work/claude_code/claudeCFD/.agents/skills/pdf-downloader/scripts/download_pdf.py "$URL" --output-dir papers/library/pdf --filename "$FILENAME"
```

Markdown fallback:

```bash
/home/younglin90/work/claude_code/claudeCFD/.agents/skills/pdf-downloader/.venv/bin/python /home/younglin90/work/claude_code/claudeCFD/.agents/skills/pdf-downloader/scripts/download_pdf.py --markdown tmp/article.md --source-url "$URL" --output-dir papers/library/pdf --filename "$FILENAME"
```

## Index JSONL Schema

Append one JSON object per paper:

```json
{
  "title": "",
  "authors": [],
  "year": null,
  "doi": "",
  "arxiv_id": "",
  "source": "",
  "query": "",
  "pdf_path": "",
  "markdown_path": "",
  "retrieval_tool": "paper-search|pdf-downloader",
  "retrieved_at": ""
}
```


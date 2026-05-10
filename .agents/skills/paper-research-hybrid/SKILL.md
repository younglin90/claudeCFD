---
name: paper-research-hybrid
description: Unified paper discovery, open-access PDF retrieval, and local paper-library workflow for CFD/scientific literature. Use when searching academic papers, downloading legal/open-access PDFs, extracting paper text, building papers/library, or preparing research-scout inputs; this replaces repo-local paper-search and pdf-downloader skills.
---

# Paper Research Hybrid

Use this as the single repo-local paper workflow. Search broadly with `paper-search`; download with source-native and legal OA fallbacks; preserve public webpages as offline PDFs only when no direct OA PDF exists.

## Commands

Search:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT=/tmp/codex-paper-search-mcp-venv uv run --directory /home/younglin90/work/claude_code/claudeCFD/.agents/tools/paper-search-mcp paper-search search "<query>" -n 5 -s arxiv,semantic,crossref,openalex,pmc,europepmc,hal,zenodo
```

Download from a known source:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT=/tmp/codex-paper-search-mcp-venv uv run --directory /home/younglin90/work/claude_code/claudeCFD/.agents/tools/paper-search-mcp paper-search download <source> <paper_id> -o papers/library/pdf
```

Download with legal OA fallback:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT=/tmp/codex-paper-search-mcp-venv uv run --directory /home/younglin90/work/claude_code/claudeCFD/.agents/tools/paper-search-mcp paper-search download-oa --source <source> --paper-id <paper_id> --doi "<doi>" --title "<title>" -o papers/library/pdf --failure-report papers/library/download_failures.jsonl --manual-dir papers/library/manual_inbox
```

Preserve a public landing page when no direct PDF is available:

```bash
/home/younglin90/.codex/skills/paper-research-hybrid/tools/paper-search-mcp/.venv/bin/python /home/younglin90/work/claude_code/claudeCFD/.agents/skills/paper-research-hybrid/scripts/download_pdf.py "$URL" --output-dir papers/library/pdf --filename "$FILENAME"
```

Convert equation-heavy PDFs to Markdown:

```bash
python3 /home/younglin90/work/claude_code/claudeCFD/.agents/skills/paper-research-hybrid/scripts/pdf_to_md.py papers/library/pdf/example.pdf --outdir papers/library/md --backend fitz
```

Use `--backend marker` when `marker-pdf` is installed and high-quality LaTeX equation extraction is needed. Use `--backend pymupdf4llm` when installed and equation/image preservation is preferred. The built-in `fitz` backend is deterministic and extracts likely display equations as PNG images.

## Workflow

1. Search targeted sources first. Prefer `arxiv,semantic,crossref,openalex,pmc,europepmc,hal,zenodo` for CFD/numerics.
2. Deduplicate by DOI, arXiv ID, normalized title, and year.
3. Present candidates with title, authors, year, source, DOI/arXiv ID, URL/PDF URL, and relevance.
4. For selected papers, try `paper-search download`.
5. If that fails, run `paper-search download-oa` with DOI/title/source metadata.
6. Prefer DOI, arXiv ID, source, and `paper_id` over title-only fallback. Title-only retrieval is best-effort and can be ambiguous when several papers reuse the same title phrase.
7. If there is still no OA PDF but the public page is accessible, use the bundled webpage-to-PDF preservation script and label it as an offline page capture, not an official PDF.
8. If all legal download paths fail, report `papers/library/download_failures.md` to the user and ask them to place the PDF in `papers/library/manual_inbox/` using the suggested filename when possible.
9. After manual PDFs are added, convert them with `scripts/pdf_to_md.py` and read only the relevant sections first: abstract, governing equations, numerical method, stability/limiter details, validation cases, and conclusions.
10. Record retrieved artifacts in `papers/library/index.jsonl`.

## Equation-Heavy Reading

- First convert only a page window around the method section when the PDF is long: `--pages 1-8`, then widen as needed.
- Prefer `marker` for LaTeX equations when installed; otherwise use `pymupdf4llm` or the built-in `fitz` backend, which preserves likely display equations as PNG images.
- Treat fallback text extraction of formulas as approximate. For equations that drive implementation, inspect the generated equation images or the original PDF page.
- Keep token use small: read the converted Markdown headings, abstract, model equations, discretization, limiter/reconstruction details, validation tables, and conclusion before reading the full paper.
- If the user manually provides failed PDFs in `papers/library/manual_inbox/`, convert them with:

```bash
python3 /home/younglin90/work/claude_code/claudeCFD/.agents/skills/paper-research-hybrid/scripts/pdf_to_md.py papers/library/manual_inbox/*.pdf --outdir papers/library/md --backend fitz
```

## Fallback Policy

Allowed fallback order:

```text
source-native download
direct PDF URL
Unpaywall OA URL
OpenAlex / PMC / EuropePMC / CORE / OpenAIRE / Zenodo / HAL / DOAJ / BASE OA links
public webpage-to-PDF preservation
```

Do not use Sci-Hub, paywall bypasses, credential sharing, or mirror scraping. If all legal/OA routes fail, report the failure with DOI/title/URL and suggest legal access paths.

## Storage

Use:

```text
papers/library/pdf/
papers/library/md/
papers/library/index.jsonl
papers/library/download_failures.jsonl
papers/library/download_failures.md
papers/library/manual_inbox/
```

Stable filenames:

```text
{year}_{first-author}_{short-title}.pdf
{year}_{first-author}_{short-title}.md
```

Index JSONL object:

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
  "retrieval_tool": "paper-search|paper-search-download-oa|pdf-page-capture",
  "retrieved_at": ""
}
```

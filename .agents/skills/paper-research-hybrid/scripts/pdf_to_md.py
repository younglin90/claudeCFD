#!/usr/bin/env python3
"""PDF to Markdown converter for equation-heavy academic papers.

Backends, in priority order when available:
1. marker: high-quality Markdown with LaTeX equations.
2. pymupdf4llm: Markdown with image extraction.
3. fitz/PyMuPDF: deterministic fallback; extracts likely display equations as PNG.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

_HAS_MARKER = False
try:
    from marker.converters.pdf import PdfConverter as _MarkerPdfConverter
    from marker.models import create_model_dict as _marker_create_model_dict
    _HAS_MARKER = True
except ImportError:
    pass

_HAS_PYMUPDF4LLM = False
try:
    import pymupdf4llm
    _HAS_PYMUPDF4LLM = True
except ImportError:
    pass

try:
    import fitz
except ImportError as exc:  # pragma: no cover - hard dependency for fallback
    raise SystemExit("PyMuPDF is required for fallback conversion: install pymupdf") from exc

_marker_model_dict = None


def _get_marker_models():
    global _marker_model_dict
    if _marker_model_dict is None:
        print("  Loading marker models (first time only)...")
        _marker_model_dict = _marker_create_model_dict()
    return _marker_model_dict


def _postprocess_common(md: str) -> str:
    md = re.sub(r"\n{4,}", "\n\n\n", md)
    md = re.sub(r"(\w)- (\w)", r"\1\2", md)
    md = re.sub(r"<span id=\"page-\d+-\d+\">\s*</span>\s*", "", md)
    for pattern in [
        r"(?m)^##\s*Physics of Fluids\s*$\n*",
        r"(?m)^##\s*ARTICLE\s*$\n*",
        r"(?m)^##\s*pubs\.aip\.org/aip/pof\s*$\n*",
        r"(?m)^Phys\. Fluids \d+,.*Published under.*$\n*",
        r"(?m)^##?\s*Journal of Computational Physics\s*$\n*",
        r"(?m)^\d+ \w+ \d{4} \d{2}:\d{2}:\d{2}\s*$\n*",
        r"(?m)^\d+, \d{6}-\d+\s*$\n*",
        r"(?m)^Published under.*$\n*",
    ]:
        md = re.sub(pattern, "", md)
    return re.sub(r"\n{4,}", "\n\n\n", md)


def _convert_with_marker(pdf_path: Path, *, page_range=None, **kwargs) -> str:
    model_dict = _get_marker_models()
    converter = _MarkerPdfConverter(artifact_dict=model_dict)
    rendered = converter(str(pdf_path))
    return _postprocess_common(rendered.markdown)


def _convert_with_pymupdf4llm(pdf_path: Path, *, page_range=None, dpi=150, image_dir=None) -> str:
    if image_dir is None:
        image_dir = pdf_path.parent / "images"
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    pages = None
    if page_range:
        start, end = page_range
        pages = list(range(start - 1, end))

    md = pymupdf4llm.to_markdown(
        str(pdf_path),
        pages=pages,
        write_images=True,
        image_path=str(image_dir) + "/",
        image_format="png",
        dpi=dpi,
        image_size_limit=0.005,
        force_text=True,
        detect_bg_color=True,
    )
    image_dir_str = str(image_dir)
    md = md.replace(image_dir_str + "/", "images/").replace(image_dir_str, "images")
    md = re.sub(r"\*\*==> picture \[\d+ x \d+\] intentionally omitted <==\*\*\n*", "", md)
    md = re.sub(
        r"\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*",
        "",
        md,
        flags=re.DOTALL,
    )
    return _postprocess_common(md)


def _detect_body_fontsize(blocks) -> float:
    sizes = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if len(text) > 20:
                    sizes.append(round(span.get("size", 10.0), 1))
    return Counter(sizes).most_common(1)[0][0] if sizes else 10.0


_MATH_CHARS = re.compile(
    r"[\u2200-\u22FF\u2A00-\u2AFF\u0370-\u03FF\u2190-\u21FF\u2100-\u214F"
    r"\u00B1\u00D7\u00F7\u2260-\u226F\u222B\u222C\u222D\u2202\u2207\u221A\u221E]"
)
_EQ_NUMBER_RE = re.compile(r"\(\s*\d+[a-z]?\s*\)\s*$")
_REF_HEADER_RE = re.compile(r"^(References|Bibliography|참고문헌)\s*$", re.IGNORECASE)
_FIG_CAPTION_RE = re.compile(r"^(Fig\.|Figure|Table|표|그림)\s*\d+", re.IGNORECASE)


def _math_char_ratio(text: str) -> float:
    return len(_MATH_CHARS.findall(text)) / len(text) if text else 0.0


def _line_text(line) -> str:
    return "".join(span.get("text", "") for span in line.get("spans", []))


def _line_fontsize(line) -> float:
    sizes = [span.get("size", 0.0) for span in line.get("spans", []) if span.get("text", "").strip()]
    return max(sizes) if sizes else 0.0


def _line_font(line) -> str:
    for span in line.get("spans", []):
        if span.get("text", "").strip():
            return span.get("font", "")
    return ""


def _looks_like_equation(text: str, span_font: str = "") -> bool:
    text = text.strip()
    if not text:
        return False
    if _EQ_NUMBER_RE.search(text) or _math_char_ratio(text) > 0.15:
        return True
    font_lower = span_font.lower()
    return any(k in font_lower for k in ("italic", "math", "symbol", "cmmi", "cmsy", "cmex")) and len(text) < 120 and _math_char_ratio(text) > 0.05


def _crop_block_to_image(page, bbox, out_path: Path, dpi: int = 150, padding: int = 4) -> None:
    x0, y0, x1, y1 = bbox
    clip = fitz.Rect(x0 - padding, y0 - padding, x1 + padding, y1 + padding) & page.rect
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), clip=clip)
    pix.save(str(out_path))


def _convert_with_fitz(pdf_path: Path, *, page_range=None, dpi=150, image_dir=None) -> str:
    if image_dir is None:
        image_dir = pdf_path.parent / "images"
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    pages = range(total_pages)
    if page_range:
        start, end = page_range
        pages = range(max(1, start) - 1, min(total_pages, end))

    all_blocks = []
    for pno in pages:
        all_blocks.extend(doc[pno].get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"])
    body_size = _detect_body_fontsize(all_blocks)

    math_fonts = set()
    for block in all_blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = span.get("font", "")
                fl = font.lower()
                if any(prefix in fl for prefix in ("advpsm", "advp4c", "advpssy", "cmmi", "cmsy", "cmex", "msam", "msbm", "symbol")):
                    math_fonts.add(font)

    md_parts = []
    in_references = False
    eq_img_counter = 0

    for pno in pages:
        page = doc[pno]
        page_width = page.rect.width
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        for block in blocks:
            if block.get("type") == 1:
                md_parts.append(f"\n[Image on page {pno + 1}]\n")
                continue
            if block.get("type") != 0:
                continue

            lines = block.get("lines", [])
            block_lines = []
            block_has_math_font = False
            for line in lines:
                text = _line_text(line)
                font = _line_font(line)
                block_lines.append((text, _line_fontsize(line), font))
                if any(span.get("font", "") in math_fonts for span in line.get("spans", [])):
                    block_has_math_font = True

            full_text = " ".join(t.strip() for t, _, _ in block_lines if t.strip())
            if not full_text:
                continue

            if _REF_HEADER_RE.match(full_text):
                in_references = True
                md_parts.append("\n## References\n\n")
                continue
            if in_references:
                md_parts.append(full_text + "\n\n")
                continue
            if _FIG_CAPTION_RE.match(full_text):
                md_parts.append(f"\n> **{full_text}**\n\n")
                continue

            first_size = block_lines[0][1]
            if first_size > body_size * 1.15 and len(full_text) < 200:
                ratio = first_size / body_size
                prefix = "#" if ratio > 1.5 else "##" if ratio > 1.25 else "###"
                md_parts.append(f"\n{prefix} {full_text}\n\n")
                continue

            bx0 = block.get("bbox", [0, 0, 0, 0])[0]
            is_centered = bx0 > page_width * 0.2
            eq_count = sum(1 for t, _, f in block_lines if t.strip() and _looks_like_equation(t.strip(), f))
            total = sum(1 for t, _, _ in block_lines if t.strip())
            is_equation_block = (
                (block_has_math_font and is_centered and len(full_text) < 300)
                or (is_centered and len(full_text) < 150 and (_math_char_ratio(full_text) > 0.05 or _EQ_NUMBER_RE.search(full_text)))
                or (total > 0 and eq_count / total > 0.5)
            )

            if is_equation_block and block.get("bbox"):
                eq_img_counter += 1
                img_name = f"{pdf_path.stem}_eq{eq_img_counter:03d}.png"
                _crop_block_to_image(page, block["bbox"], image_dir / img_name, dpi=dpi)
                md_parts.append(f"\n![Equation](images/{img_name})\n\n")
                continue

            md_parts.append(full_text + "\n\n")

    doc.close()
    return _postprocess_common("".join(md_parts))


def convert_pdf_to_markdown(pdf_path, *, page_range=None, dpi=150, image_dir=None, backend=None) -> str:
    pdf_path = Path(pdf_path)
    if backend is None:
        backend = "marker" if _HAS_MARKER else "pymupdf4llm" if _HAS_PYMUPDF4LLM else "fitz"
    if backend == "marker" and _HAS_MARKER:
        return _convert_with_marker(pdf_path, page_range=page_range)
    if backend == "pymupdf4llm" and _HAS_PYMUPDF4LLM:
        return _convert_with_pymupdf4llm(pdf_path, page_range=page_range, dpi=dpi, image_dir=image_dir)
    return _convert_with_fitz(pdf_path, page_range=page_range, dpi=dpi, image_dir=image_dir)


def _parse_page_range(value: str | None):
    if not value:
        return None
    parts = value.split("-")
    start = int(parts[0])
    end = int(parts[1]) if len(parts) > 1 else start
    return start, end


def main() -> int:
    parser = argparse.ArgumentParser(description="Academic PDF to Markdown converter with equation image fallback")
    parser.add_argument("inputs", nargs="+", help="PDF paths or glob patterns")
    parser.add_argument("-o", "--output", help="Output markdown path for a single input")
    parser.add_argument("--outdir", help="Output directory for one or more PDFs")
    parser.add_argument("--pages", help="Page range, e.g. 1-10")
    parser.add_argument("--dpi", type=int, default=150, help="Equation image DPI")
    parser.add_argument("--backend", choices=["marker", "pymupdf4llm", "fitz"], help="Conversion backend")
    parser.add_argument("--image-dir", help="Directory for extracted equation/images")
    args = parser.parse_args()

    page_range = _parse_page_range(args.pages)
    inputs = []
    for pattern in args.inputs:
        path = Path(pattern)
        if path.exists():
            inputs.append(path)
        else:
            parent = path.parent if path.parent != path else Path(".")
            inputs.extend(sorted(parent.glob(path.name)))
    if not inputs:
        print("Error: no input files found", file=sys.stderr)
        return 1

    backend_name = args.backend or ("marker" if _HAS_MARKER else "pymupdf4llm" if _HAS_PYMUPDF4LLM else "fitz")
    print(f"Backend: {backend_name}")

    for pdf_path in inputs:
        if args.output and len(inputs) == 1:
            out_path = Path(args.output)
        elif args.outdir:
            out_path = Path(args.outdir) / (pdf_path.stem + ".md")
        else:
            out_path = pdf_path.with_suffix(".md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image_dir = Path(args.image_dir) if args.image_dir else out_path.parent / "images"

        print(f"Converting: {pdf_path} -> {out_path}")
        print(f"  Images -> {image_dir}/")
        md = convert_pdf_to_markdown(pdf_path, page_range=page_range, dpi=args.dpi, image_dir=image_dir, backend=args.backend)
        out_path.write_text(md, encoding="utf-8")
        eq_blocks = md.count("![Equation]")
        latex_blocks = md.count("$$") // 2
        print(f"  Done: {md.count(chr(10))} lines, {latex_blocks} LaTeX equations, {eq_blocks} equation images")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

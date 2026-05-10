#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")


@dataclass(frozen=True)
class FontProfile:
    regular: str
    bold: str
    word_wrap: str


def slugify(value: str, fallback: str = "download") -> str:
    value = value.strip().lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-._")
    return (value or fallback)[:90]


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def cjk_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    return sum(1 for char in visible if CJK_RE.match(char)) / len(visible)


def is_winansi_compatible(text: str) -> bool:
    try:
        text.encode("cp1252")
        return True
    except UnicodeEncodeError:
        return False


def output_path(output_dir: Path, filename: str | None, url: str | None, title: str | None) -> Path:
    if filename:
        name = filename if filename.lower().endswith(".pdf") else f"{filename}.pdf"
    elif title:
        name = f"{slugify(title)}.pdf"
    elif url:
        parsed = urlparse(url)
        stem = Path(parsed.path).name
        if stem.lower().endswith(".pdf"):
            name = slugify(stem, "download.pdf")
        else:
            digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
            name = f"{slugify(parsed.netloc + parsed.path, 'download')}-{digest}.pdf"
    else:
        name = "download.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / name


def request_url(url: str) -> requests.Response:
    response = requests.get(
        url,
        timeout=45,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf,*/*"},
        allow_redirects=True,
    )
    response.raise_for_status()
    return response


def is_pdf_response(response: requests.Response) -> bool:
    content_type = response.headers.get("content-type", "").lower()
    return "application/pdf" in content_type or response.content[:5] == b"%PDF-"


def direct_download_pdf(url: str, out: Path) -> None:
    response = request_url(url)
    if not is_pdf_response(response):
        raise ValueError("URL did not resolve to a PDF")
    out.write_bytes(response.content)


def html_to_items(source_html: str) -> tuple[str, list[tuple[str, object]]]:
    soup = BeautifulSoup(source_html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    title = clean_text(
        (soup.find("h1") or soup.find("title") or soup.new_tag("span")).get_text(" ", strip=True)
    )
    root = soup.find("article") or soup.find("main") or soup.body or soup
    items: list[tuple[str, object]] = []
    accepted = {"h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre", "table", "figcaption", "img"}
    containers = {"blockquote", "pre", "table"}
    seen_title = False
    for elem in root.descendants:
        if not isinstance(elem, Tag) or elem.name not in accepted:
            continue
        if any(parent is not elem and isinstance(parent, Tag) and parent.name in containers for parent in elem.parents):
            continue
        if elem.name == "img":
            alt = clean_text(elem.get("alt", ""))
            if alt:
                items.append(("caption", f"Image: {alt}"))
            continue
        text = clean_text(elem.get_text(" ", strip=True))
        if not text:
            continue
        if elem.name == "h1":
            if seen_title or text == title:
                seen_title = True
                continue
            seen_title = True
        if elem.name == "table":
            rows: list[list[str]] = []
            for tr in elem.find_all("tr"):
                cells = [clean_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
                if cells:
                    rows.append(cells)
            if rows:
                items.append(("table", rows))
            continue
        items.append((elem.name, text))
    return title or "Offline webpage", items


def markdown_to_items(markdown_text: str) -> tuple[str, list[tuple[str, object]]]:
    items: list[tuple[str, object]] = []
    title = "Offline webpage"
    in_code = False
    code_lines: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            items.append(("p", clean_text(" ".join(paragraph))))
            paragraph = []

    for raw in markdown_text.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            if in_code:
                items.append(("pre", "\n".join(code_lines)))
                code_lines = []
                in_code = False
            else:
                flush_paragraph()
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            text = stripped[2:].strip()
            if title == "Offline webpage":
                title = text
            else:
                items.append(("h1", text))
        elif stripped.startswith("## "):
            flush_paragraph()
            items.append(("h2", stripped[3:].strip()))
        elif stripped.startswith("### "):
            flush_paragraph()
            items.append(("h3", stripped[4:].strip()))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            flush_paragraph()
            items.append(("li", stripped[2:].strip()))
        elif stripped.startswith("> "):
            flush_paragraph()
            items.append(("caption", stripped[2:].strip()))
        else:
            paragraph.append(stripped)
    flush_paragraph()
    if code_lines:
        items.append(("pre", "\n".join(code_lines)))
    return title, items


def first_existing_path(paths: Iterable[str]) -> str | None:
    for path in paths:
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            return expanded
    return None


def try_register_ttf(name: str, paths: Iterable[str]) -> bool:
    path = first_existing_path(paths)
    if not path:
        return False
    try:
        pdfmetrics.registerFont(TTFont(name, path))
        return True
    except Exception:
        return False


def register_fonts(text_sample: str) -> FontProfile:
    cjk_heavy = cjk_ratio(text_sample) >= 0.08
    has_cjk_text = CJK_RE.search(text_sample) is not None

    if cjk_heavy:
        if try_register_ttf(
            "PDFReaderCJK",
            [
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                "/Library/Fonts/Arial Unicode.ttf",
                "~/Library/Fonts/Arial Unicode.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
            ],
        ):
            return FontProfile("PDFReaderCJK", "PDFReaderCJK", "CJK")
        try:
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
            return FontProfile("STSong-Light", "STSong-Light", "CJK")
        except Exception:
            return FontProfile("Helvetica", "Helvetica-Bold", "LTR")

    if not has_cjk_text and is_winansi_compatible(text_sample):
        return FontProfile("Helvetica", "Helvetica-Bold", "LTR")

    if has_cjk_text and try_register_ttf(
        "PDFReaderMixed",
        [
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "~/Library/Fonts/Arial Unicode.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
        ],
    ):
        return FontProfile("PDFReaderMixed", "PDFReaderMixed", "LTR")

    latin_registered = try_register_ttf(
        "PDFReaderLatin",
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "~/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        ],
    )
    latin_bold_registered = try_register_ttf(
        "PDFReaderLatinBold",
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "~/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        ],
    )
    if latin_registered:
        return FontProfile(
            "PDFReaderLatin",
            "PDFReaderLatinBold" if latin_bold_registered else "PDFReaderLatin",
            "LTR",
        )
    return FontProfile("Helvetica", "Helvetica-Bold", "LTR")


def build_pdf(
    out: Path,
    title: str,
    items: Iterable[tuple[str, object]],
    source_url: str | None,
    generated_from: str,
) -> None:
    items = list(items)
    text_sample = "\n".join([title, *[str(value) for _, value in items[:80]]])
    font_profile = register_fonts(text_sample)
    base_font = font_profile.regular
    bold_font = font_profile.bold
    word_wrap = font_profile.word_wrap
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PDTitle", parent=styles["Title"], fontName=bold_font, fontSize=24, leading=31, alignment=TA_LEFT, spaceAfter=12, wordWrap=word_wrap))
    styles.add(ParagraphStyle(name="PDMeta", parent=styles["Normal"], fontName=base_font, fontSize=8.8, leading=13.5, alignment=TA_LEFT, textColor=colors.HexColor("#555555"), spaceAfter=5, wordWrap=word_wrap))
    styles.add(ParagraphStyle(name="PDBody", parent=styles["BodyText"], fontName=base_font, fontSize=10.8, leading=18.2, alignment=TA_LEFT, spaceAfter=9.5, wordWrap=word_wrap))
    styles.add(ParagraphStyle(name="PDH2", parent=styles["Heading2"], fontName=bold_font, fontSize=16.5, leading=23.5, alignment=TA_LEFT, spaceBefore=16, spaceAfter=8, wordWrap=word_wrap))
    styles.add(ParagraphStyle(name="PDH3", parent=styles["Heading3"], fontName=bold_font, fontSize=12.8, leading=18.5, alignment=TA_LEFT, spaceBefore=11, spaceAfter=6, wordWrap=word_wrap))
    styles.add(ParagraphStyle(name="PDBullet", parent=styles["BodyText"], fontName=base_font, fontSize=10.6, leading=17.6, alignment=TA_LEFT, leftIndent=17, firstLineIndent=-11, spaceAfter=6.5, wordWrap=word_wrap))
    styles.add(ParagraphStyle(name="PDCaption", parent=styles["BodyText"], fontName=base_font, fontSize=9, leading=14, alignment=TA_LEFT, leftIndent=9, rightIndent=9, textColor=colors.HexColor("#666666"), spaceAfter=8.5, wordWrap=word_wrap))
    pre_style = ParagraphStyle(name="PDPre", fontName="Courier", fontSize=8, leading=11, backColor=colors.HexColor("#f5f5f5"), borderColor=colors.HexColor("#dddddd"), borderWidth=0.5, borderPadding=6, leftIndent=8, rightIndent=8, spaceBefore=5, spaceAfter=10)

    flow = [
        Paragraph(html.escape(title), styles["PDTitle"]),
        Paragraph(f"Source: <a href=\"{html.escape(source_url or '')}\">{html.escape(source_url or generated_from)}</a>", styles["PDMeta"]),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} from {html.escape(generated_from)}", styles["PDMeta"]),
        Spacer(1, 6),
    ]

    for kind, value in items:
        if kind in {"h1", "h2"}:
            flow.append(Paragraph(html.escape(str(value)), styles["PDH2"]))
        elif kind in {"h3", "h4"}:
            flow.append(Paragraph(html.escape(str(value)), styles["PDH3"]))
        elif kind == "li":
            flow.append(Paragraph("• " + html.escape(str(value)), styles["PDBullet"]))
        elif kind in {"caption", "figcaption"}:
            flow.append(Paragraph("[Note] " + html.escape(str(value)), styles["PDCaption"]))
        elif kind == "blockquote":
            flow.append(Paragraph(html.escape(str(value)), styles["PDCaption"]))
        elif kind == "pre":
            text = str(value).translate(str.maketrans({"├": "+", "└": "+", "│": "|", "─": "-", "…": "..."}))
            wrapped = []
            for line in text.splitlines():
                wrapped.extend(__import__("textwrap").wrap(line, width=82, replace_whitespace=False) or [""])
            flow.append(Preformatted("\n".join(wrapped), pre_style))
        elif kind == "table":
            rows = value if isinstance(value, list) else []
            if rows:
                width = max(len(row) for row in rows)
                table_rows = []
                for row in rows:
                    padded = list(row) + [""] * (width - len(row))
                    table_rows.append([Paragraph(html.escape(str(cell)), styles["PDCaption"]) for cell in padded])
                table = Table(table_rows, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]))
                flow.extend([table, Spacer(1, 8)])
        else:
            flow.append(Paragraph(html.escape(str(value)), styles["PDBody"]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(base_font, 8)
        canvas.setFillColor(colors.HexColor("#777777"))
        footer_title = title[:70]
        canvas.drawString(1.7 * cm, 1.05 * cm, footer_title)
        canvas.drawRightString(A4[0] - 1.7 * cm, 1.05 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=1.9 * cm,
        rightMargin=1.9 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.7 * cm,
        title=title,
        subject=f"Offline PDF generated from {source_url or generated_from}",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin + 0.45 * cm, doc.width, doc.height - 0.45 * cm, id="normal")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=footer)])
    doc.build(flow)


def write_markdown(path: Path, title: str, items: Iterable[tuple[str, object]], source_url: str | None) -> None:
    lines = [f"# {title}", ""]
    if source_url:
        lines.extend([f"Source: {source_url}", ""])
    for kind, value in items:
        if kind in {"h1", "h2"}:
            lines.extend([f"## {value}", ""])
        elif kind in {"h3", "h4"}:
            lines.extend([f"### {value}", ""])
        elif kind == "li":
            lines.append(f"- {value}")
        elif kind == "pre":
            lines.extend(["```text", str(value), "```", ""])
        elif kind == "table":
            for row in value if isinstance(value, list) else []:
                lines.append(" | ".join(str(cell) for cell in row))
            lines.append("")
        elif kind in {"caption", "figcaption"}:
            lines.extend([f"> {value}", ""])
        else:
            lines.extend([str(value), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download direct PDFs or generate offline PDFs from webpages/Markdown.")
    parser.add_argument("url", nargs="?", help="URL to download or convert")
    parser.add_argument("--markdown", help="Markdown file to convert when direct page fetching is blocked")
    parser.add_argument("--source-url", help="Original source URL for Markdown fallback")
    parser.add_argument("--output-dir", default="output/pdf")
    parser.add_argument("--filename")
    parser.add_argument("--title")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if args.markdown:
        md_text = Path(args.markdown).read_text(encoding="utf-8")
        title, items = markdown_to_items(md_text)
        if args.title:
            title = args.title
        out = output_path(out_dir, args.filename, args.source_url or args.url, title)
        build_pdf(out, title, items, args.source_url or args.url, f"Markdown file {args.markdown}")
        print(f"generated_pdf={out}")
        return 0

    if not args.url:
        parser.error("provide a URL or --markdown")

    out = output_path(out_dir, args.filename, args.url, args.title)
    try:
        response = request_url(args.url)
    except Exception as exc:
        print(f"error=fetch_failed message={exc}", file=sys.stderr)
        return 2

    if is_pdf_response(response):
        out.write_bytes(response.content)
        print(f"downloaded_pdf={out}")
        return 0

    title, items = html_to_items(response.text)
    if args.title:
        title = args.title
    if not items:
        print("error=no_extractable_content", file=sys.stderr)
        return 3
    out = output_path(out_dir, args.filename, args.url, title)
    build_pdf(out, title, items, response.url, "webpage HTML")
    write_markdown(out.with_suffix(".md"), title, items, response.url)
    print(f"generated_pdf={out}")
    print(f"generated_markdown={out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

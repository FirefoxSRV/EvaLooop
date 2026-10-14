"""Render report.md to a print-ready PDF via headless Chrome.

The assignment caps the report at three single-column pages, so the stylesheet
is tuned for density rather than screen reading: 9.6pt body, tight leading,
narrow margins, and tables that never split across a page. Chrome is used
because it is already on the machine and renders the same box model the HTML
preview shows; no LaTeX or pandoc toolchain is needed.

Usage:
    python scripts/build_report.py            # report.md -> report.pdf
    python scripts/build_report.py --html     # stop at the HTML, for inspection
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: letter; margin: 0.62in 0.70in; }
html { -webkit-print-color-adjust: exact; }
body {
  font-family: "Charter", "Georgia", "Times New Roman", serif;
  font-size: 9.6pt; line-height: 1.34; color: #111; margin: 0;
  text-align: justify; hyphens: auto; -webkit-hyphens: auto;
}
h1 { font-size: 14.5pt; line-height: 1.2; margin: 0 0 2pt; text-align: left; }
h2 { font-size: 10.6pt; margin: 9pt 0 3pt; text-align: left;
     border-bottom: 0.5pt solid #bbb; padding-bottom: 1pt; }
p { margin: 0 0 4.5pt; }
blockquote {
  margin: 5pt 0; padding: 3pt 7pt; border-left: 2pt solid #666;
  background: #f5f5f5; font-style: italic; text-align: left;
}
blockquote p { margin: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 5pt 0 6pt;
  font-size: 8.3pt; page-break-inside: avoid;
}
th, td { border: 0.4pt solid #999; padding: 1.8pt 3.5pt; text-align: left; }
th { background: #eee; font-weight: bold; }
code, tt {
  font-family: "SF Mono", "Menlo", monospace; font-size: 8.1pt;
  background: #f0f0f0; padding: 0 1.5pt; border-radius: 1.5pt;
}
em { font-style: italic; }
strong { font-weight: bold; }
a { color: #0b3d91; text-decoration: none; }
h1 + p { font-size: 8.7pt; color: #444; text-align: left; margin-bottom: 7pt; }
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md", default="report.md")
    parser.add_argument("--pdf", default="report.pdf")
    parser.add_argument("--html", action="store_true", help="Write the HTML and stop")
    parser.add_argument("--max-pages", type=int, default=3,
                        help="Fail if the PDF exceeds this many pages")
    args = parser.parse_args()

    try:
        import markdown
    except ImportError:
        print("Needs the markdown package: pip install markdown", file=sys.stderr)
        return 2

    source = Path(args.md)
    if not source.exists():
        print(f"{source} not found", file=sys.stderr)
        return 2

    body = markdown.markdown(source.read_text(),
                             extensions=["tables", "smarty", "sane_lists"])
    html_path = source.with_suffix(".html")
    html_path.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{source.stem}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )
    print(f"Wrote {html_path}")
    if args.html:
        return 0

    if not Path(CHROME).exists():
        print(f"Chrome not found at {CHROME}; open {html_path} and print to PDF",
              file=sys.stderr)
        return 1

    pdf_path = Path(args.pdf).resolve()
    subprocess.run([
        CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri(),
    ], check=True, capture_output=True, timeout=120)

    size = pdf_path.stat().st_size
    print(f"Wrote {pdf_path} ({size // 1024} KB)")

    # Page count, so the three-page cap is checked rather than assumed. Read it
    # out of the PDF's page tree: Spotlight's kMDItemNumberOfPages is served
    # from a stale index right after a write and reported 3 for a 4-page file.
    pages = page_count(pdf_path)
    over = pages is not None and pages > args.max_pages
    print(f"Pages: {pages}" + (f"  <-- OVER THE {args.max_pages}-PAGE LIMIT" if over else ""))
    return 1 if over else 0


def page_count(pdf_path: Path):
    """Pages from the PDF page tree, or None if it cannot be determined."""
    raw = pdf_path.read_bytes()
    counts = [int(m.group(1)) for m in re.finditer(rb"/Count\s+(\d+)", raw)]
    if counts:
        return max(counts)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
    return pages or None


if __name__ == "__main__":
    raise SystemExit(main())

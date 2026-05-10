"""Smoke tests for v2 format readers.

Covers: TXT (with markdown stripping), Markdown, HTML (local file),
PDF (verify install-hint error path when pypdf is missing), EPUB
(roundtrip through a minimal in-memory zip).

URL fetch is not exercised here — the HTML reader path is the same.
"""
from __future__ import annotations

import io
import textwrap
import zipfile
from pathlib import Path

import pytest

from sara_brain.cortex.transformer.v2.format_readers import (
    detect_format, read,
)


def test_detect_format_extensions():
    assert detect_format("paper.txt") == "txt"
    assert detect_format("paper.md") == "md"
    assert detect_format("paper.markdown") == "md"
    assert detect_format("paper.pdf") == "pdf"
    assert detect_format("paper.epub") == "epub"
    assert detect_format("page.html") == "html"
    assert detect_format("page.htm") == "html"
    assert detect_format("https://example.com/page") == "url"
    assert detect_format("http://example.com/page.txt") == "url"


def test_txt_reader_strips_markdown(tmp_path: Path):
    body = textwrap.dedent(
        """
        # **Some Heading**

        This is *italic* and **bold** text. It has a [link](http://example.com).

        Another paragraph with `inline code` and ~~strikethrough~~.

        - bullet one
        - bullet two
        """
    ).strip()
    src = tmp_path / "a.txt"
    src.write_text(body, encoding="utf-8")

    segs = list(read(str(src)))
    assert len(segs) >= 2
    joined = " ".join(s.text for s in segs)
    assert "**" not in joined
    assert "[link]" not in joined and "](" not in joined
    assert "italic" in joined
    assert "bold" in joined
    assert "link" in joined  # link text retained, target dropped


def test_md_reader_tracks_headings(tmp_path: Path):
    body = textwrap.dedent(
        """
        # Top Heading

        First paragraph under top.

        ## Subsection

        Second paragraph under subsection.
        """
    ).strip()
    src = tmp_path / "doc.md"
    src.write_text(body, encoding="utf-8")

    segs = list(read(str(src)))
    assert len(segs) == 2
    # Both segments should carry the appropriate heading breadcrumb.
    assert "Top Heading" in segs[0].provenance
    assert "Top Heading > Subsection" in segs[1].provenance
    assert segs[0].text.startswith("First paragraph")
    assert segs[1].text.startswith("Second paragraph")


def test_html_reader_local_file(tmp_path: Path):
    html_body = """
    <html><body>
      <h1>Example Title</h1>
      <p>The first paragraph mentions <em>creed 2</em>.</p>
      <h2>Sub</h2>
      <p>Second paragraph here.</p>
      <script>console.log('skip me')</script>
    </body></html>
    """
    src = tmp_path / "page.html"
    src.write_text(html_body, encoding="utf-8")

    segs = list(read(str(src)))
    assert len(segs) >= 2
    texts = [s.text for s in segs]
    assert any("creed 2" in t for t in texts)
    assert any("Second paragraph" in t for t in texts)
    # Script content must not leak through.
    assert not any("console.log" in t for t in texts)
    # Heading breadcrumb tracking.
    assert any("Example Title" in s.provenance for s in segs)
    assert any("Example Title > Sub" in s.provenance for s in segs)


def test_pdf_reader_install_hint_when_pypdf_missing(tmp_path: Path, monkeypatch):
    """Without pypdf installed, the PDF reader must raise a clear
    install hint, not a cryptic ImportError."""
    src = tmp_path / "fake.pdf"
    src.write_bytes(b"%PDF-1.4\n% fake")  # not a real PDF
    # Force the import path to fail even if pypdf happens to be installed.
    import sys
    monkeypatch.setitem(sys.modules, "pypdf", None)

    from sara_brain.cortex.transformer.v2.format_readers import pdf
    with pytest.raises(RuntimeError) as exc:
        list(pdf.read(str(src)))
    assert "pypdf" in str(exc.value).lower()


def test_epub_reader_minimal(tmp_path: Path):
    """Build a minimal EPUB (zip with container.xml + content.opf + one
    XHTML chapter) and verify the reader yields its paragraph."""
    epub_path = tmp_path / "tiny.epub"
    container_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
        '  <rootfiles><rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles>\n'
        '</container>\n'
    )
    content_opf = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
        'unique-identifier="bookid">\n'
        '  <metadata/>\n'
        '  <manifest>\n'
        '    <item id="ch1" href="ch1.xhtml" '
        'media-type="application/xhtml+xml"/>\n'
        '  </manifest>\n'
        '  <spine><itemref idref="ch1"/></spine>\n'
        '</package>\n'
    )
    chapter_xhtml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
        '<head><title>Chapter 1</title></head>\n'
        '<body><h1>Chapter 1</h1>'
        '<p>Bruce Lee created Jeet Kune Do.</p></body>\n'
        '</html>\n'
    )
    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", content_opf)
        zf.writestr("OEBPS/ch1.xhtml", chapter_xhtml)

    segs = list(read(str(epub_path)))
    assert len(segs) >= 1
    # The chapter paragraph must be present.
    assert any("Bruce Lee" in s.text for s in segs)
    # Provenance should encode the chapter identifier.
    assert any("chapter=ch1" in s.provenance for s in segs)

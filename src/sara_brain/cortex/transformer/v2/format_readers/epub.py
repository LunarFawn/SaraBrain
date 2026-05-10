"""EPUB reader. Stdlib only — EPUB is a zip of XHTML files plus a manifest.

Reads the OPF spine to get reading order, parses each XHTML file with
the html reader, prefixes provenance with the spine item's filename so
chapter location is recoverable.
"""
from __future__ import annotations

import re
import urllib.parse
import zipfile
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

from . import TextSegment
from .html import _Extractor


_NS = {
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
}


def read(source: str) -> Iterator[TextSegment]:
    path = Path(source)
    with zipfile.ZipFile(path) as zf:
        opf_path = _find_opf(zf)
        opf_dir = str(Path(opf_path).parent)
        spine_files = _read_spine(zf, opf_path)
        for rel_href in spine_files:
            href = urllib.parse.unquote(rel_href)
            zip_path = _join_zip(opf_dir, href)
            try:
                raw = zf.read(zip_path).decode("utf-8", errors="replace")
            except KeyError:
                continue
            chapter_label = Path(href).stem
            parser = _Extractor()
            parser.feed(raw)
            parser._flush_block()
            for text, crumb in parser.results:
                prov = f"{path.name}#chapter={chapter_label}"
                if crumb:
                    prov += f"|{crumb}"
                yield TextSegment(text=text, provenance=prov)


def _find_opf(zf: zipfile.ZipFile) -> str:
    container_xml = zf.read("META-INF/container.xml").decode("utf-8")
    root = ET.fromstring(container_xml)
    rootfile = root.find(".//container:rootfile", _NS)
    if rootfile is None:
        raise RuntimeError("EPUB container.xml has no rootfile element")
    return rootfile.attrib["full-path"]


def _read_spine(zf: zipfile.ZipFile, opf_path: str) -> list[str]:
    opf_xml = zf.read(opf_path).decode("utf-8")
    root = ET.fromstring(opf_xml)
    manifest_items: dict[str, str] = {}
    for item in root.findall(".//opf:manifest/opf:item", _NS):
        manifest_items[item.attrib["id"]] = item.attrib["href"]
    spine_files: list[str] = []
    for ref in root.findall(".//opf:spine/opf:itemref", _NS):
        idref = ref.attrib.get("idref")
        if idref and idref in manifest_items:
            spine_files.append(manifest_items[idref])
    return spine_files


def _join_zip(base: str, rel: str) -> str:
    """Resolve an OPF-relative href into a zipfile path, normalizing
    forward slashes."""
    rel = rel.split("#", 1)[0]
    parts: list[str] = []
    if base:
        parts.extend(base.split("/"))
    for piece in rel.split("/"):
        if piece in {".", ""}:
            continue
        if piece == "..":
            if parts:
                parts.pop()
            continue
        parts.append(piece)
    return "/".join(parts)

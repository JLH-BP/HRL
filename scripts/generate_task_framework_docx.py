# Generates the Word version of the META-HRL project task framework.
"""Convert the version-controlled Markdown framework source into a dependency-free DOCX file."""

from __future__ import annotations

from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "docs" / "meta_hrl_task_framework.md"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "meta_hrl_task_framework.docx"

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

CORE_PROPERTIES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>META-HRL 项目任务框架</dc:title>
<dc:subject>混合近远场 RSMA 元分层强化学习项目规划</dc:subject>
<dc:description>由 META_HRL 项目内置生成脚本创建。</dc:description>
<dc:creator>META_HRL</dc:creator>
</cp:coreProperties>"""

APP_PROPERTIES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>META_HRL</Application></Properties>"""


def _paragraph(text: str, bold: bool = False, size: int | None = None) -> str:
    """Create a minimal WordprocessingML paragraph containing one text run."""
    properties = ""
    if bold:
        properties += "<w:b/>"
    if size is not None:
        properties += f'<w:sz w:val="{size}"/>'
    run_properties = f"<w:rPr>{properties}</w:rPr>" if properties else ""
    return f'<w:p><w:r>{run_properties}<w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'


def _document_xml(markdown: str) -> str:
    """Render constrained Markdown lines into minimal document XML paragraphs."""
    paragraphs = [
        _paragraph("META-HRL 项目任务框架", bold=True, size=36),
        _paragraph("文档用途：指导 128 阵元 ULA、6 用户混合近远场 RSMA 的 Meta-HRL 项目实施。"),
        _paragraph("当前范围：项目目录与接口骨架；不包含完整信道仿真、RSMA 优化器或训练算法。"),
    ]
    in_code_block = False
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("<!--") or not line:
            continue
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            paragraphs.append(_paragraph(raw_line))
        elif line.startswith("# "):
            paragraphs.append(_paragraph(line[2:], bold=True, size=32))
        elif line.startswith("## "):
            paragraphs.append(_paragraph(line[3:], bold=True, size=26))
        elif line.startswith("### "):
            paragraphs.append(_paragraph(line[4:], bold=True, size=22))
        elif line.startswith("- "):
            paragraphs.append(_paragraph("• " + line[2:]))
        else:
            paragraphs.append(_paragraph(line.replace("  ", "")))
    body = "".join(paragraphs)
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{body}<w:sectPr/></w:body></w:document>'''


def generate_document(source_path: Path = SOURCE_PATH, output_path: Path = OUTPUT_PATH) -> Path:
    """Generate the framework DOCX from its Markdown source and return its path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", ZIP_DEFLATED) as document:
        document.writestr("[Content_Types].xml", CONTENT_TYPES)
        document.writestr("_rels/.rels", ROOT_RELS)
        document.writestr("docProps/core.xml", CORE_PROPERTIES)
        document.writestr("docProps/app.xml", APP_PROPERTIES)
        document.writestr("word/document.xml", _document_xml(source_path.read_text(encoding="utf-8")))
    return output_path


if __name__ == "__main__":
    generated_path = generate_document()
    print(f"Generated {generated_path}")

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.colors import HexColor
from io import BytesIO
import re


# -------------------------------------------------
# Inline Markdown → ReportLab HTML
# -------------------------------------------------
def markdown_inline_to_html(text: str) -> str:
    if not text:
        return ""

    text = (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )

    # Bold
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

    # Italic
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)

    return text


# -------------------------------------------------
# Markdown → ReportLab Flowables
# -------------------------------------------------
def render_markdown(text: str, styles):
    story = []
    bullet_buffer = []

    def flush_bullets():
        nonlocal bullet_buffer
        if bullet_buffer:
            story.append(
                ListFlowable(
                    bullet_buffer,
                    bulletType="bullet",
                    leftIndent=24
                )
            )
            bullet_buffer = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            flush_bullets()
            story.append(Spacer(1, 8))
            continue

        # Normalize bullets
        if line.startswith("• "):
            line = "- " + line[2:]

        # H1
        if line.startswith("# "):
            flush_bullets()
            story.append(
                Paragraph(
                    f"<b>{markdown_inline_to_html(line[2:])}</b>",
                    styles["Title"]
                )
            )
            story.append(Spacer(1, 14))
            continue

        # H2
        if line.startswith("## "):
            flush_bullets()
            story.append(
                Paragraph(
                    markdown_inline_to_html(line[3:]),
                    styles["Heading2"]
                )
            )
            story.append(Spacer(1, 10))
            continue

        # H3
        if line.startswith("### "):
            flush_bullets()
            story.append(
                Paragraph(
                    markdown_inline_to_html(line[4:]),
                    styles["Heading3"]
                )
            )
            story.append(Spacer(1, 6))
            continue

        # Bullet
        if line.startswith("- "):
            bullet_buffer.append(
                ListItem(
                    Paragraph(
                        markdown_inline_to_html(line[2:]),
                        styles["Normal"]
                    )
                )
            )
            continue

        # Normal paragraph
        flush_bullets()
        story.append(
            Paragraph(
                markdown_inline_to_html(line),
                styles["Normal"]
            )
        )

    flush_bullets()
    return story


# -------------------------------------------------
# Main PDF Generator
# -------------------------------------------------
def generate_policy_pdf_bytes(rag_data: dict) -> bytes:
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Improve base styles
    styles["Normal"].fontSize = 10
    styles["Normal"].leading = 14

    styles["Heading2"].fontSize = 14
    styles["Heading2"].spaceAfter = 12

    styles["Heading3"].fontSize = 12
    styles["Heading3"].spaceAfter = 8

    story = []

    # -------------------------------------------------
    # Render FULL markdown report
    # -------------------------------------------------
    story.extend(
        render_markdown(
            rag_data.get("explanation", ""),
            styles
        )
    )

    doc.build(story)
    buffer.seek(0)

    return buffer.read()

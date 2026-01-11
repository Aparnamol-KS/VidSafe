from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO


def generate_policy_pdf_bytes(rag_data):
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []

    story.append(Paragraph("VidSafe – Content Moderation Policy Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>Video Name:</b> {rag_data['video_name']}", styles["Normal"]))
    story.append(Paragraph(f"<b>Analysis Time:</b> {rag_data['analysis_time']}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Detected Categories</b>", styles["Heading2"]))
    for cat in rag_data["detected_categories"]:
        story.append(Paragraph(f"- {cat}", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Severity Level:</b> {rag_data['severity_level']}", styles["Normal"]))
    story.append(Paragraph(f"<b>Policy Decision:</b> {rag_data['policy_decision']}", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Flagged Segments</b>", styles["Heading2"]))
    for seg in rag_data["flagged_segments"]:
        story.append(Paragraph(f"{seg['start']} – {seg['end']}", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Explanation</b>", styles["Heading2"]))
    story.append(Paragraph(rag_data["explanation"], styles["Normal"]))

    doc.build(story)
    buffer.seek(0)

    return buffer.read()

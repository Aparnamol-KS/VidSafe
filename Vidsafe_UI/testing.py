from policy_reducer import reduce_policy_violations_to_text
from pdf_utils import generate_policy_pdf_bytes


# -----------------------------------------
# Step 1: Generate markdown report via LLM
# -----------------------------------------
report_markdown = reduce_policy_violations_to_text({
    "video_id": "test_video",
    "fusion_severity": "High",
    "video_max_confidence": 0.68,
    "audio_max_confidence": 0.75,
    "policy_violations": [
        {
            "policy_name": "Violence and Graphic Content",
            "category": "Violence",
            "severity": "Medium",
            "modality": "video",
            "timestamp": "0:00:02 - 0:00:04",
            "reason": "Physical aggression detected"
        },
        {
            "policy_name": "Harassment and Threats",
            "category": "Abuse",
            "severity": "High",
            "modality": "audio",
            "timestamp": "0:00:06 - 0:00:08",
            "reason": "Threatening speech detected"
        }
    ]
})

print("\n=== MARKDOWN REPORT ===\n")
print(report_markdown)


# -----------------------------------------
# Step 2: Wrap markdown for PDF generator
# -----------------------------------------
rag_data = {
    "explanation": report_markdown
}


# -----------------------------------------
# Step 3: Generate PDF bytes
# -----------------------------------------
pdf_bytes = generate_policy_pdf_bytes(rag_data)


# -----------------------------------------
# Step 4: Save PDF to disk
# -----------------------------------------
with open("test_report.pdf", "wb") as f:
    f.write(pdf_bytes)

print("\n✅ PDF generated successfully → test_report.pdf")

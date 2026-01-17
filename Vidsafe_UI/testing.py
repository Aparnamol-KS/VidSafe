import json
from pathlib import Path

from policy_reducer import reduce_policy_violations_to_text
from pdf_utils import generate_policy_pdf_bytes


# -------------------------------------------------
# CONFIG: paths to REAL pipeline outputs
# -------------------------------------------------
POLICY_OUTPUT_FILE = Path("outputs/policy_violations_output.json")
PDF_OUTPUT_FILE = Path("test_report.pdf")


# -------------------------------------------------
# STEP 1: Load real policy violations output
# -------------------------------------------------
if not POLICY_OUTPUT_FILE.exists():
    raise FileNotFoundError(
        f"Policy output not found: {POLICY_OUTPUT_FILE}"
    )

with open(POLICY_OUTPUT_FILE, "r", encoding="utf-8") as f:
    raw_policy_output = json.load(f)

print("✅ Loaded policy violations output")
print(f"• Video ID: {raw_policy_output.get('video_id')}")
print(f"• Total violations: {raw_policy_output.get('total_violations')}")
print(f"• Fusion severity: {raw_policy_output.get('fusion_severity')}")


# -------------------------------------------------
# STEP 2: Generate markdown moderation report (LLM)
# -------------------------------------------------
print("\n🧠 Generating moderation report via policy reducer...")

report_markdown = reduce_policy_violations_to_text(raw_policy_output)

print("\n=== MARKDOWN REPORT (PREVIEW) ===\n")
print(report_markdown[:1200])  # preview only


# -------------------------------------------------
# STEP 3: Wrap for PDF generator
# -------------------------------------------------
rag_data = {
    "explanation": report_markdown
}


# -------------------------------------------------
# STEP 4: Generate PDF bytes
# -------------------------------------------------
print("\n📄 Generating PDF...")
pdf_bytes = generate_policy_pdf_bytes(rag_data)


# -------------------------------------------------
# STEP 5: Save PDF
# -------------------------------------------------
with open(PDF_OUTPUT_FILE, "wb") as f:
    f.write(pdf_bytes)

print(f"\n✅ PDF generated successfully → {PDF_OUTPUT_FILE.resolve()}")

import os
import json
from collections import defaultdict
from groq import Groq
from dotenv import load_dotenv


# -------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# -------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment variables")

MODEL_NAME = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY)


# -------------------------------------------------
# PRE-COMPRESSION (CRITICAL FOR TOKEN SAFETY)
# -------------------------------------------------
def compress_policy_output(
    raw_policy_output: dict,
    max_timestamps_per_policy: int = 5
) -> dict:
    """
    Compress frame-level moderation output into
    policy-level summaries BEFORE sending to LLM.
    """

    grouped = defaultdict(lambda: {
        "policy_name": "",
        "category": "",
        "severity": "",
        "timestamps": []
    })

    for v in raw_policy_output.get("policy_violations", []):
        key = f"{v['policy_name']}|{v['category']}"

        grouped[key]["policy_name"] = v["policy_name"]
        grouped[key]["category"] = v["category"]
        grouped[key]["severity"] = v["severity"]

        if len(grouped[key]["timestamps"]) < max_timestamps_per_policy:
            grouped[key]["timestamps"].append({
                "timestamp": v["timestamp"],
                "reason": v["reason"]
            })

    return {
        "video_id": raw_policy_output.get("video_id"),
        "total_policies": len(grouped),
        "policies": list(grouped.values())
    }


# -------------------------------------------------
# LLM-BASED POLICY REDUCTION & REORDERING
# -------------------------------------------------
def reduce_policy_violations_to_text(raw_policy_output: dict) -> str:
    """
    Uses Groq LLM to generate a professional moderation report
    from compressed policy-level data (token-safe).
    """

    compressed_output = compress_policy_output(raw_policy_output)
    prompt = f"""
You are a senior content moderation analyst for a video safety platform.

You are given a COMPRESSED JSON summary of video policy violations.
Each policy already contains representative timestamps.

Your task is to generate a PROFESSIONAL MODERATION REPORT
using **MARKDOWN STRUCTURE** suitable for rendering into a PDF.

----------------------------------------------------------------
OUTPUT FORMAT RULES (STRICT)
----------------------------------------------------------------
- Use MARKDOWN headings (#, ##, ###)
- Use bullet points (-)
- Do NOT use separators like ----
- Do NOT output JSON
- Keep language formal and audit-ready

----------------------------------------------------------------
REQUIRED MARKDOWN STRUCTURE
----------------------------------------------------------------

# Content Moderation Summary Report

## Video Overview
- **Total Policies Violated:** <number>
- **Overall Severity Level:** <Low / Medium / High>
- **Recommended Action:** <Allow / Age-Restrict / Limited Distribution / Remove>

## Policy Violations Summary

### <Policy Name> (<Category>)
- **Severity:** <Low / Medium / High>
- **Description:** <one-line description>
- **Representative Affected Timestamps:**
  - <start> – <end>: <short explanation>
  - <start> – <end>: <short explanation>

## Age Restriction Assessment
- **Age Restriction Required:** <Yes / No>
- **Justification:** <professional justification>

## Overall Risk Assessment
<One concise paragraph covering:
nature of content, recurrence, viewer impact, platform safety>

----------------------------------------------------------------
COMPRESSED INPUT (JSON)
----------------------------------------------------------------
{json.dumps(compressed_output, indent=2)}

FINAL REPORT:
"""


    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate professional, structured content moderation "
                    "reports suitable for compliance and audit review."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=600
    )

    return response.choices[0].message.content.strip()


# -------------------------------------------------
# LOCAL TEST (OPTIONAL)
# -------------------------------------------------
if __name__ == "__main__":
    with open(
        "policy_data/raw_policy_output.json",
        "r",
        encoding="utf-8"
    ) as f:
        raw_output = json.load(f)

    policy_report = reduce_policy_violations_to_text(raw_output)

    print("\n=== CONTENT MODERATION REPORT ===\n")
    print(policy_report)

    with open(
        "policy_data/policy_report.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(policy_report)

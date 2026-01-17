import os
import json
from collections import defaultdict
from groq import Groq
from dotenv import load_dotenv

# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found")

MODEL_NAME = "llama-3.3-70b-versatile"
client = Groq(api_key=GROQ_API_KEY)

# -------------------------------------------------
# POLICY COMPRESSION (MODALITY-AWARE)
# -------------------------------------------------
def compress_policy_output(
    raw_policy_output: dict,
    max_timestamps_per_policy: int = 5
) -> dict:
    """
    Compress violations while preserving
    VIDEO vs AUDIO meaning clearly.
    """

    grouped = defaultdict(lambda: {
        "policy_name": "",
        "category": "",
        "modality": "",
        "severity": "",
        "timestamps": []
    })

    for v in raw_policy_output.get("policy_violations", []):
        policy_name = v.get("policy_name", "Unknown Policy")
        category = v.get("category", "Uncategorized")
        modality = v.get("modality", "unknown").upper()
        severity = v.get("severity", "Medium")
        timestamp = v.get("timestamp", "N/A")
        reason = v.get("reason", "Policy violation detected")

        # ---- KEY CHANGE: MODALITY INCLUDED ----
        key = f"{policy_name}|{category}|{modality}"

        grouped[key]["policy_name"] = policy_name
        grouped[key]["category"] = category
        grouped[key]["modality"] = modality
        grouped[key]["severity"] = severity

        if len(grouped[key]["timestamps"]) < max_timestamps_per_policy:
            grouped[key]["timestamps"].append({
                "timestamp": timestamp,
                "reason": f"[{modality}] {reason}"
            })

    return {
        "video_id": raw_policy_output.get("video_id", "unknown"),
        "fusion_severity": raw_policy_output.get("fusion_severity", "Medium"),
        "video_max_confidence": raw_policy_output.get("video_max_confidence"),
        "audio_max_confidence": raw_policy_output.get("audio_max_confidence"),
        "total_policy_groups": len(grouped),
        "policies": list(grouped.values())
    }

# -------------------------------------------------
# LLM REPORT GENERATION
# -------------------------------------------------
def reduce_policy_violations_to_text(raw_policy_output: dict) -> str:
    """
    Generate a professional moderation report
    that clearly distinguishes visual violence
    from audio-based toxicity.
    """

    compressed_output = compress_policy_output(raw_policy_output)

    prompt = f"""
You are a senior content moderation analyst.

You are given a COMPRESSED JSON summary of violations detected
from BOTH video frames and audio speech.

IMPORTANT INTERPRETATION RULES:
- VIDEO violations represent physical or visual violence
- AUDIO violations represent spoken threats, abuse, or toxic language
- Visual violence must be clearly highlighted as physical harm
- Audio toxicity must be treated as secondary if visual violence exists

---------------------------------------------------------------
REQUIRED MARKDOWN STRUCTURE
---------------------------------------------------------------

# Safety Assessment Report

## Video Overview
- **Total Policy Groups Violated:** <number>
- **Overall Fusion Severity Level:** <Low / Medium / High / Critical>
- **Recommended Action:**
  - Critical → Remove Content
  - High → Age Restrict
  - Medium → Limited Distribution
  - Low → Allow

## Policy Violations Summary

### <Policy Name> (<Category>) – <VIDEO / AUDIO>
- **Severity:** <Low / Medium / High>
- **Description:** Clear explanation of why this policy is violated
- **Representative Affected Timestamps:**
  - <timestamp>: <explanation>

## Age Restriction Assessment
- **Age Restriction Required:** <Yes / No>
- **Justification:** Professional explanation

## Overall Risk Assessment
Summarize:
- presence of sustained physical violence
- recurrence of harmful speech
- potential harm to viewers
- platform safety implications

---------------------------------------------------------------
COMPRESSED INPUT
---------------------------------------------------------------
{json.dumps(compressed_output, indent=2)}

FINAL REPORT:
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate professional, compliance-ready "
                    "content moderation reports."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=700
    )

    return response.choices[0].message.content.strip()

# -------------------------------------------------
# LOCAL TEST
# -------------------------------------------------
if __name__ == "__main__":
    with open("policy_violations_output.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print(reduce_policy_violations_to_text(data))

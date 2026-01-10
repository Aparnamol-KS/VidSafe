import os
import json
from groq import Groq
from dotenv import load_dotenv

# -------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# -------------------------------------------------
load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------------------------
# LLM-BASED POLICY REDUCTION
# -------------------------------------------------
def reduce_policy_violations_to_text(raw_policy_output: dict) -> str:
    """
    Uses Groq LLM to reduce large, repetitive policy violation JSON
    into a structured, human-readable moderation report.
    """

    prompt = f"""
You are an expert content moderation analyst for a video safety platform.

You are given a raw JSON output containing many repeated policy violations
detected across video frames.

Your task is to REDUCE and ORGANIZE this information into a clear,
human-readable moderation report.

### REQUIRED OUTPUT FORMAT (PLAIN TEXT ONLY):

Policy Violations Summary:

1. <Policy Name> (<Category>)
   - Severity: <Low / Medium / High>
   - Affected Timestamps:
     • <start time> – <end time>: <short explanation>
     • <start time> – <end time>: <short explanation>

2. <Policy Name> (<Category>)
   - Severity: <Low / Medium / High>
   - Affected Timestamps:
     • <start time> – <end time>: <short explanation>

Age Restriction:
- <Yes / No>
- Reason: <short justification>

Detailed Summary:
<One short paragraph summarizing the overall safety concerns, intent of the content,
and final moderation implication in professional language.>

### IMPORTANT RULES:
- Do NOT output JSON
- Group similar violations under the same policy
- Do NOT list all timestamps; only representative ones
- Keep explanations short and clear
- Use formal, platform-style language
- Do not mention raw counts unless necessary
- Ensure the output looks suitable for a policy report or PDF

RAW INPUT:
{json.dumps(raw_policy_output, indent=2)}

OUTPUT:
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You generate professional, structured moderation reports."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=600
    )

    return response.choices[0].message.content.strip()


# -------------------------------------------------
# LOCAL TEST (OPTIONAL)
# -------------------------------------------------
if __name__ == "__main__":
    with open("policy_data/raw_policy_output.json", "r") as f:
        raw_output = json.load(f)

    policy_report = reduce_policy_violations_to_text(raw_output)

    print("\n=== POLICY REPORT ===\n")
    print(policy_report)

    with open("policy_data/policy_report.txt", "w", encoding="utf-8") as f:
        f.write(policy_report)

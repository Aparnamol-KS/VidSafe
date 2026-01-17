import streamlit as st
import os
import json
from datetime import datetime
from pathlib import Path
import time


import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from core.pipeline import VidSafePipeline
from pdf_utils import generate_policy_pdf_bytes
from policy_reducer import reduce_policy_violations_to_text


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
INPUT_VIDEO = "input_video.mp4"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

pipeline = VidSafePipeline(output_dir=Path(OUTPUT_DIR))


# -------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------
st.session_state.setdefault("policy_report", None)
st.session_state.setdefault("pdf_bytes", None)
st.session_state.setdefault("video_ready", False)
st.session_state.setdefault("analysis_done", False)
st.session_state.setdefault("output_video", None)
st.session_state.setdefault("raw_policy_json", None)


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="VidSafe – Video Safety Platform",
    page_icon="🎥",
    layout="wide"
)


# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; margin-top:10px;">
        <h1 style="font-size:72px; font-weight:800; margin-bottom:5px;">
            VidSafe
        </h1>
        <p style="font-size:20px; color:#6b7280;">
            Automated Video Safety Analysis & Policy Enforcement Platform
        </p>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# UPLOAD SECTION
# -------------------------------------------------
st.subheader("Upload Video")

uploaded_video = st.file_uploader(
    "Select a video file for safety analysis",
    type=["mp4", "avi", "mov"]
)

if uploaded_video and not st.session_state.analysis_done:
    col1, col2 = st.columns([1.2, 1])

    # -------- INPUT VIDEO PREVIEW --------
    with col1:
        st.markdown("### Input Video Preview")
        st.video(uploaded_video)

    # -------- VIDEO DETAILS + ACTION --------
    with col2:
        st.markdown("### Video Details")
        st.write(f"**Filename:** {uploaded_video.name}")
        st.write(f"**Size:** {uploaded_video.size / (1024 * 1024):.2f} MB")

        if st.button("▶ Start Analysis", use_container_width=True):
            with st.status(
                "Starting content safety analysis...",
                expanded=False
            ) as status:

                progress = st.progress(0)

                # -------------------------------------------------
                # SAVE VIDEO
                # -------------------------------------------------
                status.update(label="Saving uploaded video...")
                with open(INPUT_VIDEO, "wb") as f:
                    f.write(uploaded_video.read())
                progress.progress(10)

                # -------------------------------------------------
                # AUDIO ANALYSIS
                # -------------------------------------------------
                status.update(label="Analyzing audio for harmful or unsafe speech...")
                input_path = Path(INPUT_VIDEO).resolve()
                results = pipeline.run(input_path)
                progress.progress(40)

                # -------------------------------------------------
                # VIDEO ANALYSIS
                # -------------------------------------------------
                status.update(label="Analyzing video frames for violent or unsafe visuals...")
                progress.progress(65)

                # -------------------------------------------------
                # POLICY MAPPING
                # -------------------------------------------------
                status.update(label="Checking detected content against platform safety rules...")
                st.session_state.output_video = str(results["final_video"])
                st.session_state.raw_policy_json = str(results["policy_report"])

                with open(st.session_state.raw_policy_json, "r", encoding="utf-8") as f:
                    raw_policy_output = json.load(f)

                progress.progress(80)

                # -------------------------------------------------
                # REPORT GENERATION (RAG)
                # -------------------------------------------------
                status.update(label="Generating moderation report...")
                policy_report = reduce_policy_violations_to_text(raw_policy_output)
                st.session_state.policy_report = policy_report

                pdf_payload = {
                    "video_name": uploaded_video.name,
                    "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "detected_categories": list(
                        {v["category"] for v in raw_policy_output.get("policy_violations", [])}
                    ),
                    "severity_level": raw_policy_output.get("fusion_severity", "Medium"),
                    "policy_decision": (
                        "Remove Content"
                        if raw_policy_output.get("fusion_severity") == "Critical"
                        else "Age Restrict"
                        if raw_policy_output.get("fusion_severity") == "High"
                        else "Content Review Recommended"
                    ),
                    "flagged_segments": raw_policy_output.get("policy_violations", []),
                    "explanation": policy_report
                }

                st.session_state.pdf_bytes = generate_policy_pdf_bytes(pdf_payload)

                progress.progress(100)

                # -------------------------------------------------
                # FINALIZE
                # -------------------------------------------------
                st.session_state.video_ready = True
                st.session_state.analysis_done = True

                status.update(
                    label="Moderation completed successfully.",
                    state="complete"
                )

        if st.session_state.raw_policy_json:
            with open(st.session_state.raw_policy_json, "r", encoding="utf-8") as f:
                raw_policy_output = json.load(f)

            st.markdown("#### Overall Fusion Severity")
            st.metric(
                label="Fusion Severity",
                value=raw_policy_output.get("fusion_severity", "Unknown")
            )




# -------------------------------------------------
# RESULTS SECTION
# -------------------------------------------------
if st.session_state.video_ready:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Analysis Results")

    left, right = st.columns([1, 1])

    # -------- PROCESSED VIDEO --------
    with left:
        st.markdown("### Processed Video")
        st.video(st.session_state.output_video)

        with open(st.session_state.output_video, "rb") as f:
            st.download_button(
                "⬇ Download Processed Video",
                f.read(),
                file_name="vidsafe_processed_video.mp4",
                mime="video/mp4",
                use_container_width=True
            )

    # -------- POLICY REPORT --------
    with right:
        st.markdown("### Policy Assessment")
        st.markdown("#### 📝 Moderation Report")

        st.text_area(
            label="Moderation Report",
            value=st.session_state.policy_report,
            height=420,
            label_visibility="collapsed"
        )

        st.download_button(
            "⬇ Download Policy Report (PDF)",
            st.session_state.pdf_bytes,
            file_name="vidsafe_policy_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# -------------------------------------------------
# RESET OPTION
# -------------------------------------------------
if st.session_state.analysis_done:
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("🔄 Analyze Another Video", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

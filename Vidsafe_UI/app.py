import streamlit as st
import os
import json
import shutil
from datetime import datetime

from pdf_utils import generate_policy_pdf_bytes
from policy_reducer import reduce_policy_violations_to_text

# -------------------------------------------------
# CONFIG (UPLOAD SIZE MUST BE SET BEFORE RUN)
# -------------------------------------------------
# IMPORTANT (Windows PowerShell):
# $env:STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024
# streamlit run app.py

INPUT_VIDEO = "input_video.mp4"
OUTPUT_DIR = "outputs"
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "processed_video.mp4")
RAW_POLICY_JSON = os.path.join(OUTPUT_DIR, "raw_policy_output.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "policy_report" not in st.session_state:
    st.session_state.policy_report = None

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "video_ready" not in st.session_state:
    st.session_state.video_ready = False

# -------------------------------------------------
# PAGE SETUP
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
        <h1 style="
            font-size:72px;
            font-weight:800;
            letter-spacing:1px;
            margin-bottom:5px;
        ">
            VidSafe
        </h1>
        <p style="
            font-size:20px;
            color:#6b7280;
            margin-top:0;
        ">
            Automated Video Safety Analysis & Policy Enforcement Platform
        </p>
    </div>
    <hr style="margin-top:20px;">
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

if uploaded_video:
    col1, spacer, col2 = st.columns([1.2, 0.15, 1])

    # -------- LEFT: VIDEO PREVIEW --------
    with col1:
        st.markdown("**Input Video Preview**")
        st.video(uploaded_video)

    # Save uploaded file
    with open(INPUT_VIDEO, "wb") as f:
        f.write(uploaded_video.read())

    # -------- RIGHT: VIDEO DETAILS --------
    with col2:
        st.markdown("**Video Details**")
        st.write(f"**Filename:** {uploaded_video.name}")
        st.write(f"**Size:** {uploaded_video.size / (1024 * 1024):.2f} MB")

        if st.button("▶ Start Analysis", use_container_width=True):
            with st.spinner("Analyzing video content and generating policy assessment..."):

                # -------------------------------------------------
                # (1) DUMMY VIDEO PROCESSING
                # Replace with your real pipeline later
                # -------------------------------------------------
                shutil.copy(INPUT_VIDEO, OUTPUT_VIDEO)

                # -------------------------------------------------
                # (2) RAW POLICY OUTPUT (SIMULATED)
                # -------------------------------------------------
                raw_policy_output = {
                    "video_id": "vid1",
                    "total_violations": 1440,
                    "policy_violations": [
                        {
                            "policy_id": "YT-VIO-001",
                            "policy_name": "Violence and Graphic Content",
                            "category": "Violence",
                            "severity": "Medium",
                            "timestamp": "0:00:01.87 - 0:00:01.90",
                            "reason": "Physical aggression detected"
                        },
                        {
                            "policy_id": "YT-AGE-006",
                            "policy_name": "Age-Restricted Content",
                            "category": "Age Restriction",
                            "severity": "Medium",
                            "timestamp": "0:03:42.10 - 0:03:45.00",
                            "reason": "Content unsuitable for minors"
                        }
                    ]
                }

                with open(RAW_POLICY_JSON, "w", encoding="utf-8") as f:
                    json.dump(raw_policy_output, f, indent=2)

                # -------------------------------------------------
                # (3) LLM REDUCTION → STRUCTURED REPORT
                # -------------------------------------------------
                policy_report = reduce_policy_violations_to_text(raw_policy_output)
                st.session_state.policy_report = policy_report

                # -------------------------------------------------
                # (4) GENERATE PDF
                # -------------------------------------------------
                pdf_payload = {
                    "video_name": uploaded_video.name,
                    "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "detected_categories": ["Violence", "Age Restriction"],
                    "severity_level": "Medium",
                    "policy_decision": "Age restriction required",
                    "flagged_segments": [],
                    "explanation": policy_report
                }

                st.session_state.pdf_bytes = generate_policy_pdf_bytes(pdf_payload)
                st.session_state.video_ready = True

            st.success("Analysis completed successfully")

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
        st.video(OUTPUT_VIDEO)

        with open(OUTPUT_VIDEO, "rb") as f:
            st.download_button(
                "⬇ Download Processed Video",
                f.read(),
                "vidsafe_processed_video.mp4",
                "video/mp4",
                use_container_width=True
            )

    # -------- POLICY PREVIEW --------
    with right:
        st.markdown("### Policy Assessment")
        st.markdown("#### 📝 Moderation Report")

        # Preserve formatting from LLM
        st.text(st.session_state.policy_report)

        st.download_button(
            "⬇ Download Policy Report (PDF)",
            st.session_state.pdf_bytes,
            "vidsafe_policy_report.pdf",
            "application/pdf",
            use_container_width=True
        )

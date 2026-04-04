import streamlit as st
import os
import json
from datetime import datetime
from pathlib import Path
import tempfile
import sys

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------------------------------------------------
# IMPORTS
# -------------------------------------------------
from modules.pipeline import VidSafePipeline
from ui.pdf_utils import generate_policy_pdf_bytes
from ui.policy_reducer import reduce_policy_violations_to_text

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pipeline = VidSafePipeline(output_dir=OUTPUT_DIR)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
def init_session():
    defaults = {
        "policy_report": None,
        "pdf_bytes": None,
        "video_ready": False,
        "analysis_done": False,
        "output_video": None,
        "raw_policy_json": None,
        "uploaded_video": None,
        "seek_time": 0   # 🔥 NEW (timeline control)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="VidSafe", layout="wide")

# -------------------------------------------------
# STYLING
# -------------------------------------------------
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    background-color: #111827;
    margin-bottom: 20px;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 12px;
    color: #f9fafb;
}
.badge {
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 14px;
}
.timeline-item {
    padding: 10px;
    margin-bottom: 8px;
    border-radius: 8px;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
def render_header():
    st.markdown("""
    <div style="text-align:center; padding-bottom:25px;">
        <h1 style="font-size:64px; font-weight:800;">VidSafe</h1>
        <p style="font-size:20px; color:#9ca3af;">
            AI-powered Video Safety Analysis
        </p>
    </div>
    <hr>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# RESET
# -------------------------------------------------
def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# -------------------------------------------------
# PIPELINE
# -------------------------------------------------
def run_pipeline(uploaded_video):

    with st.status("Running analysis..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            input_path = tmp.name

        results = pipeline.run(input_path)

        st.session_state.output_video = results["final_video"]
        st.session_state.raw_policy_json = results["policy_report"]

        with open(st.session_state.raw_policy_json) as f:
            raw = json.load(f)

        st.session_state.policy_report = reduce_policy_violations_to_text(raw)

        st.session_state.video_ready = True
        st.session_state.analysis_done = True

# -------------------------------------------------
# VIDEO SECTION (UPDATED)
# -------------------------------------------------
def render_video():

    if not st.session_state.video_ready:
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Video Output</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("Original")
        st.video(st.session_state.uploaded_video)

    with col2:
        st.markdown("Processed")

        # 🔥 KEY PART — seek functionality
        st.video(st.session_state.output_video, start_time=st.session_state.seek_time)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# CLICKABLE TIMELINE 🔥
# -------------------------------------------------
def render_timeline():

    if not st.session_state.video_ready:
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Violation Timeline</div>', unsafe_allow_html=True)

    with open(st.session_state.raw_policy_json) as f:
        raw = json.load(f)

    violations = raw.get("policy_violations", [])

    if not violations:
        st.success("No violations detected")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    for i, v in enumerate(violations):

        start = int(v.get("start", 0))
        end = int(v.get("end", 0))
        category = v.get("category", "Unknown")

        if st.button(f"{category.upper()}  |  {start}s → {end}s", key=f"timeline_{i}"):

            # 🔥 THIS IS THE MAGIC
            st.session_state.seek_time = start
            st.experimental_rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# POLICY
# -------------------------------------------------
def render_policy():

    if not st.session_state.video_ready:
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Policy Report</div>', unsafe_allow_html=True)

    st.write(st.session_state.policy_report)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# UPLOAD
# -------------------------------------------------
def render_upload():

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Video</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose video", type=["mp4", "avi", "mov"])

    if uploaded:
        st.session_state.uploaded_video = uploaded
        st.video(uploaded)

        if not st.session_state.analysis_done:
            if st.button("Start Analysis"):
                run_pipeline(uploaded)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():

    render_header()

    left, right = st.columns([1, 2])

    with left:
        render_upload()

    with right:
        render_video()
        render_timeline()   # 🔥 NEW FEATURE
        render_policy()

        if st.session_state.analysis_done:
            if st.button("Analyze Another Video"):
                reset_app()

# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    main()
from pathlib import Path
import json

from modules.audio.audio_pipeline import AudioPipeline
from modules.video.video_pipeline import run_video_pipeline
from modules.fusion.aligner import fuse_modalities
from modules.reasoning.rag_engine import run_policy_rag
from modules.audio.merger import merge_audio_to_video
from modules.video.format_converter import convert_to_web_format


class VidSafePipeline:

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.audio_dir = self.output_dir / "audio"
        self.video_output = self.output_dir / "blurred_video.mp4"
        self.final_video = self.output_dir / "final_moderated_video.mp4"
        self.evidence_file = self.output_dir / "moderation_evidence.json"
        self.policy_output = self.output_dir / "policy_report.json"

        self.audio_pipeline = AudioPipeline()

    def run(self, input_video: str):

        input_video = Path(input_video).resolve()

        # ==============================
        # 1️⃣ AUDIO PIPELINE
        # ==============================
        print("\n🔊 Running audio pipeline...")
        audio_results = self.audio_pipeline.run(
            str(input_video),
            str(self.audio_dir)
        )

        # ==============================
        # 2️⃣ VIDEO PIPELINE
        # ==============================
        print("\n🎥 Running video pipeline...")
        video_results = run_video_pipeline(
            str(input_video),
            str(self.video_output)
        )

        vision_segments = video_results["violent_segments"]
        audio_segments = audio_results["word_level_toxic"]

        # ==============================
        # 3️⃣ MERGE VIDEO + AUDIO
        # ==============================
        print("\n🎬 Merging blurred video with censored audio...")

        merge_audio_to_video(
            original_video=str(self.video_output),
            new_audio=str(audio_results["censored_audio"]),
            out_video=str(self.final_video)
        )

        # ==============================
        # 3.1️⃣ FORMAT FIX (ROBUST)
        # ==============================
        print("\n🎞️ Converting video to web-compatible format...")

        web_video = self.output_dir / f"{input_video.stem}_final_web.mp4"

        convert_to_web_format(
            input_path=str(self.final_video),
            output_path=str(web_video)
        )

        # ✅ Safe fallback handling
        if web_video.exists() and web_video.stat().st_size > 0:
            final_output_video = web_video
            print(f"✅ Converted video ready → {web_video}")
        else:
            final_output_video = self.final_video
            print("⚠️ Conversion failed, using original merged video")

        # ==============================
        # 4️⃣ FUSION
        # ==============================
        print("\n🔗 Running multimodal fusion...")
        fused_events = fuse_modalities(
            vision_segments,
            audio_segments
        )

        # ==============================
        # 5️⃣ BUILD EVIDENCE
        # ==============================
        evidence = {
            "video_id": input_video.stem,
            "vision_segments": vision_segments,
            "audio_word_segments": audio_segments,
            "audio_sentence_segments": audio_results["toxic_sentences"],
            "fused_events": fused_events
        }

        with open(self.evidence_file, "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2)

        print(f"\n📄 Evidence saved → {self.evidence_file}")

        # ==============================
        # 6️⃣ POLICY REASONING
        # ==============================
        print("\n🧠 Running policy reasoning...")
        run_policy_rag(
            evidence_file=str(self.evidence_file),
            output_file=str(self.policy_output)
        )

        print(f"📄 Policy report saved → {self.policy_output}")

        # ==============================
        # 7️⃣ FINAL RETURN
        # ==============================
        return {
            "final_video": str(final_output_video),
            "evidence_file": str(self.evidence_file),
            "policy_report": str(self.policy_output)
        }
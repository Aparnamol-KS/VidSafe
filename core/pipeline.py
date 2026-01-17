from pathlib import Path
import json

from audio_processing.main import run_audio_moderation
from violence_detection.vision_pipeline import run_vision_pipeline
from audio_processing.merger import merge_audio_to_video
from Reporting.rag_vector import run_policy_rag


class VidSafePipeline:
    """
    Core orchestration pipeline.
    Responsible for:
    - Running audio + vision modules
    - Merging moderated outputs
    - Formatting evidence
    - Invoking policy-aware reporting
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.audio_dir = self.output_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)

        self.blurred_video = self.output_dir / "blurred_video.mp4"
        self.final_video = self.output_dir / "final_moderated_video.mp4"

        self.evidence_file = self.output_dir / "moderation_evidence.json"
        self.policy_output = self.output_dir / "policy_violations_output.json"

    def run(self, input_video: Path):
        """
        Executes the full multimodal moderation pipeline.
        """
        input_video = Path(input_video).resolve()

        # ==================================================
        # 1. AUDIO MODERATION
        # ==================================================
        print("\n🔊 Running audio moderation...")
        audio_results = run_audio_moderation(
            input_video=str(input_video),
            work_dir=str(self.audio_dir)
        )

        # Normalize audio evidence for reporting
        audio_violations = []
        for item in audio_results.get("word_level_toxic", []):
            audio_violations.append({
                "start": item.get("start", 0.0),
                "end": item.get("end", 0.0),
                "type": "toxic_speech",
                "confidence": item.get("confidence", 0.8),
                "text": item.get("word", "")
            })

        # ==================================================
        # 2. VISION MODERATION
        # ==================================================
        print("\n🎥 Running vision moderation...")
        vision_results = run_vision_pipeline(
            video_path=str(input_video),
            output_path=str(self.blurred_video)
        )

        if not self.blurred_video.exists():
            raise RuntimeError("❌ Blurred video was not generated.")

        violent_segments = vision_results.get("violent_segments", [])

        rtdetr_detections = vision_results.get("rtdetr_detections",[])
        

        # ==================================================
        # 3. MERGE AUDIO + VIDEO
        # ==================================================
        print("\n🎬 Merging censored audio with blurred video...")
        merge_audio_to_video(
            original_video=str(self.blurred_video),
            new_audio=str(audio_results["censored_audio"]),
            out_video=str(self.final_video)
        )

        if not self.final_video.exists():
            raise RuntimeError("❌ Final moderated video was not generated.")

        # ==================================================
        # 4. BUILD NORMALIZED EVIDENCE
        # ==================================================
        evidence = {
            "video_id": input_video.stem,
            "final_video": str(self.final_video),

            # CLIP timestamps
            "violent_segments": violent_segments,

            # RT-DETR truth signal
            "rtdetr_detections": rtdetr_detections,

            # Audio
            "audio_violations": audio_violations
        }


        with open(self.evidence_file, "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2)

        print(f"\n📄 Evidence saved → {self.evidence_file}")

        # ==================================================
        # 5. POLICY-AWARE RAG
        # ==================================================
        print("\n🧠 Running policy-aware reporting...")
        run_policy_rag(
            evidence_file=self.evidence_file,
            output_file=self.policy_output
        )

        print(f"📄 Policy report saved → {self.policy_output}")

        return {
            "final_video": self.final_video,
            "policy_report": self.policy_output,
            "evidence": self.evidence_file
        }

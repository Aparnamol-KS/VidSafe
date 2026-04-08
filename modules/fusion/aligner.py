from config import FUSION_TIME_WINDOW


def overlap(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) <= min(a_end, b_end)


def fuse_modalities(vision_segments, audio_segments):
    """
    Aligns vision and audio events by timestamp.

    Returns:
    [
        {
            "start": float,
            "end": float,
            "vision_confidence": float,
            "audio_confidence": float,
            "severity_score": float,
            "modalities": ["vision", "audio"],
            "vision_queries": [list],
            "audio_label": str
        }
    ]
    """

    fused_events = []

    for v in vision_segments:

        v_start = v["start"]
        v_end = v["end"]

        # video pipeline outputs vision_queries
        v_queries = v.get("vision_queries", [])

        # confidence key may vary depending on pipeline stage
        v_conf = v.get("confidence", v.get("vision_confidence", 0.0))

        overlapping_audio = []

        for a in audio_segments:
            if overlap(v_start, v_end, a["start"], a["end"]):
                overlapping_audio.append(a)

        if overlapping_audio:

            max_audio_conf = max(a["confidence"] for a in overlapping_audio)

            # choose audio label from highest confidence segment
            best_audio = max(overlapping_audio, key=lambda x: x["confidence"])
            audio_label = best_audio.get("label", "")

            severity = round((v_conf + max_audio_conf) / 2, 3)

            fused_events.append({
                "start": v_start,
                "end": v_end,
                "vision_confidence": v_conf,
                "audio_confidence": max_audio_conf,
                "severity_score": severity,
                "modalities": ["vision", "audio"],

                # new semantic fields
                "vision_queries": v_queries,
                "audio_label": audio_label
            })

        else:

            fused_events.append({
                "start": v_start,
                "end": v_end,
                "vision_confidence": v_conf,
                "audio_confidence": 0.0,
                "severity_score": round(v_conf, 3),
                "modalities": ["vision"],

                # propagate CLIP semantic queries
                "vision_queries": v_queries,
                "audio_label": ""
            })

    # -----------------------------
    # Audio-only toxic segments
    # -----------------------------
    for a in audio_segments:

        if not any(
            overlap(a["start"], a["end"], v["start"], v["end"])
            for v in vision_segments
        ):

            fused_events.append({
                "start": a["start"],
                "end": a["end"],
                "vision_confidence": 0.0,
                "audio_confidence": a["confidence"],
                "severity_score": round(a["confidence"], 3),
                "modalities": ["audio"],

                # no vision evidence
                "vision_queries": [],
                "audio_label": a.get("label", "")
            })

    fused_events = sorted(fused_events, key=lambda x: x["start"])

    return fused_events
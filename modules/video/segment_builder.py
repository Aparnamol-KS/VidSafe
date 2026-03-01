def build_segments_from_detections(detections, fps, gap_frames=10):
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["frame"])

    segments = []

    start_frame = detections[0]["frame"]
    end_frame = start_frame
    max_conf = detections[0]["confidence"]

    for det in detections[1:]:
        frame = det["frame"]
        conf = det["confidence"]

        if frame <= end_frame + gap_frames:
            end_frame = frame
            max_conf = max(max_conf, conf)
        else:
            segments.append({
                "start": round(start_frame / fps, 3),
                "end": round(end_frame / fps, 3),
                "confidence": round(max_conf, 3)
            })

            start_frame = frame
            end_frame = frame
            max_conf = conf

    segments.append({
        "start": round(start_frame / fps, 3),
        "end": round(end_frame / fps, 3),
        "confidence": round(max_conf, 3)
    })

    return segments
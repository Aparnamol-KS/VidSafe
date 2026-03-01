def apply_temporal_consistency(detections, min_frames=3):
    """
    Keeps detections only if they persist
    for at least `min_frames` consecutive frames.
    """

    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x["frame"])

    stable = []
    streak = [detections[0]]

    for det in detections[1:]:
        if det["frame"] == streak[-1]["frame"] + 1:
            streak.append(det)
        else:
            if len(streak) >= min_frames:
                stable.extend(streak)
            streak = [det]

    if len(streak) >= min_frames:
        stable.extend(streak)

    return stable
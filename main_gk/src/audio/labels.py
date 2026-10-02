from pathlib import Path
import numpy as np


def parse_lab(lab_path):
    path = Path(lab_path)
    if not path.is_file():
        return {
            "speech_segments": [],
            "boundaries": [],
            "f0_mean": None,
            "f0_std": None,
        }

    merged_segments = []
    f0_mean = None
    f0_std = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue

            first_word = parts[0].lower()

            if first_word == "f0mean" and len(parts) >= 2:
                f0_mean = float(parts[1])
            elif first_word == "f0std" and len(parts) >= 2:
                f0_std = float(parts[1])

            elif len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                tag = parts[2].lower()

                if tag == "sil":
                    current_label = "silence"
                else:
                    current_label = "speech"

                if (
                    len(merged_segments) > 0
                    and merged_segments[-1]["label"] == current_label
                ):
                    merged_segments[-1]["end"] = end
                else:
                    new_segment = {
                        "start": start,
                        "end": end,
                        "label": current_label,
                    }
                    merged_segments.append(new_segment)

    boundaries = []
    for i in range(1, len(merged_segments)):
        boundary_time = round(merged_segments[i]["start"], 2)
        boundaries.append(boundary_time)

    speech_segments = []
    for seg in merged_segments:
        if seg["label"] == "speech":
            speech_segments.append((seg["start"], seg["end"]))

    return {
        "speech_segments": speech_segments,
        "boundaries": boundaries,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
    }


def create_frame_labels(timestamps, speech_segments):
    labels = np.zeros(len(timestamps), dtype=int)
    ts = np.asarray(timestamps)

    for start_time, end_time in speech_segments:
        mask = (ts >= start_time) & (ts <= end_time)
        labels[mask] = 1

    return labels

import numpy as np


def filter_short_silence(
    decisions: np.ndarray,
    frame_shift_ms: float = 10.0,
    min_silence_ms: float = 200.0,
) -> np.ndarray:
    filtered: np.ndarray = np.copy(decisions)
    num_frames: int = len(filtered)
    min_sil_frames: int = int(round(min_silence_ms / frame_shift_ms))

    in_silence: bool = False
    sil_start: int = 0

    for i in range(num_frames):
        if filtered[i] == 0:
            if not in_silence:
                in_silence = True
                sil_start = i
        else:
            if in_silence:
                sil_duration_frames: int = i - sil_start
                # < 200ms
                if sil_duration_frames < min_sil_frames and sil_start > 0:
                    for k in range(sil_start, i):
                        filtered[k] = 1
                in_silence = False
    if in_silence:
        sil_duration_frames = num_frames - sil_start
        if sil_duration_frames < min_sil_frames and sil_start > 0:
            for k in range(sil_start, num_frames):
                filtered[k] = 1

    return filtered


def extract_boundaries_and_segments(
    timestamps: np.ndarray,
    decisions: np.ndarray,
) -> tuple[list[float], list[tuple[float, float, str]]]:
    num_frames: int = len(decisions)
    boundaries: list[float] = []
    segments: list[tuple[float, float, str]] = []

    if num_frames == 0:
        return boundaries, segments

    for i in range(1, num_frames):
        if decisions[i] != decisions[i - 1]:
            t_boundary: float = round(float(timestamps[i]), 2)
            boundaries.append(t_boundary)

    curr_label: int = decisions[0]
    seg_start_time: float = float(timestamps[0])

    for i in range(1, num_frames):
        if decisions[i] != curr_label:
            seg_end_time: float = float(timestamps[i])
            if curr_label == 1:
                label_str = "speech"
            else:
                label_str = "silence"

            segments.append((round(seg_start_time, 2), round(seg_end_time, 2), label_str))
            curr_label = decisions[i]
            seg_start_time = float(timestamps[i])

    seg_end_time = float(timestamps[-1])
    if curr_label == 1:
        label_str = "speech"
    else:
        label_str = "silence"
    segments.append((round(seg_start_time, 2), round(seg_end_time, 2), label_str))

    return boundaries, segments

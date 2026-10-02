import warnings
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav

from src.audio.labels import parse_lab
from src.models import AudioSignal

warnings.filterwarnings("ignore", category=wav.WavFileWarning)


def preprocessing(wav_path):
    fs, x = wav.read(wav_path)

    if x.ndim > 1:
        x = np.mean(x, axis=1)

    if np.issubdtype(x.dtype, np.floating):
        preprocessed = x.astype(np.float32)
    elif np.issubdtype(x.dtype, np.signedinteger):
        max_limit = np.iinfo(x.dtype).max + 1
        preprocessed = x.astype(np.float32) / max_limit
    elif x.dtype == np.uint8:
        preprocessed = (x.astype(np.float32) - 128) / 128
    else:
        raise ValueError(f"Không hỗ trợ định dạng dữ liệu WAV: {x.dtype}")

    return int(fs), preprocessed


def load_audio(wav_path, lab_path=None):
    w_path = Path(wav_path)
    if not w_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file WAV: {w_path}")

    if lab_path is not None:
        l_path = Path(lab_path)
    else:
        l_path = w_path.with_suffix(".lab")

    fs, x = preprocessing(w_path)
    duration = float(len(x) / fs)

    lab_data = parse_lab(l_path)
    gt_segments = lab_data["speech_segments"]
    gt_boundaries = lab_data["boundaries"]

    is_phone_signal = "phone" in w_path.stem.lower()

    return AudioSignal(
        name=w_path.stem,
        wav_path=w_path,
        fs=fs,
        signal=x,
        duration_sec=duration,
        is_phone=is_phone_signal,
        gt_segments=gt_segments,
        gt_boundaries=gt_boundaries,
    )


def load_dataset(dir_path):
    p = Path(dir_path)
    if not p.is_dir():
        raise NotADirectoryError(f"Thư mục không tồn tại: {p}")
    return [load_audio(w) for w in sorted(p.glob("*.wav"))]


from src.models import AudioSignal
from pathlib import Path
from src.audio.loader import load_audio


def load_dataset(path: Path | str) -> list[AudioSignal]:
    w_path = Path(path)
    if not w_path.is_dir():
        raise NotADirectoryError(f"Thư mục không tồn tại: {w_path}")
    res: list[AudioSignal] = []
    for file in sorted(w_path.glob("*.wav")):
        res.append(load_audio(file))
    return res

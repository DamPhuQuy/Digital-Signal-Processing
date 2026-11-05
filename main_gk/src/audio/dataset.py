from src.models import AudioSignal
from pathlib import Path
from src.audio.loader import load_audio


def load_dataset(path: Path | str) -> list[AudioSignal]:
    w_path = Path(path)
    if not w_path.is_dir():
        root = Path(__file__).resolve().parent.parent.parent
        name = w_path.name.lower()
        search_roots = [root, root / "data", Path.cwd(), Path.cwd() / "data"]
        found = False
        for s_root in search_roots:
            if s_root.is_dir():
                for item in s_root.iterdir():
                    if item.is_dir() and item.name.lower() == name:
                        w_path = item
                        found = True
                        break
            if found:
                break
        if not found:
            raise NotADirectoryError(f"Thư mục không tồn tại: {path}")

    res: list[AudioSignal] = []
    for file in sorted(w_path.glob("*.wav")):
        res.append(load_audio(file))
    return res

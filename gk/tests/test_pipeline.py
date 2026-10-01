"""Unit tests kiểm thử cho toàn bộ lifecycle VAD (Speech/Silence Discrimination)."""

import unittest
from pathlib import Path
import numpy as np

from src.models import (
    AudioSample,
    AudioSignal,
    EvaluationMetrics,
    FramedSignal,
    SegmentationResult,
    ShortTimeFeatures,
    Thresholds,
    VADResult,
    VADThresholds,
)
from src.audio import (
    get_ground_truth_segments,
    load_audio_signal,
    load_dataset,
    load_dataset_samples,
    load_wav,
    parse_lab,
)
from src.features import (
    compute_ma,
    compute_ste,
    compute_zcr,
    extract_features,
    frame_signal,
    get_window,
    normalize_feature,
)
from src.segmentation import (
    estimate_threshold_binary_search,
    estimate_threshold_gaussian,
    estimate_threshold_histogram,
    extract_boundaries,
    filter_short_silences,
    find_all_thresholds,
    find_threshold_binary_search,
    find_threshold_gaussian,
    find_threshold_histogram,
    segment_signal,
)
from src.evaluate import (
    compute_boundary_errors,
    evaluate_segmentation,
)
from src.pipeline import SpeechSegmenter, VADPipeline


class TestIoUtils(unittest.TestCase):
    """Kiểm thử cho module preprocess/audio.py."""

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.sample_wav = self.root_dir / "data" / "TinHieuHuanLuyen" / "phone_F1.wav"
        self.sample_lab = self.root_dir / "data" / "TinHieuHuanLuyen" / "phone_F1.lab"

    def test_load_wav(self):
        fs, signal = load_wav(self.sample_wav, normalize=True)
        self.assertEqual(fs, 16000)
        self.assertEqual(signal.ndim, 1)
        self.assertAlmostEqual(np.max(np.abs(signal)), 1.0, places=5)
        self.assertTrue(len(signal) > 0)

    def test_load_audio_signal(self):
        audio = load_audio_signal(self.sample_wav)
        self.assertIsInstance(audio, AudioSignal)
        self.assertEqual(audio.fs, 16000)
        self.assertTrue(audio.is_phone)
        self.assertEqual(audio.environment_name, "Phone")

    def test_parse_lab(self):
        segments = parse_lab(self.sample_lab)
        self.assertTrue(len(segments) > 0)
        # phone_F1.lab bắt đầu bằng sil từ 0.00 đến 0.53
        self.assertEqual(segments[0], (0.00, 0.53, "sil"))
        # Cuối cùng là sil từ 2.75 đến 3.23 (trước F0)
        self.assertEqual(segments[-1], (2.75, 3.23, "sil"))

    def test_get_ground_truth_segments(self):
        merged, boundaries = get_ground_truth_segments(self.sample_lab)
        # Cấu trúc: sil -> speech (gộp v & uv) -> sil => đúng 3 đoạn
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0], (0.00, 0.53, "silence"))
        self.assertEqual(merged[1], (0.53, 2.75, "speech"))
        self.assertEqual(merged[2], (2.75, 3.23, "silence"))

        # Đúng 2 biên chuyển đổi trạng thái: 0.53s và 2.75s
        self.assertEqual(len(boundaries), 2)
        self.assertAlmostEqual(boundaries[0], 0.53, places=2)
        self.assertAlmostEqual(boundaries[1], 2.75, places=2)


class TestFraming(unittest.TestCase):
    """Kiểm thử cho module preprocess/framing.py."""

    def test_get_window(self):
        rect = get_window("rectangular", 100)
        self.assertEqual(len(rect), 100)
        self.assertTrue(np.all(rect == 1.0))

        hamming = get_window("hamming", 100)
        self.assertEqual(len(hamming), 100)
        self.assertAlmostEqual(hamming[0], 0.08, places=4)
        self.assertAlmostEqual(hamming[-1], 0.08, places=4)
        self.assertAlmostEqual(hamming[50], 1.0, places=1)

    def test_frame_signal(self):
        fs = 16000
        # 1 giây tín hiệu
        sig = np.sin(2 * np.pi * 440 * np.arange(fs) / fs)
        frames, ts = frame_signal(sig, fs, frame_size_ms=20.0, frame_shift_ms=10.0, window_type="hamming")
        # Với fs = 16000: frame_len = 320 mẫu, frame_shift = 160 mẫu
        self.assertEqual(frames.shape[1], 320)
        # Số khung: 1 + floor((16000 - 320) / 160) = 1 + 98 = 99
        self.assertEqual(frames.shape[0], 99)
        self.assertEqual(len(ts), 99)
        self.assertTrue(ts[0] > 0.0)
        self.assertTrue(ts[-1] < 1.0)


class TestFeatures(unittest.TestCase):
    """Kiểm thử cho module preprocess/features.py."""

    def test_compute_ste_and_ma(self):
        # Ma trận khung đơn giản: 2 khung, mỗi khung 3 mẫu
        frames = np.array([
            [1.0, -2.0, 3.0],
            [0.0, 0.5, -0.5]
        ])
        ste = compute_ste(frames)
        # Khung 1: 1^2 + (-2)^2 + 3^2 = 1 + 4 + 9 = 14
        # Khung 2: 0 + 0.25 + 0.25 = 0.5
        np.testing.assert_allclose(ste, [14.0, 0.5])

        ma = compute_ma(frames)
        # Khung 1: |1| + |-2| + |3| = 6
        # Khung 2: 0 + 0.5 + 0.5 = 1.0
        np.testing.assert_allclose(ma, [6.0, 1.0])

    def test_compute_zcr(self):
        frames = np.array([
            [1.0, -1.0, 1.0, -1.0],  # 3 lần đổi dấu trên 3 khoảng => 3 / 3 = 1.0
            [1.0, 2.0, 3.0, 4.0]     # 0 lần đổi dấu => 0.0
        ])
        zcr = compute_zcr(frames)
        self.assertAlmostEqual(zcr[0], 1.0)
        self.assertAlmostEqual(zcr[1], 0.0)

    def test_normalize_feature(self):
        feat = np.array([10.0, 20.0, 50.0, 100.0])
        norm = normalize_feature(feat, method="minmax")
        self.assertAlmostEqual(norm[0], 0.0)
        self.assertAlmostEqual(norm[-1], 1.0)
        self.assertTrue(np.all(norm >= 0.0) and np.all(norm <= 1.0))


class TestThresholds(unittest.TestCase):
    """Kiểm thử cho module train/thresholds.py."""

    def test_find_threshold_gaussian(self):
        # Silence: mean = 0.01, std = 0.005
        sil = np.random.normal(0.01, 0.005, 500)
        # Speech: mean = 0.30, std = 0.10
        sp = np.random.normal(0.30, 0.10, 500)

        th = find_threshold_gaussian(sp, sil)
        # Ngưỡng phải nằm giữa 2 kỳ vọng
        self.assertTrue(0.01 < th < 0.30)

    def test_find_threshold_histogram(self):
        # Tạo phân bố bimodal với 2 đỉnh rõ rệt tại 0.05 và 0.40
        np.random.seed(42)
        d1 = np.random.normal(0.05, 0.01, 1000)
        d2 = np.random.normal(0.40, 0.05, 1000)
        data = np.clip(np.concatenate([d1, d2]), 0.0, 1.0)

        th = find_threshold_histogram(data, num_bins=100)
        # Thung lũng phải nằm giữa khoảng 0.05 và 0.40
        self.assertTrue(0.08 < th < 0.35)

    def test_find_threshold_binary_search(self):
        # 1 file giả định có 100 khung, 50 khung đầu là sil (0.01), 50 khung sau là sp (0.5)
        feat = np.concatenate([np.full(50, 0.01), np.full(50, 0.50)])
        lbl = np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)])

        th = find_threshold_binary_search([feat], [lbl], t_min=0.001, t_max=0.49)
        self.assertTrue(0.01 <= th <= 0.50)


class TestSegmenter(unittest.TestCase):
    """Kiểm thử cho module inference/detector.py (Hậu xử lý lọc khoảng lặng < 200ms)."""

    def test_filter_short_silences(self):
        # Khung cách nhau 10 ms (0.01s)
        ts = np.arange(100) * 0.01  # 0.0s đến 0.99s
        # Tạo chuỗi: Speech -> Silence ngắn 50ms (5 khung) -> Speech
        # Index 0..20: 1 (Speech 200ms)
        # Index 20..25: 0 (Silence 50ms < 200ms) -> Cần được gộp thành 1
        # Index 25..50: 1 (Speech 250ms)
        # Index 50..80: 0 (Silence 300ms > 200ms) -> Giữ nguyên là 0
        # Index 80..100: 1 (Speech 200ms)
        dec = np.ones(100, dtype=int)
        dec[20:25] = 0
        dec[50:80] = 0

        filt_dec = filter_short_silences(dec, ts, min_silence_duration_ms=200.0)

        # Khoảng lặng ngắn 50ms phải biến mất (chuyển thành 1)
        self.assertTrue(np.all(filt_dec[20:25] == 1))
        # Khoảng lặng dài 300ms phải được giữ nguyên (vẫn là 0)
        self.assertTrue(np.all(filt_dec[50:80] == 0))

        # Trích xuất biên
        boundaries = extract_boundaries(filt_dec, ts)
        self.assertEqual(len(boundaries), 2)  # Đúng 2 biên quanh đoạn silence dài


class TestMetrics(unittest.TestCase):
    """Kiểm thử cho module evaluate/metrics.py (Đánh giá MAE, RMSE, FER)."""

    def test_compute_boundary_errors(self):
        # Biên chuẩn: 0.50s và 2.50s
        gt = [0.50, 2.50]
        # Biên dự đoán: 0.51s và 2.48s (lệch 10ms và 20ms)
        pred = [0.51, 2.48]

        mae, rmse, _ = compute_boundary_errors(pred, gt)
        # MAE = (10 + 20) / 2 = 15.0 ms
        self.assertAlmostEqual(mae, 15.0, places=2)
        # RMSE = sqrt((10^2 + 20^2) / 2) = sqrt(250) ≈ 15.81 ms
        self.assertAlmostEqual(rmse, np.sqrt(250.0), places=2)


class TestStagesAndPipeline(unittest.TestCase):
    """Kiểm thử cho SpeechSegmenter và VADPipeline."""

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.train_dir = self.root_dir / "data" / "TinHieuHuanLuyen"
        self.test_dir = self.root_dir / "data" / "TinHieuKiemThu"

    def test_segmenter_find_and_segment(self):
        train_signals = load_dataset(self.train_dir)
        self.assertEqual(len(train_signals), 4)

        segmenter = SpeechSegmenter(frame_size_ms=20.0, frame_shift_ms=10.0, window_type="hamming")
        thresholds = segmenter.find_thresholds(train_signals)

        # Kiểm tra các giá trị ngưỡng hợp lệ trong khoảng [0, 1]
        self.assertIsInstance(thresholds, Thresholds)
        self.assertTrue(0.0 < thresholds.binary_global < 1.0)
        self.assertTrue(0.0 < thresholds.histogram_global < 1.0)
        self.assertTrue(0.0 < thresholds.gaussian_global < 1.0)
        self.assertTrue(0.0 < thresholds.binary_studio < 1.0)
        self.assertTrue(0.0 < thresholds.binary_phone < 1.0)

        # Đánh giá 1 sample test
        test_signals = load_dataset(self.test_dir)
        framed, features, seg_res, metrics = segmenter.evaluate_signal(test_signals[0], algorithm="binary")

        self.assertIsInstance(features, ShortTimeFeatures)
        self.assertIsInstance(seg_res, SegmentationResult)
        self.assertEqual(seg_res.signal_name, test_signals[0].name)
        self.assertEqual(len(framed.timestamps), len(features.ste_norm))
        self.assertTrue(metrics.f1 > 0.5)

    def test_pipeline_backwards_compatible(self):
        train_samples = load_dataset_samples(self.train_dir)
        pipeline = VADPipeline(frame_size_ms=20.0, frame_shift_ms=10.0, window_type="hamming")
        thresholds = pipeline.fit(train_samples)

        self.assertTrue(0.0 < thresholds.binary_global < 1.0)
        test_samples = load_dataset_samples(self.test_dir)
        framed, features, vad_res, metrics = pipeline.evaluate_sample(test_samples[0], algorithm="binary")
        self.assertEqual(vad_res.sample_name, test_samples[0].name)

    def test_evaluate_dataset(self):
        train_signals = load_dataset(self.train_dir)
        segmenter = SpeechSegmenter()
        segmenter.find_thresholds(train_signals)
        test_signals = load_dataset(self.test_dir)
        records = segmenter.evaluate_dataset(test_signals, algorithm="binary", use_adaptive=True)

        self.assertEqual(len(records), 4)
        for sig, framed, feat, seg_res, m in records:
            self.assertTrue(m.f1 > 0.8)
            self.assertTrue(m.mae_ms >= 0)


if __name__ == "__main__":
    unittest.main()


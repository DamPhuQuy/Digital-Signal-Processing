from pathlib import Path

from src.audio.dataset import load_dataset
from src.evaluate.metrics import evaluate_signal, print_evaluation_table
from src.models import AudioSignal, EvaluationMetrics, SegmentationResult, Threshold
from src.segmentation.segment import segment_signal
from src.segmentation.threshold import calibrate_thresholds
from src.visualization.plotting import visualize_test_dataset


def run_pipeline(
    train_dir: str | Path = "TinHieuHuanLuyen",
    test_dir: str | Path = "TinHieuKiemThu",
    algorithm: str = "binary",
    output_dir: str | Path = "reports/figures",
    generate_plots: bool = True,
    use_common: bool = True,
) -> tuple[Threshold, list[SegmentationResult], list[EvaluationMetrics]]:
    train_signals: list[AudioSignal] = load_dataset(train_dir)
    test_signals: list[AudioSignal] = load_dataset(test_dir)

    thresholds: Threshold = calibrate_thresholds(train_signals)

    results: list[SegmentationResult] = []
    metrics_list: list[EvaluationMetrics] = []

    for sig in test_signals:
        th_val: float = thresholds.get_threshold(
            algorithm=algorithm,
            is_phone=sig.is_phone,
            use_common=use_common,
        )
        seg_res: SegmentationResult = segment_signal(
            signal=sig,
            threshold=th_val,
            algorithm=algorithm,
            min_silence_ms=200.0,
        )
        eval_met: EvaluationMetrics = evaluate_signal(
            signal=sig,
            predicted_boundaries=seg_res.predicted_boundaries,
            algorithm=algorithm,
            threshold=th_val,
        )
        results.append(seg_res)
        metrics_list.append(eval_met)

    print_evaluation_table(metrics_list)

    if generate_plots:
        visualize_test_dataset(
            test_signals=test_signals,
            results=results,
            metrics_list=metrics_list,
            output_dir=output_dir,
        )

    return thresholds, results, metrics_list

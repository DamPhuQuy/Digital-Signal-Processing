"""Evaluation Package for VAD.

Cung cấp các công cụ tính sai số định lượng (MAE, RMSE, FER, F1) và trực quan hóa kết quả phân đoạn.
"""

from src.evaluate.metrics import (
    compute_boundary_errors,
    compute_frame_metrics,
    evaluate_segmentation,
    evaluate_vad_result,
    match_boundaries,
)
from src.evaluate.visualizer import (
    get_screen_dimensions,
    plot_combined_segmentation_figure,
    plot_combined_vad_figure,
    plot_segmentation_figure,
    plot_vad_result,
    plot_vad_sample_figure,
    position_figure_at_corner,
)

__all__ = [
    "match_boundaries",
    "compute_boundary_errors",
    "compute_frame_metrics",
    "evaluate_segmentation",
    "evaluate_vad_result",
    "get_screen_dimensions",
    "position_figure_at_corner",
    "plot_segmentation_figure",
    "plot_vad_sample_figure",
    "plot_vad_result",
    "plot_combined_segmentation_figure",
    "plot_combined_vad_figure",
]

import numpy as np


def find_threshold_binary_search(
    silence_vals: np.ndarray,
    speech_vals: np.ndarray,
    max_iters: int = 200,
    verbose: bool = False,
) -> tuple[float, str]:
    if len(silence_vals) == 0 or len(speech_vals) == 0:
        return 0.0, "silence"

    if np.mean(silence_vals) <= np.mean(speech_vals):
        f_all = np.sort(silence_vals)
        g_all = np.sort(speech_vals)
        lower_name = "silence"
    else:
        f_all = np.sort(speech_vals)
        g_all = np.sort(silence_vals)
        lower_name = "speech"

    f = f_all[f_all >= g_all.min()]
    g = g_all[g_all <= f_all.max()]

    if len(f) == 0 or len(g) == 0:
        t_opt = float((f_all.max() + g_all.min()) / 2.0)
        if verbose:
            print("Khong co overlap, T = trung điểm 2 phân phối")
        return t_opt, lower_name

    nf: int = len(f)
    ng: int = len(g)

    t_min: float = float(min(f.min(), g.min()))
    t_max: float = float(max(f.max(), g.max()))
    t_curr: float = 0.5 * (t_min + t_max)

    i = int(np.sum(f < t_curr))
    p = int(np.sum(g > t_curr))
    j, q = -1, -1

    n_iter = 0
    while (i != j or p != q) and n_iter < max_iters:
        lhs = (1.0 / nf) * np.sum(np.maximum(f - t_curr, 0.0)) - (1.0 / ng) * np.sum(
            np.maximum(t_curr - g, 0.0)
        )
        if lhs > 0:
            t_min = t_curr
        else:
            t_max = t_curr

        t_curr = 0.5 * (t_min + t_max)
        j, q = i, p
        i = int(np.sum(f < t_curr))
        p = int(np.sum(g > t_curr))
        n_iter += 1

    if verbose:
        print(f"T = {t_curr:.4f}, n_iter = {n_iter}")

    return float(t_curr), lower_name


def find_threshold_binary(
    ste: np.ndarray,
    labels: np.ndarray,
    max_iters: int = 200,
) -> float:
    silence_vals = ste[labels == 0]
    speech_vals = ste[labels == 1]

    if len(silence_vals) == 0 or len(speech_vals) == 0:
        return float(np.median(ste)) if len(ste) > 0 else 0.0

    t_opt, _ = find_threshold_binary_search(
        silence_vals=silence_vals,
        speech_vals=speech_vals,
        max_iters=max_iters,
    )
    return float(t_opt)

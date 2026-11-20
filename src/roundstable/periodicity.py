"""Periodicity analysis for regularly sampled one-dimensional indicator series.

The module builds on the supported one-dimensional transforms in
:mod:`roundstable.fourier` (radix-2 lengths up to 4096, direct DFT up to 64).
A prominent spectral peak alone is never treated as evidence of a business
cycle: every candidate period must be referred to an explicit white-noise null
with family-wise error control, an effective fully observed window and a
documented statistic convention.

Design rules enforced here:

* Missing observations are never zero-filled. The mask splits the series into
  fully observed runs; only complete windows are tested and gaps are reported.
* Windows are disjoint. Multiple windows never count as independent
  replications: a single joint Monte Carlo max statistic covers every window
  and every eligible frequency of one shared statistical population.
* The noise variance is estimated once, by pooling residuals from every
  window, and the same estimate structure is reproduced inside every
  simulation replicate.
* Frequencies at or above the Nyquist limit cannot be distinguished from
  aliases; the Nyquist bin is flagged and anything above Nyquist is absent by
  construction.
* A simulation budget that cannot buy the minimum number of replicates yields
  an ``incomplete`` result, which is distinct from ``no_period_found``.
"""
import math
import random

from . import fourier

MAX_WINDOW = 4096          # hard transform length limit inherited from fourier
MAX_OBS = 65536            # maximum accepted input series length
MIN_WINDOW_DEFAULT = 8     # smallest complete window that carries >=2 cycles
DEFAULT_BUDGET = 100_000   # Monte Carlo work units (eligible bin evaluations)
REPLICATE_CAP = 1000
MIN_REPLICATES = 40
SUPPORTED_NOISE = ("gaussian",)
_TREND_DOF = {"none": 0, "mean": 1, "linear": 2}


def _fisher_survival(m, x):
    """Exact survival P(g >= x) for Fisher's g statistic under white noise.

    g is the maximum of m standardized periodogram ordinates divided by their
    sum. The closed form is an alternating sum evaluated in log space so that
    large combinatorial factors cannot overflow.
    """
    if x <= 0.0:
        return 1.0
    if x >= 1.0:
        return 0.0
    # When every one of the m terms contributes (m*x < 1) the alternating
    # binomial sum collapses exactly to 1; short-circuit to avoid cancellation.
    if m * x < 1.0:
        return 1.0
    terms = []
    for j in range(1, min(m, math.floor(1.0 / x)) + 1):
        log_choose = (math.lgamma(m + 1) - math.lgamma(j + 1)
                      - math.lgamma(m - j + 1))
        log_tail = (m - 1) * math.log(max(1.0 - j * x, 1e-300))
        term = math.exp(log_choose + log_tail)
        terms.append(-term if j % 2 == 0 else term)
    return min(1.0, max(0.0, math.fsum(terms)))


def _detrend(values, trend):
    """Return residuals within a complete window and the trend degrees of freedom."""
    w = len(values)
    if trend == "none":
        return list(values), 0
    mean = math.fsum(values) / w
    if trend == "mean":
        return [y - mean for y in values], 1
    # Ordinary least squares line on a centered time axis (pure Python).
    t0 = (w - 1) / 2.0
    sxx = math.fsum((t - t0) ** 2 for t in range(w))
    sxy = math.fsum((t - t0) * (values[t] - mean) for t in range(w))
    slope = sxy / sxx if sxx > 0.0 else 0.0
    return [values[t] - (mean + slope * (t - t0)) for t in range(w)], 2


def _periodogram(residuals):
    """Unnormalized DFT periodogram |F_k|^2 via the supported fourier transform."""
    spectrum = fourier.transform(residuals, inverse=False)
    return [z.real * z.real + z.imag * z.imag for z in spectrum]


def _build_windows(mask, min_window, max_window):
    """Cut fully observed, non-overlapping power-of-two windows and record gaps."""
    n = len(mask)
    windows, excluded, runs = [], [], []
    i = 0
    while i < n:
        if not mask[i]:
            start = i
            while i < n and not mask[i]:
                i += 1
            excluded.append({"start": start, "length": i - start,
                             "reason": "missing_observations"})
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        runs.append((start, i - start))
    for start, run_len in runs:
        cursor = start
        remaining = run_len
        while remaining >= min_window:
            size = 1 << (min(remaining, max_window).bit_length() - 1)
            windows.append({"index": len(windows), "start": cursor,
                            "end_exclusive": cursor + size, "length": size})
            cursor += size
            remaining -= size
        if remaining:
            excluded.append({"start": cursor, "length": remaining,
                             "reason": "window_below_minimum_length"})
    return windows, excluded


def analyze(values, interval=1.0, mask=None, times=None, trend="linear",
            noise="gaussian", alpha=0.05, budget=None, seed=0,
            min_window=MIN_WINDOW_DEFAULT, max_periods=5, min_cycles=2):
    """Test a regularly sampled indicator series for periodic components.

    Returns a JSON-serialisable result dictionary. Hard contract violations
    raise ``ValueError``; data that simply cannot support a test are returned
    with ``status == "untestable"`` and an explanatory ``reason``.
    """
    # ---- input validation -------------------------------------------------
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("non-empty values list required")
    n = len(values)
    if n > MAX_OBS:
        raise ValueError("input length limit")
    series = []
    for y in values:
        if isinstance(y, complex) or not math.isfinite(float(y)):
            raise ValueError("finite real samples required")
        series.append(float(y))
    if not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
        raise ValueError("interval must be a positive finite number")
    interval = float(interval)
    if trend not in _TREND_DOF:
        raise ValueError("trend must be one of %r" % sorted(_TREND_DOF))
    if not isinstance(alpha, (int, float)) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if mask is None:
        mask = [True] * n
    if len(mask) != n or any(type(b) is not bool for b in mask):
        raise ValueError("mask must be a boolean list matching values")
    if not any(mask):
        raise ValueError("at least one observed sample required")
    if times is not None:
        if len(times) != n or any(not math.isfinite(float(t)) for t in times):
            raise ValueError("times must be finite and match values length")
        times = [float(t) for t in times]
    if isinstance(noise, dict):
        noise = noise.get("kind")
    if noise not in SUPPORTED_NOISE:
        return _untestable("unsupported_noise_null",
                           "only %r white-noise nulls are supported" % (SUPPORTED_NOISE,),
                           series, interval, mask, trend, noise,
                           grid=_grid_report(series, interval, mask))
    if budget is None:
        budget = DEFAULT_BUDGET
    if not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a positive integer")
    if not isinstance(min_window, int) or min_window < 4 or min_window > MAX_WINDOW:
        raise ValueError("min_window out of supported range")
    if not isinstance(max_periods, int) or max_periods < 1:
        raise ValueError("max_periods must be a positive integer")
    if not isinstance(min_cycles, int) or min_cycles < 1:
        raise ValueError("min_cycles must be a positive integer")

    grid = _grid_report(series, interval, mask)

    # ---- regularity check against the explicit time grid ------------------
    if times is not None:
        observed_idx = [k for k in range(n) if mask[k]]
        origin = times[observed_idx[0]]
        for k in observed_idx:
            expected = origin + (k - observed_idx[0]) * interval
            tol = 1e-9 * max(1.0, abs(expected))
            if abs(times[k] - expected) > tol:
                return _untestable("irregular_sampling",
                                   "observed sample at index %d is off the %g grid" % (k, interval),
                                   series, interval, mask, trend, noise, grid=grid)

    # ---- effective windows: complete, disjoint, never zero-filled ---------
    windows, excluded = _build_windows(mask, min_window, MAX_WINDOW)
    if not windows:
        return _untestable("no_valid_window",
                           "no fully observed run reaches min_window=%d" % min_window,
                           series, interval, mask, trend, noise, grid=grid,
                           excluded=excluded)

    # ---- per-window detrending, periodograms and pooled noise scale -------
    dof_total, ss_total = 0, 0.0
    for win in windows:
        segment = series[win["start"]:win["end_exclusive"]]
        residuals, dof = _detrend(segment, trend)
        power = _periodogram(residuals)
        w = win["length"]
        eligible = [k for k in range(min_cycles, w // 2 + 1)]
        if not eligible:
            excluded.append({"start": win["start"], "length": w,
                             "reason": "no_bin_completes_min_cycles"})
            win["eligible"] = []
            continue
        win["residuals"] = residuals
        win["dof"] = dof
        win["power"] = power
        win["eligible"] = eligible
        ss_total += math.fsum(r * r for r in residuals)
        dof_total += w - dof
    windows = [w for w in windows if w.get("eligible")]
    if not windows:
        return _untestable("no_valid_window",
                           "no window contains a bin completing %d cycles" % min_cycles,
                           series, interval, mask, trend, noise, grid=grid,
                           excluded=excluded)
    if dof_total <= 0 or ss_total <= 0.0:
        observed_values = [series[k] for k in range(n) if mask[k]]
        if min(observed_values) == max(observed_values):
            reason, detail = "constant_series", "all observed values are equal; no variance or null scale exists"
        else:
            reason, detail = "zero_residual_variance", "residuals carry no variance after trend removal; the noise null cannot be calibrated"
        return _untestable(reason, detail, series, interval, mask, trend, noise,
                           grid=grid, excluded=excluded)
    sigma2 = ss_total / dof_total

    # ---- observed standardized periodograms and Fisher evidence -----------
    t_obs = 0.0
    candidate_keys = []
    for win in windows:
        w = win["length"]
        standardized = {k: win["power"][k] / (w * sigma2) for k in win["eligible"]}
        win["standardized"] = standardized
        power_sum = math.fsum(win["power"][k] for k in win["eligible"])
        g_bin = max(win["eligible"], key=lambda k: win["power"][k])
        win["fisher_g"] = win["power"][g_bin] / power_sum
        win["fisher_p"] = _fisher_survival(len(win["eligible"]), win["fisher_g"])
        for k in win["eligible"]:
            t_obs = max(t_obs, standardized[k])
        top = sorted(win["eligible"], key=lambda k: standardized[k], reverse=True)[:max_periods]
        candidate_keys.extend((win["index"], k) for k in top)

    # ---- joint Monte Carlo null: one population, one replicate family -----
    total_bins = sum(len(w["eligible"]) for w in windows)
    replicates = max(1, min(REPLICATE_CAP, budget // total_bins))
    rng = random.Random(seed)
    exceed_max = 0
    exceed_marginal = {key: 0 for key in candidate_keys}
    for _ in range(replicates):
        sim_ss, sim_dof = 0.0, 0
        sim_data = []
        for win in windows:
            w = win["length"]
            sim_residuals, dof = _detrend([rng.gauss(0.0, 1.0) for _ in range(w)], trend)
            sim_ss += math.fsum(r * r for r in sim_residuals)
            sim_dof += w - dof
            sim_data.append((win, sim_residuals))
        sim_sigma2 = sim_ss / sim_dof
        family_exceeded = False
        for win, sim_residuals in sim_data:
            sim_power = _periodogram(sim_residuals)
            w = win["length"]
            for k in win["eligible"]:
                stat = sim_power[k] / (w * sim_sigma2)
                if stat >= t_obs:
                    family_exceeded = True
                key = (win["index"], k)
                if key in exceed_marginal and stat >= win["standardized"][k]:
                    exceed_marginal[key] += 1
        if family_exceeded:
            exceed_max += 1
    family_p = (1 + exceed_max) / (replicates + 1)
    completed = replicates >= MIN_REPLICATES

    # ---- candidate report with back-references and alias marking ----------
    nyquist = 1.0 / (2.0 * interval)
    window_by_index = {w["index"]: w for w in windows}
    candidates = []
    for window_index, k in candidate_keys:
        win = window_by_index[window_index]
        w = win["length"]
        f_k = k / (w * interval)
        amplitude = (1.0 if k == w // 2 else 2.0) * math.sqrt(win["power"][k]) / w
        candidates.append({
            "window": window_index,
            "bin": k,
            "frequency": f_k,
            "period": (w * interval) / k,
            "amplitude": amplitude,
            "power": win["power"][k],
            "standardized_power": win["standardized"][k],
            "nyquist_bin": k == w // 2,
            "alias_band_frequency": 1.0 / interval - f_k,
            "window_fisher_p": win["fisher_p"],
            "marginal_empirical_p": (1 + exceed_marginal[(window_index, k)]) / (replicates + 1),
            "family_significant": completed and abs(win["standardized"][k] - t_obs) <= 1e-12 * max(1.0, t_obs) and family_p <= alpha,
            "evidence": {
                "window_start": win["start"],
                "window_end_exclusive": win["end_exclusive"],
                "observed_samples": w,
                "frequency_resolution": 1.0 / (w * interval),
                "statistic": "standardized periodogram |F_k|^2/(w*sigma2)",
                "noise_null": "gaussian white, pooled sigma2 estimate",
            },
        })
    candidates.sort(key=lambda c: c["standardized_power"], reverse=True)

    window_report = [{
        "index": w["index"], "start": w["start"], "end_exclusive": w["end_exclusive"],
        "length": w["length"], "observed_samples": w["length"],
        "resolution_frequency": 1.0 / (w["length"] * interval),
        "eligible_bins": len(w["eligible"]), "fisher_g": w["fisher_g"],
        "fisher_p": w["fisher_p"],
    } for w in windows]

    result = {
        "status": "completed" if completed else "incomplete",
        "grid": grid,
        "trend": trend,
        "trend_dof": _TREND_DOF[trend],
        "noise": {"null": "gaussian_white", "variance": sigma2,
                  "variance_dof": dof_total,
                  "estimation": "pooled over residuals of every effective window"},
        "windows": window_report,
        "excluded_segments": excluded,
        "test": {
            "statistic": "family max standardized periodogram over all windows and bins",
            "replicates": replicates,
            "min_replicates": MIN_REPLICATES,
            "budget": budget,
            "budget_used": replicates * total_bins,
            "family_statistic_observed": t_obs,
            "family_p": family_p if completed else None,
            "alpha": alpha,
            "fisher_method": "exact closed-form survival, scale invariant",
            "rng": "python random.Random.gauss",
            "seed": seed,
            "valid_for_significance": completed,
        },
        "candidates": candidates,
        "period_detection": (
            "significant_period" if completed and family_p <= alpha
            else "no_period_found" if completed
            else "inconclusive_incomplete"
        ),
        "aliasing_note": (
            "frequencies above Nyquist %g are untestable; each bin k is "
            "indistinguishable from alias fs-f, and the Nyquist bin is flagged" % nyquist
        ),
        "window_note": (
            "windows are disjoint and jointly tested; overlapping windows are "
            "never created and repeated windows are not independent samples"
        ),
    }
    if not completed:
        result["reason"] = "insufficient_budget"
        result["detail"] = (
            "budget buys %d replicates, below min_replicates=%d; descriptive "
            "peaks are reported but no significance is claimed" % (replicates, MIN_REPLICATES))
    return result


def _grid_report(series, interval, mask):
    return {
        "interval": interval,
        "samples": len(series),
        "observed_samples": sum(1 for b in mask if b),
        "sampling_frequency": 1.0 / interval,
        "nyquist_frequency": 1.0 / (2.0 * interval),
        "regular": True,
    }


def _untestable(reason, detail, series, interval, mask, trend, noise,
                grid=None, excluded=None):
    return {
        "status": "untestable",
        "reason": reason,
        "detail": detail,
        "grid": grid if grid is not None else _grid_report(series, interval, mask),
        "trend": trend,
        "trend_dof": _TREND_DOF.get(trend, 0),
        "noise": {"null": noise if noise in SUPPORTED_NOISE else None,
                  "supported": list(SUPPORTED_NOISE)},
        "windows": [],
        "excluded_segments": excluded or [],
        "candidates": [],
        "period_detection": "untestable",
    }

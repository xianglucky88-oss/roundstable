"""Tests for the regularly sampled periodicity analysis.

Every statistical claim below is cross-checked by an independent computation:
a direct DFT (module code and numpy), the analytic Fisher g distribution and
Monte Carlo null calibration, plus synthetic observations with known periods.
"""
import cmath
import math
import random

import numpy as np
import pytest

from roundstable import periodicity as P
from roundstable import fourier


# --------------------------------------------------------------------------
# Independent building blocks
# --------------------------------------------------------------------------
def test_periodogram_matches_independent_direct_dft_and_numpy():
    rng = random.Random(11)
    values = [rng.gauss(0.0, 1.0) for _ in range(32)]
    residuals, _ = P._detrend(values, "mean")
    got = P._periodogram(residuals)
    direct = [
        abs(sum(residuals[t] * cmath.exp(-2j * math.pi * k * t / 32)
                for t in range(32))) ** 2
        for k in range(32)
    ]
    scale = max(direct)
    assert max(abs(a - b) for a, b in zip(got, direct)) < 1e-12 * scale
    assert np.allclose(got, np.abs(np.fft.fft(residuals)) ** 2, atol=1e-12 * scale)


def test_fisher_survival_analytic_and_calibrated():
    # m=2 closed form: P(g >= x) = 2(1-x) for 1/2 <= x <= 1, and 1 below 1/2.
    for x in (0.3, 0.5, 0.6, 0.75, 0.9):
        expected = 1.0 if x < 0.5 else 2.0 * (1.0 - x)
        assert abs(P._fisher_survival(2, x) - expected) < 1e-12
    assert P._fisher_survival(5, 0.0) == 1.0 and P._fisher_survival(5, 1.0) == 0.0
    # monotone over the whole domain, including the cancellation-prone tail.
    for m in (2, 5, 15, 31):
        probs = [P._fisher_survival(m, i / 200) for i in range(1, 200)]
        assert all(probs[i] >= probs[i + 1] - 1e-15 for i in range(len(probs) - 1))
    # Independent Monte Carlo calibration of the exact survival curve.
    rng = random.Random(321)
    reps, m, quantile = 40000, 15, 0.2
    hits = 0
    for _ in range(reps):
        ordinates = [rng.gauss(0, 1) ** 2 + rng.gauss(0, 1) ** 2 for _ in range(m)]
        g = max(ordinates) / sum(ordinates)
        hits += g >= quantile
    empirical = hits / reps
    exact = P._fisher_survival(m, quantile)
    se = math.sqrt(empirical * (1 - empirical) / reps)
    assert abs(empirical - exact) <= 3.5 * se


# --------------------------------------------------------------------------
# Known-period observations must be recovered with traceable evidence
# --------------------------------------------------------------------------
def test_known_period_recovered_with_amplitude_and_window_evidence():
    rng = random.Random(12)
    amplitude, period_steps = 2.5, 8
    values = [amplitude * math.sin(2 * math.pi * t / period_steps)
              + 0.3 * rng.gauss(0, 1) for t in range(64)]
    result = P.analyze(values, interval=0.5, trend="mean", budget=40000, seed=12)
    top = result["candidates"][0]
    assert result["period_detection"] == "significant_period"
    assert top["family_significant"] and top["bin"] == 8
    assert top["period"] == pytest.approx(4.0)           # 8 steps * 0.5 time units
    assert top["frequency"] == pytest.approx(0.25)
    assert abs(top["amplitude"] - amplitude) < 0.15
    # amplitude convention 2|F_k|/w independently reproduced
    reference = 2.0 * abs(np.fft.fft([v - np.mean(values) for v in values])[8]) / 64
    assert top["amplitude"] == pytest.approx(reference, rel=1e-10)
    # every judgement points back at real observations, an effective window, a statistic
    evidence = top["evidence"]
    assert evidence["window_start"] == 0 and evidence["observed_samples"] == 64
    assert evidence["statistic"].startswith("standardized periodogram")


def test_linear_trend_alone_is_not_a_period():
    rng = random.Random(5)
    values = [0.4 * t + rng.gauss(0, 1.0) for t in range(128)]
    result = P.analyze(values, trend="linear", budget=30000, seed=5)
    assert result["period_detection"] == "no_period_found"
    assert not any(c["family_significant"] for c in result["candidates"])


def test_nyquist_bin_is_flagged_with_alias_band():
    rng = random.Random(6)
    values = [(1.0 if t % 2 == 0 else -1.0) + 0.05 * rng.gauss(0, 1)
              for t in range(64)]
    result = P.analyze(values, budget=20000, seed=6)
    top = result["candidates"][0]
    assert top["nyquist_bin"] and top["bin"] == 32
    assert top["alias_band_frequency"] == result["grid"]["nyquist_frequency"]
    assert "Nyquist" in result["aliasing_note"]


# --------------------------------------------------------------------------
# Untestable data must explain why, never report significance
# --------------------------------------------------------------------------
def test_constant_and_zero_residual_series_are_untestable():
    result = P.analyze([7.0] * 16)
    assert result["status"] == "untestable" and result["reason"] == "constant_series"
    assert result["period_detection"] == "untestable" and result["candidates"] == []
    linear = P.analyze([float(t) for t in range(16)])
    assert linear["status"] == "untestable"
    assert linear["reason"] == "zero_residual_variance"


def test_irregular_sampling_is_rejected_but_masked_times_are_ignored():
    n, mask = 40, [True] * 16 + [False] * 8 + [True] * 16
    times = [2.0 * t for t in range(n)]
    regular = P.analyze([math.sin(t) for t in range(n)], interval=2.0,
                        mask=mask, times=times, budget=8000)
    assert regular["status"] == "completed"
    irregular_times = times.copy()
    irregular_times[5] += 0.7
    rejected = P.analyze([math.sin(t) for t in range(n)], interval=2.0,
                         mask=mask, times=irregular_times, budget=8000)
    assert rejected["status"] == "untestable" and rejected["reason"] == "irregular_sampling"
    # a deviant timestamp at a missing index does not break the observed grid
    masked_deviant = times.copy()
    masked_deviant[20] = 999.0
    still_regular = P.analyze([math.sin(t) for t in range(n)], interval=2.0,
                              mask=mask, times=masked_deviant, budget=8000)
    assert still_regular["status"] == "completed"


def test_gaps_never_zero_filled_and_short_runs_are_excluded():
    mask = [True] * 20 + [False] * 4 + [True] * 40
    result = P.analyze([math.sin(0.5 * t) for t in range(64)], mask=mask, budget=5000)
    windows = result["windows"]
    # windows are disjoint and fully observed
    for window in windows:
        assert all(mask[i] for i in range(window["start"], window["end_exclusive"]))
    assert all(a["end_exclusive"] <= b["start"]
               for a, b in zip(windows, windows[1:]))
    reasons = {item["reason"] for item in result["excluded_segments"]}
    assert "missing_observations" in reasons
    short = P.analyze([math.sin(t) for t in range(10)], min_window=16, budget=1000)
    assert short["status"] == "untestable" and short["reason"] == "no_valid_window"
    assert short["excluded_segments"][0]["reason"] == "window_below_minimum_length"


def test_unsupported_noise_null_is_untestable_with_reason():
    result = P.analyze([math.sin(t) for t in range(32)], noise="ar1")
    assert result["status"] == "untestable"
    assert result["reason"] == "unsupported_noise_null"
    assert result["noise"]["supported"] == ["gaussian"]


# --------------------------------------------------------------------------
# Error control, budgets and shared statistical population
# --------------------------------------------------------------------------
def test_family_wise_error_rate_under_white_noise_null():
    rng = random.Random(99)
    alpha, trials = 0.1, 120
    false_positive, p_values = 0, []
    for _ in range(trials):
        result = P.analyze([rng.gauss(0, 1) for _ in range(64)],
                           alpha=alpha, budget=20000, seed=rng.randrange(10 ** 9))
        p_values.append(result["test"]["family_p"])
        false_positive += result["period_detection"] == "significant_period"
    rate = false_positive / trials
    se = math.sqrt(alpha * (1 - alpha) / trials)
    assert abs(rate - alpha) <= 3 * se + 1 / trials
    assert 0.35 < sum(p_values) / len(p_values) < 0.65


def test_incomplete_budget_is_distinct_from_no_period_found():
    rng = random.Random(2)
    result = P.analyze([rng.gauss(0, 1) for _ in range(32)], budget=10, seed=2)
    assert result["status"] == "incomplete"
    assert result["period_detection"] == "inconclusive_incomplete"
    assert result["test"]["family_p"] is None
    assert result["test"]["valid_for_significance"] is False
    assert result["reason"] == "insufficient_budget"
    # descriptive peaks remain but cannot claim significance
    assert not any(c["family_significant"] for c in result["candidates"])
    adequate = P.analyze([rng.gauss(0, 1) for _ in range(32)], budget=20000, seed=2)
    assert adequate["status"] == "completed"
    assert adequate["period_detection"] == "no_period_found"


def test_disjoint_windows_share_one_joint_test_and_one_noise_population():
    rng = random.Random(14)
    data = [0.5 * rng.gauss(0, 1) for _ in range(192)]
    for t in range(128, 192):                       # signal confined to window 2
        data[t] += 2.0 * math.sin(2 * math.pi * t / 16)
    result = P.analyze(data, budget=120000, seed=14)
    assert len(result["windows"]) >= 2
    assert result["period_detection"] == "significant_period"
    significant = [c for c in result["candidates"] if c["family_significant"]]
    assert len(significant) == 1                     # one family decision, not one per window
    window = next(w for w in result["windows"]
                  if w["index"] == significant[0]["window"])
    assert window["start"] == 128 and significant[0]["bin"] == 4
    test = result["test"]
    assert test["budget_used"] == test["replicates"] * sum(
        w["eligible_bins"] for w in result["windows"])
    assert result["noise"]["estimation"].startswith("pooled")


def test_chunked_execution_respects_transform_limits():
    rng = random.Random(4)
    result = P.analyze([rng.gauss(0, 1) for _ in range(10000)],
                       budget=300000, seed=4)
    assert result["status"] == "completed"
    assert all(w["length"] <= 4096 for w in result["windows"])
    assert all(w["length"] & (w["length"] - 1) == 0 for w in result["windows"])
    # a single Monte Carlo replicate covers every window jointly
    assert result["test"]["budget_used"] == result["test"]["replicates"] * sum(
        w["eligible_bins"] for w in result["windows"])


def test_seeded_run_is_deterministic_and_inputs_are_validated():
    rng = random.Random(7)
    data = [rng.gauss(0, 1) for _ in range(64)]
    first = P.analyze(data, budget=5000, seed=99)
    second = P.analyze(data, budget=5000, seed=99)
    assert first["test"]["family_p"] == second["test"]["family_p"]
    with pytest.raises(ValueError):
        P.analyze(data, interval=0)
    with pytest.raises(ValueError):
        P.analyze(data, alpha=1.5)
    with pytest.raises(ValueError):
        P.analyze(data, budget=0)
    with pytest.raises(ValueError):
        P.analyze(data, trend="quadratic")
    with pytest.raises(ValueError):
        P.analyze([1.0, float("nan")] + data[2:])
    with pytest.raises(ValueError):
        P.analyze(data, mask=[False] * len(data))


def test_transform_length_limits_are_inherited_not_bypassed():
    # the analysis only schedules lengths the fourier component accepts
    assert fourier.transform.__module__ == "roundstable.fourier"
    with pytest.raises(ValueError):
        fourier.transform([0.0] * 4097)

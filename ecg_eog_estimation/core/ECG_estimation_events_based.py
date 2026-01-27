"""
ECG event validation (REAL ECG reference vs MNE ECG-from-MEG baseline) + ICA UNSUP ECG proxy (MEG-only)

================================================================================
GOAL
================================================================================
For each MEG recording, we want to validate R-peak (ECG) event detection.

We will compute R-peak events in 3 different ways:

REFERENCE (ground truth for benchmarking in this script)
  (R) MNE ECG trace derived from the REAL ECG channel:
      mne.preprocessing.find_ecg_events(raw, ch_name=<real_ecg>, return_ecg=True)

TEST #1 (baseline, MEG-only method already provided by MNE)
  (M) MNE ECG trace derived from MEG channels (no ECG channel present):
      find_ecg_events(raw_meg_only, ch_name=None, return_ecg=True)

TEST #2 (new, MEG-only ICA unsupervised proxy)
  (I) Fit ICA ONCE on MEG-only data, pick the "most ECG-like" IC using MEG-only
      heuristics (NO ECG channel is used for selection), then:
      - create a RawArray with this IC time course as an ECG channel
      - run find_ecg_events on it to detect R-peaks

We then compare detected event sample indices (R-peaks) against the reference using a
one-to-one matching with tolerance MATCH_TOL_SEC:
  - Precision / Recall / F1
  - False positives per minute
  - Jitter statistics (test - ref) for matched events (ms)

IMPORTANT DESIGN CHOICE (EVENT METRICS)
================================================================================
Event matching uses the event sample indices directly, WITHOUT shifting events based on
any waveform lag. This avoids "cheating" by aligning time series; we measure detection
timing exactly as produced by each detector.

Waveform alignment (cross-correlation lag) is used ONLY for visualization/correlation,
so overlay plots are readable.

OUTPUTS PER FILE
================================================================================
1) Publication-grade figure: reference vs baseline (MNE-from-MEG)
2) Publication-grade figure: reference vs ICA-unsupervised proxy
3) ICA score ranking bar plot (top K ICs)
4) ICA diagnostics panel for the selected IC

And a CSV summary across files.

DEPENDENCIES
================================================================================
- mne, numpy, pandas, matplotlib, scipy

NOTE ABOUT SIGN / LAG FOR ICA (BENCHMARK-ONLY)
================================================================================
ICA sign is arbitrary. For the ICA waveform overlay figure we optionally flip the ICA
derived MNE-ECG trace to correlate positively with the reference. This affects ONLY
plots/correlation (secondary). Event detection & matching does not use this sign flip.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from scipy.signal import correlate, find_peaks
from scipy.stats import pearsonr, kurtosis


# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset search root. We will recursively look for "*_meg.fif"
DATASET_ROOT = "/Users/karelo/Development/datasets/ds_small"

# Where to write figures + CSV
OUTPUT_DIR = "/Users/karelo/Development/datasets/ds_small/derivatives/ecg_results_event_validation_with_ica_unsup_v2"

# Plot window in seconds for time series panels (A/B/C/D)
PLOT_SECONDS = 60

# Parallel processing: set >1 to enable joblib across files
N_JOBS = 1

# Matching tolerance to consider a detected R-peak correct (seconds)
MATCH_TOL_SEC = 0.05  # 50 ms

# --------------------------
# ICA (unsupervised) config
# --------------------------
ICA_N_COMPONENTS = 0.99
ICA_METHOD = "fastica"
ICA_RANDOM_STATE = 97
ICA_MAX_ITER = 1000

# Band-pass for ICA stability / focusing on ECG-like content (typical QRS energy)
ICA_L_FREQ = 5.0
ICA_H_FREQ = 40.0

# --------------------------
# Unsupervised ECG scoring
# --------------------------
# Plausible heart rate range
HR_MIN_BPM = 40
HR_MAX_BPM = 180
HR_BAND_HZ = (HR_MIN_BPM / 60.0, HR_MAX_BPM / 60.0)

# Score on first N seconds (None uses full recording)
UNSUP_SCORE_SECONDS = None

# Peak detection parameters used in ECG-like heuristics
PEAK_MIN_DISTANCE_SEC = 0.30
PEAK_PROMINENCE = 1.3

# "Spike" heuristics (high |z| points)
SPIKE_Z_ABS_THRESHOLD = 4.0
SPIKE_RATE_MIN_PER_MIN = 20
SPIKE_RATE_MAX_PER_MIN = 220

# RR variability plausibility (MAD thresholds)
RR_MAD_TOO_SMALL_SEC = 0.005
RR_MAD_TOO_LARGE_SEC = 0.25

# Weighted sum of ECG-likeness sub-scores
W_HR_BANDPOWER = 0.20
W_AUTOCORR = 0.20
W_PEAK_TRAIN = 0.25
W_KURT = 0.10
W_SPIKE_STRENGTH = 0.25

# Diagnostic plots
UNSUP_SCORE_TOPK_PLOT = 12       # bar plot: show top-K ICs
UNSUP_DIAG_SECONDS = 30          # how many seconds to show in diagnostics time panel
UNSUP_DIAG_MAX_AUTOCORR_LAG_SEC = 2.5  # autocorr plot max lag

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# BASIC HELPERS (normalization, lag, shifting)
# =============================================================================
def safe_zscore(x: np.ndarray) -> np.ndarray:
    """
    Z-score a 1D array safely.

    Why "safe"?
    - If std==0 (constant signal) or std is NaN, z-scoring would break.
    - In that case, we fall back to mean-centering only.
    """
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sd = np.std(x)
    if sd == 0 or np.isnan(sd):
        return x - mu
    return (x - mu) / sd


def best_lag_via_xcorr(x_ref: np.ndarray, y: np.ndarray) -> int:
    """
    Find lag (samples) that maximizes cross-correlation between y and x_ref.

    Used ONLY for waveform overlay plots / correlations.

    Convention:
      c = correlate(y, x_ref, mode='full')
      best_lag > 0 means shifting y LEFT by best_lag aligns better with x_ref.
    """
    n = min(len(x_ref), len(y))
    x = x_ref[:n] - np.mean(x_ref[:n])
    yy = y[:n] - np.mean(y[:n])

    c = correlate(yy, x, mode="full")
    lags = np.arange(-n + 1, n)
    return int(lags[np.argmax(c)])


def shift_with_zeros(y: np.ndarray, lag: int) -> np.ndarray:
    """
    Shift a 1D array by 'lag' samples, padding with zeros to keep length unchanged.

    lag > 0: shift LEFT (drop first lag samples, pad zeros at end)
    lag < 0: shift RIGHT (pad zeros at start, drop last -lag samples)
    lag = 0: unchanged
    """
    y = np.asarray(y)
    if lag > 0:
        return np.concatenate([y[lag:], np.zeros(lag)])
    elif lag < 0:
        k = -lag
        return np.concatenate([np.zeros(k), y[:-k]])
    return y.copy()


# =============================================================================
# EVENT MATCHING HELPERS (PRIMARY METRICS)
# =============================================================================
def extract_event_samples(events: np.ndarray) -> np.ndarray:
    """
    MNE events are (n_events, 3) and the sample index is column 0.
    Returns a 1D int array of sample indices.
    """
    if events is None or len(events) == 0:
        return np.array([], dtype=int)
    return np.asarray(events[:, 0], dtype=int)


def match_events_one_to_one(
    ref_samp: np.ndarray,
    test_samp: np.ndarray,
    sfreq: float,
    tol_sec: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One-to-one greedy matching between reference and test event sample indices.

    We walk through both sorted lists and match the closest pair if within tolerance.

    Returns:
      matched_ref   : sample indices from reference that were matched (TP)
      matched_test  : corresponding sample indices from test (TP)
      unmatched_ref : ref events not matched (FN)
      unmatched_test: test events not matched (FP)
    """
    ref_samp = np.sort(np.asarray(ref_samp, dtype=int))
    test_samp = np.sort(np.asarray(test_samp, dtype=int))
    tol = int(round(tol_sec * sfreq))

    i, j = 0, 0
    matched_ref, matched_test = [], []

    while i < len(ref_samp) and j < len(test_samp):
        r = ref_samp[i]
        t = test_samp[j]
        d = t - r

        if abs(d) <= tol:
            matched_ref.append(r)
            matched_test.append(t)
            i += 1
            j += 1
        elif t < r - tol:
            # test is too early; move test forward
            j += 1
        else:
            # ref is too early; move ref forward
            i += 1

    matched_ref = np.array(matched_ref, dtype=int)
    matched_test = np.array(matched_test, dtype=int)

    matched_ref_set = set(matched_ref.tolist())
    matched_test_set = set(matched_test.tolist())

    unmatched_ref = np.array([r for r in ref_samp if r not in matched_ref_set], dtype=int)
    unmatched_test = np.array([t for t in test_samp if t not in matched_test_set], dtype=int)

    return matched_ref, matched_test, unmatched_ref, unmatched_test


def compute_detection_metrics(
    matched_ref: np.ndarray,
    matched_test: np.ndarray,
    unmatched_ref: np.ndarray,
    unmatched_test: np.ndarray,
    sfreq: float
) -> Dict:
    """
    Compute standard detection metrics and jitter stats.

    - TP = number of matched reference events
    - FN = unmatched reference events
    - FP = unmatched test events

    Jitter is (test - ref) in seconds for matched events.

    Returns a dict with:
      precision, recall, f1, miss_rate, jitter summaries, etc.
    """
    tp = len(matched_ref)
    fn = len(unmatched_ref)
    fp = len(unmatched_test)

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = np.nan
    miss_rate = fn / (tp + fn) if (tp + fn) > 0 else np.nan

    if tp > 0:
        jitter_sec = (matched_test - matched_ref) / sfreq
        jitter_abs_sec = np.abs(jitter_sec)
        jitter_mean_ms = 1e3 * np.mean(jitter_sec)            # signed mean
        jitter_std_ms = 1e3 * np.std(jitter_sec)              # signed std
        jitter_mae_ms = 1e3 * np.mean(jitter_abs_sec)         # mean absolute error
        jitter_median_ms = 1e3 * np.median(jitter_abs_sec)
        jitter_p95_ms = 1e3 * np.percentile(jitter_abs_sec, 95)
    else:
        jitter_sec = np.array([], dtype=float)
        jitter_mean_ms = jitter_std_ms = jitter_mae_ms = jitter_median_ms = jitter_p95_ms = np.nan

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "miss_rate": miss_rate,
        "jitter_mean_ms_signed": jitter_mean_ms,
        "jitter_std_ms_signed": jitter_std_ms,
        "jitter_mae_ms": jitter_mae_ms,
        "jitter_median_abs_ms": jitter_median_ms,
        "jitter_p95_abs_ms": jitter_p95_ms,
        "jitter_sec_signed": jitter_sec,
    }


# =============================================================================
# ICA UNSUPERVISED SCORING HELPERS (MEG-only IC ranking)
# =============================================================================
def segment_for_scoring(x: np.ndarray, sfreq: float, seconds: Optional[float]) -> np.ndarray:
    """Return x[:seconds] in samples (or full x if seconds is None)."""
    if seconds is None:
        return x
    n_seg = min(len(x), int(round(seconds * sfreq)))
    return x[:n_seg]


def autocorr_full(x: np.ndarray) -> np.ndarray:
    """
    Normalized autocorrelation for non-negative lags.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    if len(x) < 3:
        return np.array([1.0])

    ac = correlate(x, x, mode="full")
    ac = ac[ac.size // 2:]  # keep >=0 lags

    if ac[0] == 0:
        return np.zeros_like(ac)
    return ac / ac[0]


def bandpower_ratio_hr(x: np.ndarray, sfreq: float, band_hz=(0.8, 2.5)) -> float:
    """
    Compute fraction of total power in HR band (band_hz) using FFT PSD.
    This favors components with periodic energy near plausible HR frequencies.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)

    # Need enough samples to estimate frequency content robustly
    if n < int(sfreq * 5):
        return 0.0

    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    psd = np.abs(np.fft.rfft(x)) ** 2

    # Ignore DC
    valid = freqs > 0
    freqs = freqs[valid]
    psd = psd[valid]

    denom = psd.sum()
    if denom <= 0:
        return 0.0

    lo, hi = band_hz
    band_mask = (freqs >= lo) & (freqs <= hi)
    return float(psd[band_mask].sum() / denom)


def autocorr_periodicity_score(x: np.ndarray, sfreq: float, hr_band_hz=(0.75, 2.5)) -> float:
    """
    Score periodicity by taking the max autocorrelation within HR lag window.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < int(sfreq * 5):
        return 0.0

    ac = autocorr_full(x)

    lo, hi = hr_band_hz
    min_period = 1.0 / hi
    max_period = 1.0 / lo

    min_lag = int(round(min_period * sfreq))
    max_lag = int(round(max_period * sfreq))

    if max_lag <= min_lag or max_lag >= len(ac):
        return 0.0

    return float(np.max(ac[min_lag:max_lag + 1]))


def peak_train_plausibility_ecg(
    x: np.ndarray,
    sfreq: float,
    min_bpm: float = 40.0,
    max_bpm: float = 180.0,
    peak_prominence: float = 1.3,
    min_distance_sec: float = 0.30,
) -> Tuple[float, float, int, float, np.ndarray]:
    """
    Detect peaks in |z| and judge whether the peak train looks like plausible heartbeats.

    Returns:
      score, bpm_est, n_peaks, rr_mad_sec, peaks_samples
    """
    z = safe_zscore(x)
    z_abs = np.abs(z)

    min_dist = max(1, int(round(min_distance_sec * sfreq)))
    peaks, _ = find_peaks(z_abs, prominence=peak_prominence, distance=min_dist)

    duration_sec = len(z) / sfreq
    if duration_sec <= 0 or len(peaks) < 2:
        return 0.0, float("nan"), int(len(peaks)), float("nan"), peaks

    bpm = (len(peaks) / duration_sec) * 60.0

    # HR plausibility score
    if min_bpm <= bpm <= max_bpm:
        hr_score = 1.0
    else:
        if bpm < min_bpm:
            d = (min_bpm - bpm) / min_bpm
        else:
            d = (bpm - max_bpm) / max_bpm
        hr_score = float(np.exp(-3.0 * d))

    # RR variability plausibility
    rr = np.diff(peaks) / sfreq
    rr_med = np.median(rr)
    rr_mad = np.median(np.abs(rr - rr_med)) + 1e-12

    rr_score = 1.0
    if rr_mad < RR_MAD_TOO_SMALL_SEC:
        rr_score = 0.3
    elif rr_mad > RR_MAD_TOO_LARGE_SEC:
        rr_score = 0.3

    return float(hr_score * rr_score), float(bpm), int(len(peaks)), float(rr_mad), peaks


def spike_strength_score(x: np.ndarray, sfreq: float) -> Tuple[float, float, float, int, np.ndarray]:
    """
    ECG has sharp QRS transients. We measure spike strength using |z| threshold events.

    Returns:
      score, rate_per_min, frac_above_thr, n_spikes, spikes_samples
    """
    z = safe_zscore(x)
    z_abs = np.abs(z)

    min_dist = max(1, int(round(PEAK_MIN_DISTANCE_SEC * sfreq)))
    spikes, _ = find_peaks(z_abs, height=SPIKE_Z_ABS_THRESHOLD, distance=min_dist)

    duration_sec = len(z) / sfreq
    if duration_sec <= 0:
        return 0.0, 0.0, 0.0, 0, spikes

    rate_per_min = (len(spikes) / duration_sec) * 60.0
    frac_above = float(np.mean(z_abs >= SPIKE_Z_ABS_THRESHOLD))

    # Rate plausibility
    if SPIKE_RATE_MIN_PER_MIN <= rate_per_min <= SPIKE_RATE_MAX_PER_MIN:
        rate_score = 1.0
    else:
        if rate_per_min < SPIKE_RATE_MIN_PER_MIN:
            d = (SPIKE_RATE_MIN_PER_MIN - rate_per_min) / max(1e-9, SPIKE_RATE_MIN_PER_MIN)
        else:
            d = (rate_per_min - SPIKE_RATE_MAX_PER_MIN) / max(1e-9, SPIKE_RATE_MAX_PER_MIN)
        rate_score = float(np.exp(-3.0 * d))

    # Fraction above threshold saturating score
    frac_score = float(np.tanh(frac_above * 50.0))

    # Combine: rate is more important than raw fraction
    score = float(0.7 * rate_score + 0.3 * frac_score)
    return score, float(rate_per_min), float(frac_above), int(len(spikes)), spikes


def unsupervised_ecg_ic_score(ic: np.ndarray, sfreq: float) -> Dict:
    """
    Compute an ECG-likeness score for one ICA component using ONLY MEG-derived IC.

    Sub-scores:
      - p_hr   : power fraction in HR band
      - p_ac   : autocorr periodicity in HR lag window
      - p_peaks: peak train plausibility (rate + RR variability)
      - p_kurt : peaky distribution (kurtosis)
      - p_spike: spike strength plausibility

    Final score = weighted sum.
    """
    x = segment_for_scoring(ic, sfreq, UNSUP_SCORE_SECONDS)
    z = safe_zscore(x)

    p_hr = bandpower_ratio_hr(z, sfreq, band_hz=HR_BAND_HZ)
    p_ac = autocorr_periodicity_score(z, sfreq, hr_band_hz=HR_BAND_HZ)

    p_peaks, bpm, n_peaks, rr_mad, peaks = peak_train_plausibility_ecg(
        z,
        sfreq,
        min_bpm=HR_MIN_BPM,
        max_bpm=HR_MAX_BPM,
        peak_prominence=PEAK_PROMINENCE,
        min_distance_sec=PEAK_MIN_DISTANCE_SEC,
    )

    k = float(kurtosis(z, fisher=False, bias=False)) if len(z) > 10 else 0.0
    p_kurt = float(np.tanh((k - 3.0) / 5.0))
    p_kurt = max(0.0, p_kurt)

    p_spike, spike_rate_per_min, frac_above_thr, n_spikes, spikes = spike_strength_score(z, sfreq)

    score = (
        W_HR_BANDPOWER * p_hr +
        W_AUTOCORR * p_ac +
        W_PEAK_TRAIN * p_peaks +
        W_KURT * p_kurt +
        W_SPIKE_STRENGTH * p_spike
    )

    return dict(
        score=float(score),
        p_hr=float(p_hr),
        p_ac=float(p_ac),
        p_peaks=float(p_peaks),
        p_kurt=float(p_kurt),
        p_spike=float(p_spike),
        bpm_est=float(bpm),
        n_peaks=int(n_peaks),
        rr_mad_sec=float(rr_mad) if np.isfinite(rr_mad) else float("nan"),
        kurtosis=float(k),
        spike_rate_per_min=float(spike_rate_per_min),
        frac_absz_above_thr=float(frac_above_thr),
        n_spikes=int(n_spikes),
        peaks_samples=peaks.astype(int) if peaks is not None else np.array([], dtype=int),
        spikes_samples=spikes.astype(int) if spikes is not None else np.array([], dtype=int),
        scoring_n_samples=int(len(z)),
    )


def fit_ica_and_get_sources(raw_meg: mne.io.BaseRaw) -> Tuple[mne.preprocessing.ICA, np.ndarray]:
    """
    Fit ICA ONCE on MEG-only Raw and return:
      - ica object
      - sources array (n_ic, n_times)

    We filter (ICA_L_FREQ..ICA_H_FREQ) before fitting to improve stability.
    """
    tmp = raw_meg.copy()
    if ICA_L_FREQ is not None or ICA_H_FREQ is not None:
        tmp.filter(ICA_L_FREQ, ICA_H_FREQ, fir_design="firwin", verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=ICA_N_COMPONENTS,
        method=ICA_METHOD,
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER,
    )
    ica.fit(tmp, verbose=False)

    sources = ica.get_sources(tmp).get_data()
    if sources.size == 0:
        raise RuntimeError("ICA produced no sources.")
    return ica, sources


def pick_best_ic_unsupervised(sources: np.ndarray, sfreq: float) -> Tuple[int, Dict, List[Dict]]:
    """
    Score each IC and return:
      best_ic_index, best_details, all_details_sorted(desc by score)
    """
    all_details: List[Dict] = []
    for i in range(sources.shape[0]):
        d_pos = unsupervised_ecg_ic_score(sources[i, :], sfreq)
        d_neg = unsupervised_ecg_ic_score(-sources[i, :], sfreq)
        if d_neg["score"] > d_pos["score"]:
            d = d_neg
            d["flip_for_score"] = True
        else:
            d = d_pos
            d["flip_for_score"] = False
        d["ic"] = int(i)
        all_details.append(d)

    all_sorted = sorted(all_details, key=lambda x: x["score"], reverse=True)
    best = all_sorted[0]
    return int(best["ic"]), best, all_sorted


# =============================================================================
# PLOTTING: PUBLICATION-STYLE FIGURE (A/B/C/D)
# =============================================================================
def make_publication_figure(
    out_png: str,
    subject_id: str,
    sfreq: float,
    t: np.ndarray,
    ref_trace: np.ndarray,
    test_trace_aligned: np.ndarray,
    ref_events_samp_abs: np.ndarray,
    test_events_samp_abs: np.ndarray,
    matched_ref_abs: np.ndarray,
    matched_test_abs: np.ndarray,
    metrics: Dict,
    match_tol_sec: float,
    first_samp: int,
    test_label: str,
):
    """
    Create the publication-style figure with 4 panels:

      A) Waveform overlay (z-scored) + vertical lines for events
      B) Event raster (ref on top, test on bottom)
      C) Jitter over time for matched events (ms) + tolerance bounds
      D) Histogram of |jitter| (ms) + tolerance line

    Note:
    - test_trace_aligned is aligned ONLY for waveform readability.
    - Events are plotted in absolute sample indices converted to the plotted window.
    """
    n_plot = len(t)

    z_ref = safe_zscore(ref_trace[:n_plot])
    z_test = safe_zscore(test_trace_aligned[:n_plot])

    # Convert absolute event sample indices to "relative to first_samp in file"
    ref_rel = ref_events_samp_abs - first_samp
    test_rel = test_events_samp_abs - first_samp
    matched_ref_rel = matched_ref_abs - first_samp
    matched_test_rel = matched_test_abs - first_samp

    # Utility: keep events that fall inside the plotted window
    def in_win(x):
        return x[(x >= 0) & (x < n_plot)]

    ref_in = in_win(ref_rel)
    test_in = in_win(test_rel)

    # Jitter arrays for panel C & D
    if len(matched_ref_rel) > 0:
        mask = (matched_ref_rel >= 0) & (matched_ref_rel < n_plot)
        mref = matched_ref_rel[mask]
        mtest = matched_test_rel[mask]

        jitter_sec = (mtest - mref) / sfreq
        jitter_ms = 1e3 * jitter_sec
        jitter_abs_ms = np.abs(jitter_ms)
        jitter_time_s = mref / sfreq
    else:
        jitter_ms = np.array([])
        jitter_abs_ms = np.array([])
        jitter_time_s = np.array([])

    tol_ms = 1e3 * match_tol_sec

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.1, 1.3, 1.3])

    # Panel A: waveform + event lines
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, z_ref, label="Ref: MNE ECG from REAL ECG")
    ax1.plot(t, z_test, label=test_label)

    for s in ref_in:
        ax1.axvline(s / sfreq, linestyle="--", linewidth=0.8, alpha=0.5)
    for s in test_in:
        ax1.axvline(s / sfreq, linestyle=":", linewidth=0.8, alpha=0.5)

    # Highlight matched reference events
    if len(matched_ref_rel) > 0:
        for s in in_win(matched_ref_rel):
            ax1.axvline(s / sfreq, linestyle="-", linewidth=1.2, alpha=0.35)

    ax1.set_title("A) Signal overlay (z) + events (ref dashed, test dotted; matched highlighted)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("a.u. (z-score)")
    ax1.legend(loc="upper right")
    ax1.set_xlim(t[0], t[-1])

    # Panel B: raster view
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    if len(ref_in) > 0:
        ax2.vlines(ref_in / sfreq, 0.65, 1.00, linewidth=1.2)
    if len(test_in) > 0:
        ax2.vlines(test_in / sfreq, 0.00, 0.35, linewidth=1.2)

    ax2.set_yticks([0.175, 0.825])
    ax2.set_yticklabels([test_label, "Ref (ECG)"])
    ax2.set_title("B) R-peak raster (windowed)")
    ax2.set_xlabel("Time (s)")

    # Panel C: jitter over time
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    if len(jitter_time_s) > 0:
        ax3.plot(jitter_time_s, jitter_ms, marker="o", linestyle="None")
        ax3.axhline(+tol_ms, linestyle="--", linewidth=1.0)
        ax3.axhline(-tol_ms, linestyle="--", linewidth=1.0)
        ax3.set_ylim(-max(1.1 * tol_ms, 5), max(1.1 * tol_ms, 5))
    ax3.set_title("C) Jitter for matched R-peaks (test - ref) [ms]")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Jitter (ms)")

    # Panel D: |jitter| histogram
    ax4 = fig.add_subplot(gs[3, 0])
    if len(jitter_abs_ms) > 0:
        ax4.hist(jitter_abs_ms, bins=30)
        ax4.axvline(tol_ms, linestyle="--", linewidth=1.0)
        ax4.set_xlim(0, max(tol_ms * 1.1, np.percentile(jitter_abs_ms, 99) * 1.1))
    ax4.set_title("D) |Jitter| distribution [ms] (dashed = match tolerance)")
    ax4.set_xlabel("|Jitter| (ms)")
    ax4.set_ylabel("Count")

    header_1 = (
        f"{subject_id} | sfreq={sfreq:.2f} Hz | match_tol={tol_ms:.0f} ms | "
        f"TP={metrics['TP']} FP={metrics['FP']} FN={metrics['FN']}"
    )
    header_2 = (
        f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f} miss_rate={metrics['miss_rate']:.3f} | "
        f"jitter(MAE)={metrics['jitter_mae_ms']:.2f} ms "
        f"p95(|·|)={metrics['jitter_p95_abs_ms']:.2f} ms"
    )
    fig.suptitle(f"{header_1}\n{header_2}", fontsize=11)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =============================================================================
# PLOTTING: ICA ranking + diagnostics
# =============================================================================
def plot_unsup_ic_ranking(
    out_png: str,
    subject_id: str,
    all_details_sorted: List[Dict],
    topk: int = 12
):
    """
    Bar plot of the top-K ICA components by unsupervised ECG-likeness score.
    """
    top = all_details_sorted[:max(1, min(topk, len(all_details_sorted)))]
    ics = [d["ic"] for d in top]
    scores = [d["score"] for d in top]

    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(top)), scores)
    plt.xticks(np.arange(len(top)), [f"IC{ic}" for ic in ics], rotation=0)
    plt.ylabel("Unsupervised ECG score")
    plt.title(f"{subject_id} | Unsupervised ICA score ranking (top {len(top)})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_unsup_ic_diagnostics(
    out_png: str,
    subject_id: str,
    sfreq: float,
    ic_signal: np.ndarray,
    unsup_details: Dict,
):
    """
    Multi-panel diagnostic figure for the chosen IC:
      A) time series (z) with detected 'peaks'
      B) |z| with spike threshold and detected spikes
      C/D) RR interval series + histogram (based on peak samples)
      E) autocorrelation with HR lag window
      F) PSD with HR band highlighted
      G) a text panel with numeric diagnostics

    This helps to understand WHY an IC got selected.
    """
    x = segment_for_scoring(ic_signal, sfreq, UNSUP_SCORE_SECONDS)
    z = safe_zscore(x)
    z_abs = np.abs(z)

    # Window for time-domain panels (A/B)
    n_diag = min(len(z), int(round(UNSUP_DIAG_SECONDS * sfreq)))
    t = np.arange(n_diag) / sfreq

    peaks = np.asarray(unsup_details.get("peaks_samples", []), dtype=int)
    spikes = np.asarray(unsup_details.get("spikes_samples", []), dtype=int)

    peaks_in = peaks[(peaks >= 0) & (peaks < n_diag)]
    spikes_in = spikes[(spikes >= 0) & (spikes < n_diag)]

    # RR stats (computed from peak indices)
    if len(peaks) >= 2:
        rr = np.diff(peaks) / sfreq
        rr_med = np.median(rr)
        rr_mad = np.median(np.abs(rr - rr_med))
    else:
        rr = np.array([], dtype=float)
        rr_mad = np.nan

    # Autocorrelation
    ac = autocorr_full(z)
    max_lag = min(len(ac) - 1, int(round(UNSUP_DIAG_MAX_AUTOCORR_LAG_SEC * sfreq)))
    lags = np.arange(max_lag + 1) / sfreq
    ac_plot = ac[:max_lag + 1]

    # HR band -> lag window
    lo_hz, hi_hz = HR_BAND_HZ
    min_period = 1.0 / hi_hz
    max_period = 1.0 / lo_hz

    # PSD (for visualization only)
    n = len(z)
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    psd = np.abs(np.fft.rfft(z - np.mean(z))) ** 2
    valid = freqs > 0
    freqs = freqs[valid]
    psd = psd[valid]
    psd_db = 10.0 * np.log10(psd + 1e-30)

    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs = fig.add_gridspec(5, 2, height_ratios=[1.3, 1.0, 1.0, 1.0, 1.0])

    # A) time course + peaks
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, z[:n_diag], label="IC z(t)")
    if len(peaks_in) > 0:
        ax1.plot(peaks_in / sfreq, z[peaks_in], "o", linestyle="None", label="Peaks (|z| prominence)")
    ax1.set_title("A) Selected UNSUP IC time course (z) + detected peaks")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("z")
    ax1.legend(loc="upper right")

    # B) |z| + spikes
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax2.plot(t, z_abs[:n_diag], label="|IC z(t)|")
    ax2.axhline(SPIKE_Z_ABS_THRESHOLD, linestyle="--", linewidth=1.0, label=f"Spike thr |z|≥{SPIKE_Z_ABS_THRESHOLD}")
    if len(spikes_in) > 0:
        ax2.plot(spikes_in / sfreq, z_abs[spikes_in], "o", linestyle="None", label="Spikes")
    ax2.set_title("B) Spike strength view (|z| + threshold + detected spikes)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("|z|")
    ax2.legend(loc="upper right")

    # C) RR series
    ax3 = fig.add_subplot(gs[2, 0])
    if len(rr) > 0:
        ax3.plot(rr, marker="o", linestyle="-")
        ax3.axhline(np.median(rr), linestyle="--", linewidth=1.0)
    ax3.set_title("C) RR intervals from peaks (sec)")
    ax3.set_xlabel("Beat index")
    ax3.set_ylabel("RR (s)")

    # D) RR histogram
    ax4 = fig.add_subplot(gs[2, 1])
    if len(rr) > 0:
        ax4.hist(rr, bins=20)
    ax4.set_title(f"D) RR histogram | rrMAD={rr_mad:.4f}s")
    ax4.set_xlabel("RR (s)")
    ax4.set_ylabel("Count")

    # E) autocorrelation
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.plot(lags, ac_plot)
    ax5.axvspan(min_period, max_period, alpha=0.2, label="HR lag window")
    ax5.set_title("E) Autocorrelation (normalized)")
    ax5.set_xlabel("Lag (s)")
    ax5.set_ylabel("ACF")
    ax5.legend(loc="upper right")

    # F) PSD
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.plot(freqs, psd_db)
    ax6.axvspan(HR_BAND_HZ[0], HR_BAND_HZ[1], alpha=0.2, label="HR band (Hz)")
    ax6.set_title("F) PSD of IC (dB) + HR band")
    ax6.set_xlabel("Frequency (Hz)")
    ax6.set_ylabel("Power (dB)")
    ax6.set_xlim(0, min(10.0, np.max(freqs)))
    ax6.legend(loc="upper right")

    # G) Text panel
    ax7 = fig.add_subplot(gs[4, :])
    ax7.axis("off")
    txt = (
        f"UNSUP IC diagnostics | subject={subject_id} | sfreq={sfreq:.2f} Hz\n"
        f"IC={unsup_details.get('ic','NA')} | score={unsup_details['score']:.3f} | bpm_est={unsup_details['bpm_est']:.1f}\n"
        f"p_hr={unsup_details['p_hr']:.3f} | p_ac={unsup_details['p_ac']:.3f} | p_peaks={unsup_details['p_peaks']:.3f} | "
        f"p_kurt={unsup_details['p_kurt']:.3f} | p_spike={unsup_details['p_spike']:.3f}\n"
        f"kurtosis={unsup_details['kurtosis']:.2f} | n_peaks={unsup_details['n_peaks']} | rrMAD={unsup_details['rr_mad_sec']:.4f}s\n"
        f"spike_rate/min={unsup_details['spike_rate_per_min']:.1f} | n_spikes={unsup_details['n_spikes']} | "
        f"frac(|z|≥{SPIKE_Z_ABS_THRESHOLD:.1f})={unsup_details['frac_absz_above_thr']:.4f}\n"
        f"Scoring window: {UNSUP_SCORE_SECONDS if UNSUP_SCORE_SECONDS is not None else 'FULL'} sec"
    )
    ax7.text(0.01, 0.95, txt, va="top", ha="left", fontsize=10)

    fig.suptitle(f"{subject_id} | Unsupervised ICA diagnostic panel", fontsize=12)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

# Discover files (FIF + CTF)
fif_files = [
    p
    for p in Path(DATASET_ROOT).rglob("*_meg.fif")
    if p.is_file() and "derivatives" not in str(p) and ".git" not in str(p)
]
ctf_files = [
    p
    for p in Path(DATASET_ROOT).rglob("*.ds")
    if p.is_dir() and "derivatives" not in str(p) and ".git" not in str(p)
]
data_paths = sorted(fif_files + ctf_files)

print(f"Found {len(data_paths)} FIF/CTF files")

def process_file(data_path: Path):
    subject_id = data_path.stem
    print(f"\nProcessing {subject_id}")
    
    # -------------------------------------------------------------------------
    # 1) Load raw data
    # -------------------------------------------------------------------------
    try:
        if data_path.suffix == ".fif":
            raw = mne.io.read_raw_fif(data_path, preload=True, verbose=False)
        elif data_path.suffix == ".ds":
            raw = mne.io.read_raw_ctf(data_path, preload=True, verbose=False)
        else:
            print(f"  Unsupported file format: {data_path}")
            return None
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return None
    
    sfreq = float(raw.info["sfreq"])
    
    # -------------------------------------------------------------------------
    # 2) Find REAL ECG channel (needed for reference)
    # -------------------------------------------------------------------------
    # If there's no ECG channel, we cannot benchmark event metrics in this script.
    ecg_picks = mne.pick_types(raw.info, ecg=True)
    if len(ecg_picks) == 0:
        print("  No real ECG channel found -> skipping")
        return None
    
    ecg_idx = int(ecg_picks[0])
    ecg_ch_name = raw.ch_names[ecg_idx]
    ecg_raw_real_full = raw.get_data(picks=[ecg_idx])[0]
    
    # -------------------------------------------------------------------------
    # 3) Reference: find_ecg_events using REAL ECG channel
    #    - returns events + an "MNE ECG trace" (a processed helper signal)
    # -------------------------------------------------------------------------
    try:
        ecg_events_ref, _, _, ecg_mne_from_ecg = mne.preprocessing.find_ecg_events(
            raw,
            ch_name=ecg_ch_name,
            return_ecg=True,
            verbose=False,
        )
        ecg_mne_from_ecg = ecg_mne_from_ecg[0]  # shape (n_times,)
        ref_events_samp = extract_event_samples(ecg_events_ref)  # absolute samples
    except Exception as e:
        print(f"  ERROR find_ecg_events using real ECG: {e}")
        return None
    
    # -------------------------------------------------------------------------
    # 4) Baseline test: find_ecg_events on MEG-only raw (MNE's baseline)
    # -------------------------------------------------------------------------
    raw_meg_only = raw.copy().pick_types(meg=True, eeg=False, eog=False, ecg=False, stim=False)
    
    try:
        ecg_events_test, _, _, ecg_mne_from_meg = mne.preprocessing.find_ecg_events(
            raw_meg_only,
            ch_name=None,        # let MNE pick a good ECG projection from MEG
            return_ecg=True,
            verbose=False,
        )
        ecg_mne_from_meg = ecg_mne_from_meg[0]
        test_events_samp = extract_event_samples(ecg_events_test)  # absolute samples (same raw)
    except Exception as e:
        print(f"  ERROR find_ecg_events deriving from MEG: {e}")
        return None
    
    # -------------------------------------------------------------------------
    # 5) NEW: Fit ICA ONCE on MEG-only and select an ECG-like IC UNSUPERVISED
    # -------------------------------------------------------------------------
    try:
        _ica, sources = fit_ica_and_get_sources(raw_meg_only)
    except Exception as e:
        print(f"  ERROR ICA fit: {e}")
        return None
    
    unsup_ic, unsup_details, all_details_sorted = pick_best_ic_unsupervised(sources, sfreq)
    unsup_details["ic"] = int(unsup_ic)
    
    # IC time course in sample space of raw_meg_only (same n_times as raw_meg_only)
    ecg_ica_unsup = sources[unsup_ic, :]
    if unsup_details.get("flip_for_score", False):
        ecg_ica_unsup = -ecg_ica_unsup
    
    # -------------------------------------------------------------------------
    # 6) Detect ECG events from the ICA-derived IC trace
    #    Strategy:
    #      - Put IC into RawArray as an ECG channel
    #      - Run find_ecg_events on that RawArray
    #
    # IMPORTANT:
    # - RawArray events start at sample 0.
    # - We convert its event samples to absolute samples by adding raw.first_samp.
    # -------------------------------------------------------------------------
    try:
        info = mne.create_info(ch_names=["ICA_ECG"], sfreq=sfreq, ch_types=["ecg"])
        raw_ica_ecg = mne.io.RawArray(ecg_ica_unsup[np.newaxis, :], info, verbose=False)
    
        ecg_events_ica, _, _, ecg_mne_from_ica = mne.preprocessing.find_ecg_events(
            raw_ica_ecg,
            ch_name="ICA_ECG",
            return_ecg=True,
            verbose=False,
        )
        ecg_mne_from_ica = ecg_mne_from_ica[0]  # "MNE ECG trace" derived from ICA channel
    
        ica_events_samp = extract_event_samples(ecg_events_ica)          # relative to RawArray (starts at 0)
        ica_events_samp_abs = ica_events_samp + raw.first_samp          # convert to absolute samples
    except Exception as e:
        print(f"  ERROR find_ecg_events on ICA-derived trace: {e}")
        return None
    
    # -------------------------------------------------------------------------
    # 7) Length match for waveform comparisons (plots/correlation only)
    # -------------------------------------------------------------------------
    n = min(
        len(ecg_raw_real_full),
        len(ecg_mne_from_ecg),
        len(ecg_mne_from_meg),
        len(ecg_mne_from_ica),
    )
    if n < int(round(2 * sfreq)):
        print("  Too short after length matching -> skipping")
        return None
    
    ecg_raw_real = ecg_raw_real_full[:n]
    ecg_mne_from_ecg = ecg_mne_from_ecg[:n]
    ecg_mne_from_meg = ecg_mne_from_meg[:n]
    ecg_mne_from_ica = ecg_mne_from_ica[:n]
    
    # -------------------------------------------------------------------------
    # 8) Waveform alignment (SECONDARY; for plots only)
    #    Baseline: align MNE-from-MEG trace to reference
    # -------------------------------------------------------------------------
    lag_mne = best_lag_via_xcorr(ecg_mne_from_ecg, ecg_mne_from_meg)
    ecg_mne_from_meg_aligned = shift_with_zeros(ecg_mne_from_meg, lag_mne)
    
    r_mne_before, _ = pearsonr(ecg_mne_from_ecg, ecg_mne_from_meg)
    r_mne_after, _ = pearsonr(ecg_mne_from_ecg, ecg_mne_from_meg_aligned)
    
    # -------------------------------------------------------------------------
    # 9) Waveform alignment for ICA-derived MNE ECG (SECONDARY; plots only)
    #    Optional sign flip for nicer overlay/correlation (benchmark only).
    # -------------------------------------------------------------------------
    if np.corrcoef(ecg_mne_from_ica, ecg_mne_from_ecg)[0, 1] < 0:
        ecg_mne_from_ica = -ecg_mne_from_ica  # benchmark-only sign flip
    
    lag_ica = best_lag_via_xcorr(ecg_mne_from_ecg, ecg_mne_from_ica)
    ecg_mne_from_ica_aligned = shift_with_zeros(ecg_mne_from_ica, lag_ica)
    
    r_ica_before, _ = pearsonr(ecg_mne_from_ecg, ecg_mne_from_ica)
    r_ica_after, _ = pearsonr(ecg_mne_from_ecg, ecg_mne_from_ica_aligned)
    
    # -------------------------------------------------------------------------
    # 10) PRIMARY: Event matching metrics (NO lag correction)
    # -------------------------------------------------------------------------
    # Baseline events vs reference
    matched_ref_mne, matched_test_mne, unmatched_ref_mne, unmatched_test_mne = match_events_one_to_one(
        ref_samp=ref_events_samp,
        test_samp=test_events_samp,
        sfreq=sfreq,
        tol_sec=MATCH_TOL_SEC,
    )
    metrics_mne = compute_detection_metrics(
        matched_ref=matched_ref_mne,
        matched_test=matched_test_mne,
        unmatched_ref=unmatched_ref_mne,
        unmatched_test=unmatched_test_mne,
        sfreq=sfreq,
    )
    
    # ICA-derived events vs reference
    matched_ref_ica, matched_test_ica, unmatched_ref_ica, unmatched_test_ica = match_events_one_to_one(
        ref_samp=ref_events_samp,
        test_samp=ica_events_samp_abs,  # already absolute
        sfreq=sfreq,
        tol_sec=MATCH_TOL_SEC,
    )
    metrics_ica = compute_detection_metrics(
        matched_ref=matched_ref_ica,
        matched_test=matched_test_ica,
        unmatched_ref=unmatched_ref_ica,
        unmatched_test=unmatched_test_ica,
        sfreq=sfreq,
    )
    
    duration_sec = n / sfreq
    fp_per_min_mne = metrics_mne["FP"] / (duration_sec / 60.0) if duration_sec > 0 else np.nan
    fp_per_min_ica = metrics_ica["FP"] / (duration_sec / 60.0) if duration_sec > 0 else np.nan
    
    # -------------------------------------------------------------------------
    # 11) Figures
    # -------------------------------------------------------------------------
    n_plot = min(n, int(round(PLOT_SECONDS * sfreq)))
    t = np.arange(n_plot) / sfreq
    
    fig_path_baseline = os.path.join(OUTPUT_DIR, f"{subject_id}_pub_MNEfromMEG.png")
    make_publication_figure(
        out_png=fig_path_baseline,
        subject_id=subject_id,
        sfreq=sfreq,
        t=t,
        ref_trace=ecg_mne_from_ecg,
        test_trace_aligned=ecg_mne_from_meg_aligned,
        ref_events_samp_abs=ref_events_samp,
        test_events_samp_abs=test_events_samp,
        matched_ref_abs=matched_ref_mne,
        matched_test_abs=matched_test_mne,
        metrics=metrics_mne,
        match_tol_sec=MATCH_TOL_SEC,
        first_samp=raw.first_samp,
        test_label="Test: MNE-from-MEG",
    )
    
    fig_path_ica = os.path.join(OUTPUT_DIR, f"{subject_id}_pub_ICAunsup.png")
    make_publication_figure(
        out_png=fig_path_ica,
        subject_id=subject_id,
        sfreq=sfreq,
        t=t,
        ref_trace=ecg_mne_from_ecg,
        test_trace_aligned=ecg_mne_from_ica_aligned,
        ref_events_samp_abs=ref_events_samp,
        test_events_samp_abs=ica_events_samp_abs,
        matched_ref_abs=matched_ref_ica,
        matched_test_abs=matched_test_ica,
        metrics=metrics_ica,
        match_tol_sec=MATCH_TOL_SEC,
        first_samp=raw.first_samp,
        test_label=f"Test: ICA-unsup (IC{unsup_ic})",
    )
    
    # ICA ranking + diagnostics
    rank_png = os.path.join(OUTPUT_DIR, f"{subject_id}_ICAunsup_ranking.png")
    diag_png = os.path.join(OUTPUT_DIR, f"{subject_id}_ICAunsup_diag_IC{unsup_ic}.png")
    
    try:
        plot_unsup_ic_ranking(rank_png, subject_id, all_details_sorted, topk=UNSUP_SCORE_TOPK_PLOT)
    except Exception as e:
        print(f"  WARNING: could not write ICA ranking plot: {e}")
        rank_png = ""
    
    try:
        plot_unsup_ic_diagnostics(
            diag_png,
            subject_id,
            sfreq,
            ic_signal=ecg_ica_unsup,
            unsup_details=unsup_details,
        )
    except Exception as e:
        print(f"  WARNING: could not write ICA diagnostics plot: {e}")
        diag_png = ""
    
    # -------------------------------------------------------------------------
    # 12) Store summary row (baseline + ICA)
    # -------------------------------------------------------------------------
    result = {
        "subject": subject_id,
        "file": str(data_path),
        "sfreq_hz": sfreq,
        "duration_sec": duration_sec,
        "ecg_channel": ecg_ch_name,
    
        # Baseline waveform (secondary)
        "baseline_waveform_lag_samples": lag_mne,
        "baseline_waveform_lag_sec": lag_mne / sfreq,
        "baseline_r_waveform_before": r_mne_before,
        "baseline_r_waveform_after": r_mne_after,
    
        # Baseline event metrics (primary)
        "baseline_match_tol_ms": 1e3 * MATCH_TOL_SEC,
        "baseline_TP": metrics_mne["TP"],
        "baseline_FP": metrics_mne["FP"],
        "baseline_FN": metrics_mne["FN"],
        "baseline_precision": metrics_mne["precision"],
        "baseline_recall": metrics_mne["recall"],
        "baseline_f1": metrics_mne["f1"],
        "baseline_miss_rate": metrics_mne["miss_rate"],
        "baseline_fp_per_min": fp_per_min_mne,
        "baseline_jitter_mean_ms_signed": metrics_mne["jitter_mean_ms_signed"],
        "baseline_jitter_std_ms_signed": metrics_mne["jitter_std_ms_signed"],
        "baseline_jitter_mae_ms": metrics_mne["jitter_mae_ms"],
        "baseline_jitter_median_abs_ms": metrics_mne["jitter_median_abs_ms"],
        "baseline_jitter_p95_abs_ms": metrics_mne["jitter_p95_abs_ms"],
        "baseline_pubgrade_figure": fig_path_baseline,
    
        # ICA waveform (secondary)
        "ica_unsup_ic": unsup_ic,
        "ica_waveform_lag_samples": lag_ica,
        "ica_waveform_lag_sec": lag_ica / sfreq,
        "ica_r_waveform_before": r_ica_before,
        "ica_r_waveform_after": r_ica_after,
    
        # ICA event metrics (primary)
        "ica_match_tol_ms": 1e3 * MATCH_TOL_SEC,
        "ica_TP": metrics_ica["TP"],
        "ica_FP": metrics_ica["FP"],
        "ica_FN": metrics_ica["FN"],
        "ica_precision": metrics_ica["precision"],
        "ica_recall": metrics_ica["recall"],
        "ica_f1": metrics_ica["f1"],
        "ica_miss_rate": metrics_ica["miss_rate"],
        "ica_fp_per_min": fp_per_min_ica,
        "ica_jitter_mean_ms_signed": metrics_ica["jitter_mean_ms_signed"],
        "ica_jitter_std_ms_signed": metrics_ica["jitter_std_ms_signed"],
        "ica_jitter_mae_ms": metrics_ica["jitter_mae_ms"],
        "ica_jitter_median_abs_ms": metrics_ica["jitter_median_abs_ms"],
        "ica_jitter_p95_abs_ms": metrics_ica["jitter_p95_abs_ms"],
    
        # ICA unsupervised score components
        "ica_unsup_score": unsup_details["score"],
        "ica_unsup_bpm_est": unsup_details["bpm_est"],
        "ica_unsup_p_hr": unsup_details["p_hr"],
        "ica_unsup_p_ac": unsup_details["p_ac"],
        "ica_unsup_p_peaks": unsup_details["p_peaks"],
        "ica_unsup_p_kurt": unsup_details["p_kurt"],
        "ica_unsup_p_spike": unsup_details["p_spike"],
        "ica_unsup_kurtosis": unsup_details["kurtosis"],
        "ica_unsup_rr_mad_sec": unsup_details["rr_mad_sec"],
        "ica_unsup_spike_rate_per_min": unsup_details["spike_rate_per_min"],
        "ica_unsup_frac_absz_above_thr": unsup_details["frac_absz_above_thr"],
        "ica_unsup_n_spikes": unsup_details["n_spikes"],
    
        # Figure paths
        "ica_pubgrade_figure": fig_path_ica,
        "ica_score_ranking_plot": rank_png,
        "ica_diagnostics_plot": diag_png,
    }
    
    print(
        f"  Done | baseline(MNE-from-MEG): F1={metrics_mne['f1']:.3f}, miss={metrics_mne['miss_rate']:.3f}, "
        f"jMAE={metrics_mne['jitter_mae_ms']:.2f}ms, FP/min={fp_per_min_mne:.2f} | "
        f"ICA-unsup(IC{unsup_ic}): F1={metrics_ica['f1']:.3f}, miss={metrics_ica['miss_rate']:.3f}, "
        f"jMAE={metrics_ica['jitter_mae_ms']:.2f}ms, FP/min={fp_per_min_ica:.2f} | "
        f"ICA score={unsup_details['score']:.3f}, bpm≈{unsup_details['bpm_est']:.1f}"
    )
    return result
    

# Run processing (serial or parallel)
if N_JOBS == 1:
    rows = [r for r in (process_file(p) for p in data_paths) if r is not None]
else:
    rows = Parallel(n_jobs=N_JOBS)(delayed(process_file)(p) for p in data_paths)
    rows = [r for r in rows if r is not None]

    
# =============================================================================
# SAVE CSV SUMMARY
# =============================================================================
df = pd.DataFrame(rows)
csv_path = os.path.join(OUTPUT_DIR, "ecg_event_validation_with_ica_unsup_summary.csv")
df.to_csv(csv_path, index=False)

print("\n=== SUMMARY (BASELINE vs ICA) ===")
if len(df) > 0:
    show_cols = [
        "baseline_precision", "baseline_recall", "baseline_f1", "baseline_miss_rate",
        "baseline_jitter_mae_ms", "baseline_fp_per_min",
        "ica_precision", "ica_recall", "ica_f1", "ica_miss_rate",
        "ica_jitter_mae_ms", "ica_fp_per_min",
        "ica_unsup_score", "ica_unsup_bpm_est", "ica_unsup_spike_rate_per_min"
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].describe())
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved figures in: {OUTPUT_DIR}")
else:
    print("No valid files processed.")

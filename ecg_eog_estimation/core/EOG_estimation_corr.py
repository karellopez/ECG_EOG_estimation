"""
MEG → EOG estimation / comparison script (supervised benchmark + MEG-only production)
with a *graceful fallback* when no REAL EOG channel is present.

-------------------------------------------------------------------------------
PURPOSE
-------------------------------------------------------------------------------
This script compares multiple strategies to estimate an EOG-like (blink) trace
from MEG data, and (when real EOG exists) benchmarks them against the real EOG.

We produce plots and a CSV summary for each MEG FIF file.

-------------------------------------------------------------------------------
METHODS
-------------------------------------------------------------------------------
When a REAL EOG channel exists (benchmark mode):
  1) Real EOG (raw)                           [reference for sanity]
  2) Real EOG processed (band-pass + z-score) [reference for scoring/metrics]
  3) Global PCA (MEG-only)                    [unsupervised; uses all MEG]
  4) Frontal PCA supervised                   [uses real EOG to rank sensors]
  5) ICA supervised                           [uses real EOG to select IC]
  6) ICA unsupervised                         [MEG-only selection + sign convention]

When NO REAL EOG channel exists (MEG-only mode):
  1) Global PCA (MEG-only)
  2) Frontal PCA unsupervised
  3) ICA unsupervised
and we generate plots + CSV row with only MEG-only metrics (no correlations to real EOG).

-------------------------------------------------------------------------------
CRITICAL DESIGN DECISION
-------------------------------------------------------------------------------
ICA is fit ONCE per file, producing a single "sources" array (n_ic, n_times).
Both supervised and unsupervised ICA selections are taken from that same array.
This prevents "IC0 mismatch" across repeated ICA runs.

-------------------------------------------------------------------------------
DEPENDENCIES
-------------------------------------------------------------------------------
- mne, numpy, pandas, matplotlib, scipy
"""

import os
from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import correlate, find_peaks
from scipy.stats import pearsonr, kurtosis

# =============================================================================
# CONFIGURATION
# =============================================================================

# Root folder to search recursively for "*_meg.fif"
DATASET_ROOT = "/Users/karelo/Development/datasets/ds_small"

# Where plots + CSV will be written
OUTPUT_DIR = "/Users/karelo/Development/datasets/ds_small/derivatives/eog_compare_methods_supervised_unsupervised_v7"

# Seconds shown in each plot
PLOT_SECONDS = 60

# Band-pass used for "blink-sensitive" content (applied to reference and proxies)
EOG_L_FREQ = 1.0
EOG_H_FREQ = 10.0

# --------------------------
# PCA settings
# --------------------------
# Global PCA: choose top N variance channels for PCA
N_TOP_CHANNELS_FOR_PCA = 30
FALLBACK_USE_ALL_MEG = True

# Supervised "frontal" PCA: rank MEG channels by abs corr to real EOG and PCA the top K
FRONTAL_TOPK_BY_CORR = 40

# Unsupervised frontal PCA channel selection method (MEG-only)
#   "layout": pick channels with highest "y" coordinate in device space (front)
#   "regex" : pick channels by name pattern
#   "variance": pick top variance channels in the EOG band (not truly "frontal", but simple)
FRONTAL_PCA_UNSUPERVISED_MODE = "layout"  # "layout" | "regex" | "variance"
FRONTAL_TOPK_UNSUPERVISED = 40
FRONTAL_REGEX = r"MEG0(1|2|3)"  # used if mode == "regex"

# --------------------------
# ICA settings
# --------------------------
ICA_N_COMPONENTS = 0.99
ICA_METHOD = "fastica"
ICA_RANDOM_STATE = 97
ICA_MAX_ITER = 1000

# Score window for unsupervised ICA heuristics (None uses full data)
UNSUP_SCORE_SECONDS = 120

# Unsupervised ICA selection mode:
#   "heuristic": choose IC by blink-likeness score
#   "fixed": always choose a fixed index (useful for debugging or reproducibility tests)
ICA_UNSUP_MODE = "heuristic"  # "heuristic" | "fixed"
ICA_UNSUP_FIXED_IC = 0

# Sign convention for unsupervised IC (MEG-only):
#   "frontal_proxy": flip sign to correlate positively with MEG-only frontal PCA proxy
#   "peak_polarity": flip sign so the strongest transient peaks tend to be positive
UNSUP_SIGN_MODE = "peak_polarity"  # "frontal_proxy" | "peak_polarity"

# --------------------------
# Peak/heuristic parameters for blink-likeness
# --------------------------
BLINK_MIN_PER_MIN = 2
BLINK_MAX_PER_MIN = 80
PEAK_PROMINENCE = 1.0
PEAK_MIN_DISTANCE_SEC = 0.05

# Blink morphology heuristics (typical vertical EOG has strong POSITIVE blinks)
EOG_POS_SPIKE_Z_THR = 2.5
EOG_NEG_SPIKE_Z_THR = 2.0
EOG_POS_SPIKE_MIN_PER_MIN = 4
EOG_POS_SPIKE_MAX_PER_MIN = 80
EOG_NEG_SPIKE_MAX_PER_MIN = 6

# Weights for the unsupervised ICA score
W_EOG_KURT = 0.25
W_EOG_RATE = 0.25
W_EOG_POS_SPIKE = 0.40
W_EOG_NEG_PENALTY = 0.10

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# HELPERS: normalization & alignment
# =============================================================================
def safe_zscore(x: np.ndarray) -> np.ndarray:
    """
    Z-score a 1D array safely.

    Why "safe"?
    - If a trace is constant (std = 0) or contains NaNs that cause std = NaN,
      z-scoring would fail or produce NaNs.
    - In that case, we fall back to mean-centering only.

    Returns:
      z-scored (or mean-centered) array of same length.
    """
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sd = np.std(x)
    if sd == 0 or np.isnan(sd):
        return x - mu
    return (x - mu) / sd


def best_lag_via_xcorr(x_ref: np.ndarray, y: np.ndarray) -> int:
    """
    Find the integer lag (in samples) that maximizes cross-correlation between y and x_ref.

    This is used ONLY for waveform alignment in plots and for reporting a single correlation
    value that is robust to small temporal offsets.

    Important:
    - This is not event-level alignment.
    - We do not alter the raw data, only the plotted/compared proxy signals.

    Returns:
      lag (int). Positive means shifting y LEFT by 'lag' samples.
    """
    n = min(len(x_ref), len(y))
    x = x_ref[:n] - np.mean(x_ref[:n])
    yy = y[:n] - np.mean(y[:n])
    c = correlate(yy, x, mode="full")
    lags = np.arange(-n + 1, n)
    return int(lags[np.argmax(c)])


def shift_with_zeros(y: np.ndarray, lag: int) -> np.ndarray:
    """
    Shift a 1D array by 'lag' samples and pad with zeros to keep same length.

    lag > 0: shift y LEFT  (drop first lag samples, pad zeros at end)
    lag < 0: shift y RIGHT (pad zeros at start, drop last k samples)

    Returns:
      shifted array with same length as input.
    """
    y = np.asarray(y)
    if lag > 0:
        return np.concatenate([y[lag:], np.zeros(lag)])
    elif lag < 0:
        k = -lag
        return np.concatenate([np.zeros(k), y[:-k]])
    return y.copy()


# =============================================================================
# HELPERS: EOG reference processing
# =============================================================================
def process_eog_trace(x: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    """
    Standardize an EOG-like 1D trace for fair comparison.

    Steps:
    1) Band-pass filter to blink band (EOG_L_FREQ..EOG_H_FREQ).
    2) Z-score result.

    Notes:
    - We use mne.filter.filter_data for a stable FIR filter.
    - We always return a 1D array.
    """
    x = np.asarray(x, dtype=float)
    xf = mne.filter.filter_data(
        x[np.newaxis, :],
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design="firwin",
        verbose=False,
    )[0]
    return safe_zscore(xf)


def pick_prefer_vertical_eog(raw: mne.io.BaseRaw) -> int:
    """
    Choose a "best guess" EOG channel index when multiple EOG channels exist.

    Rationale:
    - Vertical EOG is generally most blink-sensitive.
    - Depending on acquisition systems, naming varies.

    Strategy:
    - If any EOG channels exist, try to find a name matching common "vertical" patterns.
    - Otherwise return the first EOG channel.
    - If no EOG channels exist, return -1.
    """
    eog_picks = mne.pick_types(raw.info, eog=True)
    if len(eog_picks) == 0:
        return -1

    names = [raw.ch_names[i].lower() for i in eog_picks]
    for key in ["veog", "eog061", "eog1", "eog_v", "v-eog", "eogv"]:
        for idx, nm in zip(eog_picks, names):
            if key in nm:
                return int(idx)

    return int(eog_picks[0])


# =============================================================================
# HELPERS: PCA estimators
# =============================================================================
def pca_first_component(data_ch_by_time: np.ndarray) -> np.ndarray:
    """
    Compute the first principal component time-course from a channels×time matrix.

    Implementation:
    - Mean-center each channel.
    - SVD on X (channels×time) -> Vt contains time components.
    - PC1 is Vt[0, :].

    Sign convention:
    - PCA sign is arbitrary. We flip PC1 so it correlates positively with the mean
      across channels (a weak but consistent heuristic).

    Returns:
      pc1 time-course (1D array).
    """
    X = np.asarray(data_ch_by_time, dtype=float)
    X = X - np.mean(X, axis=1, keepdims=True)

    # SVD: X = U S Vt. Vt rows are time-domain components.
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = Vt[0, :]

    # Stabilize sign using correlation with mean signal
    mean_sig = np.mean(X, axis=0)
    if np.corrcoef(pc1, mean_sig)[0, 1] < 0:
        pc1 = -pc1

    return pc1


def build_synth_eog_pca_all(raw_meg: mne.io.BaseRaw) -> np.ndarray:
    """
    Build a MEG-only synthetic EOG trace using a "global PCA" approach.

    Steps:
    1) Band-pass MEG data to blink band.
    2) Compute channel variance and pick top N channels (high variance tends to include blinks).
    3) Return PC1 across selected channels.

    Returns:
      1D array: synthetic EOG (raw PC1, not yet processed with process_eog_trace).
    """
    tmp = raw_meg.copy().filter(EOG_L_FREQ, EOG_H_FREQ, fir_design="firwin", verbose=False)
    data = tmp.get_data()
    if data.size == 0:
        raise RuntimeError("No MEG data available.")

    ch_var = np.var(data, axis=1)
    order = np.argsort(ch_var)[::-1]

    if len(order) >= N_TOP_CHANNELS_FOR_PCA:
        picks = order[:N_TOP_CHANNELS_FOR_PCA]
    else:
        if not FALLBACK_USE_ALL_MEG:
            raise RuntimeError("Not enough channels for PCA subset.")
        picks = order

    return pca_first_component(data[picks, :])


def build_synth_eog_pca_frontal_supervised(raw_meg: mne.io.BaseRaw, eog_ref_proc: np.ndarray) -> np.ndarray:
    """
    Build a synthetic EOG trace using a supervised "frontal PCA" approach.

    Why "supervised"?
    - We use real EOG reference to rank MEG channels by how EOG-like they are.
    - This is NOT MEG-only production; it's used for benchmarking.

    Steps:
    1) Filter MEG to blink band.
    2) Compute abs correlation of each MEG channel with processed real EOG reference.
    3) Select top K channels and compute PC1 across them.

    Returns:
      1D array: synthetic EOG (raw PC1 time-course, unprocessed).
    """
    tmp = raw_meg.copy().filter(EOG_L_FREQ, EOG_H_FREQ, fir_design="firwin", verbose=False)
    data = tmp.get_data()
    if data.size == 0:
        raise RuntimeError("No MEG data available.")

    n = min(data.shape[1], len(eog_ref_proc))
    X = data[:, :n]
    y = eog_ref_proc[:n]

    # Efficient abs correlation (dot products on demeaned, normalized vectors)
    corrs = np.zeros(X.shape[0], dtype=float)
    y0 = y - np.mean(y)
    ysd = np.std(y0) + 1e-12
    for i in range(X.shape[0]):
        xi = X[i, :] - np.mean(X[i, :])
        xsd = np.std(xi) + 1e-12
        corrs[i] = np.abs(np.dot(xi, y0) / (len(y0) * xsd * ysd))

    order = np.argsort(corrs)[::-1]
    k = min(FRONTAL_TOPK_BY_CORR, len(order))
    picks = order[:k]

    return pca_first_component(X[picks, :])


def pick_frontal_channels_unsupervised(raw_meg: mne.io.BaseRaw) -> np.ndarray:
    """
    Pick a set of "frontal" MEG channels without using real EOG (MEG-only).

    Supported modes:
    - "layout": uses channel 3D locations and selects channels with largest y-coordinate.
               Assumption: +y corresponds roughly to "front" (depends on coordinate frame).
    - "regex":  selects channels whose names match FRONTAL_REGEX.
    - "variance": selects top variance channels in blink band (not necessarily frontal).

    Returns:
      1D int array of channel indices.
    """
    if FRONTAL_PCA_UNSUPERVISED_MODE == "layout":
        picks = mne.pick_types(raw_meg.info, meg=True)

        # Gather valid positions
        pos = []
        keep = []
        for p in picks:
            loc = raw_meg.info["chs"][p]["loc"][:3]
            if np.any(np.isnan(loc)) or np.allclose(loc, 0):
                continue
            keep.append(p)
            pos.append(loc)

        # If positions are missing, fallback to first K MEG channels
        if len(keep) == 0:
            return np.array(picks[:min(len(picks), FRONTAL_TOPK_UNSUPERVISED)], dtype=int)

        pos = np.array(pos)
        y_coord = pos[:, 1]
        order = np.argsort(y_coord)[::-1]
        sel = np.array(keep, dtype=int)[order[:min(len(order), FRONTAL_TOPK_UNSUPERVISED)]]
        return sel

    if FRONTAL_PCA_UNSUPERVISED_MODE == "regex":
        picks = []
        for i, name in enumerate(raw_meg.ch_names):
            if re.match(FRONTAL_REGEX, name):
                picks.append(i)

        # If regex finds nothing, fallback to first K MEG channels
        if len(picks) == 0:
            picks = mne.pick_types(raw_meg.info, meg=True)

        return np.array(picks[:min(len(picks), FRONTAL_TOPK_UNSUPERVISED)], dtype=int)

    # "variance" mode
    tmp = raw_meg.copy().filter(EOG_L_FREQ, EOG_H_FREQ, fir_design="firwin", verbose=False)
    data = tmp.get_data()
    ch_var = np.var(data, axis=1)
    order = np.argsort(ch_var)[::-1]
    return np.array(order[:min(len(order), FRONTAL_TOPK_UNSUPERVISED)], dtype=int)


def build_synth_eog_pca_frontal_unsupervised(raw_meg: mne.io.BaseRaw) -> np.ndarray:
    """
    Build a MEG-only "frontal PCA" proxy.

    Steps:
    1) Filter MEG to blink band.
    2) Pick frontal channels via pick_frontal_channels_unsupervised().
    3) Compute PC1 across those channels.

    Returns:
      1D array: synthetic EOG (raw PC1 time-course, unprocessed).
    """
    tmp = raw_meg.copy().filter(EOG_L_FREQ, EOG_H_FREQ, fir_design="firwin", verbose=False)
    data = tmp.get_data()
    if data.size == 0:
        raise RuntimeError("No MEG data available.")

    picks = pick_frontal_channels_unsupervised(raw_meg)
    picks = picks[picks < data.shape[0]]
    if len(picks) == 0:
        raise RuntimeError("Unsupervised frontal picks returned no channels.")

    return pca_first_component(data[picks, :])


# =============================================================================
# HELPERS: ICA (FIT ONCE PER FILE)
# =============================================================================
def fit_ica_sources_once(raw_meg: mne.io.BaseRaw) -> np.ndarray:
    """
    Fit ICA ONCE per file and return the sources matrix (n_ic, n_times).

    Steps:
    1) Filter MEG to blink band (helps ICA focus on blink-like activity).
    2) Fit ICA.
    3) Extract IC time courses via ica.get_sources(...).get_data().

    Returns:
      sources: ndarray with shape (n_ic, n_times)

    Note:
    - We return only sources because this script's goal is to compare time courses.
    - If later you want to actually remove ICs, you should return the ICA object too.
    """
    tmp = raw_meg.copy().filter(EOG_L_FREQ, EOG_H_FREQ, fir_design="firwin", verbose=False)

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

    return sources


def pick_best_ic_supervised_from_sources(sources: np.ndarray, eog_ref_proc: np.ndarray) -> tuple[int, float]:
    """
    Supervised IC selection (benchmark only).

    We choose the IC whose time course has the highest ABS correlation with the
    processed real EOG reference.

    Returns:
      best_idx    : int, IC index
      best_abs_r  : float, the abs correlation score used for selection
    """
    n = min(sources.shape[1], len(eog_ref_proc))
    S = sources[:, :n]
    y = eog_ref_proc[:n]

    y0 = y - np.mean(y)
    ysd = np.std(y0) + 1e-12

    scores = np.zeros(S.shape[0], dtype=float)
    for i in range(S.shape[0]):
        si = S[i, :] - np.mean(S[i, :])
        ssd = np.std(si) + 1e-12
        scores[i] = np.abs(np.dot(si, y0) / (len(y0) * ssd * ysd))

    best_idx = int(np.argmax(scores))
    return best_idx, float(scores[best_idx])


# =============================================================================
# UNSUPERVISED ICA scoring (heuristics)
# =============================================================================
def eog_spike_strength_score(
        x: np.ndarray,
        sfreq: float,
        pos_thr_z: float = 3.5,
        neg_thr_z: float = 3.0,
        min_distance_sec: float = 0.20,
) -> dict:
    """
    Compute spike-based features for an EOG-like signal.

    A typical vertical blink trace shows strong positive peaks; negative peaks are rarer
    and often indicate other artifacts or inverted polarity.

    We compute:
      - positive peak rate above pos_thr_z
      - negative peak rate above neg_thr_z on -z (i.e., negative deflections)
      - a positive "reward" score p_pos based on plausible blink rate bounds
      - a negative "penalty" term p_neg_penalty that decreases if too many negative spikes exist

    Returns:
      dict with p_pos, p_neg_penalty and rates/counts.
    """
    z = safe_zscore(x)
    min_dist = max(1, int(round(min_distance_sec * sfreq)))

    pos_peaks, _ = find_peaks(z, height=pos_thr_z, distance=min_dist)
    neg_peaks, _ = find_peaks(-z, height=neg_thr_z, distance=min_dist)

    dur_sec = len(z) / sfreq
    if dur_sec <= 0:
        return dict(
            p_pos=0.0,
            p_neg_penalty=1.0,
            pos_rate_per_min=float("nan"),
            neg_rate_per_min=float("nan"),
            n_pos=0,
            n_neg=0,
        )

    pos_rate = (len(pos_peaks) / dur_sec) * 60.0
    neg_rate = (len(neg_peaks) / dur_sec) * 60.0

    # Reward a plausible positive blink spike rate
    if EOG_POS_SPIKE_MIN_PER_MIN <= pos_rate <= EOG_POS_SPIKE_MAX_PER_MIN:
        p_pos = 1.0
    else:
        if pos_rate < EOG_POS_SPIKE_MIN_PER_MIN:
            d = (EOG_POS_SPIKE_MIN_PER_MIN - pos_rate) / max(EOG_POS_SPIKE_MIN_PER_MIN, 1e-6)
        else:
            d = (pos_rate - EOG_POS_SPIKE_MAX_PER_MIN) / max(EOG_POS_SPIKE_MAX_PER_MIN, 1e-6)
        p_pos = float(np.exp(-3.0 * d))

    # Penalize too many negative spikes (likely non-blink artifact)
    if neg_rate <= EOG_NEG_SPIKE_MAX_PER_MIN:
        p_neg_penalty = 1.0
    else:
        d = (neg_rate - EOG_NEG_SPIKE_MAX_PER_MIN) / max(EOG_NEG_SPIKE_MAX_PER_MIN, 1e-6)
        p_neg_penalty = float(np.exp(-3.0 * d))

    return dict(
        p_pos=float(p_pos),
        p_neg_penalty=float(p_neg_penalty),
        pos_rate_per_min=float(pos_rate),
        neg_rate_per_min=float(neg_rate),
        n_pos=int(len(pos_peaks)),
        n_neg=int(len(neg_peaks)),
    )


def blink_likeness_score(ic: np.ndarray, sfreq: float) -> dict:
    """
    Compute an unsupervised "blink-likeness" score for a single ICA component.

    The goal is to identify an IC that:
      - is peaky (high kurtosis) -> blink transients
      - contains a plausible number of blink-like peaks per minute
      - has strong positive spikes (more likely vertical EOG blinks)
      - does not contain too many negative spikes

    Returns:
      dict with combined score + sub-scores + diagnostics.
    """
    # Optionally score only the first UNSUP_SCORE_SECONDS seconds
    if UNSUP_SCORE_SECONDS is not None:
        n_seg = min(len(ic), int(round(UNSUP_SCORE_SECONDS * sfreq)))
        x = ic[:n_seg]
    else:
        x = ic

    z = safe_zscore(x)

    # (1) Kurtosis: blink-like is typically peaky
    k = float(kurtosis(z, fisher=False, bias=False)) if len(z) > 10 else 0.0
    p_kurt = float(np.tanh((k - 3.0) / 5.0))
    p_kurt = max(0.0, p_kurt)

    # (2) Prominent peak rate (any polarity but measured on z with prominence)
    min_dist = max(1, int(round(PEAK_MIN_DISTANCE_SEC * sfreq)))
    peaks, _ = find_peaks(z, prominence=PEAK_PROMINENCE, distance=min_dist)

    dur_sec = len(z) / sfreq
    if dur_sec <= 0:
        rate = float("nan")
        p_rate = 0.0
    else:
        rate = (len(peaks) / dur_sec) * 60.0
        if BLINK_MIN_PER_MIN <= rate <= BLINK_MAX_PER_MIN:
            p_rate = 1.0
        else:
            if rate < BLINK_MIN_PER_MIN:
                d = (BLINK_MIN_PER_MIN - rate) / max(BLINK_MIN_PER_MIN, 1e-6)
            else:
                d = (rate - BLINK_MAX_PER_MIN) / max(BLINK_MAX_PER_MIN, 1e-6)
            p_rate = float(np.exp(-3.0 * d))

    # (3) Explicit spike features (positive reward, negative penalty)
    spike = eog_spike_strength_score(
        z,
        sfreq,
        pos_thr_z=EOG_POS_SPIKE_Z_THR,
        neg_thr_z=EOG_NEG_SPIKE_Z_THR,
        min_distance_sec=PEAK_MIN_DISTANCE_SEC,
    )

    p_pos = spike["p_pos"]
    p_neg_penalty = spike["p_neg_penalty"]

    # Combined score: weights are user-tunable constants above
    score = (
            W_EOG_KURT * p_kurt
            + W_EOG_RATE * p_rate
            + W_EOG_POS_SPIKE * p_pos
            + W_EOG_NEG_PENALTY * p_neg_penalty
    )

    return dict(
        score=float(score),
        p_kurt=float(p_kurt),
        p_rate=float(p_rate),
        p_pos=float(p_pos),
        p_neg_penalty=float(p_neg_penalty),
        rate_per_min=float(rate),
        kurtosis=float(k),
        n_prom_peaks=int(len(peaks)),
        pos_rate_per_min=float(spike["pos_rate_per_min"]),
        neg_rate_per_min=float(spike["neg_rate_per_min"]),
        n_pos_spikes=int(spike["n_pos"]),
        n_neg_spikes=int(spike["n_neg"]),
    )


def pick_best_ic_unsupervised_from_sources(
        sources: np.ndarray,
        sfreq: float,
        mode: str = "heuristic",
        fixed_ic: int = 0,
) -> tuple[int, dict]:
    """
    Unsupervised IC selection from the *same* sources array.

    Options:
      - mode="fixed": return fixed_ic (clipped to valid range). Useful for debugging.
      - mode="heuristic": compute blink_likeness_score for each IC and pick best score.

    Returns:
      (best_ic_index, details_dict)
    """
    if mode == "fixed":
        idx = int(np.clip(fixed_ic, 0, sources.shape[0] - 1))
        details = blink_likeness_score(sources[idx, :], sfreq)
        details["selection_mode"] = "fixed"
        return idx, details

    best_idx = -1
    best_score = -np.inf
    best_details = None

    for i in range(sources.shape[0]):
        details = blink_likeness_score(sources[i, :], sfreq)
        if details["score"] > best_score:
            best_score = details["score"]
            best_idx = i
            best_details = details

    best_details["selection_mode"] = "heuristic"
    return int(best_idx), best_details


# =============================================================================
# MEG-only sign convention for UNSUPERVISED IC
# =============================================================================
def apply_unsup_sign_convention_peak_polarity(ic: np.ndarray, sfreq: float) -> np.ndarray:
    """
    MEG-only sign convention: flip the IC so that "the strongest blink-like peaks"
    are predominantly POSITIVE.

    How:
    - Look for prominent positive and negative peaks on a scored segment.
    - Compare maximum positive peak height vs maximum negative deflection magnitude.
    - If negative dominates, flip sign.

    Why:
    - ICA sign is arbitrary.
    - For readability and consistent metrics, we prefer blinks to be positive.
    """
    if UNSUP_SCORE_SECONDS is not None:
        n_seg = min(len(ic), int(round(UNSUP_SCORE_SECONDS * sfreq)))
        x = ic[:n_seg]
    else:
        x = ic

    z = safe_zscore(x)

    min_dist = max(1, int(round(PEAK_MIN_DISTANCE_SEC * sfreq)))
    pos_peaks, _ = find_peaks(z, prominence=PEAK_PROMINENCE, distance=min_dist)
    neg_peaks, _ = find_peaks(-z, prominence=PEAK_PROMINENCE, distance=min_dist)

    pos_max = np.max(z[pos_peaks]) if len(pos_peaks) else 0.0
    neg_max = np.max((-z)[neg_peaks]) if len(neg_peaks) else 0.0

    if neg_max > pos_max:
        return -ic
    return ic


def apply_unsup_sign_convention_frontal_proxy(ic: np.ndarray, frontal_proxy: np.ndarray) -> np.ndarray:
    """
    MEG-only sign convention using a MEG-only reference proxy (frontal PCA).

    We flip the IC so that it correlates positively with the frontal proxy.

    Why:
    - The frontal proxy tends to have consistent blink polarity (depending on selection).
    - This can be more stable than peak polarity when the IC is noisy.

    Note:
    - Still MEG-only (no real EOG used).
    """
    n = min(len(ic), len(frontal_proxy))
    if n < 10:
        return ic

    r = np.corrcoef(ic[:n], frontal_proxy[:n])[0, 1]
    if np.isfinite(r) and r < 0:
        return -ic
    return ic


# =============================================================================
# PLOTTING: benchmark mode (real EOG exists)
# =============================================================================
def plot_methods_stacked_benchmark(
        out_png: str,
        t: np.ndarray,
        eog_raw_real: np.ndarray,
        eog_ref_proc: np.ndarray,
        traces: dict,
        subject_id: str,
        sfreq: float,
        lag_info: dict,
        corr_info: dict,
        extra_info: dict,
):
    """
    Create the 6-panel stacked figure used in benchmark mode.

    Panels:
      1) Real EOG raw (z)
      2) Real EOG processed (band-pass + z)
      3) Global PCA aligned to real EOG processed
      4) Frontal PCA supervised aligned
      5) ICA supervised aligned
      6) ICA unsupervised aligned (MEG-only selection + sign rule; benchmark-only alignment)

    The title includes compact diagnostics from unsupervised ICA scoring.
    """
    n_plot = len(t)
    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(t, safe_zscore(eog_raw_real[:n_plot]))
    axes[0].set_title("1) Real EOG channel (raw) [z-score]")
    axes[0].set_ylabel("a.u.")

    axes[1].plot(t, eog_ref_proc[:n_plot])
    axes[1].set_title(f"2) Reference processed EOG ({EOG_L_FREQ}-{EOG_H_FREQ} Hz, z-score)")
    axes[1].set_ylabel("a.u.")

    axes[2].plot(t, traces["pca_all_aligned"][:n_plot])
    axes[2].set_title(f"3) Global PCA (MEG-only) aligned | lag={lag_info['pca_all']} | r={corr_info['pca_all']:.3f}")
    axes[2].set_ylabel("a.u.")

    axes[3].plot(t, traces["pca_frontal_sup_aligned"][:n_plot])
    axes[3].set_title(
        f"4) Frontal PCA SUPERVISED (corr-ranked sensors) | lag={lag_info['pca_frontal_sup']} | r={corr_info['pca_frontal_sup']:.3f}"
    )
    axes[3].set_ylabel("a.u.")

    axes[4].plot(t, traces["ica_sup_aligned"][:n_plot])
    axes[4].set_title(
        f"5) ICA SUPERVISED (bestIC={extra_info['ica_sup_best_ic']}, absCorr={extra_info['ica_sup_abs_corr']:.3f}) | "
        f"lag={lag_info['ica_sup']} | r={corr_info['ica_sup']:.3f}"
    )
    axes[4].set_ylabel("a.u.")

    axes[5].plot(t, traces["ica_unsup_aligned"][:n_plot])
    axes[5].set_title(
        f"6) ICA UNSUPERVISED ({extra_info['ica_unsup_selection_mode']}; bestIC={extra_info['ica_unsup_best_ic']}, score={extra_info['ica_unsup_score']:.3f}) | "
        f"sign={extra_info['ica_unsup_sign_mode']} | lag={lag_info['ica_unsup']} | r={corr_info['ica_unsup']:.3f}"
    )
    axes[5].set_ylabel("a.u.")
    axes[5].set_xlabel("Time (s)")

    header_1 = f"{subject_id} | sfreq={sfreq:.2f} Hz | band={EOG_L_FREQ}-{EOG_H_FREQ} Hz"
    header_2 = (
        f"ICA-unsup parts: p_pos={extra_info['ica_unsup_p_pos']:.2f} "
        f"p_negPen={extra_info['ica_unsup_p_neg_penalty']:.2f} "
        f"p_rate={extra_info['ica_unsup_p_rate']:.2f} "
        f"p_kurt={extra_info['ica_unsup_p_kurt']:.2f}"
    )
    fig.suptitle(f"{header_1}\n{header_2}", y=0.995, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_methods_overlay_benchmark(
        out_png: str,
        t: np.ndarray,
        eog_ref_proc: np.ndarray,
        traces: dict,
        subject_id: str,
        corr_info: dict,
):
    """
    Overlay plot for benchmark mode:
      - real EOG processed reference
      - global PCA
      - supervised frontal PCA
      - ICA supervised
      - ICA unsupervised
    """
    n_plot = len(t)

    plt.figure(figsize=(14, 6))
    plt.plot(t, eog_ref_proc[:n_plot], label="Reference processed REAL EOG")
    plt.plot(t, traces["pca_all_aligned"][:n_plot], label=f"Global PCA r={corr_info['pca_all']:.3f}")
    plt.plot(t, traces["pca_frontal_sup_aligned"][:n_plot],
             label=f"Frontal PCA SUP r={corr_info['pca_frontal_sup']:.3f}")
    plt.plot(t, traces["ica_sup_aligned"][:n_plot], label=f"ICA SUP r={corr_info['ica_sup']:.3f}")
    plt.plot(t, traces["ica_unsup_aligned"][:n_plot], label=f"ICA UNSUP r={corr_info['ica_unsup']:.3f}")

    plt.legend()
    plt.title(f"{subject_id} | Overlay (aligned to real EOG)")
    plt.xlabel("Time (s)")
    plt.ylabel("a.u. (z-score)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# =============================================================================
# PLOTTING: MEG-only mode (no real EOG)
# =============================================================================
def plot_methods_stacked_meg_only(
        out_png: str,
        t: np.ndarray,
        traces: dict,
        subject_id: str,
        sfreq: float,
        extra_info: dict,
):
    """
    3-panel stacked plot when no real EOG channel exists.

    We show only MEG-only methods:
      1) Global PCA (processed)
      2) Frontal PCA unsupervised (processed)
      3) ICA unsupervised (processed)

    There is no "reference", so we do not compute correlations to real EOG here.
    """
    n_plot = len(t)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, traces["pca_all_proc"][:n_plot])
    axes[0].set_title("1) Global PCA (MEG-only) processed (band-pass + z)")
    axes[0].set_ylabel("a.u.")

    axes[1].plot(t, traces["pca_frontal_unsup_proc"][:n_plot])
    axes[1].set_title(f"2) Frontal PCA UNSUPERVISED ({FRONTAL_PCA_UNSUPERVISED_MODE}) processed")
    axes[1].set_ylabel("a.u.")

    axes[2].plot(t, traces["ica_unsup_proc"][:n_plot])
    axes[2].set_title(
        f"3) ICA UNSUPERVISED ({extra_info['ica_unsup_selection_mode']}; bestIC={extra_info['ica_unsup_best_ic']}, score={extra_info['ica_unsup_score']:.3f}) | "
        f"sign={extra_info['ica_unsup_sign_mode']}"
    )
    axes[2].set_ylabel("a.u.")
    axes[2].set_xlabel("Time (s)")

    header_1 = f"{subject_id} | sfreq={sfreq:.2f} Hz | band={EOG_L_FREQ}-{EOG_H_FREQ} Hz | (NO real EOG)"
    header_2 = (
        f"ICA-unsup parts: p_pos={extra_info['ica_unsup_p_pos']:.2f} "
        f"p_negPen={extra_info['ica_unsup_p_neg_penalty']:.2f} "
        f"p_rate={extra_info['ica_unsup_p_rate']:.2f} "
        f"p_kurt={extra_info['ica_unsup_p_kurt']:.2f}"
    )
    fig.suptitle(f"{header_1}\n{header_2}", y=0.995, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_methods_overlay_meg_only(
        out_png: str,
        t: np.ndarray,
        traces: dict,
        subject_id: str,
        extra_info: dict,
):
    """
    Overlay plot in MEG-only mode:
      - Global PCA processed
      - Frontal PCA unsupervised processed
      - ICA unsupervised processed
    """
    n_plot = len(t)

    plt.figure(figsize=(14, 6))
    plt.plot(t, traces["pca_all_proc"][:n_plot], label="Global PCA (processed)")
    plt.plot(t, traces["pca_frontal_unsup_proc"][:n_plot], label=f"Frontal PCA UNSUP ({FRONTAL_PCA_UNSUPERVISED_MODE})")
    plt.plot(t, traces["ica_unsup_proc"][:n_plot],
             label=f"ICA UNSUP (IC{extra_info['ica_unsup_best_ic']}, score={extra_info['ica_unsup_score']:.3f})")

    plt.legend()
    plt.title(f"{subject_id} | MEG-only overlay (no real EOG)")
    plt.xlabel("Time (s)")
    plt.ylabel("a.u. (z-score)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
results = []

# Discover files to process
fif_files = sorted(
    p for p in Path(DATASET_ROOT).rglob("*_meg.fif")
    if p.is_file() and "derivatives" not in str(p) and ".git" not in str(p)
)

print(f"Found {len(fif_files)} FIF files")

for fif_path in fif_files:
    subject_id = fif_path.stem
    print(f"\nProcessing {subject_id}")

    # -------------------------------------------------------------------------
    # Step 1) Load FIF
    # -------------------------------------------------------------------------
    try:
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        continue

    sfreq = float(raw.info["sfreq"])

    # -------------------------------------------------------------------------
    # Step 2) Check whether a real EOG channel exists
    #   - If yes: benchmark mode (compute correlations/align to real EOG)
    #   - If no:  MEG-only mode (plot only MEG-only proxies)
    # -------------------------------------------------------------------------
    eog_idx = pick_prefer_vertical_eog(raw)
    has_real_eog = eog_idx >= 0

    eog_ch_name = ""
    eog_raw_real = None
    eog_ref_proc = None

    if has_real_eog:
        # Extract the real EOG trace and process it (band-pass + z)
        eog_ch_name = raw.ch_names[eog_idx]
        eog_raw_real = raw.get_data(picks=[eog_idx])[0]
        eog_ref_proc = process_eog_trace(eog_raw_real, sfreq, EOG_L_FREQ, EOG_H_FREQ)
    else:
        print("  No real EOG channel -> MEG-only fallback mode (plots: Global PCA, Frontal PCA, ICA unsupervised).")

    # -------------------------------------------------------------------------
    # Step 3) Create MEG-only Raw (always needed)
    # -------------------------------------------------------------------------
    raw_meg_only = raw.copy().pick_types(meg=True, eeg=False, eog=False, ecg=False, stim=False)

    # Determine a common length for signals and operations
    if has_real_eog:
        n_total = min(len(eog_raw_real), len(eog_ref_proc), raw_meg_only.n_times)
    else:
        n_total = raw_meg_only.n_times

    if n_total < int(round(5 * sfreq)):
        print("  Too short -> skipping")
        continue

    if has_real_eog:
        eog_raw_real = eog_raw_real[:n_total]
        eog_ref_proc = eog_ref_proc[:n_total]

    # -------------------------------------------------------------------------
    # Step 4) Global PCA proxy (MEG-only)
    # -------------------------------------------------------------------------
    try:
        syn_pca_all_raw = build_synth_eog_pca_all(raw_meg_only)
        syn_pca_all_proc = process_eog_trace(syn_pca_all_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)[:n_total]
    except Exception as e:
        print(f"  ERROR Global PCA: {e}")
        continue

    # -------------------------------------------------------------------------
    # Step 5) Frontal PCA unsupervised proxy (MEG-only)
    #   - Needed always for MEG-only plots
    #   - Also useful as sign proxy for unsupervised ICA if UNSUP_SIGN_MODE="frontal_proxy"
    # -------------------------------------------------------------------------
    try:
        syn_pca_front_unsup_raw = build_synth_eog_pca_frontal_unsupervised(raw_meg_only)
        syn_pca_front_unsup_proc = process_eog_trace(syn_pca_front_unsup_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)[:n_total]
    except Exception as e:
        # In MEG-only mode, if this fails we can still proceed with ICA peak polarity sign.
        # In benchmark mode, we can proceed without the frontal unsup trace (it's not used in plots).
        print(f"  WARNING Frontal PCA unsupervised failed: {e}")
        syn_pca_front_unsup_raw = None
        syn_pca_front_unsup_proc = None

    # -------------------------------------------------------------------------
    # Step 6) If we are in benchmark mode, compute Frontal PCA supervised
    # -------------------------------------------------------------------------
    if has_real_eog:
        try:
            syn_pca_front_sup_raw = build_synth_eog_pca_frontal_supervised(raw_meg_only, eog_ref_proc)
            syn_pca_front_sup_proc = process_eog_trace(syn_pca_front_sup_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)[:n_total]
        except Exception as e:
            print(f"  ERROR Frontal PCA supervised: {e}")
            continue

    # -------------------------------------------------------------------------
    # Step 7) ICA: fit ONCE and extract sources
    # -------------------------------------------------------------------------
    try:
        sources = fit_ica_sources_once(raw_meg_only)
    except Exception as e:
        print(f"  ERROR ICA fit: {e}")
        continue

    # -------------------------------------------------------------------------
    # Step 8) ICA supervised selection (benchmark only)
    # -------------------------------------------------------------------------
    if has_real_eog:
        sup_ic, sup_abs_corr = pick_best_ic_supervised_from_sources(sources, eog_ref_proc)
        syn_ica_sup_raw = sources[sup_ic, :n_total]

        # Benchmark-only sign stabilization: make it correlate positively with real EOG reference
        if np.corrcoef(syn_ica_sup_raw, eog_ref_proc)[0, 1] < 0:
            syn_ica_sup_raw = -syn_ica_sup_raw

        syn_ica_sup_proc = process_eog_trace(syn_ica_sup_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)[:n_total]

    # -------------------------------------------------------------------------
    # Step 9) ICA unsupervised selection (MEG-only)
    # -------------------------------------------------------------------------
    unsup_ic, unsup_details = pick_best_ic_unsupervised_from_sources(
        sources,
        sfreq=sfreq,
        mode=ICA_UNSUP_MODE,
        fixed_ic=ICA_UNSUP_FIXED_IC,
    )
    syn_ica_unsup_raw = sources[unsup_ic, :n_total]

    # Apply MEG-only sign convention (no real EOG is used here)
    if UNSUP_SIGN_MODE == "frontal_proxy" and syn_pca_front_unsup_proc is not None:
        syn_ica_unsup_raw = apply_unsup_sign_convention_frontal_proxy(syn_ica_unsup_raw, syn_pca_front_unsup_proc)
    else:
        syn_ica_unsup_raw = apply_unsup_sign_convention_peak_polarity(syn_ica_unsup_raw, sfreq)

    syn_ica_unsup_proc = process_eog_trace(syn_ica_unsup_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)[:n_total]

    # Collect unsupervised diagnostics into a unified dict for plots/CSV
    extra_info_common = dict(
        ica_unsup_best_ic=unsup_ic,
        ica_unsup_score=unsup_details.get("score", float("nan")),
        ica_unsup_p_kurt=unsup_details.get("p_kurt", float("nan")),
        ica_unsup_p_rate=unsup_details.get("p_rate", float("nan")),
        ica_unsup_p_pos=unsup_details.get("p_pos", float("nan")),
        ica_unsup_p_neg_penalty=unsup_details.get("p_neg_penalty", float("nan")),
        ica_unsup_selection_mode=unsup_details.get("selection_mode", ICA_UNSUP_MODE),
        ica_unsup_sign_mode=UNSUP_SIGN_MODE,
    )

    # -------------------------------------------------------------------------
    # Step 10) MEG-only mode plotting + CSV row
    # -------------------------------------------------------------------------
    if not has_real_eog:
        if syn_pca_front_unsup_proc is None:
            # If frontal PCA unsupervised failed, we still plot global PCA + ICA unsup,
            # but the user asked explicitly: "Global PCA, Frontal PCA, ICA unsupervised".
            # So here we enforce a fallback: use global PCA as frontal proxy (clearly labeled).
            syn_pca_front_unsup_proc = syn_pca_all_proc.copy()

        n_plot = min(n_total, int(round(PLOT_SECONDS * sfreq)))
        t = np.arange(n_plot) / sfreq

        traces_meg_only = dict(
            pca_all_proc=syn_pca_all_proc,
            pca_frontal_unsup_proc=syn_pca_front_unsup_proc,
            ica_unsup_proc=syn_ica_unsup_proc,
        )

        stacked_png = os.path.join(OUTPUT_DIR, f"{subject_id}_MEGonly_stacked_GlobalPCA_FrontalPCA_ICAunsup.png")
        overlay_png = os.path.join(OUTPUT_DIR, f"{subject_id}_MEGonly_overlay_GlobalPCA_FrontalPCA_ICAunsup.png")

        plot_methods_stacked_meg_only(
            out_png=stacked_png,
            t=t,
            traces=traces_meg_only,
            subject_id=subject_id,
            sfreq=sfreq,
            extra_info=extra_info_common,
        )

        plot_methods_overlay_meg_only(
            out_png=overlay_png,
            t=t,
            traces=traces_meg_only,
            subject_id=subject_id,
            extra_info=extra_info_common,
        )

        # Save MEG-only results row (no reference correlations available)
        results.append(
            dict(
                subject=subject_id,
                file=str(fif_path),
                sfreq_hz=sfreq,
                n_samples=n_total,
                has_real_eog=False,
                eog_channel="",

                # Global PCA (MEG-only) — no benchmark correlations
                pca_all_corr_before=np.nan,
                pca_all_corr_after=np.nan,
                pca_all_lag_samples=np.nan,
                pca_all_lag_seconds=np.nan,

                # Frontal PCA supervised — not computed
                pca_frontal_sup_corr_before=np.nan,
                pca_frontal_sup_corr_after=np.nan,
                pca_frontal_sup_lag_samples=np.nan,
                pca_frontal_sup_lag_seconds=np.nan,

                # ICA supervised — not computed
                ica_sup_best_ic=np.nan,
                ica_sup_best_ic_abs_corr=np.nan,
                ica_sup_corr_before=np.nan,
                ica_sup_corr_after=np.nan,
                ica_sup_lag_samples=np.nan,
                ica_sup_lag_seconds=np.nan,

                # ICA unsupervised — always computed (MEG-only)
                ica_unsup_mode=ICA_UNSUP_MODE,
                ica_unsup_fixed_ic=ICA_UNSUP_FIXED_IC if ICA_UNSUP_MODE == "fixed" else np.nan,
                ica_unsup_sign_mode=UNSUP_SIGN_MODE,
                ica_unsup_best_ic=unsup_ic,
                ica_unsup_score=unsup_details.get("score", float("nan")),
                ica_unsup_p_kurt=unsup_details.get("p_kurt", float("nan")),
                ica_unsup_p_rate=unsup_details.get("p_rate", float("nan")),
                ica_unsup_p_pos=unsup_details.get("p_pos", float("nan")),
                ica_unsup_p_neg_penalty=unsup_details.get("p_neg_penalty", float("nan")),
                ica_unsup_pos_rate_per_min=unsup_details.get("pos_rate_per_min", float("nan")),
                ica_unsup_neg_rate_per_min=unsup_details.get("neg_rate_per_min", float("nan")),
                ica_unsup_corr_before=np.nan,
                ica_unsup_corr_after=np.nan,
                ica_unsup_lag_samples=np.nan,
                ica_unsup_lag_seconds=np.nan,

                stacked_plot=stacked_png,
                overlay_plot=overlay_png,
            )
        )

        print(
            f"  Done (MEG-only) | "
            f"GlobalPCA processed | "
            f"FrontalPCA({FRONTAL_PCA_UNSUPERVISED_MODE}) processed | "
            f"ICA UNSUP IC={unsup_ic} (mode={ICA_UNSUP_MODE}, sign={UNSUP_SIGN_MODE}, score={unsup_details.get('score', np.nan):.3f})"
        )
        continue

    # -------------------------------------------------------------------------
    # Step 11) BENCHMARK MODE: compute alignments + correlations to real EOG
    # -------------------------------------------------------------------------

    # A) Global PCA alignment/correlation
    lag_pca_all = best_lag_via_xcorr(eog_ref_proc, syn_pca_all_proc)
    syn_pca_all_aligned = shift_with_zeros(syn_pca_all_proc, lag_pca_all)
    r_pca_all_before, _ = pearsonr(eog_ref_proc, syn_pca_all_proc)
    r_pca_all_after, _ = pearsonr(eog_ref_proc, syn_pca_all_aligned)

    # B) Frontal PCA supervised alignment/correlation
    lag_pca_front_sup = best_lag_via_xcorr(eog_ref_proc, syn_pca_front_sup_proc)
    syn_pca_front_sup_aligned = shift_with_zeros(syn_pca_front_sup_proc, lag_pca_front_sup)
    r_pca_front_sup_before, _ = pearsonr(eog_ref_proc, syn_pca_front_sup_proc)
    r_pca_front_sup_after, _ = pearsonr(eog_ref_proc, syn_pca_front_sup_aligned)

    # C) ICA supervised alignment/correlation
    lag_ica_sup = best_lag_via_xcorr(eog_ref_proc, syn_ica_sup_proc)
    syn_ica_sup_aligned = shift_with_zeros(syn_ica_sup_proc, lag_ica_sup)
    r_ica_sup_before, _ = pearsonr(eog_ref_proc, syn_ica_sup_proc)
    r_ica_sup_after, _ = pearsonr(eog_ref_proc, syn_ica_sup_aligned)

    # D) ICA unsupervised alignment/correlation (benchmark-only alignment)
    lag_ica_unsup = best_lag_via_xcorr(eog_ref_proc, syn_ica_unsup_proc)
    syn_ica_unsup_aligned = shift_with_zeros(syn_ica_unsup_proc, lag_ica_unsup)
    r_ica_unsup_before, _ = pearsonr(eog_ref_proc, syn_ica_unsup_proc)
    r_ica_unsup_after, _ = pearsonr(eog_ref_proc, syn_ica_unsup_aligned)

    # -------------------------------------------------------------------------
    # Step 12) BENCHMARK MODE plots (6 stacked + overlay)
    # -------------------------------------------------------------------------
    n_plot = min(n_total, int(round(PLOT_SECONDS * sfreq)))
    t = np.arange(n_plot) / sfreq

    traces_benchmark = dict(
        pca_all_aligned=syn_pca_all_aligned,
        pca_frontal_sup_aligned=syn_pca_front_sup_aligned,
        ica_sup_aligned=syn_ica_sup_aligned,
        ica_unsup_aligned=syn_ica_unsup_aligned,
    )

    lag_info = dict(
        pca_all=lag_pca_all,
        pca_frontal_sup=lag_pca_front_sup,
        ica_sup=lag_ica_sup,
        ica_unsup=lag_ica_unsup,
    )

    corr_info = dict(
        pca_all=r_pca_all_after,
        pca_frontal_sup=r_pca_front_sup_after,
        ica_sup=r_ica_sup_after,
        ica_unsup=r_ica_unsup_after,
    )

    extra_info_benchmark = dict(
        ica_sup_best_ic=sup_ic,
        ica_sup_abs_corr=sup_abs_corr,
        **extra_info_common,
    )

    stacked_png = os.path.join(OUTPUT_DIR, f"{subject_id}_BENCH_stacked_methods_sup_unsup.png")
    overlay_png = os.path.join(OUTPUT_DIR, f"{subject_id}_BENCH_overlay_methods_sup_unsup.png")

    plot_methods_stacked_benchmark(
        out_png=stacked_png,
        t=t,
        eog_raw_real=eog_raw_real,
        eog_ref_proc=eog_ref_proc,
        traces=traces_benchmark,
        subject_id=subject_id,
        sfreq=sfreq,
        lag_info=lag_info,
        corr_info=corr_info,
        extra_info=extra_info_benchmark,
    )

    plot_methods_overlay_benchmark(
        out_png=overlay_png,
        t=t,
        eog_ref_proc=eog_ref_proc,
        traces=traces_benchmark,
        subject_id=subject_id,
        corr_info=corr_info,
    )

    # -------------------------------------------------------------------------
    # Step 13) BENCHMARK MODE: Save results row (includes correlations)
    # -------------------------------------------------------------------------
    results.append(
        dict(
            subject=subject_id,
            file=str(fif_path),
            sfreq_hz=sfreq,
            n_samples=n_total,
            has_real_eog=True,
            eog_channel=eog_ch_name,

            # Global PCA
            pca_all_corr_before=r_pca_all_before,
            pca_all_corr_after=r_pca_all_after,
            pca_all_lag_samples=lag_pca_all,
            pca_all_lag_seconds=lag_pca_all / sfreq,

            # Supervised frontal PCA
            pca_frontal_sup_corr_before=r_pca_front_sup_before,
            pca_frontal_sup_corr_after=r_pca_front_sup_after,
            pca_frontal_sup_lag_samples=lag_pca_front_sup,
            pca_frontal_sup_lag_seconds=lag_pca_front_sup / sfreq,

            # ICA supervised
            ica_sup_best_ic=sup_ic,
            ica_sup_best_ic_abs_corr=sup_abs_corr,
            ica_sup_corr_before=r_ica_sup_before,
            ica_sup_corr_after=r_ica_sup_after,
            ica_sup_lag_samples=lag_ica_sup,
            ica_sup_lag_seconds=lag_ica_sup / sfreq,

            # ICA unsupervised
            ica_unsup_mode=ICA_UNSUP_MODE,
            ica_unsup_fixed_ic=ICA_UNSUP_FIXED_IC if ICA_UNSUP_MODE == "fixed" else np.nan,
            ica_unsup_sign_mode=UNSUP_SIGN_MODE,
            ica_unsup_best_ic=unsup_ic,
            ica_unsup_score=unsup_details.get("score", float("nan")),
            ica_unsup_p_kurt=unsup_details.get("p_kurt", float("nan")),
            ica_unsup_p_rate=unsup_details.get("p_rate", float("nan")),
            ica_unsup_p_pos=unsup_details.get("p_pos", float("nan")),
            ica_unsup_p_neg_penalty=unsup_details.get("p_neg_penalty", float("nan")),
            ica_unsup_pos_rate_per_min=unsup_details.get("pos_rate_per_min", float("nan")),
            ica_unsup_neg_rate_per_min=unsup_details.get("neg_rate_per_min", float("nan")),
            ica_unsup_corr_before=r_ica_unsup_before,
            ica_unsup_corr_after=r_ica_unsup_after,
            ica_unsup_lag_samples=lag_ica_unsup,
            ica_unsup_lag_seconds=lag_ica_unsup / sfreq,

            stacked_plot=stacked_png,
            overlay_plot=overlay_png,
        )
    )

    print(
        f"  Done (benchmark) | EOG={eog_ch_name} | "
        f"GlobalPCA r_after={r_pca_all_after:.3f} | "
        f"FrontalPCA SUP r_after={r_pca_front_sup_after:.3f} | "
        f"ICA SUP r_after={r_ica_sup_after:.3f} (IC={sup_ic}) | "
        f"ICA UNSUP r_after={r_ica_unsup_after:.3f} (IC={unsup_ic}, mode={ICA_UNSUP_MODE}, sign={UNSUP_SIGN_MODE})"
    )

# =============================================================================
# SAVE CSV SUMMARY
# =============================================================================
df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "eog_compare_methods_supervised_unsupervised_summary_v7.csv")
df.to_csv(csv_path, index=False)

print("\n=== SUMMARY ===")
if len(df) > 0:
    # Show a small descriptive overview. In MEG-only rows, corr columns will be NaN.
    cols = [
        "has_real_eog",
        "pca_all_corr_after",
        "pca_frontal_sup_corr_after",
        "ica_sup_corr_after",
        "ica_unsup_corr_after",
        "ica_unsup_score",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].describe(include="all"))
    print(f"\nSaved summary CSV: {csv_path}")
else:
    print("No valid files processed.")
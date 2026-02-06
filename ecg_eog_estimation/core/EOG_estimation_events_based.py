"""
EOG EVENT VALIDATION (REAL EOG ground truth) + MEG-only proxies
with the FULL functionality of your older v7 script:
  - ICA unsupervised: mode fixed vs heuristic
  - ICA unsupervised: sign convention (frontal_proxy vs peak_polarity)
  - Frontal PCA unsupervised: layout vs regex vs variance
  - Optional supervised methods (when real EOG exists): frontal PCA supervised + ICA supervised
  - ICA fit ONCE per file (critical to avoid IC mismatch)
  - Event-level benchmark metrics using mne.preprocessing.find_eog_events

-------------------------------------------------------------------------------
WHAT IS "GROUND TRUTH" HERE?
-------------------------------------------------------------------------------
Ground truth is defined as:
  1) REAL EOG channel processed with MNE-style band-pass + z-score (for waveform comparisons)
  2) Blink events extracted from REAL EOG channel using:
        mne.preprocessing.find_eog_events(raw, ch_name=<real_eog>)

The MEG-only methods are compared against those REAL-EOG blink events.

-------------------------------------------------------------------------------
PRIMARY METRICS (EVENT-LEVEL)
-------------------------------------------------------------------------------
We match event sample indices using a tolerance window MATCH_TOL_SEC.
We compute:
  - TP/FP/FN
  - precision/recall/F1/miss_rate
  - jitter stats (test - ref) in ms
  - FP/min (normalized by recording duration)

IMPORTANT:
- We DO NOT shift events by waveform lag.
- Waveform lag (xcorr) is ONLY used for visualization and correlation reporting.

-------------------------------------------------------------------------------
MEG-ONLY PROXIES
-------------------------------------------------------------------------------
Always computed:
  - PCA global (MEG-only)
  - PCA frontal unsupervised (MEG-only)
  - ICA unsupervised (MEG-only, selection fixed/heuristic; sign convention applied)
  - Multi PCA unsupervised (align global/frontal/ICA + PCA)

If a REAL EOG channel exists, we additionally compute (benchmark extras):
  - PCA frontal supervised (uses real EOG for sensor ranking)
  - ICA supervised (uses real EOG to choose IC)

-------------------------------------------------------------------------------
DEPENDENCIES
-------------------------------------------------------------------------------
mne, numpy, pandas, matplotlib, scipy

"""

import os
from pathlib import Path
import re
from typing import Optional, Dict, Tuple, List

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from scipy.signal import correlate, find_peaks, welch
from scipy.stats import pearsonr, kurtosis, skew


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_ROOT = "/Users/karelo/Development/datasets/ds_small"
OUTPUT_DIR = "/Users/karelo/Development/datasets/ds_small/derivatives/eog_event_validation_full_v7style"

PLOT_SECONDS = 100

# Parallel processing: set >1 to enable joblib across files
N_JOBS = 1

# Band-pass used for blink-sensitive content (applied to reference and proxies for waveform comparisons)
EOG_L_FREQ = 1.0
EOG_H_FREQ = 10.0

# Event matching tolerance (seconds)
MATCH_TOL_SEC = 0.08

# --------------------------
# PCA settings
# --------------------------
# Global PCA: choose top N variance channels for PCA
N_TOP_CHANNELS_FOR_PCA = 30
FALLBACK_USE_ALL_MEG = True

# Supervised "frontal" PCA: rank MEG channels by abs corr to real EOG and PCA the top K
FRONTAL_TOPK_BY_CORR = 40

# Unsupervised frontal PCA channel selection method (MEG-only)
#   "layout": pick channels with highest "y" coordinate in device/head space (front)
#   "regex" : pick channels by name pattern
#   "variance": pick top variance channels in the EOG band (not truly "frontal", but simple)
FRONTAL_PCA_UNSUPERVISED_MODE = "layout"  # "layout" | "regex" | "variance"
FRONTAL_TOPK_UNSUPERVISED = 40
FRONTAL_REGEX = r"MEG0(1|2|3)"  # used if mode == "regex"

# --------------------------
# ICA settings (fit ONCE)
# --------------------------
ICA_N_COMPONENTS = 0.99
ICA_METHOD = "fastica"
ICA_RANDOM_STATE = 97
ICA_MAX_ITER = 1000

# Score window for unsupervised ICA heuristics (None uses full data)
UNSUP_SCORE_SECONDS = None

# ICA unsupervised selection mode:
#   "heuristic": choose IC by blink-likeness score
#   "fixed": always choose a fixed index (useful for debugging)
#   "megnet": classify ICs with MEGNet
#   "hybrid": MEGNet + heuristics (weighted)
ICA_UNSUP_MODE = "heuristic"  # "heuristic" | "fixed" | "megnet" | "hybrid"
ICA_UNSUP_FIXED_IC = 0

# MEGNet configuration (used when ICA_UNSUP_MODE is "megnet" or "hybrid")
MEGNET_MODEL_PATH = ""
MEGNET_INPUT_SAMPLES = 2048
MEGNET_OUTPUT_INDEX = 0
MEGNET_SCORE_WEIGHT = 0.5

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
W_EOG_KURT = 0.20
W_EOG_RATE = 0.20
W_EOG_POS_SPIKE = 0.30
W_EOG_NEG_PENALTY = 0.10
W_EOG_BANDPOWER = 0.10
W_EOG_SKEW = 0.10

# Spectral shape heuristics: blinks are low-frequency dominant
EOG_LOW_BAND = (0.5, 4.0)
EOG_HIGH_BAND = (8.0, 30.0)
EOG_BANDPOWER_SIGMOID_K = 1.5

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# HELPERS: normalization & alignment
# =============================================================================
def safe_zscore(x: np.ndarray) -> np.ndarray:
    """
    Safe z-score for 1D arrays.

    - If std==0 or NaN, return mean-centered values to avoid NaN cascades.
    """
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sd = np.std(x)
    if sd == 0 or np.isnan(sd):
        return x - mu
    return (x - mu) / sd


def best_lag_via_xcorr(x_ref: np.ndarray, y: np.ndarray) -> int:
    """
    Find integer lag (samples) maximizing xcorr(y, x_ref).

    Used ONLY for waveform alignment in plots/correlation.
    Not used for event-matching.
    """
    n = min(len(x_ref), len(y))
    x = x_ref[:n] - np.mean(x_ref[:n])
    yy = y[:n] - np.mean(y[:n])
    c = correlate(yy, x, mode="full")
    lags = np.arange(-n + 1, n)
    return int(lags[np.argmax(c)])


def shift_with_zeros(y: np.ndarray, lag: int) -> np.ndarray:
    """
    Shift a 1D array by lag samples, padding with zeros to keep same length.

    lag>0: shift LEFT (drop first lag, pad zeros at end)
    lag<0: shift RIGHT (pad zeros at start, drop last k)
    """
    y = np.asarray(y)
    if lag > 0:
        return np.concatenate([y[lag:], np.zeros(lag)])
    elif lag < 0:
        k = -lag
        return np.concatenate([np.zeros(k), y[:-k]])
    return y.copy()


# =============================================================================
# MEGNet helpers (optional ICA classification)
# =============================================================================
_MEGNET_MODEL = None


def _resample_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a 1D array to target_len using linear interpolation.
    """
    if target_len <= 0:
        raise ValueError("target_len must be positive.")
    x = np.asarray(x, dtype=float)
    if len(x) == target_len:
        return x
    if len(x) < 2:
        return np.full(target_len, float(x[0]) if len(x) == 1 else 0.0, dtype=float)
    xp = np.linspace(0, len(x) - 1, num=target_len)
    return np.interp(xp, np.arange(len(x)), x)


def _load_megnet_model(model_path: str):
    """
    Load a MEGNet model from disk using TensorFlow/Keras.
    """
    if not model_path:
        raise ValueError("MEGNet model path is empty. Set MEGNET_MODEL_PATH.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MEGNet model not found: {model_path}")
    from tensorflow import keras

    return keras.models.load_model(model_path)


def _get_megnet_model():
    """
    Lazy-load and cache the MEGNet model.
    """
    global _MEGNET_MODEL
    if _MEGNET_MODEL is None:
        _MEGNET_MODEL = _load_megnet_model(MEGNET_MODEL_PATH)
    return _MEGNET_MODEL


def megnet_ic_score(ic: np.ndarray, model) -> float:
    """
    Score a single IC using MEGNet. Returns the requested output index.
    """
    x = safe_zscore(ic)
    x = _resample_to_length(x, MEGNET_INPUT_SAMPLES)
    x = x.reshape(1, -1, 1)
    pred = model.predict(x, verbose=0)
    pred = np.asarray(pred).ravel()
    if MEGNET_OUTPUT_INDEX >= len(pred):
        raise IndexError("MEGNET_OUTPUT_INDEX is out of bounds for model output.")
    return float(pred[MEGNET_OUTPUT_INDEX])


# =============================================================================
# REFERENCE: EOG processing
# =============================================================================
def process_trace_bandpass_z(x: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    """
    Standard preprocessing for EOG(-like) traces:
      1) band-pass (l_freq..h_freq)
      2) z-score

    We use mne.filter.filter_data for a stable FIR filter.
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
    Choose a best-guess vertical EOG channel index if multiple EOG channels exist.

    - Prefer common "vertical" naming patterns.
    - Otherwise return the first EOG channel.
    - If no EOG exists return -1.
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
# EVENT UTILITIES (benchmark core)
# =============================================================================
def extract_event_samples(events: np.ndarray) -> np.ndarray:
    """Return event sample indices as int array (events[:,0])."""
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
    Greedy one-to-one matching of event samples within ±tol_sec.

    This is the same style you used previously:
      - sort both lists
      - walk with i/j pointers
      - when within tolerance => match
      - else advance the side that is too early

    Returns:
      matched_ref, matched_test, unmatched_ref, unmatched_test
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
            j += 1
        else:
            i += 1

    matched_ref = np.array(matched_ref, dtype=int)
    matched_test = np.array(matched_test, dtype=int)

    mr_set = set(matched_ref.tolist())
    mt_set = set(matched_test.tolist())

    unmatched_ref = np.array([r for r in ref_samp if r not in mr_set], dtype=int)
    unmatched_test = np.array([t for t in test_samp if t not in mt_set], dtype=int)

    return matched_ref, matched_test, unmatched_ref, unmatched_test


def compute_detection_metrics(
    matched_ref: np.ndarray,
    matched_test: np.ndarray,
    unmatched_ref: np.ndarray,
    unmatched_test: np.ndarray,
    sfreq: float
) -> Dict:
    """
    Compute event-level detection metrics.

    - TP = matched
    - FN = reference unmatched
    - FP = test unmatched
    - precision / recall / f1
    - jitter stats for matched events (test - ref) in ms
    """
    tp = len(matched_ref)
    fn = len(unmatched_ref)
    fp = len(unmatched_test)

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else np.nan
    miss_rate = fn / (tp + fn) if (tp + fn) > 0 else np.nan

    if tp > 0:
        jitter_sec = (matched_test - matched_ref) / sfreq
        jitter_abs = np.abs(jitter_sec)
        jitter_mean_ms = 1e3 * np.mean(jitter_sec)
        jitter_std_ms = 1e3 * np.std(jitter_sec)
        jitter_mae_ms = 1e3 * np.mean(jitter_abs)
        jitter_med_ms = 1e3 * np.median(jitter_abs)
        jitter_p95_ms = 1e3 * np.percentile(jitter_abs, 95)
    else:
        jitter_sec = np.array([], dtype=float)
        jitter_mean_ms = jitter_std_ms = jitter_mae_ms = jitter_med_ms = jitter_p95_ms = np.nan

    return dict(
        TP=int(tp), FP=int(fp), FN=int(fn),
        precision=float(precision) if np.isfinite(precision) else np.nan,
        recall=float(recall) if np.isfinite(recall) else np.nan,
        f1=float(f1) if np.isfinite(f1) else np.nan,
        miss_rate=float(miss_rate) if np.isfinite(miss_rate) else np.nan,
        jitter_mean_ms_signed=float(jitter_mean_ms) if np.isfinite(jitter_mean_ms) else np.nan,
        jitter_std_ms_signed=float(jitter_std_ms) if np.isfinite(jitter_std_ms) else np.nan,
        jitter_mae_ms=float(jitter_mae_ms) if np.isfinite(jitter_mae_ms) else np.nan,
        jitter_median_abs_ms=float(jitter_med_ms) if np.isfinite(jitter_med_ms) else np.nan,
        jitter_p95_abs_ms=float(jitter_p95_ms) if np.isfinite(jitter_p95_ms) else np.nan,
        jitter_sec_signed=jitter_sec,
    )


def events_from_trace_via_mne_find_eog(
    trace: np.ndarray,
    sfreq: float,
    first_samp_abs: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect EOG events on a synthetic proxy trace using MNE's find_eog_events.

    Implementation:
      - Wrap the 1D trace into a RawArray with ch_type='eog'
      - Call find_eog_events on this synthetic raw
      - Convert returned event sample indices to ABSOLUTE indices by adding first_samp_abs

    Returns:
      (events_array, event_samples_abs)
    """
    info = mne.create_info(ch_names=["SYNTH_EOG"], sfreq=sfreq, ch_types=["eog"])
    raw_syn = mne.io.RawArray(trace[np.newaxis, :], info, verbose=False)
    ev = mne.preprocessing.find_eog_events(raw_syn, ch_name="SYNTH_EOG", verbose=False)
    samp_abs = extract_event_samples(ev) + int(first_samp_abs)
    return ev, samp_abs


# =============================================================================
# PCA estimators
# =============================================================================
def pca_first_component(data_ch_by_time: np.ndarray) -> np.ndarray:
    """
    Compute PC1 time-course from channels×time matrix via SVD.

    Sign convention:
      - PCA sign is arbitrary.
      - We flip PC1 so it correlates positively with the channel-mean signal
        (a weak but stable rule for reproducibility in plots).
    """
    X = np.asarray(data_ch_by_time, dtype=float)
    X = X - np.mean(X, axis=1, keepdims=True)

    # X = U S Vt ; Vt rows are time components
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = Vt[0, :]

    mean_sig = np.mean(X, axis=0)
    if np.corrcoef(pc1, mean_sig)[0, 1] < 0:
        pc1 = -pc1
    return pc1


def select_best_scored_polarity(x: np.ndarray, sfreq: float) -> Tuple[np.ndarray, Dict]:
    """
    Pick the polarity (original or flipped) that yields the better blink-likeness score.

    Returns:
      best_signal, details dict including score and flip flag.
    """
    details_pos = blink_likeness_score(x, sfreq)
    details_neg = blink_likeness_score(-x, sfreq)
    if details_neg["score"] > details_pos["score"]:
        details = details_neg
        details["flip_for_score"] = True
        return -x, details
    details = details_pos
    details["flip_for_score"] = False
    return x, details


def align_signals_to_reference(ref: np.ndarray, others: List[np.ndarray]) -> List[np.ndarray]:
    """
    Align each signal in "others" to ref using xcorr lag.
    """
    aligned = []
    for sig in others:
        lag = best_lag_via_xcorr(ref, sig)
        aligned.append(shift_with_zeros(sig, lag))
    return aligned


def build_multi_pca_unsupervised(
    global_pca: np.ndarray,
    frontal_pca: np.ndarray,
    ica_unsup: np.ndarray,
    sfreq: float,
) -> Tuple[np.ndarray, Dict]:
    """
    Build "Multi PCA unsupervised" by aligning + z-scoring the three MEG-only proxies,
    then extracting PC1 across them. Polarity is chosen by blink-likeness score.
    """
    ref = global_pca
    aligned_frontal, aligned_ica = align_signals_to_reference(ref, [frontal_pca, ica_unsup])
    X = np.vstack(
        [
            safe_zscore(ref),
            safe_zscore(aligned_frontal),
            safe_zscore(aligned_ica),
        ]
    )
    multi_raw = pca_first_component(X)
    multi_best, details = select_best_scored_polarity(multi_raw, sfreq)
    return multi_best, details


def build_synth_eog_pca_all(raw_meg: mne.io.BaseRaw) -> np.ndarray:
    """
    Global PCA (MEG-only):
      - filter MEG to EOG band
      - take top-N channels by variance
      - return PC1 across them
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
            raise RuntimeError("Not enough channels for PCA subset and fallback disabled.")
        picks = order

    return pca_first_component(data[picks, :])


def build_synth_eog_pca_frontal_supervised(raw_meg: mne.io.BaseRaw, eog_ref_proc: np.ndarray) -> np.ndarray:
    """
    Frontal PCA SUPERVISED (benchmark only):
      - filter MEG to EOG band
      - rank channels by abs corr to processed real EOG
      - PCA PC1 on top-K ranked channels
    """
    tmp = raw_meg.copy().filter(EOG_L_FREQ, EOG_H_FREQ, fir_design="firwin", verbose=False)
    data = tmp.get_data()
    if data.size == 0:
        raise RuntimeError("No MEG data available.")

    n = min(data.shape[1], len(eog_ref_proc))
    X = data[:, :n]
    y = eog_ref_proc[:n]

    # Efficient abs correlation using demeaned dot products
    y0 = y - np.mean(y)
    ysd = np.std(y0) + 1e-12

    corrs = np.zeros(X.shape[0], dtype=float)
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
    Pick a "frontal" channel subset without using real EOG.

    Modes:
      - layout: select channels with largest y-coordinate (front-most)
      - regex:  select channels matching FRONTAL_REGEX
      - variance: select top variance channels in EOG band
    """
    if FRONTAL_PCA_UNSUPERVISED_MODE == "layout":
        picks = mne.pick_types(raw_meg.info, meg=True)

        pos = []
        keep = []
        for p in picks:
            loc = raw_meg.info["chs"][p]["loc"][:3]
            if np.any(np.isnan(loc)) or np.allclose(loc, 0):
                continue
            keep.append(p)
            pos.append(loc)

        if len(keep) == 0:
            return np.array(picks[:min(len(picks), FRONTAL_TOPK_UNSUPERVISED)], dtype=int)

        pos = np.array(pos)
        y = pos[:, 1]
        order = np.argsort(y)[::-1]
        sel = np.array(keep, dtype=int)[order[:min(len(order), FRONTAL_TOPK_UNSUPERVISED)]]
        return sel

    if FRONTAL_PCA_UNSUPERVISED_MODE == "regex":
        picks = []
        for i, name in enumerate(raw_meg.ch_names):
            if re.match(FRONTAL_REGEX, name):
                picks.append(i)
        if len(picks) == 0:
            picks = mne.pick_types(raw_meg.info, meg=True)
        return np.array(picks[:min(len(picks), FRONTAL_TOPK_UNSUPERVISED)], dtype=int)

    # variance
    tmp = raw_meg.copy().filter(EOG_L_FREQ, EOG_H_FREQ, fir_design="firwin", verbose=False)
    data = tmp.get_data()
    ch_var = np.var(data, axis=1)
    order = np.argsort(ch_var)[::-1]
    return np.array(order[:min(len(order), FRONTAL_TOPK_UNSUPERVISED)], dtype=int)


def build_synth_eog_pca_frontal_unsupervised(raw_meg: mne.io.BaseRaw) -> np.ndarray:
    """
    Frontal PCA UNSUPERVISED (MEG-only):
      - filter MEG to EOG band
      - pick frontal channels via pick_frontal_channels_unsupervised()
      - return PC1
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
# ICA: fit ONCE per file and select ICs
# =============================================================================
def fit_ica_sources_once(raw_meg: mne.io.BaseRaw) -> np.ndarray:
    """
    Fit ICA ONCE per file in the EOG band and return sources (n_ic, n_times).

    This is critical: both supervised and unsupervised selections must come from
    the SAME sources array to avoid IC-index mismatch across separate ICA runs.
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


def pick_best_ic_supervised_from_sources(sources: np.ndarray, eog_ref_proc: np.ndarray) -> Tuple[int, float]:
    """
    Supervised IC selection (benchmark only):
      - pick IC with highest ABS correlation to processed real EOG reference.
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
# UNSUPERVISED ICA scoring (heuristics) + sign conventions
# =============================================================================
def eog_spike_strength_score(
    x: np.ndarray,
    sfreq: float,
    pos_thr_z: float = 3.5,
    neg_thr_z: float = 3.0,
    min_distance_sec: float = 0.20,
) -> Dict:
    """
    Spike features for an EOG-like trace:
      - strong positive spikes are rewarded (typical vertical EOG blinks)
      - too many strong negative spikes are penalized (often non-blink artifacts or inverted polarity)
    """
    z = safe_zscore(x)
    min_dist = max(1, int(round(min_distance_sec * sfreq)))

    pos_peaks, _ = find_peaks(z, height=pos_thr_z, distance=min_dist)
    neg_peaks, _ = find_peaks(-z, height=neg_thr_z, distance=min_dist)

    dur_sec = len(z) / sfreq
    if dur_sec <= 0:
        return dict(
            p_pos=0.0, p_neg_penalty=1.0,
            pos_rate_per_min=float("nan"), neg_rate_per_min=float("nan"),
            n_pos=0, n_neg=0
        )

    pos_rate = (len(pos_peaks) / dur_sec) * 60.0
    neg_rate = (len(neg_peaks) / dur_sec) * 60.0

    # Reward plausible positive spike rate
    if EOG_POS_SPIKE_MIN_PER_MIN <= pos_rate <= EOG_POS_SPIKE_MAX_PER_MIN:
        p_pos = 1.0
    else:
        if pos_rate < EOG_POS_SPIKE_MIN_PER_MIN:
            d = (EOG_POS_SPIKE_MIN_PER_MIN - pos_rate) / max(EOG_POS_SPIKE_MIN_PER_MIN, 1e-6)
        else:
            d = (pos_rate - EOG_POS_SPIKE_MAX_PER_MIN) / max(EOG_POS_SPIKE_MAX_PER_MIN, 1e-6)
        p_pos = float(np.exp(-3.0 * d))

    # Penalize too many negative spikes
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


def eog_bandpower_ratio_score(
    x: np.ndarray,
    sfreq: float,
    low_band: Tuple[float, float] = EOG_LOW_BAND,
    high_band: Tuple[float, float] = EOG_HIGH_BAND,
) -> Dict:
    """
    Low-frequency dominance score using Welch PSD.

    Returns:
      p_band: sigmoid score (0..1) for low/high bandpower ratio
      bandpower_ratio: raw low/high ratio (unitless)
    """
    if len(x) < 4 or sfreq <= 0:
        return dict(p_band=0.0, bandpower_ratio=float("nan"))

    nperseg = min(len(x), int(round(4.0 * sfreq)))
    if nperseg < 4:
        return dict(p_band=0.0, bandpower_ratio=float("nan"))

    freqs, psd = welch(x, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    low_mask = (freqs >= low_band[0]) & (freqs <= low_band[1])
    high_mask = (freqs >= high_band[0]) & (freqs <= high_band[1])

    if not np.any(low_mask) or not np.any(high_mask):
        return dict(p_band=0.0, bandpower_ratio=float("nan"))

    low_power = float(np.trapezoid(psd[low_mask], freqs[low_mask]))
    high_power = float(np.trapezoid(psd[high_mask], freqs[high_mask]))
    ratio = low_power / (high_power + 1e-12)

    log_ratio = np.log10(ratio + 1e-12)
    p_band = 1.0 / (1.0 + np.exp(-EOG_BANDPOWER_SIGMOID_K * log_ratio))

    return dict(p_band=float(p_band), bandpower_ratio=float(ratio))


def blink_likeness_score(ic: np.ndarray, sfreq: float) -> Dict:
    """
    Unsupervised blink-likeness score for a single IC.

    Components:
      - kurtosis (peaky transientness)
      - peak rate plausibility (prominence-based)
      - explicit positive-spike reward
      - negative-spike penalty
      - positive skewness (blink polarity)

    Returns a dict with total score and diagnostics.
    """
    # score only first UNSUP_SCORE_SECONDS if requested
    if UNSUP_SCORE_SECONDS is not None:
        n_seg = min(len(ic), int(round(UNSUP_SCORE_SECONDS * sfreq)))
        x = ic[:n_seg]
    else:
        x = ic

    z = safe_zscore(x)

    # (1) Kurtosis
    k = float(kurtosis(z, fisher=False, bias=False)) if len(z) > 10 else 0.0
    p_kurt = float(np.tanh((k - 3.0) / 5.0))
    p_kurt = max(0.0, p_kurt)

    # (2) Prominent-peak rate (polarity-agnostic)
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

    # (3) Spike features
    spike = eog_spike_strength_score(
        z, sfreq,
        pos_thr_z=EOG_POS_SPIKE_Z_THR,
        neg_thr_z=EOG_NEG_SPIKE_Z_THR,
        min_distance_sec=PEAK_MIN_DISTANCE_SEC,
    )

    p_pos = spike["p_pos"]
    p_neg_penalty = spike["p_neg_penalty"]

    # (4) Low-frequency dominance (blink energy tends to be low-freq)
    band = eog_bandpower_ratio_score(z, sfreq)
    p_band = band["p_band"]

    # (5) Positive skewness (reward positive-going blink transients)
    sk = float(skew(z, bias=False)) if len(z) > 10 else 0.0
    p_skew = float(np.tanh(sk / 2.0))
    p_skew = max(0.0, p_skew)

    # Weighted combination
    score = (
        W_EOG_KURT * p_kurt
        + W_EOG_RATE * p_rate
        + W_EOG_POS_SPIKE * p_pos
        + W_EOG_NEG_PENALTY * p_neg_penalty
        + W_EOG_BANDPOWER * p_band
        + W_EOG_SKEW * p_skew
    )

    return dict(
        score=float(score),
        p_kurt=float(p_kurt),
        p_rate=float(p_rate),
        p_pos=float(p_pos),
        p_neg_penalty=float(p_neg_penalty),
        p_bandpower=float(p_band),
        p_skew=float(p_skew),
        rate_per_min=float(rate),
        kurtosis=float(k),
        skewness=float(sk),
        n_prom_peaks=int(len(peaks)),
        pos_rate_per_min=float(spike["pos_rate_per_min"]),
        neg_rate_per_min=float(spike["neg_rate_per_min"]),
        n_pos_spikes=int(spike["n_pos"]),
        n_neg_spikes=int(spike["n_neg"]),
        bandpower_ratio=float(band["bandpower_ratio"]),
    )


def pick_best_ic_unsupervised_from_sources(
    sources: np.ndarray,
    sfreq: float,
    mode: str = "heuristic",
    fixed_ic: int = 0,
) -> Tuple[int, Dict]:
    """
    Unsupervised IC selection from the SAME sources array.

    - mode="fixed": choose fixed index (clipped)
    - mode="heuristic": score each IC and pick max score
    """
    mode = mode.lower()
    if mode == "fixed":
        idx = int(np.clip(fixed_ic, 0, sources.shape[0] - 1))
        details_pos = blink_likeness_score(sources[idx, :], sfreq)
        details_neg = blink_likeness_score(-sources[idx, :], sfreq)
        if details_neg["score"] > details_pos["score"]:
            details = details_neg
            details["flip_for_score"] = True
        else:
            details = details_pos
            details["flip_for_score"] = False
        details["selection_mode"] = "fixed"
        details["heuristic_score"] = float(details["score"])
        details["megnet_score"] = float("nan")
        details["megnet_flip_for_score"] = False
        return idx, details

    use_megnet = mode in {"megnet", "hybrid"}
    model = _get_megnet_model() if use_megnet else None
    best_idx = -1
    best_score = -np.inf
    best_details = None

    for i in range(sources.shape[0]):
        details_pos = blink_likeness_score(sources[i, :], sfreq)
        details_neg = blink_likeness_score(-sources[i, :], sfreq)
        if details_neg["score"] > details_pos["score"]:
            details = details_neg
            details["flip_for_score"] = True
        else:
            details = details_pos
            details["flip_for_score"] = False
        details["heuristic_score"] = float(details["score"])

        if use_megnet:
            m_pos = megnet_ic_score(sources[i, :], model)
            m_neg = megnet_ic_score(-sources[i, :], model)
            if m_neg > m_pos:
                megnet_score = float(m_neg)
                megnet_flip = True
            else:
                megnet_score = float(m_pos)
                megnet_flip = False
            details["megnet_score"] = megnet_score
            details["megnet_flip_for_score"] = megnet_flip
            if mode == "megnet":
                details["score"] = megnet_score
            else:
                details["score"] = (
                    (1.0 - MEGNET_SCORE_WEIGHT) * details["heuristic_score"]
                    + MEGNET_SCORE_WEIGHT * megnet_score
                )
        else:
            details["megnet_score"] = float("nan")
            details["megnet_flip_for_score"] = False

        if details["score"] > best_score:
            best_score = details["score"]
            best_idx = i
            best_details = details

    best_details["selection_mode"] = mode
    return int(best_idx), best_details


def apply_unsup_sign_convention_peak_polarity(ic: np.ndarray, sfreq: float) -> np.ndarray:
    """
    MEG-only sign rule: ensure strongest blink-like peaks are mostly POSITIVE.

    Compare max positive vs max negative prominent deflection on the scoring segment.
    If negative dominates -> flip sign.
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

    return -ic if neg_max > pos_max else ic


def apply_unsup_sign_convention_frontal_proxy(ic: np.ndarray, frontal_proxy_proc: np.ndarray) -> np.ndarray:
    """
    MEG-only sign rule: flip IC so it correlates positively with MEG-only frontal PCA proxy.
    """
    n = min(len(ic), len(frontal_proxy_proc))
    if n < 10:
        return ic
    r = np.corrcoef(ic[:n], frontal_proxy_proc[:n])[0, 1]
    return -ic if np.isfinite(r) and r < 0 else ic


# =============================================================================
# PLOTTING (event-aware, compact)
# =============================================================================
def plot_event_overlay(
    out_png: str,
    subject_id: str,
    sfreq: float,
    first_samp: int,
    ref_trace_proc: np.ndarray,
    test_trace_proc: np.ndarray,
    ref_events_abs: np.ndarray,
    test_events_abs: np.ndarray,
    matched_ref_abs: np.ndarray,
    matched_test_abs: np.ndarray,
    metrics: Dict,
    label_test: str,
    match_tol_sec: float,
    seconds: float = 60.0,
):
    """
    Publication-style plot:
      - overlay ref vs test traces (z / processed)
      - draw event markers: ref dashed, test dotted, matched emphasized
      - show jitter time series + |jitter| histogram
    """
    n_total = min(len(ref_trace_proc), len(test_trace_proc))
    n_plot = min(n_total, int(round(seconds * sfreq)))
    t = np.arange(n_plot) / sfreq

    ref = ref_trace_proc[:n_plot]
    test = test_trace_proc[:n_plot]

    # Convert abs event samples to window-relative samples
    ref_rel = ref_events_abs - first_samp
    test_rel = test_events_abs - first_samp
    mref_rel = matched_ref_abs - first_samp
    mtest_rel = matched_test_abs - first_samp

    def in_win(x):
        return x[(x >= 0) & (x < n_plot)]

    ref_in = in_win(ref_rel)
    test_in = in_win(test_rel)
    mref_in = in_win(mref_rel)

    # jitter vectors in ms for matched events in window
    if len(mref_rel) > 0:
        mask = (mref_rel >= 0) & (mref_rel < n_plot)
        mref = mref_rel[mask]
        mtest = mtest_rel[mask]
        jitter_ms = 1e3 * (mtest - mref) / sfreq
        jitter_abs_ms = np.abs(jitter_ms)
        jitter_time_s = mref / sfreq
    else:
        jitter_ms = np.array([])
        jitter_abs_ms = np.array([])
        jitter_time_s = np.array([])

    tol_ms = 1e3 * match_tol_sec

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.0, 1.2, 1.2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, ref, label="Ref: REAL EOG processed")
    ax1.plot(t, test, label=label_test)

    for s in ref_in:
        ax1.axvline(s / sfreq, linestyle="--", linewidth=0.8, alpha=0.5)
    for s in test_in:
        ax1.axvline(s / sfreq, linestyle=":", linewidth=0.8, alpha=0.5)
    for s in mref_in:
        ax1.axvline(s / sfreq, linestyle="-", linewidth=1.1, alpha=0.25)

    ax1.set_title("A) Processed traces + blink events (ref dashed, test dotted; matched highlighted)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("a.u. (z)")
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    if len(ref_in) > 0:
        ax2.vlines(ref_in / sfreq, 0.60, 1.00, linewidth=1.2)
    if len(test_in) > 0:
        ax2.vlines(test_in / sfreq, 0.00, 0.40, linewidth=1.2)
    ax2.set_yticks([0.2, 0.8])
    ax2.set_yticklabels([label_test, "Ref"])
    ax2.set_title("B) Event raster")
    ax2.set_xlabel("Time (s)")

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    if len(jitter_time_s) > 0:
        ax3.plot(jitter_time_s, jitter_ms, marker="o", linestyle="None")
        ax3.axhline(+tol_ms, linestyle="--", linewidth=1.0)
        ax3.axhline(-tol_ms, linestyle="--", linewidth=1.0)
        ax3.set_ylim(-max(1.1 * tol_ms, 10), max(1.1 * tol_ms, 10))
    ax3.set_title("C) Jitter over time for matched events (test - ref) [ms]")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Jitter (ms)")

    ax4 = fig.add_subplot(gs[3, 0])
    if len(jitter_abs_ms) > 0:
        ax4.hist(jitter_abs_ms, bins=30)
        ax4.axvline(tol_ms, linestyle="--", linewidth=1.0)
        ax4.set_xlim(0, max(tol_ms * 1.1, np.percentile(jitter_abs_ms, 99) * 1.1))
    ax4.set_title("D) |Jitter| distribution [ms] (dashed = tolerance)")
    ax4.set_xlabel("|Jitter| (ms)")
    ax4.set_ylabel("Count")

    fig.suptitle(
        f"{subject_id} | {label_test} | TP={metrics['TP']} FP={metrics['FP']} FN={metrics['FN']} | "
        f"prec={metrics['precision']:.3f} rec={metrics['recall']:.3f} F1={metrics['f1']:.3f} | "
        f"MAE={metrics['jitter_mae_ms']:.2f}ms p95={metrics['jitter_p95_abs_ms']:.2f}ms | tol={tol_ms:.0f}ms",
        fontsize=11
    )
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_meg_only_stacked(
    out_png: str,
    t: np.ndarray,
    traces: Dict[str, np.ndarray],
    subject_id: str,
    sfreq: float,
):
    """
    Simple stacked plot for MEG-only mode (no real EOG).
    """
    n_plot = len(t)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t, traces["pca_all_proc"][:n_plot])
    axes[0].set_title("1) Global PCA (MEG-only) processed")
    axes[0].set_ylabel("a.u.")

    axes[1].plot(t, traces["pca_frontal_unsup_proc"][:n_plot])
    axes[1].set_title(f"2) Frontal PCA UNSUP ({FRONTAL_PCA_UNSUPERVISED_MODE}) processed")
    axes[1].set_ylabel("a.u.")

    axes[2].plot(t, traces["ica_unsup_proc"][:n_plot])
    axes[2].set_title("3) ICA UNSUP processed")
    axes[2].set_ylabel("a.u.")

    axes[3].plot(t, traces["multi_pca_unsup_proc"][:n_plot])
    axes[3].set_title("4) Multi PCA UNSUP processed")
    axes[3].set_ylabel("a.u.")
    axes[3].set_xlabel("Time (s)")

    fig.suptitle(f"{subject_id} | sfreq={sfreq:.2f} Hz | NO real EOG", y=0.995, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

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
    
    # 1) Load
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
    
    # 2) Real EOG?
    eog_idx = pick_prefer_vertical_eog(raw)
    has_real_eog = eog_idx >= 0
    
    if not has_real_eog:
        print("  No real EOG channel -> MEG-only mode (generate proxies + Multi PCA unsupervised).")
    
    if has_real_eog:
        eog_ch_name = raw.ch_names[eog_idx]
        eog_raw_real = raw.get_data(picks=[eog_idx])[0]
        eog_ref_proc = process_trace_bandpass_z(eog_raw_real, sfreq, EOG_L_FREQ, EOG_H_FREQ)

        # Ground-truth events from REAL EOG
        try:
            eog_events_ref = mne.preprocessing.find_eog_events(raw, ch_name=eog_ch_name, verbose=False)
            ref_events_abs = extract_event_samples(eog_events_ref)
        except Exception as e:
            print(f"  ERROR find_eog_events on real EOG: {e}")
            return None
    else:
        eog_ch_name = ""
        eog_ref_proc = None
        ref_events_abs = np.array([], dtype=int)

    # 3) MEG-only raw
    raw_meg = raw.copy().pick_types(meg=True, eeg=False, eog=False, ecg=False, stim=False)

    # common length
    n_total = raw_meg.n_times if not has_real_eog else min(len(eog_ref_proc), raw_meg.n_times)
    if n_total < int(round(10 * sfreq)):
        print("  Too short -> skipping")
        return None

    if has_real_eog:
        eog_ref_proc = eog_ref_proc[:n_total]
    
    # 4) Global PCA (MEG-only) + unsupervised polarity selection
    try:
        pca_all_raw = build_synth_eog_pca_all(raw_meg)[:n_total]
        pca_all_raw, pca_all_details = select_best_scored_polarity(pca_all_raw, sfreq)
        pca_all_proc = process_trace_bandpass_z(pca_all_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)[:n_total]
        _, pca_all_events_abs = events_from_trace_via_mne_find_eog(pca_all_raw[:n_total], sfreq, raw.first_samp)
    except Exception as e:
        print(f"  ERROR Global PCA: {e}")
        return None
    
    # 5) Frontal PCA UNSUP (MEG-only) + polarity selection
    try:
        pca_front_unsup_raw = build_synth_eog_pca_frontal_unsupervised(raw_meg)[:n_total]
        pca_front_unsup_raw, pca_front_unsup_details = select_best_scored_polarity(pca_front_unsup_raw, sfreq)
        pca_front_unsup_proc = process_trace_bandpass_z(pca_front_unsup_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)[:n_total]
        _, pca_front_unsup_events_abs = events_from_trace_via_mne_find_eog(pca_front_unsup_raw[:n_total], sfreq, raw.first_samp)
    except Exception as e:
        print(f"  WARNING Frontal PCA unsupervised failed: {e}")
        pca_front_unsup_raw = None
        pca_front_unsup_proc = None
        pca_front_unsup_events_abs = np.array([], dtype=int)
    
    # 6) Supervised Frontal PCA (benchmark extra)
    if has_real_eog:
        try:
            pca_front_sup_raw = build_synth_eog_pca_frontal_supervised(raw_meg, eog_ref_proc)[:n_total]
            pca_front_sup_proc = process_trace_bandpass_z(pca_front_sup_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)[:n_total]
            _, pca_front_sup_events_abs = events_from_trace_via_mne_find_eog(pca_front_sup_raw[:n_total], sfreq, raw.first_samp)
        except Exception as e:
            print(f"  ERROR Frontal PCA supervised: {e}")
            return None
    else:
        pca_front_sup_raw = None
        pca_front_sup_proc = None
        pca_front_sup_events_abs = np.array([], dtype=int)
    
    # 7) ICA sources ONCE
    try:
        sources = fit_ica_sources_once(raw_meg)[:, :n_total]
    except Exception as e:
        print(f"  ERROR ICA fit: {e}")
        return None
    
    # 8) ICA supervised (benchmark extra)
    if has_real_eog:
        try:
            ica_sup_ic, ica_sup_abs_corr = pick_best_ic_supervised_from_sources(sources, eog_ref_proc)
            ica_sup_raw = sources[ica_sup_ic, :n_total]

            # sign stabilization (benchmark-only): positive corr to real EOG
            if np.corrcoef(ica_sup_raw, eog_ref_proc)[0, 1] < 0:
                ica_sup_raw = -ica_sup_raw

            ica_sup_proc = process_trace_bandpass_z(ica_sup_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)[:n_total]
            _, ica_sup_events_abs = events_from_trace_via_mne_find_eog(ica_sup_raw, sfreq, raw.first_samp)
        except Exception as e:
            print(f"  ERROR ICA supervised: {e}")
            return None
    else:
        ica_sup_ic = None
        ica_sup_abs_corr = np.nan
        ica_sup_raw = None
        ica_sup_proc = None
        ica_sup_events_abs = np.array([], dtype=int)
    
    # 9) ICA unsupervised (MEG-only)
    try:
        ica_unsup_ic, unsup_details = pick_best_ic_unsupervised_from_sources(
            sources, sfreq=sfreq, mode=ICA_UNSUP_MODE, fixed_ic=ICA_UNSUP_FIXED_IC
        )
        ica_unsup_raw = sources[ica_unsup_ic, :n_total]
        if unsup_details.get("flip_for_score", False):
            ica_unsup_raw = -ica_unsup_raw
    
        # MEG-only sign convention
        if UNSUP_SIGN_MODE == "frontal_proxy" and pca_front_unsup_proc is not None:
            ica_unsup_raw = apply_unsup_sign_convention_frontal_proxy(ica_unsup_raw, pca_front_unsup_proc)
        else:
            ica_unsup_raw = apply_unsup_sign_convention_peak_polarity(ica_unsup_raw, sfreq)
    
        ica_unsup_proc = process_trace_bandpass_z(ica_unsup_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)[:n_total]
        _, ica_unsup_events_abs = events_from_trace_via_mne_find_eog(ica_unsup_raw, sfreq, raw.first_samp)
    except Exception as e:
        print(f"  ERROR ICA unsupervised: {e}")
        return None

    # 9b) Multi PCA unsupervised (align + z-score + PCA across 3 proxies)
    frontal_for_multi = pca_front_unsup_raw if pca_front_unsup_raw is not None else pca_all_raw
    multi_pca_raw, multi_pca_details = build_multi_pca_unsupervised(
        pca_all_raw[:n_total],
        frontal_for_multi[:n_total],
        ica_unsup_raw[:n_total],
        sfreq,
    )
    multi_pca_proc = process_trace_bandpass_z(multi_pca_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)[:n_total]
    _, multi_pca_events_abs = events_from_trace_via_mne_find_eog(multi_pca_raw[:n_total], sfreq, raw.first_samp)

    if not has_real_eog:
        n_plot = min(n_total, int(round(PLOT_SECONDS * sfreq)))
        t = np.arange(n_plot) / sfreq
        stacked_png = os.path.join(OUTPUT_DIR, f"{subject_id}_MEGonly_stacked_MultiPCA.png")
        plot_meg_only_stacked(
            out_png=stacked_png,
            t=t,
            traces=dict(
                pca_all_proc=pca_all_proc,
                pca_frontal_unsup_proc=pca_front_unsup_proc if pca_front_unsup_proc is not None else pca_all_proc,
                ica_unsup_proc=ica_unsup_proc,
                multi_pca_unsup_proc=multi_pca_proc,
            ),
            subject_id=subject_id,
            sfreq=sfreq,
        )
        return dict(
            subject=subject_id,
            file=str(data_path),
            sfreq_hz=sfreq,
            duration_sec=n_total / sfreq,
            has_real_eog=False,
            eog_channel="",
            stacked_plot=stacked_png,
            multi_pca_unsup_score=multi_pca_details.get("score", np.nan),
        )
    
# =============================================================================
    # EVENT METRICS (PRIMARY)
# =============================================================================
    duration_sec = n_total / sfreq
    
    # PCA global
    mref_g, mtest_g, uref_g, utest_g = match_events_one_to_one(ref_events_abs, pca_all_events_abs, sfreq, MATCH_TOL_SEC)
    met_g = compute_detection_metrics(mref_g, mtest_g, uref_g, utest_g, sfreq)
    fpmin_g = met_g["FP"] / (duration_sec / 60.0) if duration_sec > 0 else np.nan
    
    # PCA frontal unsup (if available; else NaN metrics)
    if pca_front_unsup_raw is not None:
        mref_fu, mtest_fu, uref_fu, utest_fu = match_events_one_to_one(ref_events_abs, pca_front_unsup_events_abs, sfreq, MATCH_TOL_SEC)
        met_fu = compute_detection_metrics(mref_fu, mtest_fu, uref_fu, utest_fu, sfreq)
        fpmin_fu = met_fu["FP"] / (duration_sec / 60.0) if duration_sec > 0 else np.nan
    else:
        met_fu = dict(TP=np.nan, FP=np.nan, FN=np.nan, precision=np.nan, recall=np.nan, f1=np.nan, miss_rate=np.nan,
                      jitter_mean_ms_signed=np.nan, jitter_std_ms_signed=np.nan, jitter_mae_ms=np.nan,
                      jitter_median_abs_ms=np.nan, jitter_p95_abs_ms=np.nan, jitter_sec_signed=np.array([]))
        fpmin_fu = np.nan
        mref_fu = mtest_fu = np.array([], dtype=int)
    
    # PCA frontal supervised
    mref_fs, mtest_fs, uref_fs, utest_fs = match_events_one_to_one(ref_events_abs, pca_front_sup_events_abs, sfreq, MATCH_TOL_SEC)
    met_fs = compute_detection_metrics(mref_fs, mtest_fs, uref_fs, utest_fs, sfreq)
    fpmin_fs = met_fs["FP"] / (duration_sec / 60.0) if duration_sec > 0 else np.nan
    
    # ICA supervised
    mref_is, mtest_is, uref_is, utest_is = match_events_one_to_one(ref_events_abs, ica_sup_events_abs, sfreq, MATCH_TOL_SEC)
    met_is = compute_detection_metrics(mref_is, mtest_is, uref_is, utest_is, sfreq)
    fpmin_is = met_is["FP"] / (duration_sec / 60.0) if duration_sec > 0 else np.nan
    
    # ICA unsupervised
    mref_iu, mtest_iu, uref_iu, utest_iu = match_events_one_to_one(ref_events_abs, ica_unsup_events_abs, sfreq, MATCH_TOL_SEC)
    met_iu = compute_detection_metrics(mref_iu, mtest_iu, uref_iu, utest_iu, sfreq)
    fpmin_iu = met_iu["FP"] / (duration_sec / 60.0) if duration_sec > 0 else np.nan

    # Multi PCA unsupervised
    mref_mu, mtest_mu, uref_mu, utest_mu = match_events_one_to_one(ref_events_abs, multi_pca_events_abs, sfreq, MATCH_TOL_SEC)
    met_mu = compute_detection_metrics(mref_mu, mtest_mu, uref_mu, utest_mu, sfreq)
    fpmin_mu = met_mu["FP"] / (duration_sec / 60.0) if duration_sec > 0 else np.nan
    
# =============================================================================
    # WAVEFORM LAG + CORR (SECONDARY, like your v7)
# =============================================================================
    def corr_with_lag(refx, testx):
        r0, _ = pearsonr(refx, testx)
        lag = best_lag_via_xcorr(refx, testx)
        aligned = shift_with_zeros(testx, lag)
        r1, _ = pearsonr(refx, aligned)
        return r0, r1, lag, aligned
    
    # Global PCA
    r_g0, r_g1, lag_g, pca_all_aligned = corr_with_lag(eog_ref_proc, pca_all_proc)
    
    # Frontal PCA unsup
    if pca_front_unsup_proc is not None:
        r_fu0, r_fu1, lag_fu, pca_front_unsup_aligned = corr_with_lag(eog_ref_proc, pca_front_unsup_proc)
    else:
        r_fu0 = r_fu1 = np.nan
        lag_fu = 0
        pca_front_unsup_aligned = np.zeros_like(eog_ref_proc)
    
    # Frontal PCA supervised
    r_fs0, r_fs1, lag_fs, pca_front_sup_aligned = corr_with_lag(eog_ref_proc, pca_front_sup_proc)
    
    # ICA supervised
    r_is0, r_is1, lag_is, ica_sup_aligned = corr_with_lag(eog_ref_proc, ica_sup_proc)
    
    # ICA unsupervised
    r_iu0, r_iu1, lag_iu, ica_unsup_aligned = corr_with_lag(eog_ref_proc, ica_unsup_proc)

    # Multi PCA unsupervised
    r_mu0, r_mu1, lag_mu, multi_pca_aligned = corr_with_lag(eog_ref_proc, multi_pca_proc)
    
# =============================================================================
    # PLOTS (event-aware)
# =============================================================================
    # One figure per method (so you can compare directly)
    plot_event_overlay(
        out_png=os.path.join(OUTPUT_DIR, f"{subject_id}_EV_PCAglobal.png"),
        subject_id=subject_id, sfreq=sfreq, first_samp=raw.first_samp,
        ref_trace_proc=eog_ref_proc, test_trace_proc=pca_all_proc,
        ref_events_abs=ref_events_abs, test_events_abs=pca_all_events_abs,
        matched_ref_abs=mref_g, matched_test_abs=mtest_g,
        metrics=met_g, label_test="Test: PCA global", match_tol_sec=MATCH_TOL_SEC,
        seconds=PLOT_SECONDS
    )
    
    if pca_front_unsup_proc is not None:
        plot_event_overlay(
            out_png=os.path.join(OUTPUT_DIR, f"{subject_id}_EV_PCAfrontal_UNSUP.png"),
            subject_id=subject_id, sfreq=sfreq, first_samp=raw.first_samp,
            ref_trace_proc=eog_ref_proc, test_trace_proc=pca_front_unsup_proc,
            ref_events_abs=ref_events_abs, test_events_abs=pca_front_unsup_events_abs,
            matched_ref_abs=mref_fu, matched_test_abs=mtest_fu,
            metrics=met_fu, label_test=f"Test: PCA frontal UNSUP ({FRONTAL_PCA_UNSUPERVISED_MODE})",
            match_tol_sec=MATCH_TOL_SEC, seconds=PLOT_SECONDS
        )
    
    plot_event_overlay(
        out_png=os.path.join(OUTPUT_DIR, f"{subject_id}_EV_PCAfrontal_SUP.png"),
        subject_id=subject_id, sfreq=sfreq, first_samp=raw.first_samp,
        ref_trace_proc=eog_ref_proc, test_trace_proc=pca_front_sup_proc,
        ref_events_abs=ref_events_abs, test_events_abs=pca_front_sup_events_abs,
        matched_ref_abs=mref_fs, matched_test_abs=mtest_fs,
        metrics=met_fs, label_test="Test: PCA frontal SUP (corr-ranked sensors)",
        match_tol_sec=MATCH_TOL_SEC, seconds=PLOT_SECONDS
    )
    
    plot_event_overlay(
        out_png=os.path.join(OUTPUT_DIR, f"{subject_id}_EV_ICAsup_IC{ica_sup_ic}.png"),
        subject_id=subject_id, sfreq=sfreq, first_samp=raw.first_samp,
        ref_trace_proc=eog_ref_proc, test_trace_proc=ica_sup_proc,
        ref_events_abs=ref_events_abs, test_events_abs=ica_sup_events_abs,
        matched_ref_abs=mref_is, matched_test_abs=mtest_is,
        metrics=met_is, label_test=f"Test: ICA SUP (IC{ica_sup_ic})",
        match_tol_sec=MATCH_TOL_SEC, seconds=PLOT_SECONDS
    )
    
    plot_event_overlay(
        out_png=os.path.join(OUTPUT_DIR, f"{subject_id}_EV_ICAunsup_IC{ica_unsup_ic}.png"),
        subject_id=subject_id, sfreq=sfreq, first_samp=raw.first_samp,
        ref_trace_proc=eog_ref_proc, test_trace_proc=ica_unsup_proc,
        ref_events_abs=ref_events_abs, test_events_abs=ica_unsup_events_abs,
        matched_ref_abs=mref_iu, matched_test_abs=mtest_iu,
        metrics=met_iu, label_test=f"Test: ICA UNSUP ({ICA_UNSUP_MODE}; IC{ica_unsup_ic}; sign={UNSUP_SIGN_MODE})",
        match_tol_sec=MATCH_TOL_SEC, seconds=PLOT_SECONDS
    )

    plot_event_overlay(
        out_png=os.path.join(OUTPUT_DIR, f"{subject_id}_EV_MultiPCAunsup.png"),
        subject_id=subject_id, sfreq=sfreq, first_samp=raw.first_samp,
        ref_trace_proc=eog_ref_proc, test_trace_proc=multi_pca_proc,
        ref_events_abs=ref_events_abs, test_events_abs=multi_pca_events_abs,
        matched_ref_abs=mref_mu, matched_test_abs=mtest_mu,
        metrics=met_mu, label_test="Test: Multi PCA UNSUP",
        match_tol_sec=MATCH_TOL_SEC, seconds=PLOT_SECONDS
    )
    
# =============================================================================
    # CSV ROW
# =============================================================================
    result = dict(
        subject=subject_id,
        file=str(data_path),
        sfreq_hz=sfreq,
        duration_sec=duration_sec,
        has_real_eog=True,
        eog_channel=eog_ch_name,
        match_tol_ms=1e3 * MATCH_TOL_SEC,
    
        # Waveform (secondary)
        pca_global_r_before=r_g0, pca_global_r_after=r_g1, pca_global_lag_samples=lag_g, pca_global_lag_sec=lag_g/sfreq,
        pca_frontal_unsup_mode=FRONTAL_PCA_UNSUPERVISED_MODE,
        pca_frontal_unsup_r_before=r_fu0, pca_frontal_unsup_r_after=r_fu1, pca_frontal_unsup_lag_samples=lag_fu, pca_frontal_unsup_lag_sec=lag_fu/sfreq,
        pca_frontal_sup_r_before=r_fs0, pca_frontal_sup_r_after=r_fs1, pca_frontal_sup_lag_samples=lag_fs, pca_frontal_sup_lag_sec=lag_fs/sfreq,
        ica_sup_ic=ica_sup_ic, ica_sup_abs_corr=ica_sup_abs_corr,
        ica_sup_r_before=r_is0, ica_sup_r_after=r_is1, ica_sup_lag_samples=lag_is, ica_sup_lag_sec=lag_is/sfreq,
    
        ica_unsup_mode=ICA_UNSUP_MODE,
        ica_unsup_fixed_ic=ICA_UNSUP_FIXED_IC if ICA_UNSUP_MODE == "fixed" else np.nan,
        ica_unsup_sign_mode=UNSUP_SIGN_MODE,
        ica_unsup_ic=ica_unsup_ic,
        ica_unsup_score=unsup_details.get("score", np.nan),
        ica_unsup_p_kurt=unsup_details.get("p_kurt", np.nan),
        ica_unsup_p_rate=unsup_details.get("p_rate", np.nan),
        ica_unsup_p_pos=unsup_details.get("p_pos", np.nan),
        ica_unsup_p_neg_penalty=unsup_details.get("p_neg_penalty", np.nan),
        ica_unsup_pos_rate_per_min=unsup_details.get("pos_rate_per_min", np.nan),
        ica_unsup_neg_rate_per_min=unsup_details.get("neg_rate_per_min", np.nan),
        ica_unsup_r_before=r_iu0, ica_unsup_r_after=r_iu1, ica_unsup_lag_samples=lag_iu, ica_unsup_lag_sec=lag_iu/sfreq,

        # Multi PCA unsupervised
        multi_pca_unsup_score=multi_pca_details.get("score", np.nan),
        multi_pca_unsup_r_before=r_mu0, multi_pca_unsup_r_after=r_mu1,
        multi_pca_unsup_lag_samples=lag_mu, multi_pca_unsup_lag_sec=lag_mu / sfreq,
    
        # Event metrics (primary) — PCA global
        pca_global_TP=met_g["TP"], pca_global_FP=met_g["FP"], pca_global_FN=met_g["FN"],
        pca_global_precision=met_g["precision"], pca_global_recall=met_g["recall"], pca_global_f1=met_g["f1"],
        pca_global_miss_rate=met_g["miss_rate"], pca_global_fp_per_min=fpmin_g,
        pca_global_jitter_mae_ms=met_g["jitter_mae_ms"], pca_global_jitter_p95_ms=met_g["jitter_p95_abs_ms"],
    
        # Event metrics — PCA frontal unsup
        pca_frontal_unsup_TP=met_fu["TP"], pca_frontal_unsup_FP=met_fu["FP"], pca_frontal_unsup_FN=met_fu["FN"],
        pca_frontal_unsup_precision=met_fu["precision"], pca_frontal_unsup_recall=met_fu["recall"], pca_frontal_unsup_f1=met_fu["f1"],
        pca_frontal_unsup_miss_rate=met_fu["miss_rate"], pca_frontal_unsup_fp_per_min=fpmin_fu,
        pca_frontal_unsup_jitter_mae_ms=met_fu["jitter_mae_ms"], pca_frontal_unsup_jitter_p95_ms=met_fu["jitter_p95_abs_ms"],
    
        # Event metrics — PCA frontal sup
        pca_frontal_sup_TP=met_fs["TP"], pca_frontal_sup_FP=met_fs["FP"], pca_frontal_sup_FN=met_fs["FN"],
        pca_frontal_sup_precision=met_fs["precision"], pca_frontal_sup_recall=met_fs["recall"], pca_frontal_sup_f1=met_fs["f1"],
        pca_frontal_sup_miss_rate=met_fs["miss_rate"], pca_frontal_sup_fp_per_min=fpmin_fs,
        pca_frontal_sup_jitter_mae_ms=met_fs["jitter_mae_ms"], pca_frontal_sup_jitter_p95_ms=met_fs["jitter_p95_abs_ms"],
    
        # Event metrics — ICA sup
        ica_sup_TP=met_is["TP"], ica_sup_FP=met_is["FP"], ica_sup_FN=met_is["FN"],
        ica_sup_precision=met_is["precision"], ica_sup_recall=met_is["recall"], ica_sup_f1=met_is["f1"],
        ica_sup_miss_rate=met_is["miss_rate"], ica_sup_fp_per_min=fpmin_is,
        ica_sup_jitter_mae_ms=met_is["jitter_mae_ms"], ica_sup_jitter_p95_ms=met_is["jitter_p95_abs_ms"],
    
        # Event metrics — ICA unsup
        ica_unsup_TP=met_iu["TP"], ica_unsup_FP=met_iu["FP"], ica_unsup_FN=met_iu["FN"],
        ica_unsup_precision=met_iu["precision"], ica_unsup_recall=met_iu["recall"], ica_unsup_f1=met_iu["f1"],
        ica_unsup_miss_rate=met_iu["miss_rate"], ica_unsup_fp_per_min=fpmin_iu,
        ica_unsup_jitter_mae_ms=met_iu["jitter_mae_ms"], ica_unsup_jitter_p95_ms=met_iu["jitter_p95_abs_ms"],

        # Event metrics — Multi PCA unsup
        multi_pca_unsup_TP=met_mu["TP"], multi_pca_unsup_FP=met_mu["FP"], multi_pca_unsup_FN=met_mu["FN"],
        multi_pca_unsup_precision=met_mu["precision"], multi_pca_unsup_recall=met_mu["recall"], multi_pca_unsup_f1=met_mu["f1"],
        multi_pca_unsup_miss_rate=met_mu["miss_rate"], multi_pca_unsup_fp_per_min=fpmin_mu,
        multi_pca_unsup_jitter_mae_ms=met_mu["jitter_mae_ms"], multi_pca_unsup_jitter_p95_ms=met_mu["jitter_p95_abs_ms"],
    )
    
    print(
        f"  Done | Events F1: PCAglob={met_g['f1']:.3f} | "
        f"PCAfrontUNSUP={met_fu['f1'] if np.isfinite(met_fu['f1']) else np.nan} | "
        f"PCAfrontSUP={met_fs['f1']:.3f} | "
        f"ICAsup={met_is['f1']:.3f} | "
        f"ICAunsup={met_iu['f1']:.3f} (IC{ica_unsup_ic}, mode={ICA_UNSUP_MODE}, sign={UNSUP_SIGN_MODE}) | "
        f"MultiPCAunsup={met_mu['f1']:.3f}"
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
csv_path = os.path.join(OUTPUT_DIR, "eog_event_validation_full_v7style_summary.csv")
df.to_csv(csv_path, index=False)

print("\n=== SUMMARY (EVENT METRICS) ===")
if len(df) > 0:
    cols = [
        "pca_global_f1", "pca_frontal_unsup_f1", "pca_frontal_sup_f1",
        "ica_sup_f1", "ica_unsup_f1",
        "multi_pca_unsup_f1",
        "ica_unsup_score", "multi_pca_unsup_score"
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].describe(include="all"))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved figures in: {OUTPUT_DIR}")
else:
    print("No valid files processed.")

"""
Unified ECG/EOG estimation pipeline.

This script consolidates helper utilities from the four legacy scripts:
  - ECG_estimation_corr.py
  - ECG_estimation_events_based.py
  - EOG_estimation_corr.py
  - EOG_estimation_events_based.py

It provides a single CLI entry point that can run one or more pipelines
in a single pass over the dataset to avoid repeated file scans and loads.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import correlate, find_peaks, welch
from scipy.stats import kurtosis, pearsonr, skew

# =============================================================================
# COMMON HELPERS
# =============================================================================


def safe_zscore(x: np.ndarray) -> np.ndarray:
    """Return z-scored x with a safe fallback for constant or NaN signals."""
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sd = np.std(x)
    if sd == 0 or np.isnan(sd):
        return x - mu
    return (x - mu) / sd


def best_lag_via_xcorr(x_ref: np.ndarray, y: np.ndarray) -> int:
    """Return integer lag (samples) maximizing xcorr(y, x_ref)."""
    n = min(len(x_ref), len(y))
    x = x_ref[:n] - np.mean(x_ref[:n])
    yy = y[:n] - np.mean(y[:n])
    c = correlate(yy, x, mode="full")
    lags = np.arange(-n + 1, n)
    return int(lags[np.argmax(c)])


def shift_with_zeros(y: np.ndarray, lag: int) -> np.ndarray:
    """Shift 1D array by lag samples, padding with zeros to preserve length."""
    y = np.asarray(y)
    if lag > 0:
        return np.concatenate([y[lag:], np.zeros(lag)])
    if lag < 0:
        k = -lag
        return np.concatenate([np.zeros(k), y[:-k]])
    return y.copy()


def extract_event_samples(events: np.ndarray) -> np.ndarray:
    """Return event sample indices from an MNE events array."""
    if events is None or len(events) == 0:
        return np.array([], dtype=int)
    return np.asarray(events[:, 0], dtype=int)


def match_events_one_to_one(
    ref_events: np.ndarray,
    test_events: np.ndarray,
    sfreq: float,
    tol_sec: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Greedy one-to-one match with tolerance window."""
    if len(ref_events) == 0 and len(test_events) == 0:
        empty = np.array([], dtype=int)
        return empty, empty, empty, empty

    ref = np.sort(np.asarray(ref_events, dtype=int))
    test = np.sort(np.asarray(test_events, dtype=int))

    tol_samp = int(round(tol_sec * sfreq))
    matched_ref = []
    matched_test = []
    used_test = set()

    for r in ref:
        lo = r - tol_samp
        hi = r + tol_samp
        candidates = [t for t in test if lo <= t <= hi and t not in used_test]
        if not candidates:
            continue
        best = min(candidates, key=lambda t: abs(t - r))
        matched_ref.append(r)
        matched_test.append(best)
        used_test.add(best)

    matched_ref = np.asarray(matched_ref, dtype=int)
    matched_test = np.asarray(matched_test, dtype=int)
    unmatched_ref = np.setdiff1d(ref, matched_ref, assume_unique=False)
    unmatched_test = np.setdiff1d(test, matched_test, assume_unique=False)
    return matched_ref, matched_test, unmatched_ref, unmatched_test


def compute_detection_metrics(
    matched_ref: np.ndarray,
    matched_test: np.ndarray,
    unmatched_ref: np.ndarray,
    unmatched_test: np.ndarray,
    sfreq: float,
) -> Dict[str, float]:
    """Compute TP/FP/FN, precision/recall/F1, and jitter stats."""
    tp = len(matched_ref)
    fp = len(unmatched_test)
    fn = len(unmatched_ref)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    miss_rate = fn / (tp + fn) if tp + fn > 0 else 0.0

    if tp > 0:
        jitter_sec = (matched_test - matched_ref) / sfreq
        jitter_ms = 1e3 * jitter_sec
        jitter_abs_ms = np.abs(jitter_ms)
        jitter_mean_ms = float(np.mean(jitter_ms))
        jitter_std_ms = float(np.std(jitter_ms))
        jitter_mae_ms = float(np.mean(jitter_abs_ms))
        jitter_median_abs_ms = float(np.median(jitter_abs_ms))
        jitter_p95_abs_ms = float(np.percentile(jitter_abs_ms, 95))
    else:
        jitter_sec = np.array([])
        jitter_mean_ms = jitter_std_ms = jitter_mae_ms = jitter_median_abs_ms = jitter_p95_abs_ms = np.nan

    return dict(
        TP=int(tp),
        FP=int(fp),
        FN=int(fn),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        miss_rate=float(miss_rate),
        jitter_mean_ms_signed=float(jitter_mean_ms),
        jitter_std_ms_signed=float(jitter_std_ms),
        jitter_mae_ms=float(jitter_mae_ms),
        jitter_median_abs_ms=float(jitter_median_abs_ms),
        jitter_p95_abs_ms=float(jitter_p95_abs_ms),
        jitter_sec_signed=jitter_sec,
    )


def discover_data_paths(dataset_root: str) -> List[Path]:
    """Discover MEG FIF/CTF files under the dataset root."""
    root = Path(dataset_root)
    fif_files = [
        p
        for p in root.rglob("*_meg.fif")
        if p.is_file() and "derivatives" not in str(p) and ".git" not in str(p)
    ]
    ctf_files = [
        p
        for p in root.rglob("*.ds")
        if p.is_dir() and "derivatives" not in str(p) and ".git" not in str(p)
    ]
    return sorted(fif_files + ctf_files)


def load_raw(data_path: Path) -> mne.io.BaseRaw:
    """Load FIF or CTF data with preload enabled."""
    if data_path.suffix == ".fif":
        return mne.io.read_raw_fif(data_path, preload=True, verbose=False)
    if data_path.suffix == ".ds":
        return mne.io.read_raw_ctf(data_path, preload=True, verbose=False)
    raise ValueError(f"Unsupported file format: {data_path}")


# =============================================================================
# ECG HELPERS (corr + events)
# =============================================================================

ECG_ICA_N_COMPONENTS = 0.99
ECG_ICA_METHOD = "fastica"
ECG_ICA_RANDOM_STATE = 97
ECG_ICA_MAX_ITER = 1000
ECG_ICA_L_FREQ = 5.0
ECG_ICA_H_FREQ = 40.0

ECG_HR_MIN_BPM = 40
ECG_HR_MAX_BPM = 180
ECG_HR_BAND_HZ = (ECG_HR_MIN_BPM / 60.0, ECG_HR_MAX_BPM / 60.0)
ECG_UNSUP_SCORE_SECONDS = None
ECG_PEAK_MIN_DISTANCE_SEC = 0.30
ECG_PEAK_PROMINENCE = 1.3
ECG_SPIKE_Z_ABS_THRESHOLD = 4.0
ECG_SPIKE_RATE_MIN_PER_MIN = 20
ECG_SPIKE_RATE_MAX_PER_MIN = 220
ECG_RR_MAD_TOO_SMALL_SEC = 0.005
ECG_RR_MAD_TOO_LARGE_SEC = 0.25
ECG_W_HR_BANDPOWER = 0.20
ECG_W_AUTOCORR = 0.20
ECG_W_PEAK_TRAIN = 0.25
ECG_W_KURT = 0.10
ECG_W_SPIKE_STRENGTH = 0.25


def detect_ecg_events_from_1d_trace(trace_1d: np.ndarray, sfreq: float):
    """Run find_ecg_events on a 1D trace wrapped as an ECG RawArray."""
    info = mne.create_info(ch_names=["ICA_ECG"], sfreq=sfreq, ch_types=["ecg"])
    raw_tmp = mne.io.RawArray(trace_1d[np.newaxis, :], info, verbose=False)
    events, _, _, ecg_mne = mne.preprocessing.find_ecg_events(
        raw_tmp,
        ch_name="ICA_ECG",
        return_ecg=True,
        verbose=False,
    )
    return extract_event_samples(events), ecg_mne[0]


def _ecg_segment_for_scoring(x: np.ndarray, sfreq: float) -> np.ndarray:
    if ECG_UNSUP_SCORE_SECONDS is None:
        return np.asarray(x, dtype=float)
    n_seg = min(len(x), int(round(ECG_UNSUP_SCORE_SECONDS * sfreq)))
    return np.asarray(x[:n_seg], dtype=float)


def ecg_bandpower_ratio_hr(x: np.ndarray, sfreq: float, band_hz=(0.8, 2.5)) -> float:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < int(sfreq * 5):
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    psd = np.abs(np.fft.rfft(x)) ** 2
    valid = freqs > 0
    freqs = freqs[valid]
    psd = psd[valid]
    denom = psd.sum()
    if denom <= 0:
        return 0.0
    lo, hi = band_hz
    band_mask = (freqs >= lo) & (freqs <= hi)
    return float(psd[band_mask].sum() / denom)


def ecg_autocorr_periodicity_score(x: np.ndarray, sfreq: float, hr_band_hz=(0.75, 2.5)) -> float:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < int(sfreq * 5):
        return 0.0
    ac = correlate(x, x, mode="full")
    ac = ac[ac.size // 2:]
    if ac[0] == 0:
        return 0.0
    ac = ac / ac[0]
    lo, hi = hr_band_hz
    min_period = 1.0 / hi
    max_period = 1.0 / lo
    min_lag = int(round(min_period * sfreq))
    max_lag = int(round(max_period * sfreq))
    if max_lag <= min_lag or max_lag >= len(ac):
        return 0.0
    return float(np.max(ac[min_lag:max_lag + 1]))


def ecg_peak_train_plausibility(
    x: np.ndarray,
    sfreq: float,
    min_bpm: float = 40.0,
    max_bpm: float = 180.0,
    peak_prominence: float = 1.3,
    min_distance_sec: float = 0.30,
):
    z = safe_zscore(x)
    z_abs = np.abs(z)
    min_dist = max(1, int(round(min_distance_sec * sfreq)))
    peaks, _ = find_peaks(z_abs, prominence=peak_prominence, distance=min_dist)
    duration_sec = len(z) / sfreq
    if duration_sec <= 0 or len(peaks) < 2:
        return 0.0, float("nan"), int(len(peaks)), float("nan")
    bpm = (len(peaks) / duration_sec) * 60.0
    if min_bpm <= bpm <= max_bpm:
        hr_score = 1.0
    else:
        if bpm < min_bpm:
            d = (min_bpm - bpm) / min_bpm
        else:
            d = (bpm - max_bpm) / max_bpm
        hr_score = float(np.exp(-3.0 * d))
    rr = np.diff(peaks) / sfreq
    rr_med = np.median(rr)
    rr_mad = np.median(np.abs(rr - rr_med)) + 1e-12
    rr_score = 1.0
    if rr_mad < ECG_RR_MAD_TOO_SMALL_SEC:
        rr_score = 0.3
    elif rr_mad > ECG_RR_MAD_TOO_LARGE_SEC:
        rr_score = 0.3
    return float(hr_score * rr_score), float(bpm), int(len(peaks)), float(rr_mad)


def ecg_spike_strength_score(x: np.ndarray, sfreq: float):
    z = safe_zscore(x)
    z_abs = np.abs(z)
    min_dist = max(1, int(round(ECG_PEAK_MIN_DISTANCE_SEC * sfreq)))
    spikes, _ = find_peaks(z_abs, height=ECG_SPIKE_Z_ABS_THRESHOLD, distance=min_dist)
    duration_sec = len(z) / sfreq
    if duration_sec <= 0:
        return 0.0, 0.0, 0.0, 0
    rate_per_min = (len(spikes) / duration_sec) * 60.0
    frac_above = float(np.mean(z_abs >= ECG_SPIKE_Z_ABS_THRESHOLD))
    if ECG_SPIKE_RATE_MIN_PER_MIN <= rate_per_min <= ECG_SPIKE_RATE_MAX_PER_MIN:
        rate_score = 1.0
    else:
        if rate_per_min < ECG_SPIKE_RATE_MIN_PER_MIN:
            d = (ECG_SPIKE_RATE_MIN_PER_MIN - rate_per_min) / max(1e-9, ECG_SPIKE_RATE_MIN_PER_MIN)
        else:
            d = (rate_per_min - ECG_SPIKE_RATE_MAX_PER_MIN) / max(1e-9, ECG_SPIKE_RATE_MAX_PER_MIN)
        rate_score = float(np.exp(-3.0 * d))
    frac_score = float(np.tanh(frac_above * 50.0))
    score = float(0.7 * rate_score + 0.3 * frac_score)
    return score, float(rate_per_min), float(frac_above), int(len(spikes))


def ecg_unsupervised_ic_score(ic: np.ndarray, sfreq: float) -> Dict:
    x = _ecg_segment_for_scoring(ic, sfreq)
    z = safe_zscore(x)
    p_hr = ecg_bandpower_ratio_hr(z, sfreq, band_hz=ECG_HR_BAND_HZ)
    p_ac = ecg_autocorr_periodicity_score(z, sfreq, hr_band_hz=ECG_HR_BAND_HZ)
    p_peaks, bpm, n_peaks, rr_mad = ecg_peak_train_plausibility(
        z,
        sfreq,
        min_bpm=ECG_HR_MIN_BPM,
        max_bpm=ECG_HR_MAX_BPM,
        peak_prominence=ECG_PEAK_PROMINENCE,
        min_distance_sec=ECG_PEAK_MIN_DISTANCE_SEC,
    )
    k = float(kurtosis(z, fisher=False, bias=False)) if len(z) > 10 else 0.0
    p_kurt = float(np.tanh((k - 3.0) / 5.0))
    p_kurt = max(0.0, p_kurt)
    p_spike, spike_rate_per_min, frac_above_thr, n_spikes = ecg_spike_strength_score(z, sfreq)
    score = (
        ECG_W_HR_BANDPOWER * p_hr
        + ECG_W_AUTOCORR * p_ac
        + ECG_W_PEAK_TRAIN * p_peaks
        + ECG_W_KURT * p_kurt
        + ECG_W_SPIKE_STRENGTH * p_spike
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
    )


def fit_ecg_ica_and_sources(raw_meg: mne.io.BaseRaw) -> Tuple[mne.preprocessing.ICA, np.ndarray]:
    tmp = raw_meg.copy()
    if ECG_ICA_L_FREQ is not None or ECG_ICA_H_FREQ is not None:
        tmp.filter(ECG_ICA_L_FREQ, ECG_ICA_H_FREQ, fir_design="firwin", verbose=False)
    ica = mne.preprocessing.ICA(
        n_components=ECG_ICA_N_COMPONENTS,
        method=ECG_ICA_METHOD,
        random_state=ECG_ICA_RANDOM_STATE,
        max_iter=ECG_ICA_MAX_ITER,
    )
    ica.fit(tmp, verbose=False)
    sources = ica.get_sources(tmp).get_data()
    if sources.size == 0:
        raise RuntimeError("ICA produced no sources.")
    return ica, sources


def pick_best_ecg_ic_supervised(sources: np.ndarray, ecg_ref: np.ndarray) -> Tuple[int, float]:
    n = min(sources.shape[1], len(ecg_ref))
    S = sources[:, :n]
    y = ecg_ref[:n]
    y0 = y - np.mean(y)
    ysd = np.std(y0) + 1e-12
    scores = np.zeros(S.shape[0], dtype=float)
    for i in range(S.shape[0]):
        si = S[i, :] - np.mean(S[i, :])
        ssd = np.std(si) + 1e-12
        scores[i] = np.abs(np.dot(si, y0) / (len(y0) * ssd * ysd))
    best = int(np.argmax(scores))
    return best, float(scores[best])


def pick_best_ecg_ic_unsupervised(sources: np.ndarray, sfreq: float) -> Tuple[int, Dict, List[Dict]]:
    all_details = []
    for i in range(sources.shape[0]):
        d_pos = ecg_unsupervised_ic_score(sources[i, :], sfreq)
        d_neg = ecg_unsupervised_ic_score(-sources[i, :], sfreq)
        if d_neg["score"] > d_pos["score"]:
            d = d_neg
            d["flip_for_score"] = True
        else:
            d = d_pos
            d["flip_for_score"] = False
        d["ic"] = int(i)
        all_details.append(d)
    all_details_sorted = sorted(all_details, key=lambda x: x["score"], reverse=True)
    best = all_details_sorted[0]
    return int(best["ic"]), best, all_details_sorted


# =============================================================================
# EOG HELPERS (corr + events)
# =============================================================================

EOG_L_FREQ = 1.0
EOG_H_FREQ = 10.0
EOG_N_TOP_CHANNELS_FOR_PCA = 30
EOG_FALLBACK_USE_ALL_MEG = True
EOG_FRONTAL_TOPK_BY_CORR = 40
EOG_FRONTAL_PCA_UNSUPERVISED_MODE = "layout"
EOG_FRONTAL_TOPK_UNSUPERVISED = 40
EOG_FRONTAL_REGEX = r"MEG0(1|2|3)"

EOG_ICA_N_COMPONENTS = 0.99
EOG_ICA_METHOD = "fastica"
EOG_ICA_RANDOM_STATE = 97
EOG_ICA_MAX_ITER = 1000

EOG_UNSUP_SCORE_SECONDS = None
EOG_ICA_UNSUP_MODE = "heuristic"
EOG_ICA_UNSUP_FIXED_IC = 0
EOG_UNSUP_SIGN_MODE = "peak_polarity"

EOG_BLINK_MIN_PER_MIN = 2
EOG_BLINK_MAX_PER_MIN = 80
EOG_PEAK_PROMINENCE = 1.0
EOG_PEAK_MIN_DISTANCE_SEC = 0.05

EOG_POS_SPIKE_Z_THR = 2.5
EOG_NEG_SPIKE_Z_THR = 2.0
EOG_POS_SPIKE_MIN_PER_MIN = 4
EOG_POS_SPIKE_MAX_PER_MIN = 80
EOG_NEG_SPIKE_MAX_PER_MIN = 6

EOG_W_KURT = 0.20
EOG_W_RATE = 0.20
EOG_W_POS_SPIKE = 0.30
EOG_W_NEG_PENALTY = 0.10
EOG_W_BANDPOWER = 0.10
EOG_W_SKEW = 0.10

EOG_LOW_BAND = (0.5, 4.0)
EOG_HIGH_BAND = (8.0, 30.0)
EOG_BANDPOWER_SIGMOID_K = 1.5


def process_trace_bandpass_z(x: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    """Band-pass filter a trace and z-score it."""
    x = np.asarray(x, dtype=float)
    if l_freq is not None or h_freq is not None:
        x = mne.filter.filter_data(x, sfreq, l_freq, h_freq, verbose=False)
    return safe_zscore(x)


def pick_prefer_vertical_eog(raw: mne.io.BaseRaw) -> int:
    """Pick a likely vertical EOG channel if available."""
    eog_picks = mne.pick_types(raw.info, eog=True)
    if len(eog_picks) == 0:
        return -1
    for idx in eog_picks:
        name = raw.ch_names[idx].lower()
        if "veog" in name or "v-eog" in name or "v_eog" in name:
            return int(idx)
    return int(eog_picks[0])


def pca_first_component(data_ch_by_time: np.ndarray) -> np.ndarray:
    """Return first principal component of ch x time data."""
    data = data_ch_by_time - np.mean(data_ch_by_time, axis=1, keepdims=True)
    u, s, vt = np.linalg.svd(data, full_matrices=False)
    pc1 = u[:, 0] @ np.diag(s[:1]) @ vt[:1, :]
    return pc1.squeeze()


def build_synth_eog_pca_all(raw_meg: mne.io.BaseRaw) -> np.ndarray:
    data = raw_meg.get_data()
    if data.shape[0] == 0:
        raise RuntimeError("No MEG channels to build PCA.")
    variances = np.var(data, axis=1)
    order = np.argsort(variances)[::-1]
    if len(order) < EOG_N_TOP_CHANNELS_FOR_PCA and not EOG_FALLBACK_USE_ALL_MEG:
        raise RuntimeError("Not enough channels for PCA.")
    idx = order[: min(len(order), EOG_N_TOP_CHANNELS_FOR_PCA)]
    return pca_first_component(data[idx, :])


def build_synth_eog_pca_frontal_supervised(raw_meg: mne.io.BaseRaw, eog_ref_proc: np.ndarray) -> np.ndarray:
    data = raw_meg.get_data()
    n = min(data.shape[1], len(eog_ref_proc))
    data = data[:, :n]
    y = eog_ref_proc[:n]
    y = y - np.mean(y)
    ysd = np.std(y) + 1e-12
    corrs = []
    for ch in range(data.shape[0]):
        x = data[ch, :] - np.mean(data[ch, :])
        xsd = np.std(x) + 1e-12
        corrs.append(np.abs(np.dot(x, y) / (n * xsd * ysd)))
    corrs = np.asarray(corrs)
    order = np.argsort(corrs)[::-1]
    idx = order[: min(len(order), EOG_FRONTAL_TOPK_BY_CORR)]
    return pca_first_component(data[idx, :])


def pick_frontal_channels_unsupervised(raw_meg: mne.io.BaseRaw) -> np.ndarray:
    if EOG_FRONTAL_PCA_UNSUPERVISED_MODE == "regex":
        import re

        pat = re.compile(EOG_FRONTAL_REGEX)
        picks = [i for i, name in enumerate(raw_meg.ch_names) if pat.match(name)]
        return np.asarray(picks, dtype=int)

    if EOG_FRONTAL_PCA_UNSUPERVISED_MODE == "variance":
        data = raw_meg.get_data()
        variances = np.var(data, axis=1)
        order = np.argsort(variances)[::-1]
        return order[: min(len(order), EOG_FRONTAL_TOPK_UNSUPERVISED)]

    layout = mne.find_layout(raw_meg.info)
    y_coords = layout.pos[:, 1]
    order = np.argsort(y_coords)[::-1]
    return order[: min(len(order), EOG_FRONTAL_TOPK_UNSUPERVISED)]


def build_synth_eog_pca_frontal_unsupervised(raw_meg: mne.io.BaseRaw) -> np.ndarray:
    data = raw_meg.get_data()
    idx = pick_frontal_channels_unsupervised(raw_meg)
    if len(idx) == 0:
        raise RuntimeError("No frontal channels for unsupervised PCA.")
    return pca_first_component(data[idx, :])


def fit_ica_sources_once(raw_meg: mne.io.BaseRaw) -> np.ndarray:
    ica = mne.preprocessing.ICA(
        n_components=EOG_ICA_N_COMPONENTS,
        method=EOG_ICA_METHOD,
        random_state=EOG_ICA_RANDOM_STATE,
        max_iter=EOG_ICA_MAX_ITER,
    )
    ica.fit(raw_meg, verbose=False)
    sources = ica.get_sources(raw_meg).get_data()
    if sources.size == 0:
        raise RuntimeError("ICA produced no sources.")
    return sources


def pick_best_ic_supervised_from_sources(sources: np.ndarray, eog_ref_proc: np.ndarray) -> Tuple[int, float]:
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
    best = int(np.argmax(scores))
    return best, float(scores[best])


def eog_spike_strength_score(x: np.ndarray, sfreq: float):
    z = safe_zscore(x)
    pos_peaks, _ = find_peaks(z, height=EOG_POS_SPIKE_Z_THR, distance=max(1, int(round(EOG_PEAK_MIN_DISTANCE_SEC * sfreq))))
    neg_peaks, _ = find_peaks(-z, height=EOG_NEG_SPIKE_Z_THR, distance=max(1, int(round(EOG_PEAK_MIN_DISTANCE_SEC * sfreq))))
    duration_sec = len(z) / sfreq
    if duration_sec <= 0:
        return 0.0, 0.0, 0.0, 0.0
    pos_rate = (len(pos_peaks) / duration_sec) * 60.0
    neg_rate = (len(neg_peaks) / duration_sec) * 60.0
    if EOG_POS_SPIKE_MIN_PER_MIN <= pos_rate <= EOG_POS_SPIKE_MAX_PER_MIN:
        pos_score = 1.0
    else:
        pos_score = float(np.exp(-0.1 * abs(pos_rate - EOG_POS_SPIKE_MIN_PER_MIN)))
    if neg_rate <= EOG_NEG_SPIKE_MAX_PER_MIN:
        neg_score = 1.0
    else:
        neg_score = float(np.exp(-0.1 * (neg_rate - EOG_NEG_SPIKE_MAX_PER_MIN)))
    return pos_score, neg_score, float(pos_rate), float(neg_rate)


def eog_bandpower_ratio_score(x: np.ndarray, sfreq: float):
    x = x - np.mean(x)
    freqs, psd = welch(x, fs=sfreq, nperseg=min(len(x), int(sfreq * 2)))
    if len(freqs) == 0:
        return 0.0
    low_mask = (freqs >= EOG_LOW_BAND[0]) & (freqs <= EOG_LOW_BAND[1])
    high_mask = (freqs >= EOG_HIGH_BAND[0]) & (freqs <= EOG_HIGH_BAND[1])
    low_pow = np.sum(psd[low_mask])
    high_pow = np.sum(psd[high_mask]) + 1e-9
    ratio = low_pow / high_pow
    return float(1.0 / (1.0 + np.exp(-EOG_BANDPOWER_SIGMOID_K * (ratio - 1.0))))


def blink_likeness_score(ic: np.ndarray, sfreq: float) -> Dict:
    x = ic[: int(round(EOG_UNSUP_SCORE_SECONDS * sfreq))] if EOG_UNSUP_SCORE_SECONDS else ic
    z = safe_zscore(x)
    z_abs = np.abs(z)
    min_dist = max(1, int(round(EOG_PEAK_MIN_DISTANCE_SEC * sfreq)))
    peaks, _ = find_peaks(z_abs, prominence=EOG_PEAK_PROMINENCE, distance=min_dist)
    duration_sec = len(z) / sfreq
    if duration_sec <= 0:
        return dict(score=0.0, blink_rate_per_min=0.0, n_peaks=0)
    blink_rate = (len(peaks) / duration_sec) * 60.0
    if EOG_BLINK_MIN_PER_MIN <= blink_rate <= EOG_BLINK_MAX_PER_MIN:
        rate_score = 1.0
    else:
        rate_score = float(np.exp(-0.1 * abs(blink_rate - EOG_BLINK_MIN_PER_MIN)))
    pos_score, neg_score, pos_rate, neg_rate = eog_spike_strength_score(z, sfreq)
    band_score = eog_bandpower_ratio_score(z, sfreq)
    k = float(kurtosis(z, fisher=False, bias=False)) if len(z) > 10 else 0.0
    k_score = float(np.tanh((k - 3.0) / 5.0))
    k_score = max(0.0, k_score)
    skew_score = float(np.tanh(skew(z)))
    skew_score = max(0.0, skew_score)
    score = (
        EOG_W_KURT * k_score
        + EOG_W_RATE * rate_score
        + EOG_W_POS_SPIKE * pos_score
        - EOG_W_NEG_PENALTY * (1.0 - neg_score)
        + EOG_W_BANDPOWER * band_score
        + EOG_W_SKEW * skew_score
    )
    return dict(
        score=float(score),
        blink_rate_per_min=float(blink_rate),
        n_peaks=int(len(peaks)),
        pos_spike_rate_per_min=float(pos_rate),
        neg_spike_rate_per_min=float(neg_rate),
        bandpower_score=float(band_score),
        kurtosis=float(k),
        skew=float(skew(z)),
        pos_score=float(pos_score),
        neg_score=float(neg_score),
    )


def pick_best_ic_unsupervised_from_sources(
    sources: np.ndarray,
    sfreq: float,
    mode: str = "heuristic",
    fixed_ic: int = 0,
) -> Tuple[int, Dict]:
    if mode == "fixed":
        return int(fixed_ic), dict(score=np.nan, ic=int(fixed_ic))
    all_details = []
    for i in range(sources.shape[0]):
        d_pos = blink_likeness_score(sources[i, :], sfreq)
        d_neg = blink_likeness_score(-sources[i, :], sfreq)
        if d_neg["score"] > d_pos["score"]:
            d = d_neg
            d["flip_for_score"] = True
        else:
            d = d_pos
            d["flip_for_score"] = False
        d["ic"] = int(i)
        all_details.append(d)
    best = sorted(all_details, key=lambda x: x["score"], reverse=True)[0]
    return int(best["ic"]), best


def apply_unsup_sign_convention_peak_polarity(ic: np.ndarray, sfreq: float) -> np.ndarray:
    z = safe_zscore(ic)
    pos_peak = np.max(z)
    neg_peak = np.min(z)
    return -ic if abs(neg_peak) > abs(pos_peak) else ic


def apply_unsup_sign_convention_frontal_proxy(ic: np.ndarray, frontal_proxy: np.ndarray) -> np.ndarray:
    return -ic if np.corrcoef(ic, frontal_proxy)[0, 1] < 0 else ic


def events_from_trace_via_mne_find_eog(trace: np.ndarray, sfreq: float, first_samp: int):
    info = mne.create_info(ch_names=["EOG_SYNTH"], sfreq=sfreq, ch_types=["eog"])
    raw_tmp = mne.io.RawArray(trace[np.newaxis, :], info, verbose=False)
    events = mne.preprocessing.find_eog_events(raw_tmp, ch_name="EOG_SYNTH", verbose=False)
    events[:, 0] += first_samp
    return events, extract_event_samples(events)


# =============================================================================
# ECG CORR PIPELINE
# =============================================================================


def plot_ecg_corr_present(
    out_png: str,
    t: np.ndarray,
    ecg_raw_real: np.ndarray,
    ecg_ref_mne: np.ndarray,
    ecg_mne_meg_aligned: np.ndarray,
    ecg_ica_sup_aligned: np.ndarray,
    ecg_ica_unsup_aligned: np.ndarray,
    subject_id: str,
    sfreq: float,
    lag_mne: int,
    lag_sup: int,
    lag_unsup: int,
    r_mne_after: float,
    r_sup_after: float,
    r_unsup_after: float,
    sup_ic: int,
    sup_abs_corr: float,
    unsup_ic: int,
    unsup_details: dict,
):
    n_plot = len(t)
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(t, safe_zscore(ecg_raw_real[:n_plot]))
    axes[0].set_title("1) Real ECG channel (raw) [z-score]")
    axes[1].plot(t, safe_zscore(ecg_ref_mne[:n_plot]))
    axes[1].set_title("2) Reference: MNE ECG from REAL ECG [z-score]")
    axes[2].plot(t, safe_zscore(ecg_mne_meg_aligned[:n_plot]))
    axes[2].set_title(f"3) MNE ECG from MEG (aligned) | lag={lag_mne} | r={r_mne_after:.3f}")
    axes[3].plot(t, safe_zscore(ecg_ica_sup_aligned[:n_plot]))
    axes[3].set_title(
        f"4) ICA ECG (SUPERVISED; bestIC={sup_ic}, absCorr={sup_abs_corr:.3f}) "
        f"| lag={lag_sup} | r={r_sup_after:.3f}"
    )
    axes[4].plot(t, safe_zscore(ecg_ica_unsup_aligned[:n_plot]))
    axes[4].set_title(
        f"5) ICA ECG (UNSUPERVISED; bestIC={unsup_ic}, score={unsup_details['score']:.3f}, "
        f"bpm≈{unsup_details['bpm_est']:.1f}) | lag={lag_unsup} | r={r_unsup_after:.3f}"
    )
    axes[4].set_xlabel("Time (s)")
    header = (
        f"{subject_id} | sfreq={sfreq:.2f} Hz | "
        f"MNE r={r_mne_after:.3f} | ICA-sup r={r_sup_after:.3f} | ICA-unsup r={r_unsup_after:.3f}"
    )
    fig.suptitle(header, y=0.995, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_ecg_overlay_present(
    out_png: str,
    t: np.ndarray,
    ecg_ref_mne: np.ndarray,
    ecg_mne_meg_aligned: np.ndarray,
    ecg_ica_sup_aligned: np.ndarray,
    ecg_ica_unsup_aligned: np.ndarray,
    subject_id: str,
    r_mne_after: float,
    r_sup_after: float,
    r_unsup_after: float,
):
    n_plot = len(t)
    plt.figure(figsize=(14, 5))
    plt.plot(t, safe_zscore(ecg_ref_mne[:n_plot]), label="Reference (MNE from REAL ECG)")
    plt.plot(t, safe_zscore(ecg_mne_meg_aligned[:n_plot]), label=f"MNE from MEG (r={r_mne_after:.3f})")
    plt.plot(t, safe_zscore(ecg_ica_sup_aligned[:n_plot]), label=f"ICA supervised (r={r_sup_after:.3f})")
    plt.plot(t, safe_zscore(ecg_ica_unsup_aligned[:n_plot]), label=f"ICA unsupervised (r={r_unsup_after:.3f})")
    plt.legend()
    plt.title(f"{subject_id} | Overlay (aligned) vs reference")
    plt.xlabel("Time (s)")
    plt.ylabel("a.u. (z-score)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_ecg_corr_meg_only(
    out_png: str,
    t: np.ndarray,
    ecg_mne_from_meg: np.ndarray,
    ecg_ica_unsup: np.ndarray,
    subject_id: str,
    sfreq: float,
    unsup_ic: int,
    unsup_details: dict,
):
    n_plot = len(t)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(t, safe_zscore(ecg_mne_from_meg[:n_plot]))
    axes[0].set_title("1) MNE ECG proxy from MEG-only (z-score)")
    axes[1].plot(t, safe_zscore(ecg_ica_unsup[:n_plot]))
    axes[1].set_title(
        f"2) ICA ECG proxy (UNSUPERVISED; IC={unsup_ic}, score={unsup_details['score']:.3f}, "
        f"bpm≈{unsup_details['bpm_est']:.1f})"
    )
    axes[1].set_xlabel("Time (s)")
    header = f"{subject_id} | sfreq={sfreq:.2f} Hz"
    fig.suptitle(header, y=0.995, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_ecg_overlay_meg_only(
    out_png: str,
    t: np.ndarray,
    ecg_mne_from_meg: np.ndarray,
    ecg_ica_unsup: np.ndarray,
    subject_id: str,
    unsup_ic: int,
    unsup_details: dict,
):
    n_plot = len(t)
    plt.figure(figsize=(14, 5))
    plt.plot(t, safe_zscore(ecg_mne_from_meg[:n_plot]), label="MNE-from-MEG proxy")
    plt.plot(t, safe_zscore(ecg_ica_unsup[:n_plot]), label=f"ICA-unsup proxy (IC{unsup_ic}, score={unsup_details['score']:.3f})")
    plt.legend()
    plt.title(f"{subject_id} | MEG-only overlay (no ECG reference)")
    plt.xlabel("Time (s)")
    plt.ylabel("a.u. (z-score)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def process_ecg_corr(
    raw: mne.io.BaseRaw,
    data_path: Path,
    output_dir: str,
    plot_seconds: float,
) -> Optional[Dict]:
    subject_id = data_path.stem
    sfreq = float(raw.info["sfreq"])
    ecg_picks = mne.pick_types(raw.info, ecg=True)
    has_ecg = len(ecg_picks) > 0

    ecg_ch_name = ""
    ecg_raw_real = None
    ecg_ref_mne = None

    if has_ecg:
        ecg_idx = int(ecg_picks[0])
        ecg_ch_name = raw.ch_names[ecg_idx]
        ecg_raw_real = raw.get_data(picks=[ecg_idx])[0]
        try:
            _, _, _, ecg_ref_mne = mne.preprocessing.find_ecg_events(
                raw,
                ch_name=ecg_ch_name,
                return_ecg=True,
                verbose=False,
            )
            ecg_ref_mne = ecg_ref_mne[0]
        except Exception as exc:
            print(f"  ERROR ECG reference for {subject_id}: {exc}")
            return None

    raw_meg_only = raw.copy().pick_types(meg=True, eeg=False, eog=False, ecg=False, stim=False)

    try:
        _, _, _, ecg_mne_from_meg = mne.preprocessing.find_ecg_events(
            raw_meg_only,
            ch_name=None,
            return_ecg=True,
            verbose=False,
        )
        ecg_mne_from_meg = ecg_mne_from_meg[0]
    except Exception as exc:
        print(f"  ERROR ECG from MEG for {subject_id}: {exc}")
        return None

    if has_ecg:
        n = min(len(ecg_raw_real), len(ecg_ref_mne), len(ecg_mne_from_meg), raw_meg_only.n_times)
    else:
        n = min(len(ecg_mne_from_meg), raw_meg_only.n_times)
    if n < int(round(5 * sfreq)):
        return None
    ecg_mne_from_meg = ecg_mne_from_meg[:n]
    if has_ecg:
        ecg_raw_real = ecg_raw_real[:n]
        ecg_ref_mne = ecg_ref_mne[:n]

    try:
        _, sources = fit_ecg_ica_and_sources(raw_meg_only)
    except Exception as exc:
        print(f"  ERROR ECG ICA for {subject_id}: {exc}")
        return None

    unsup_ic, unsup_details, _ = pick_best_ecg_ic_unsupervised(sources, sfreq)
    ecg_ica_unsup = sources[unsup_ic, :n]
    if unsup_details.get("flip_for_score", False):
        ecg_ica_unsup = -ecg_ica_unsup
    if np.abs(np.min(ecg_ica_unsup)) > np.abs(np.max(ecg_ica_unsup)):
        ecg_ica_unsup = -ecg_ica_unsup

    if not has_ecg:
        n_plot = min(n, int(round(plot_seconds * sfreq)))
        t = np.arange(n_plot) / sfreq
        stacked_png = os.path.join(output_dir, f"{subject_id}_MEGonly_stacked_MNEfromMEG_vs_ICAunsup.png")
        overlay_png = os.path.join(output_dir, f"{subject_id}_MEGonly_overlay_MNEfromMEG_vs_ICAunsup.png")
        plot_ecg_corr_meg_only(
            out_png=stacked_png,
            t=t,
            ecg_mne_from_meg=ecg_mne_from_meg,
            ecg_ica_unsup=ecg_ica_unsup,
            subject_id=subject_id,
            sfreq=sfreq,
            unsup_ic=unsup_ic,
            unsup_details=unsup_details,
        )
        plot_ecg_overlay_meg_only(
            out_png=overlay_png,
            t=t,
            ecg_mne_from_meg=ecg_mne_from_meg,
            ecg_ica_unsup=ecg_ica_unsup,
            subject_id=subject_id,
            unsup_ic=unsup_ic,
            unsup_details=unsup_details,
        )
        try:
            ev_rel, _ = detect_ecg_events_from_1d_trace(ecg_ica_unsup, sfreq)
            ica_unsup_n_events_detected = int(len(ev_rel))
        except Exception:
            ica_unsup_n_events_detected = 0
        return dict(
            subject=subject_id,
            file=str(data_path),
            sfreq_hz=sfreq,
            n_samples=n,
            ecg_channel="",
            mne_meg_corr_before=np.nan,
            mne_meg_corr_after=np.nan,
            mne_meg_lag_samples=np.nan,
            mne_meg_lag_seconds=np.nan,
            ica_sup_best_ic=np.nan,
            ica_sup_best_ic_abs_corr=np.nan,
            ica_sup_corr_before=np.nan,
            ica_sup_corr_after=np.nan,
            ica_sup_lag_samples=np.nan,
            ica_sup_lag_seconds=np.nan,
            ica_unsup_best_ic=unsup_ic,
            ica_unsup_score=unsup_details["score"],
            ica_unsup_bpm_est=unsup_details["bpm_est"],
            ica_unsup_p_hr=unsup_details["p_hr"],
            ica_unsup_p_ac=unsup_details["p_ac"],
            ica_unsup_p_peaks=unsup_details["p_peaks"],
            ica_unsup_p_kurt=unsup_details["p_kurt"],
            ica_unsup_p_spike=unsup_details["p_spike"],
            ica_unsup_kurtosis=unsup_details["kurtosis"],
            ica_unsup_rr_mad_sec=unsup_details["rr_mad_sec"],
            ica_unsup_spike_rate_per_min=unsup_details["spike_rate_per_min"],
            ica_unsup_frac_absz_above_thr=unsup_details["frac_absz_above_thr"],
            ica_unsup_n_spikes=unsup_details["n_spikes"],
            ica_unsup_n_events_detected=ica_unsup_n_events_detected,
            ica_unsup_corr_before=np.nan,
            ica_unsup_corr_after=np.nan,
            ica_unsup_lag_samples=np.nan,
            ica_unsup_lag_seconds=np.nan,
            stacked_plot=stacked_png,
            overlay_plot=overlay_png,
        )

    lag_mne = best_lag_via_xcorr(ecg_ref_mne, ecg_mne_from_meg)
    ecg_mne_meg_aligned = shift_with_zeros(ecg_mne_from_meg, lag_mne)
    r_mne_before, _ = pearsonr(ecg_ref_mne, ecg_mne_from_meg)
    r_mne_after, _ = pearsonr(ecg_ref_mne, ecg_mne_meg_aligned)

    sup_ic, sup_abs_corr = pick_best_ecg_ic_supervised(sources, ecg_ref_mne)
    ecg_ica_sup = sources[sup_ic, :n]
    if np.corrcoef(ecg_ica_sup, ecg_ref_mne)[0, 1] < 0:
        ecg_ica_sup = -ecg_ica_sup
    lag_sup = best_lag_via_xcorr(ecg_ref_mne, ecg_ica_sup)
    ecg_ica_sup_aligned = shift_with_zeros(ecg_ica_sup, lag_sup)
    r_sup_before, _ = pearsonr(ecg_ref_mne, ecg_ica_sup)
    r_sup_after, _ = pearsonr(ecg_ref_mne, ecg_ica_sup_aligned)

    if np.corrcoef(ecg_ica_unsup, ecg_ref_mne)[0, 1] < 0:
        ecg_ica_unsup = -ecg_ica_unsup
    lag_unsup = best_lag_via_xcorr(ecg_ref_mne, ecg_ica_unsup)
    ecg_ica_unsup_aligned = shift_with_zeros(ecg_ica_unsup, lag_unsup)
    r_unsup_before, _ = pearsonr(ecg_ref_mne, ecg_ica_unsup)
    r_unsup_after, _ = pearsonr(ecg_ref_mne, ecg_ica_unsup_aligned)

    n_plot = min(n, int(round(plot_seconds * sfreq)))
    t = np.arange(n_plot) / sfreq
    stacked_png = os.path.join(output_dir, f"{subject_id}_stacked_ref_mne_sup_unsup.png")
    overlay_png = os.path.join(output_dir, f"{subject_id}_overlay_ref_mne_sup_unsup.png")

    plot_ecg_corr_present(
        out_png=stacked_png,
        t=t,
        ecg_raw_real=ecg_raw_real,
        ecg_ref_mne=ecg_ref_mne,
        ecg_mne_meg_aligned=ecg_mne_meg_aligned,
        ecg_ica_sup_aligned=ecg_ica_sup_aligned,
        ecg_ica_unsup_aligned=ecg_ica_unsup_aligned,
        subject_id=subject_id,
        sfreq=sfreq,
        lag_mne=lag_mne,
        lag_sup=lag_sup,
        lag_unsup=lag_unsup,
        r_mne_after=r_mne_after,
        r_sup_after=r_sup_after,
        r_unsup_after=r_unsup_after,
        sup_ic=sup_ic,
        sup_abs_corr=sup_abs_corr,
        unsup_ic=unsup_ic,
        unsup_details=unsup_details,
    )
    plot_ecg_overlay_present(
        out_png=overlay_png,
        t=t,
        ecg_ref_mne=ecg_ref_mne,
        ecg_mne_meg_aligned=ecg_mne_meg_aligned,
        ecg_ica_sup_aligned=ecg_ica_sup_aligned,
        ecg_ica_unsup_aligned=ecg_ica_unsup_aligned,
        subject_id=subject_id,
        r_mne_after=r_mne_after,
        r_sup_after=r_sup_after,
        r_unsup_after=r_unsup_after,
    )

    return dict(
        subject=subject_id,
        file=str(data_path),
        sfreq_hz=sfreq,
        n_samples=n,
        ecg_channel=ecg_ch_name,
        mne_meg_corr_before=r_mne_before,
        mne_meg_corr_after=r_mne_after,
        mne_meg_lag_samples=lag_mne,
        mne_meg_lag_seconds=lag_mne / sfreq,
        ica_sup_best_ic=sup_ic,
        ica_sup_best_ic_abs_corr=sup_abs_corr,
        ica_sup_corr_before=r_sup_before,
        ica_sup_corr_after=r_sup_after,
        ica_sup_lag_samples=lag_sup,
        ica_sup_lag_seconds=lag_sup / sfreq,
        ica_unsup_best_ic=unsup_ic,
        ica_unsup_score=unsup_details["score"],
        ica_unsup_bpm_est=unsup_details["bpm_est"],
        ica_unsup_p_hr=unsup_details["p_hr"],
        ica_unsup_p_ac=unsup_details["p_ac"],
        ica_unsup_p_peaks=unsup_details["p_peaks"],
        ica_unsup_p_kurt=unsup_details["p_kurt"],
        ica_unsup_p_spike=unsup_details["p_spike"],
        ica_unsup_kurtosis=unsup_details["kurtosis"],
        ica_unsup_rr_mad_sec=unsup_details["rr_mad_sec"],
        ica_unsup_spike_rate_per_min=unsup_details["spike_rate_per_min"],
        ica_unsup_frac_absz_above_thr=unsup_details["frac_absz_above_thr"],
        ica_unsup_n_spikes=unsup_details["n_spikes"],
        ica_unsup_corr_before=r_unsup_before,
        ica_unsup_corr_after=r_unsup_after,
        ica_unsup_lag_samples=lag_unsup,
        ica_unsup_lag_seconds=lag_unsup / sfreq,
        stacked_plot=stacked_png,
        overlay_plot=overlay_png,
    )


# =============================================================================
# ECG EVENTS PIPELINE
# =============================================================================


def plot_ecg_event_overlay(
    out_png: str,
    subject_id: str,
    sfreq: float,
    first_samp: int,
    ref_trace: np.ndarray,
    test_trace: np.ndarray,
    ref_events_abs: np.ndarray,
    test_events_abs: np.ndarray,
    matched_ref_abs: np.ndarray,
    matched_test_abs: np.ndarray,
    metrics: Dict,
    label_test: str,
    match_tol_sec: float,
    seconds: float,
):
    n_plot = int(round(seconds * sfreq))
    t = np.arange(n_plot) / sfreq
    ref = ref_trace[:n_plot]
    test = test_trace[:n_plot]
    ref_rel = ref_events_abs - first_samp
    test_rel = test_events_abs - first_samp
    mref_rel = matched_ref_abs - first_samp
    mtest_rel = matched_test_abs - first_samp

    def in_win(x):
        return x[(x >= 0) & (x < n_plot)]

    ref_in = in_win(ref_rel)
    test_in = in_win(test_rel)
    mref_in = in_win(mref_rel)
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
    ax1.plot(t, ref, label="Ref: REAL ECG processed")
    ax1.plot(t, test, label=label_test)
    for s in ref_in:
        ax1.axvline(s / sfreq, linestyle="--", linewidth=0.8, alpha=0.5)
    for s in test_in:
        ax1.axvline(s / sfreq, linestyle=":", linewidth=0.8, alpha=0.5)
    for s in mref_in:
        ax1.axvline(s / sfreq, linestyle="-", linewidth=1.1, alpha=0.25)
    ax1.set_title("A) Processed traces + ECG events (ref dashed, test dotted; matched highlighted)")
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
        fontsize=11,
    )
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def process_ecg_events(
    raw: mne.io.BaseRaw,
    data_path: Path,
    output_dir: str,
    plot_seconds: float,
    match_tol_sec: float,
) -> Optional[Dict]:
    subject_id = data_path.stem
    sfreq = float(raw.info["sfreq"])
    ecg_picks = mne.pick_types(raw.info, ecg=True)
    if len(ecg_picks) == 0:
        return None

    ecg_idx = int(ecg_picks[0])
    ecg_ch_name = raw.ch_names[ecg_idx]
    ecg_raw = raw.get_data(picks=[ecg_idx])[0]
    ecg_ref_proc = safe_zscore(ecg_raw)

    try:
        ecg_events_ref = mne.preprocessing.find_ecg_events(raw, ch_name=ecg_ch_name, verbose=False)
        ref_events_abs = extract_event_samples(ecg_events_ref)
    except Exception as exc:
        print(f"  ERROR ECG events for {subject_id}: {exc}")
        return None

    raw_meg = raw.copy().pick_types(meg=True, eeg=False, eog=False, ecg=False, stim=False)
    n_total = min(len(ecg_ref_proc), raw_meg.n_times)
    if n_total < int(round(10 * sfreq)):
        return None
    ecg_ref_proc = ecg_ref_proc[:n_total]

    try:
        _, sources = fit_ecg_ica_and_sources(raw_meg)
    except Exception as exc:
        print(f"  ERROR ECG ICA for {subject_id}: {exc}")
        return None

    unsup_ic, unsup_details, _ = pick_best_ecg_ic_unsupervised(sources, sfreq)
    ecg_ica_unsup = sources[unsup_ic, :n_total]
    if unsup_details.get("flip_for_score", False):
        ecg_ica_unsup = -ecg_ica_unsup
    ecg_ica_unsup_proc = safe_zscore(ecg_ica_unsup)
    events_unsup, _ = detect_ecg_events_from_1d_trace(ecg_ica_unsup, sfreq)
    unsup_events_abs = events_unsup + raw.first_samp

    try:
        _, _, _, ecg_mne_from_meg = mne.preprocessing.find_ecg_events(
            raw_meg,
            ch_name=None,
            return_ecg=True,
            verbose=False,
        )
        ecg_mne_from_meg = ecg_mne_from_meg[0][:n_total]
    except Exception as exc:
        print(f"  ERROR ECG MNE baseline for {subject_id}: {exc}")
        return None

    mne_proc = safe_zscore(ecg_mne_from_meg)
    events_mne, _ = detect_ecg_events_from_1d_trace(ecg_mne_from_meg, sfreq)
    mne_events_abs = events_mne + raw.first_samp

    duration_sec = n_total / sfreq
    mref_mne, mtest_mne, uref_mne, utest_mne = match_events_one_to_one(
        ref_events_abs, mne_events_abs, sfreq, match_tol_sec
    )
    met_mne = compute_detection_metrics(mref_mne, mtest_mne, uref_mne, utest_mne, sfreq)
    fpmin_mne = met_mne["FP"] / (duration_sec / 60.0) if duration_sec > 0 else np.nan

    mref_uns, mtest_uns, uref_uns, utest_uns = match_events_one_to_one(
        ref_events_abs, unsup_events_abs, sfreq, match_tol_sec
    )
    met_uns = compute_detection_metrics(mref_uns, mtest_uns, uref_uns, utest_uns, sfreq)
    fpmin_uns = met_uns["FP"] / (duration_sec / 60.0) if duration_sec > 0 else np.nan

    plot_ecg_event_overlay(
        out_png=os.path.join(output_dir, f"{subject_id}_EV_MNEfromMEG.png"),
        subject_id=subject_id,
        sfreq=sfreq,
        first_samp=raw.first_samp,
        ref_trace=ecg_ref_proc,
        test_trace=mne_proc,
        ref_events_abs=ref_events_abs,
        test_events_abs=mne_events_abs,
        matched_ref_abs=mref_mne,
        matched_test_abs=mtest_mne,
        metrics=met_mne,
        label_test="Test: MNE from MEG",
        match_tol_sec=match_tol_sec,
        seconds=plot_seconds,
    )
    plot_ecg_event_overlay(
        out_png=os.path.join(output_dir, f"{subject_id}_EV_ICAunsup.png"),
        subject_id=subject_id,
        sfreq=sfreq,
        first_samp=raw.first_samp,
        ref_trace=ecg_ref_proc,
        test_trace=ecg_ica_unsup_proc,
        ref_events_abs=ref_events_abs,
        test_events_abs=unsup_events_abs,
        matched_ref_abs=mref_uns,
        matched_test_abs=mtest_uns,
        metrics=met_uns,
        label_test="Test: ICA unsup",
        match_tol_sec=match_tol_sec,
        seconds=plot_seconds,
    )

    return dict(
        subject=subject_id,
        file=str(data_path),
        sfreq_hz=sfreq,
        n_samples=n_total,
        ecg_channel=ecg_ch_name,
        mne_tp=met_mne["TP"],
        mne_fp=met_mne["FP"],
        mne_fn=met_mne["FN"],
        mne_precision=met_mne["precision"],
        mne_recall=met_mne["recall"],
        mne_f1=met_mne["f1"],
        mne_fp_per_min=fpmin_mne,
        mne_jitter_mae_ms=met_mne["jitter_mae_ms"],
        ica_unsup_tp=met_uns["TP"],
        ica_unsup_fp=met_uns["FP"],
        ica_unsup_fn=met_uns["FN"],
        ica_unsup_precision=met_uns["precision"],
        ica_unsup_recall=met_uns["recall"],
        ica_unsup_f1=met_uns["f1"],
        ica_unsup_fp_per_min=fpmin_uns,
        ica_unsup_jitter_mae_ms=met_uns["jitter_mae_ms"],
        ica_unsup_best_ic=unsup_ic,
        ica_unsup_score=unsup_details["score"],
    )


# =============================================================================
# EOG CORR PIPELINE
# =============================================================================


def plot_eog_methods_stacked_benchmark(
    out_png: str,
    t: np.ndarray,
    eog_raw_real: np.ndarray,
    eog_ref_proc: np.ndarray,
    pca_all_proc: np.ndarray,
    pca_front_sup_proc: np.ndarray,
    ica_sup_proc: np.ndarray,
    ica_unsup_proc: np.ndarray,
    subject_id: str,
    sfreq: float,
    metrics: Dict[str, float],
):
    n_plot = len(t)
    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(t, safe_zscore(eog_raw_real[:n_plot]))
    axes[0].set_title("1) Real EOG (raw) [z-score]")
    axes[1].plot(t, eog_ref_proc[:n_plot])
    axes[1].set_title("2) Real EOG (processed)")
    axes[2].plot(t, pca_all_proc[:n_plot])
    axes[2].set_title(f"3) PCA global (r={metrics['r_pca_all']:.3f})")
    axes[3].plot(t, pca_front_sup_proc[:n_plot])
    axes[3].set_title(f"4) PCA frontal supervised (r={metrics['r_pca_front_sup']:.3f})")
    axes[4].plot(t, ica_sup_proc[:n_plot])
    axes[4].set_title(f"5) ICA supervised (r={metrics['r_ica_sup']:.3f})")
    axes[5].plot(t, ica_unsup_proc[:n_plot])
    axes[5].set_title(f"6) ICA unsupervised (r={metrics['r_ica_unsup']:.3f})")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{subject_id} | sfreq={sfreq:.2f} Hz", y=0.995, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_eog_methods_overlay_benchmark(
    out_png: str,
    t: np.ndarray,
    eog_ref_proc: np.ndarray,
    pca_all_proc: np.ndarray,
    pca_front_sup_proc: np.ndarray,
    ica_sup_proc: np.ndarray,
    ica_unsup_proc: np.ndarray,
    subject_id: str,
):
    n_plot = len(t)
    plt.figure(figsize=(14, 5))
    plt.plot(t, eog_ref_proc[:n_plot], label="Ref (real EOG processed)")
    plt.plot(t, pca_all_proc[:n_plot], label="PCA global")
    plt.plot(t, pca_front_sup_proc[:n_plot], label="PCA frontal supervised")
    plt.plot(t, ica_sup_proc[:n_plot], label="ICA supervised")
    plt.plot(t, ica_unsup_proc[:n_plot], label="ICA unsupervised")
    plt.legend()
    plt.title(f"{subject_id} | EOG overlay (aligned)")
    plt.xlabel("Time (s)")
    plt.ylabel("a.u.")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_eog_methods_stacked_meg_only(
    out_png: str,
    t: np.ndarray,
    pca_all_proc: np.ndarray,
    pca_front_unsup_proc: np.ndarray,
    ica_unsup_proc: np.ndarray,
    subject_id: str,
    sfreq: float,
):
    n_plot = len(t)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(t, pca_all_proc[:n_plot])
    axes[0].set_title("1) PCA global (MEG-only)")
    axes[1].plot(t, pca_front_unsup_proc[:n_plot])
    axes[1].set_title("2) PCA frontal unsupervised (MEG-only)")
    axes[2].plot(t, ica_unsup_proc[:n_plot])
    axes[2].set_title("3) ICA unsupervised (MEG-only)")
    axes[2].set_xlabel("Time (s)")
    fig.suptitle(f"{subject_id} | sfreq={sfreq:.2f} Hz", y=0.995, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_eog_methods_overlay_meg_only(
    out_png: str,
    t: np.ndarray,
    pca_all_proc: np.ndarray,
    pca_front_unsup_proc: np.ndarray,
    ica_unsup_proc: np.ndarray,
    subject_id: str,
):
    n_plot = len(t)
    plt.figure(figsize=(14, 5))
    plt.plot(t, pca_all_proc[:n_plot], label="PCA global")
    plt.plot(t, pca_front_unsup_proc[:n_plot], label="PCA frontal unsup")
    plt.plot(t, ica_unsup_proc[:n_plot], label="ICA unsup")
    plt.legend()
    plt.title(f"{subject_id} | EOG MEG-only overlay")
    plt.xlabel("Time (s)")
    plt.ylabel("a.u.")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def process_eog_corr(
    raw: mne.io.BaseRaw,
    data_path: Path,
    output_dir: str,
    plot_seconds: float,
) -> Optional[Dict]:
    subject_id = data_path.stem
    sfreq = float(raw.info["sfreq"])
    eog_idx = pick_prefer_vertical_eog(raw)
    has_real_eog = eog_idx >= 0
    if not has_real_eog:
        eog_ref_proc = None
    else:
        eog_raw_real = raw.get_data(picks=[eog_idx])[0]
        eog_ref_proc = process_trace_bandpass_z(eog_raw_real, sfreq, EOG_L_FREQ, EOG_H_FREQ)

    raw_meg = raw.copy().pick_types(meg=True, eeg=False, eog=False, ecg=False, stim=False)
    n_total = raw_meg.n_times
    if has_real_eog:
        n_total = min(n_total, len(eog_ref_proc))
    if n_total < int(round(10 * sfreq)):
        return None

    pca_all_raw = build_synth_eog_pca_all(raw_meg)[:n_total]
    pca_all_proc = process_trace_bandpass_z(pca_all_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)

    try:
        pca_front_unsup_raw = build_synth_eog_pca_frontal_unsupervised(raw_meg)[:n_total]
        pca_front_unsup_proc = process_trace_bandpass_z(pca_front_unsup_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)
    except Exception:
        pca_front_unsup_raw = None
        pca_front_unsup_proc = None

    if has_real_eog:
        pca_front_sup_raw = build_synth_eog_pca_frontal_supervised(raw_meg, eog_ref_proc)[:n_total]
        pca_front_sup_proc = process_trace_bandpass_z(pca_front_sup_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)
    else:
        pca_front_sup_proc = None

    sources = fit_ica_sources_once(raw_meg)[:, :n_total]
    if has_real_eog:
        ica_sup_ic, ica_sup_abs_corr = pick_best_ic_supervised_from_sources(sources, eog_ref_proc)
        ica_sup_raw = sources[ica_sup_ic, :n_total]
        if np.corrcoef(ica_sup_raw, eog_ref_proc)[0, 1] < 0:
            ica_sup_raw = -ica_sup_raw
        ica_sup_proc = process_trace_bandpass_z(ica_sup_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)
    else:
        ica_sup_ic = np.nan
        ica_sup_abs_corr = np.nan
        ica_sup_proc = None

    ica_unsup_ic, unsup_details = pick_best_ic_unsupervised_from_sources(
        sources, sfreq=sfreq, mode=EOG_ICA_UNSUP_MODE, fixed_ic=EOG_ICA_UNSUP_FIXED_IC
    )
    ica_unsup_raw = sources[ica_unsup_ic, :n_total]
    if unsup_details.get("flip_for_score", False):
        ica_unsup_raw = -ica_unsup_raw
    if EOG_UNSUP_SIGN_MODE == "frontal_proxy" and pca_front_unsup_proc is not None:
        ica_unsup_raw = apply_unsup_sign_convention_frontal_proxy(ica_unsup_raw, pca_front_unsup_proc)
    else:
        ica_unsup_raw = apply_unsup_sign_convention_peak_polarity(ica_unsup_raw, sfreq)
    ica_unsup_proc = process_trace_bandpass_z(ica_unsup_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)

    n_plot = min(n_total, int(round(plot_seconds * sfreq)))
    t = np.arange(n_plot) / sfreq
    if not has_real_eog:
        stacked_png = os.path.join(output_dir, f"{subject_id}_MEGonly_stacked.png")
        overlay_png = os.path.join(output_dir, f"{subject_id}_MEGonly_overlay.png")
        if pca_front_unsup_proc is None:
            pca_front_unsup_proc = np.zeros_like(pca_all_proc)
        plot_eog_methods_stacked_meg_only(
            out_png=stacked_png,
            t=t,
            pca_all_proc=pca_all_proc,
            pca_front_unsup_proc=pca_front_unsup_proc,
            ica_unsup_proc=ica_unsup_proc,
            subject_id=subject_id,
            sfreq=sfreq,
        )
        plot_eog_methods_overlay_meg_only(
            out_png=overlay_png,
            t=t,
            pca_all_proc=pca_all_proc,
            pca_front_unsup_proc=pca_front_unsup_proc,
            ica_unsup_proc=ica_unsup_proc,
            subject_id=subject_id,
        )
        return dict(
            subject=subject_id,
            file=str(data_path),
            sfreq_hz=sfreq,
            n_samples=n_total,
            eog_channel="",
            pca_all_corr=np.nan,
            pca_front_sup_corr=np.nan,
            ica_sup_corr=np.nan,
            ica_unsup_corr=np.nan,
            ica_unsup_best_ic=ica_unsup_ic,
            ica_unsup_score=unsup_details["score"],
            stacked_plot=stacked_png,
            overlay_plot=overlay_png,
        )

    eog_ref_proc = eog_ref_proc[:n_total]
    r_pca_all, _ = pearsonr(eog_ref_proc, pca_all_proc)
    r_pca_front_sup, _ = pearsonr(eog_ref_proc, pca_front_sup_proc)
    r_ica_sup, _ = pearsonr(eog_ref_proc, ica_sup_proc)
    r_ica_unsup, _ = pearsonr(eog_ref_proc, ica_unsup_proc)

    stacked_png = os.path.join(output_dir, f"{subject_id}_stacked_benchmark.png")
    overlay_png = os.path.join(output_dir, f"{subject_id}_overlay_benchmark.png")
    metrics = dict(
        r_pca_all=r_pca_all,
        r_pca_front_sup=r_pca_front_sup,
        r_ica_sup=r_ica_sup,
        r_ica_unsup=r_ica_unsup,
    )
    plot_eog_methods_stacked_benchmark(
        out_png=stacked_png,
        t=t,
        eog_raw_real=raw.get_data(picks=[eog_idx])[0],
        eog_ref_proc=eog_ref_proc,
        pca_all_proc=pca_all_proc,
        pca_front_sup_proc=pca_front_sup_proc,
        ica_sup_proc=ica_sup_proc,
        ica_unsup_proc=ica_unsup_proc,
        subject_id=subject_id,
        sfreq=sfreq,
        metrics=metrics,
    )
    plot_eog_methods_overlay_benchmark(
        out_png=overlay_png,
        t=t,
        eog_ref_proc=eog_ref_proc,
        pca_all_proc=pca_all_proc,
        pca_front_sup_proc=pca_front_sup_proc,
        ica_sup_proc=ica_sup_proc,
        ica_unsup_proc=ica_unsup_proc,
        subject_id=subject_id,
    )

    return dict(
        subject=subject_id,
        file=str(data_path),
        sfreq_hz=sfreq,
        n_samples=n_total,
        eog_channel=raw.ch_names[eog_idx],
        pca_all_corr=r_pca_all,
        pca_front_sup_corr=r_pca_front_sup,
        ica_sup_corr=r_ica_sup,
        ica_unsup_corr=r_ica_unsup,
        ica_sup_best_ic=ica_sup_ic,
        ica_sup_best_ic_abs_corr=ica_sup_abs_corr,
        ica_unsup_best_ic=ica_unsup_ic,
        ica_unsup_score=unsup_details["score"],
        stacked_plot=stacked_png,
        overlay_plot=overlay_png,
    )


# =============================================================================
# EOG EVENTS PIPELINE
# =============================================================================


def plot_eog_event_overlay(
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
    seconds: float,
):
    plot_ecg_event_overlay(
        out_png=out_png,
        subject_id=subject_id,
        sfreq=sfreq,
        first_samp=first_samp,
        ref_trace=ref_trace_proc,
        test_trace=test_trace_proc,
        ref_events_abs=ref_events_abs,
        test_events_abs=test_events_abs,
        matched_ref_abs=matched_ref_abs,
        matched_test_abs=matched_test_abs,
        metrics=metrics,
        label_test=label_test,
        match_tol_sec=match_tol_sec,
        seconds=seconds,
    )


def process_eog_events(
    raw: mne.io.BaseRaw,
    data_path: Path,
    output_dir: str,
    plot_seconds: float,
    match_tol_sec: float,
) -> Optional[Dict]:
    subject_id = data_path.stem
    sfreq = float(raw.info["sfreq"])
    eog_idx = pick_prefer_vertical_eog(raw)
    if eog_idx < 0:
        return None

    eog_ch_name = raw.ch_names[eog_idx]
    eog_raw_real = raw.get_data(picks=[eog_idx])[0]
    eog_ref_proc = process_trace_bandpass_z(eog_raw_real, sfreq, EOG_L_FREQ, EOG_H_FREQ)

    try:
        eog_events_ref = mne.preprocessing.find_eog_events(raw, ch_name=eog_ch_name, verbose=False)
        ref_events_abs = extract_event_samples(eog_events_ref)
    except Exception as exc:
        print(f"  ERROR EOG events for {subject_id}: {exc}")
        return None

    raw_meg = raw.copy().pick_types(meg=True, eeg=False, eog=False, ecg=False, stim=False)
    n_total = min(len(eog_ref_proc), raw_meg.n_times)
    if n_total < int(round(10 * sfreq)):
        return None
    eog_ref_proc = eog_ref_proc[:n_total]

    pca_all_raw = build_synth_eog_pca_all(raw_meg)[:n_total]
    pca_all_proc = process_trace_bandpass_z(pca_all_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)
    _, pca_all_events_abs = events_from_trace_via_mne_find_eog(pca_all_raw[:n_total], sfreq, raw.first_samp)

    try:
        pca_front_unsup_raw = build_synth_eog_pca_frontal_unsupervised(raw_meg)[:n_total]
        pca_front_unsup_proc = process_trace_bandpass_z(pca_front_unsup_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)
        _, pca_front_unsup_events_abs = events_from_trace_via_mne_find_eog(pca_front_unsup_raw[:n_total], sfreq, raw.first_samp)
    except Exception:
        pca_front_unsup_raw = None
        pca_front_unsup_proc = None
        pca_front_unsup_events_abs = np.array([], dtype=int)

    pca_front_sup_raw = build_synth_eog_pca_frontal_supervised(raw_meg, eog_ref_proc)[:n_total]
    pca_front_sup_proc = process_trace_bandpass_z(pca_front_sup_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)
    _, pca_front_sup_events_abs = events_from_trace_via_mne_find_eog(pca_front_sup_raw[:n_total], sfreq, raw.first_samp)

    sources = fit_ica_sources_once(raw_meg)[:, :n_total]
    ica_sup_ic, ica_sup_abs_corr = pick_best_ic_supervised_from_sources(sources, eog_ref_proc)
    ica_sup_raw = sources[ica_sup_ic, :n_total]
    if np.corrcoef(ica_sup_raw, eog_ref_proc)[0, 1] < 0:
        ica_sup_raw = -ica_sup_raw
    ica_sup_proc = process_trace_bandpass_z(ica_sup_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)
    _, ica_sup_events_abs = events_from_trace_via_mne_find_eog(ica_sup_raw, sfreq, raw.first_samp)

    ica_unsup_ic, unsup_details = pick_best_ic_unsupervised_from_sources(
        sources, sfreq=sfreq, mode=EOG_ICA_UNSUP_MODE, fixed_ic=EOG_ICA_UNSUP_FIXED_IC
    )
    ica_unsup_raw = sources[ica_unsup_ic, :n_total]
    if unsup_details.get("flip_for_score", False):
        ica_unsup_raw = -ica_unsup_raw
    if EOG_UNSUP_SIGN_MODE == "frontal_proxy" and pca_front_unsup_proc is not None:
        ica_unsup_raw = apply_unsup_sign_convention_frontal_proxy(ica_unsup_raw, pca_front_unsup_proc)
    else:
        ica_unsup_raw = apply_unsup_sign_convention_peak_polarity(ica_unsup_raw, sfreq)
    ica_unsup_proc = process_trace_bandpass_z(ica_unsup_raw, sfreq, EOG_L_FREQ, EOG_H_FREQ)
    _, ica_unsup_events_abs = events_from_trace_via_mne_find_eog(ica_unsup_raw, sfreq, raw.first_samp)

    duration_sec = n_total / sfreq
    mref_g, mtest_g, uref_g, utest_g = match_events_one_to_one(ref_events_abs, pca_all_events_abs, sfreq, match_tol_sec)
    met_g = compute_detection_metrics(mref_g, mtest_g, uref_g, utest_g, sfreq)
    fpmin_g = met_g["FP"] / (duration_sec / 60.0) if duration_sec > 0 else np.nan

    if pca_front_unsup_raw is not None:
        mref_fu, mtest_fu, uref_fu, utest_fu = match_events_one_to_one(
            ref_events_abs, pca_front_unsup_events_abs, sfreq, match_tol_sec
        )
        met_fu = compute_detection_metrics(mref_fu, mtest_fu, uref_fu, utest_fu, sfreq)
        fpmin_fu = met_fu["FP"] / (duration_sec / 60.0) if duration_sec > 0 else np.nan
    else:
        met_fu = dict(TP=np.nan, FP=np.nan, FN=np.nan, precision=np.nan, recall=np.nan, f1=np.nan, miss_rate=np.nan,
                      jitter_mean_ms_signed=np.nan, jitter_std_ms_signed=np.nan, jitter_mae_ms=np.nan,
                      jitter_median_abs_ms=np.nan, jitter_p95_abs_ms=np.nan, jitter_sec_signed=np.array([]))
        fpmin_fu = np.nan
        mref_fu = mtest_fu = np.array([], dtype=int)

    mref_fs, mtest_fs, uref_fs, utest_fs = match_events_one_to_one(ref_events_abs, pca_front_sup_events_abs, sfreq, match_tol_sec)
    met_fs = compute_detection_metrics(mref_fs, mtest_fs, uref_fs, utest_fs, sfreq)
    fpmin_fs = met_fs["FP"] / (duration_sec / 60.0) if duration_sec > 0 else np.nan

    mref_is, mtest_is, uref_is, utest_is = match_events_one_to_one(ref_events_abs, ica_sup_events_abs, sfreq, match_tol_sec)
    met_is = compute_detection_metrics(mref_is, mtest_is, uref_is, utest_is, sfreq)
    fpmin_is = met_is["FP"] / (duration_sec / 60.0) if duration_sec > 0 else np.nan

    mref_iu, mtest_iu, uref_iu, utest_iu = match_events_one_to_one(ref_events_abs, ica_unsup_events_abs, sfreq, match_tol_sec)
    met_iu = compute_detection_metrics(mref_iu, mtest_iu, uref_iu, utest_iu, sfreq)
    fpmin_iu = met_iu["FP"] / (duration_sec / 60.0) if duration_sec > 0 else np.nan

    plot_eog_event_overlay(
        out_png=os.path.join(output_dir, f"{subject_id}_EV_PCAglobal.png"),
        subject_id=subject_id,
        sfreq=sfreq,
        first_samp=raw.first_samp,
        ref_trace_proc=eog_ref_proc,
        test_trace_proc=pca_all_proc,
        ref_events_abs=ref_events_abs,
        test_events_abs=pca_all_events_abs,
        matched_ref_abs=mref_g,
        matched_test_abs=mtest_g,
        metrics=met_g,
        label_test="Test: PCA global",
        match_tol_sec=match_tol_sec,
        seconds=plot_seconds,
    )
    if pca_front_unsup_proc is not None:
        plot_eog_event_overlay(
            out_png=os.path.join(output_dir, f"{subject_id}_EV_PCAfrontal_unsup.png"),
            subject_id=subject_id,
            sfreq=sfreq,
            first_samp=raw.first_samp,
            ref_trace_proc=eog_ref_proc,
            test_trace_proc=pca_front_unsup_proc,
            ref_events_abs=ref_events_abs,
            test_events_abs=pca_front_unsup_events_abs,
            matched_ref_abs=mref_fu,
            matched_test_abs=mtest_fu,
            metrics=met_fu,
            label_test="Test: PCA frontal unsup",
            match_tol_sec=match_tol_sec,
            seconds=plot_seconds,
        )
    plot_eog_event_overlay(
        out_png=os.path.join(output_dir, f"{subject_id}_EV_PCAfrontal_sup.png"),
        subject_id=subject_id,
        sfreq=sfreq,
        first_samp=raw.first_samp,
        ref_trace_proc=eog_ref_proc,
        test_trace_proc=pca_front_sup_proc,
        ref_events_abs=ref_events_abs,
        test_events_abs=pca_front_sup_events_abs,
        matched_ref_abs=mref_fs,
        matched_test_abs=mtest_fs,
        metrics=met_fs,
        label_test="Test: PCA frontal supervised",
        match_tol_sec=match_tol_sec,
        seconds=plot_seconds,
    )
    plot_eog_event_overlay(
        out_png=os.path.join(output_dir, f"{subject_id}_EV_ICAsup.png"),
        subject_id=subject_id,
        sfreq=sfreq,
        first_samp=raw.first_samp,
        ref_trace_proc=eog_ref_proc,
        test_trace_proc=ica_sup_proc,
        ref_events_abs=ref_events_abs,
        test_events_abs=ica_sup_events_abs,
        matched_ref_abs=mref_is,
        matched_test_abs=mtest_is,
        metrics=met_is,
        label_test="Test: ICA supervised",
        match_tol_sec=match_tol_sec,
        seconds=plot_seconds,
    )
    plot_eog_event_overlay(
        out_png=os.path.join(output_dir, f"{subject_id}_EV_ICAunsup.png"),
        subject_id=subject_id,
        sfreq=sfreq,
        first_samp=raw.first_samp,
        ref_trace_proc=eog_ref_proc,
        test_trace_proc=ica_unsup_proc,
        ref_events_abs=ref_events_abs,
        test_events_abs=ica_unsup_events_abs,
        matched_ref_abs=mref_iu,
        matched_test_abs=mtest_iu,
        metrics=met_iu,
        label_test="Test: ICA unsup",
        match_tol_sec=match_tol_sec,
        seconds=plot_seconds,
    )

    return dict(
        subject=subject_id,
        file=str(data_path),
        sfreq_hz=sfreq,
        n_samples=n_total,
        eog_channel=eog_ch_name,
        pca_global_f1=met_g["f1"],
        pca_global_fp_per_min=fpmin_g,
        pca_front_unsup_f1=met_fu["f1"],
        pca_front_unsup_fp_per_min=fpmin_fu,
        pca_front_sup_f1=met_fs["f1"],
        pca_front_sup_fp_per_min=fpmin_fs,
        ica_sup_f1=met_is["f1"],
        ica_sup_fp_per_min=fpmin_is,
        ica_unsup_f1=met_iu["f1"],
        ica_unsup_fp_per_min=fpmin_iu,
        ica_sup_best_ic=ica_sup_ic,
        ica_sup_best_ic_abs_corr=ica_sup_abs_corr,
        ica_unsup_best_ic=ica_unsup_ic,
        ica_unsup_score=unsup_details["score"],
    )


# =============================================================================
# RUNNER
# =============================================================================


@dataclass
class RunConfig:
    dataset_root: str
    output_root: str
    runs: List[str]
    n_jobs: int
    plot_seconds: Optional[float]
    match_tol_sec: Optional[float]


def process_file_for_runs(data_path: Path, cfg: RunConfig, output_dirs: Dict[str, str]) -> Dict[str, Optional[Dict]]:
    try:
        raw = load_raw(data_path)
    except Exception as exc:
        print(f"  ERROR reading {data_path}: {exc}")
        return {run: None for run in cfg.runs}

    results: Dict[str, Optional[Dict]] = {}
    for run in cfg.runs:
        if run == "ecg-corr":
            results[run] = process_ecg_corr(
                raw=raw,
                data_path=data_path,
                output_dir=output_dirs[run],
                plot_seconds=cfg.plot_seconds or 60,
            )
        elif run == "ecg-events":
            results[run] = process_ecg_events(
                raw=raw,
                data_path=data_path,
                output_dir=output_dirs[run],
                plot_seconds=cfg.plot_seconds or 60,
                match_tol_sec=cfg.match_tol_sec or 0.05,
            )
        elif run == "eog-corr":
            results[run] = process_eog_corr(
                raw=raw,
                data_path=data_path,
                output_dir=output_dirs[run],
                plot_seconds=cfg.plot_seconds or 100,
            )
        elif run == "eog-events":
            results[run] = process_eog_events(
                raw=raw,
                data_path=data_path,
                output_dir=output_dirs[run],
                plot_seconds=cfg.plot_seconds or 100,
                match_tol_sec=cfg.match_tol_sec or 0.08,
            )
    return results


def run_pipelines(cfg: RunConfig) -> None:
    data_paths = discover_data_paths(cfg.dataset_root)
    if not data_paths:
        print("No data files found.")
        return

    output_dirs = {}
    for run in cfg.runs:
        out_dir = os.path.join(cfg.output_root, run.replace("-", "_"))
        os.makedirs(out_dir, exist_ok=True)
        output_dirs[run] = out_dir

    print(f"Found {len(data_paths)} files. Runs: {', '.join(cfg.runs)}")

    if cfg.n_jobs == 1:
        all_results = [process_file_for_runs(p, cfg, output_dirs) for p in data_paths]
    else:
        all_results = Parallel(n_jobs=cfg.n_jobs)(
            delayed(process_file_for_runs)(p, cfg, output_dirs) for p in data_paths
        )

    for run in cfg.runs:
        results = [res[run] for res in all_results if res.get(run) is not None]
        if not results:
            print(f"No results for {run}")
            continue
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dirs[run], f"{run.replace('-', '_')}_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {run} summary CSV: {csv_path}")


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Unified ECG/EOG estimation pipeline.")
    parser.add_argument("--dataset-root", required=True, help="Root folder to search for MEG files.")
    parser.add_argument("--output-root", required=True, help="Root folder to write outputs.")
    parser.add_argument(
        "--runs",
        default="ecg-corr,ecg-events,eog-corr,eog-events",
        help="Comma-separated list: ecg-corr, ecg-events, eog-corr, eog-events.",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs for file-level processing.")
    parser.add_argument("--plot-seconds", type=float, default=None, help="Override plot window in seconds.")
    parser.add_argument("--match-tol-sec", type=float, default=None, help="Override event matching tolerance.")
    args = parser.parse_args()

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    return RunConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        runs=runs,
        n_jobs=args.jobs,
        plot_seconds=args.plot_seconds,
        match_tol_sec=args.match_tol_sec,
    )


def main() -> None:
    cfg = parse_args()
    run_pipelines(cfg)


if __name__ == "__main__":
    main()

"""
ECG estimation / validation with supervised + improved unsupervised ICA selection.
PLUS: graceful MEG-only mode (when no real ECG channel is present).

---------------------------------------------------------------------------
WHAT THIS SCRIPT DOES (two modes)
---------------------------------------------------------------------------

A) "ECG-present" mode (benchmark / validation mode)
   If the FIF contains a real ECG channel:
   1) Builds a REFERENCE ECG proxy using MNE's find_ecg_events(..., return_ecg=True)
      on the real ECG channel.
   2) Builds a BASELINE ECG proxy using MNE's find_ecg_events on MEG-only data.
   3) Fits ICA once on MEG-only data and derives two ICA ECG proxies:
        - ICA SUPERVISED: best IC by abs correlation with the reference trace
          (uses the real ECG-derived reference only for benchmarking).
        - ICA UNSUPERVISED: best IC by ECG-likeness heuristics (NO ECG used for selection).
   4) Aligns candidate waveforms to the reference for nicer plotting and correlation reporting
      (alignment is waveform-only, not event shifting).
   5) Produces:
        - Stacked plot (5 panels): real ECG, reference, MNE-from-MEG, ICA-sup, ICA-unsup
        - Overlay plot: reference + the three candidates
        - CSV summary row with correlations, lags, and unsupervised diagnostics scores.

B) "ECG-missing" mode (production / MEG-only mode)
   If no real ECG channel exists:
   1) Builds the MNE-from-MEG baseline ECG proxy (MEG-only).
   2) Fits ICA once and selects ICA UNSUPERVISED (no ECG involved).
   3) Produces:
        - MEG-only stacked plot (2 panels): MNE-from-MEG and ICA-unsup
        - MEG-only overlay plot: those two
        - CSV summary row with unsupervised diagnostics (no supervised/ref metrics).

---------------------------------------------------------------------------
NOTES ON UNSUPERVISED ICA ECG SELECTION
---------------------------------------------------------------------------

We score each IC time course with ECG-likeness heuristics designed for MEG ICA:
- QRS complexes are sharp spikes; ICA sign is arbitrary -> use |z| and explicit spike scoring.
- ECG is quasi-periodic in a plausible HR range -> bandpower ratio and autocorr periodicity.
- Peak train plausibility -> peak rate within [HR_MIN_BPM, HR_MAX_BPM] and RR MAD sanity.
- Peaky morphology -> kurtosis.

The final score is a weighted sum of these sub-scores.

---------------------------------------------------------------------------
REQUIREMENTS
---------------------------------------------------------------------------
- mne, numpy, pandas, matplotlib, scipy

"""

import os
from pathlib import Path

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

# Root folder of your dataset (will search recursively for "*_meg.fif")
DATASET_ROOT = "/Users/karelo/Development/datasets/ds_small"

# Where all outputs are written
OUTPUT_DIR = "/Users/karelo/Development/datasets/ds_small/derivatives/ecg_results_supervised_unsupervised_ica_v6"

# Seconds to show in plots (time axis)
PLOT_SECONDS = 60

# Parallel processing: set >1 to enable joblib across files
N_JOBS = 1


# --------------------------
# ICA settings
# --------------------------
# Fraction of variance or int. Here: keep enough components to explain 99% of variance.
ICA_N_COMPONENTS = 0.99

# ICA algorithm. "fastica" is fine; you can also try "picard" if installed.
ICA_METHOD = "fastica"

# Reproducibility
ICA_RANDOM_STATE = 97
ICA_MAX_ITER = 1000

# Filtering for ICA stability: QRS energy is often ~5–40 Hz
ICA_L_FREQ = 5.0
ICA_H_FREQ = 40.0


# --------------------------
# Unsupervised ECG scoring
# --------------------------
# Unsupervised ICA selection mode: "heuristic" | "megnet" | "hybrid"
ICA_UNSUP_MODE = "heuristic"

# MEGNet configuration (used when ICA_UNSUP_MODE is "megnet" or "hybrid")
MEGNET_MODEL_PATH = ""
MEGNET_INPUT_SAMPLES = 2048
MEGNET_OUTPUT_INDEX = 0
MEGNET_SCORE_WEIGHT = 0.5

# Plausible heart rate range (bpm) used by multiple heuristics.
HR_MIN_BPM = 40
HR_MAX_BPM = 180
HR_BAND_HZ = (HR_MIN_BPM / 60.0, HR_MAX_BPM / 60.0)  # convert bpm -> Hz

# Score only the first N seconds for speed/stability (set None to use full length)
UNSUP_SCORE_SECONDS = None

# Peak detection parameters (run on abs(z))
PEAK_MIN_DISTANCE_SEC = 0.30
PEAK_PROMINENCE = 1.3

# "Spike strength" constraints (|z| >= threshold)
SPIKE_Z_ABS_THRESHOLD = 4.0
SPIKE_RATE_MIN_PER_MIN = 20
SPIKE_RATE_MAX_PER_MIN = 220

# Penalize too-regular sinusoid trains or very chaotic trains using RR MAD
RR_MAD_TOO_SMALL_SEC = 0.005
RR_MAD_TOO_LARGE_SEC = 0.25

# Weights for combined unsupervised score (tweak here)
W_HR_BANDPOWER = 0.20
W_AUTOCORR = 0.20
W_PEAK_TRAIN = 0.25
W_KURT = 0.10
W_SPIKE_STRENGTH = 0.25


# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# HELPERS: normalization & alignment
# =============================================================================
def safe_zscore(x: np.ndarray) -> np.ndarray:
    """
    Return a z-scored copy of x for visualization/comparability.

    If std == 0 or NaN (e.g., flat signal), fall back to mean-centering only.
    """
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sd = np.std(x)
    if sd == 0 or np.isnan(sd):
        return x - mu
    return (x - mu) / sd


def best_lag_via_xcorr(x_ref: np.ndarray, y: np.ndarray) -> int:
    """
    Compute the sample lag (integer) that maximizes cross-correlation between y and x_ref.

    Convention:
      c = correlate(y, x_ref, mode='full')
      best_lag > 0 means shifting y LEFT by best_lag aligns better with x_ref.

    This is used only for waveform visualization and correlation reporting,
    NOT for event matching.
    """
    n = min(len(x_ref), len(y))
    x = x_ref[:n] - np.mean(x_ref[:n])
    yy = y[:n] - np.mean(y[:n])

    c = correlate(yy, x, mode="full")
    lags = np.arange(-n + 1, n)
    return int(lags[np.argmax(c)])


def shift_with_zeros(y: np.ndarray, lag: int) -> np.ndarray:
    """
    Shift a 1D vector by 'lag' samples, padding with zeros.

    lag > 0: y_aligned = [y[lag:], zeros(lag)]   (shift left)
    lag < 0: y_aligned = [zeros(k), y[:-k]]     (shift right)
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
# HELPERS: MEGNet inference (optional ICA classification)
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
# HELPERS: event extraction (optional debugging)
# =============================================================================
def extract_event_samples(events: np.ndarray) -> np.ndarray:
    """
    MNE events array is shape (n_events, 3) with sample index in column 0.
    Returns the event sample indices as int array.
    """
    if events is None or len(events) == 0:
        return np.array([], dtype=int)
    return np.asarray(events[:, 0], dtype=int)


def detect_ecg_events_from_1d_trace(trace_1d: np.ndarray, sfreq: float):
    """
    Convenience wrapper to run MNE's ECG event detection on an arbitrary 1D trace.

    We wrap the trace into a RawArray with ch_type='ecg' so that find_ecg_events
    works in the same way as with real ECG channel data.

    Returns:
      events_samples_rel : sample indices relative to RawArray start (0-based)
      ecg_mne_trace      : the 'ecg' proxy returned by find_ecg_events(return_ecg=True)
    """
    info = mne.create_info(ch_names=["ICA_ECG"], sfreq=sfreq, ch_types=["ecg"])
    raw_tmp = mne.io.RawArray(trace_1d[np.newaxis, :], info, verbose=False)

    events, _, _, ecg_mne = mne.preprocessing.find_ecg_events(
        raw_tmp,
        ch_name="ICA_ECG",
        return_ecg=True,
        verbose=False,
    )
    return extract_event_samples(events), ecg_mne[0]


# =============================================================================
# HELPERS: Unsupervised ECG-likeness scoring (MEG-only)
# =============================================================================
def _segment_for_scoring(x: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Return the portion of x used for scoring.

    We often score only the first UNSUP_SCORE_SECONDS seconds to reduce runtime
    and reduce sensitivity to late-session artifacts.
    """
    x = np.asarray(x, dtype=float)
    if UNSUP_SCORE_SECONDS is None:
        return x
    n_seg = min(len(x), int(round(UNSUP_SCORE_SECONDS * sfreq)))
    return x[:n_seg]


def bandpower_ratio_hr(x: np.ndarray, sfreq: float, band_hz=(0.8, 2.5)) -> float:
    """
    Compute a simple FFT-based ratio:
      (power in HR band) / (total power excluding DC)

    ECG-like ICs often contain strong energy around the HR frequency.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < int(sfreq * 5):
        return 0.0

    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    psd = np.abs(np.fft.rfft(x)) ** 2

    valid = freqs > 0  # drop DC
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
    Compute normalized autocorrelation and return the max ACF value within lags
    corresponding to plausible HR periods.

    ECG-like signals are quasi-periodic (but not sinusoidal).
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < int(sfreq * 5):
        return 0.0

    ac = correlate(x, x, mode="full")
    ac = ac[ac.size // 2:]  # non-negative lags
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


def peak_train_plausibility_ecg(
    x: np.ndarray,
    sfreq: float,
    min_bpm: float = 40.0,
    max_bpm: float = 180.0,
    peak_prominence: float = 1.3,
    min_distance_sec: float = 0.30,
):
    """
    Detect peaks on abs(z) and score whether the resulting peak train looks like ECG.

    Outputs:
      score   : [0,1] plausibility score
      bpm_est : estimated BPM from peak count
      n_peaks : number of peaks
      rr_mad  : median absolute deviation of RR intervals (seconds)
    """
    z = safe_zscore(x)
    z_abs = np.abs(z)

    min_dist = max(1, int(round(min_distance_sec * sfreq)))
    peaks, _ = find_peaks(z_abs, prominence=peak_prominence, distance=min_dist)

    duration_sec = len(z) / sfreq
    if duration_sec <= 0 or len(peaks) < 2:
        return 0.0, float("nan"), int(len(peaks)), float("nan")

    bpm = (len(peaks) / duration_sec) * 60.0

    # HR plausibility: soft penalty outside [min_bpm, max_bpm]
    if min_bpm <= bpm <= max_bpm:
        hr_score = 1.0
    else:
        if bpm < min_bpm:
            d = (min_bpm - bpm) / min_bpm
        else:
            d = (bpm - max_bpm) / max_bpm
        hr_score = float(np.exp(-3.0 * d))

    # RR variability: penalize too-perfect periodicity (sinusoid) or too-chaotic trains
    rr = np.diff(peaks) / sfreq
    rr_med = np.median(rr)
    rr_mad = np.median(np.abs(rr - rr_med)) + 1e-12

    rr_score = 1.0
    if rr_mad < RR_MAD_TOO_SMALL_SEC:
        rr_score = 0.3
    elif rr_mad > RR_MAD_TOO_LARGE_SEC:
        rr_score = 0.3

    return float(hr_score * rr_score), float(bpm), int(len(peaks)), float(rr_mad)


def spike_strength_score(x: np.ndarray, sfreq: float):
    """
    Explicitly reward strong spike-like morphology, polarity-invariant.

    Steps:
    - z-score x
    - find local maxima of abs(z) above SPIKE_Z_ABS_THRESHOLD
    - compute spike rate (per minute) and fraction above threshold

    Outputs:
      score             : [0,1] spike strength score
      spike_rate_per_min: spikes/min
      frac_above_thr    : fraction of samples with |z| >= threshold
      n_spikes          : number of spikes found
    """
    z = safe_zscore(x)
    z_abs = np.abs(z)

    min_dist = max(1, int(round(PEAK_MIN_DISTANCE_SEC * sfreq)))
    spikes, _ = find_peaks(z_abs, height=SPIKE_Z_ABS_THRESHOLD, distance=min_dist)

    duration_sec = len(z) / sfreq
    if duration_sec <= 0:
        return 0.0, 0.0, 0.0, 0

    rate_per_min = (len(spikes) / duration_sec) * 60.0
    frac_above = float(np.mean(z_abs >= SPIKE_Z_ABS_THRESHOLD))

    # Rate plausibility gate (soft)
    if SPIKE_RATE_MIN_PER_MIN <= rate_per_min <= SPIKE_RATE_MAX_PER_MIN:
        rate_score = 1.0
    else:
        if rate_per_min < SPIKE_RATE_MIN_PER_MIN:
            d = (SPIKE_RATE_MIN_PER_MIN - rate_per_min) / max(1e-9, SPIKE_RATE_MIN_PER_MIN)
        else:
            d = (rate_per_min - SPIKE_RATE_MAX_PER_MIN) / max(1e-9, SPIKE_RATE_MAX_PER_MIN)
        rate_score = float(np.exp(-3.0 * d))

    # Encourage some content above threshold (mild)
    frac_score = float(np.tanh(frac_above * 50.0))
    score = float(0.7 * rate_score + 0.3 * frac_score)

    return score, float(rate_per_min), float(frac_above), int(len(spikes))


def unsupervised_ecg_ic_score(ic: np.ndarray, sfreq: float) -> dict:
    """
    Compute ECG-likeness heuristics for an ICA component time course (MEG-only).

    Returns a dict containing:
      - combined 'score'
      - sub-scores p_hr, p_ac, p_peaks, p_kurt, p_spike
      - diagnostics like bpm_est, rr_mad_sec, kurtosis, spike_rate_per_min, etc.
    """
    x = _segment_for_scoring(ic, sfreq)
    z = safe_zscore(x)

    # (1) HR band power ratio
    p_hr = bandpower_ratio_hr(z, sfreq, band_hz=HR_BAND_HZ)

    # (2) Autocorrelation periodicity score
    p_ac = autocorr_periodicity_score(z, sfreq, hr_band_hz=HR_BAND_HZ)

    # (3) Peak train plausibility on abs(z)
    p_peaks, bpm, n_peaks, rr_mad = peak_train_plausibility_ecg(
        z,
        sfreq,
        min_bpm=HR_MIN_BPM,
        max_bpm=HR_MAX_BPM,
        peak_prominence=PEAK_PROMINENCE,
        min_distance_sec=PEAK_MIN_DISTANCE_SEC,
    )

    # (4) Kurtosis as peaky morphology
    k = float(kurtosis(z, fisher=False, bias=False)) if len(z) > 10 else 0.0
    p_kurt = float(np.tanh((k - 3.0) / 5.0))  # map roughly to [0,1] for k>3
    p_kurt = max(0.0, p_kurt)

    # (5) Explicit spike strength
    p_spike, spike_rate_per_min, frac_above_thr, n_spikes = spike_strength_score(z, sfreq)

    # Weighted combined score
    score = (
        W_HR_BANDPOWER * p_hr
        + W_AUTOCORR * p_ac
        + W_PEAK_TRAIN * p_peaks
        + W_KURT * p_kurt
        + W_SPIKE_STRENGTH * p_spike
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


# =============================================================================
# HELPERS: ICA extraction and IC selection
# =============================================================================
def fit_ica_and_get_sources(raw_meg: mne.io.BaseRaw) -> tuple[mne.preprocessing.ICA, np.ndarray]:
    """
    Fit ICA on MEG-only data and return:
      - ica object
      - sources matrix shape (n_ic, n_times)

    We filter first (ICA_L_FREQ..ICA_H_FREQ) to improve ICA stability and focus
    on QRS-like content.
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

    # get_sources returns a Raw object; we convert to NumPy array
    sources = ica.get_sources(tmp).get_data()  # (n_ic, n_times)
    if sources.size == 0:
        raise RuntimeError("ICA produced no sources.")
    return ica, sources


def pick_best_ic_supervised(sources: np.ndarray, ecg_ref: np.ndarray) -> tuple[int, float]:
    """
    SUPERVISED IC selection (benchmarking):
    pick IC with maximum absolute correlation to the reference ECG proxy.

    This uses the real-ECG-derived reference trace, so it is NOT MEG-only.
    """
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


def pick_best_ic_unsupervised(sources: np.ndarray, sfreq: float) -> tuple[int, dict, list[dict]]:
    """
    UNSUPERVISED IC selection (MEG-only):
    pick IC with maximum ECG-likeness score from unsupervised_ecg_ic_score()
    or MEGNet (if enabled).

    Returns:
      best_idx,
      best_details,
      all_details_sorted (for debugging / optional ranking plots)
    """
    mode = ICA_UNSUP_MODE.lower()
    use_megnet = mode in {"megnet", "hybrid"}
    model = _get_megnet_model() if use_megnet else None
    all_details = []
    for i in range(sources.shape[0]):
        d_pos = unsupervised_ecg_ic_score(sources[i, :], sfreq)
        d_neg = unsupervised_ecg_ic_score(-sources[i, :], sfreq)
        if d_neg["score"] > d_pos["score"]:
            d = d_neg
            d["flip_for_score"] = True
        else:
            d = d_pos
            d["flip_for_score"] = False
        d["heuristic_score"] = float(d["score"])

        if use_megnet:
            m_pos = megnet_ic_score(sources[i, :], model)
            m_neg = megnet_ic_score(-sources[i, :], model)
            if m_neg > m_pos:
                megnet_score = float(m_neg)
                megnet_flip = True
            else:
                megnet_score = float(m_pos)
                megnet_flip = False
            d["megnet_score"] = megnet_score
            d["megnet_flip_for_score"] = megnet_flip

            if mode == "megnet":
                d["score"] = megnet_score
            else:
                d["score"] = (
                    (1.0 - MEGNET_SCORE_WEIGHT) * d["heuristic_score"]
                    + MEGNET_SCORE_WEIGHT * megnet_score
                )
        else:
            d["megnet_score"] = float("nan")
            d["megnet_flip_for_score"] = False

        d["ic"] = int(i)
        d["selection_mode"] = mode
        all_details.append(d)

    all_details_sorted = sorted(all_details, key=lambda x: x["score"], reverse=True)
    best = all_details_sorted[0]
    return int(best["ic"]), best, all_details_sorted


# =============================================================================
# PLOTTING: ECG-present mode
# =============================================================================
def plot_stacked_ecg_present(
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
    """
    5-panel stacked plot when ECG is available:
      1) real ECG raw channel (z)
      2) reference MNE ECG proxy from real ECG (z)
      3) MNE ECG-from-MEG aligned to reference (z)
      4) ICA supervised aligned to reference (z)
      5) ICA unsupervised aligned to reference (z)
    """
    n_plot = len(t)

    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t, safe_zscore(ecg_raw_real[:n_plot]))
    axes[0].set_title("1) Real ECG channel (raw) [z-score]")
    axes[0].set_ylabel("a.u.")

    axes[1].plot(t, safe_zscore(ecg_ref_mne[:n_plot]))
    axes[1].set_title("2) Reference: MNE ECG trace from REAL ECG (return_ecg=True) [z-score]")
    axes[1].set_ylabel("a.u.")

    axes[2].plot(t, safe_zscore(ecg_mne_meg_aligned[:n_plot]))
    axes[2].set_title(f"3) MNE ECG from MEG (aligned) | lag={lag_mne} | r={r_mne_after:.3f}")
    axes[2].set_ylabel("a.u.")

    axes[3].plot(t, safe_zscore(ecg_ica_sup_aligned[:n_plot]))
    axes[3].set_title(
        f"4) ICA ECG (SUPERVISED; bestIC={sup_ic}, absCorr={sup_abs_corr:.3f}) | lag={lag_sup} | r={r_sup_after:.3f}"
    )
    axes[3].set_ylabel("a.u.")

    axes[4].plot(t, safe_zscore(ecg_ica_unsup_aligned[:n_plot]))
    axes[4].set_title(
        f"5) ICA ECG (UNSUPERVISED; bestIC={unsup_ic}, score={unsup_details['score']:.3f}, bpm≈{unsup_details['bpm_est']:.1f}) "
        f"| lag={lag_unsup} | r={r_unsup_after:.3f}"
    )
    axes[4].set_ylabel("a.u.")
    axes[4].set_xlabel("Time (s)")

    header_1 = f"{subject_id} | sfreq={sfreq:.2f} Hz"
    header_2 = (
        f"MNE r={r_mne_after:.3f} | ICA-sup r={r_sup_after:.3f} | ICA-unsup r={r_unsup_after:.3f} | "
        f"unsup: p_hr={unsup_details['p_hr']:.2f} p_ac={unsup_details['p_ac']:.2f} "
        f"p_peaks={unsup_details['p_peaks']:.2f} p_kurt={unsup_details['p_kurt']:.2f} p_spike={unsup_details['p_spike']:.2f}"
    )
    header_3 = (
        f"unsup: spike_rate/min={unsup_details['spike_rate_per_min']:.1f} "
        f"n_spikes={unsup_details['n_spikes']} frac(|z|>={SPIKE_Z_ABS_THRESHOLD:.1f})={unsup_details['frac_absz_above_thr']:.4f} "
        f"rrMAD={unsup_details['rr_mad_sec']:.4f}s"
    )

    fig.suptitle(f"{header_1}\n{header_2}\n{header_3}", y=0.995, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_overlay_ecg_present(
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
    """
    Overlay plot for ECG-present mode:
      reference + MNE-from-MEG + ICA-sup + ICA-unsup (all aligned to reference).
    """
    n_plot = len(t)
    z_ref = safe_zscore(ecg_ref_mne[:n_plot])
    z_mne = safe_zscore(ecg_mne_meg_aligned[:n_plot])
    z_sup = safe_zscore(ecg_ica_sup_aligned[:n_plot])
    z_unsup = safe_zscore(ecg_ica_unsup_aligned[:n_plot])

    plt.figure(figsize=(14, 5))
    plt.plot(t, z_ref, label="Reference (MNE from REAL ECG)")
    plt.plot(t, z_mne, label=f"MNE from MEG (r={r_mne_after:.3f})")
    plt.plot(t, z_sup, label=f"ICA supervised (r={r_sup_after:.3f})")
    plt.plot(t, z_unsup, label=f"ICA unsupervised (r={r_unsup_after:.3f})")
    plt.legend()
    plt.title(f"{subject_id} | Overlay (aligned) vs reference")
    plt.xlabel("Time (s)")
    plt.ylabel("a.u. (z-score)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# =============================================================================
# PLOTTING: MEG-only mode (no reference ECG)
# =============================================================================
def plot_stacked_meg_only(
    out_png: str,
    t: np.ndarray,
    ecg_mne_from_meg: np.ndarray,
    ecg_ica_unsup: np.ndarray,
    subject_id: str,
    sfreq: float,
    unsup_ic: int,
    unsup_details: dict,
):
    """
    Stacked plot for MEG-only mode:
      1) MNE ECG proxy derived from MEG-only data
      2) ICA unsupervised ECG proxy (best IC by heuristics)

    There is no reference, so we do not compute correlations to ECG reference here.
    """
    n_plot = len(t)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(t, safe_zscore(ecg_mne_from_meg[:n_plot]))
    axes[0].set_title("1) MNE ECG proxy from MEG-only (z-score)")
    axes[0].set_ylabel("a.u.")

    axes[1].plot(t, safe_zscore(ecg_ica_unsup[:n_plot]))
    axes[1].set_title(
        f"2) ICA ECG proxy (UNSUPERVISED; IC={unsup_ic}, score={unsup_details['score']:.3f}, bpm≈{unsup_details['bpm_est']:.1f})"
    )
    axes[1].set_ylabel("a.u.")
    axes[1].set_xlabel("Time (s)")

    header = (
        f"{subject_id} | sfreq={sfreq:.2f} Hz | "
        f"unsup: p_hr={unsup_details['p_hr']:.2f} p_ac={unsup_details['p_ac']:.2f} "
        f"p_peaks={unsup_details['p_peaks']:.2f} p_kurt={unsup_details['p_kurt']:.2f} p_spike={unsup_details['p_spike']:.2f} | "
        f"spikes/min={unsup_details['spike_rate_per_min']:.1f}"
    )
    fig.suptitle(header, y=0.995, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_overlay_meg_only(
    out_png: str,
    t: np.ndarray,
    ecg_mne_from_meg: np.ndarray,
    ecg_ica_unsup: np.ndarray,
    subject_id: str,
    unsup_ic: int,
    unsup_details: dict,
):
    """
    Overlay plot for MEG-only mode:
      MNE-from-MEG proxy + ICA-unsup proxy
    """
    n_plot = len(t)
    z_mne = safe_zscore(ecg_mne_from_meg[:n_plot])
    z_unsup = safe_zscore(ecg_ica_unsup[:n_plot])

    plt.figure(figsize=(14, 5))
    plt.plot(t, z_mne, label="MNE-from-MEG proxy")
    plt.plot(t, z_unsup, label=f"ICA-unsup proxy (IC{unsup_ic}, score={unsup_details['score']:.3f})")
    plt.legend()
    plt.title(f"{subject_id} | MEG-only overlay (no ECG reference)")
    plt.xlabel("Time (s)")
    plt.ylabel("a.u. (z-score)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

# Find all MEG FIF/CTF files under DATASET_ROOT (excluding derivatives and git folders)
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
    # 1) Load data
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
    # 2) Determine whether we have a REAL ECG channel
    #    - If yes, we can run benchmarking (reference + supervised).
    #    - If no, we run MEG-only unsupervised mode.
    # -------------------------------------------------------------------------
    ecg_picks = mne.pick_types(raw.info, ecg=True)
    has_ecg = len(ecg_picks) > 0
    
    ecg_ch_name = ""
    ecg_raw_real = None
    ecg_ref_mne = None
    
    if has_ecg:
        # Grab first ECG channel (you could extend this if multiple ECG channels exist)
        ecg_idx = int(ecg_picks[0])
        ecg_ch_name = raw.ch_names[ecg_idx]
        ecg_raw_real = raw.get_data(picks=[ecg_idx])[0]
    
        # Build the reference ECG proxy from real ECG via find_ecg_events(return_ecg=True)
        try:
            _, _, _, ecg_ref_mne = mne.preprocessing.find_ecg_events(
                raw,
                ch_name=ecg_ch_name,
                return_ecg=True,
                verbose=False,
            )
            ecg_ref_mne = ecg_ref_mne[0]
        except Exception as e:
            print(f"  ERROR find_ecg_events using real ECG: {e}")
            return None
    else:
        print("  No real ECG channel found -> running MEG-only ICA UNSUPERVISED mode.")
    
    # -------------------------------------------------------------------------
    # 3) Build MEG-only raw (this is the common path for baseline + ICA)
    # -------------------------------------------------------------------------
    raw_meg_only = raw.copy().pick_types(meg=True, eeg=False, eog=False, ecg=False, stim=False)
    
    # -------------------------------------------------------------------------
    # 4) Baseline ECG proxy from MEG using MNE's approach
    # -------------------------------------------------------------------------
    try:
        _, _, _, ecg_mne_from_meg = mne.preprocessing.find_ecg_events(
            raw_meg_only,
            ch_name=None,
            return_ecg=True,
            verbose=False,
        )
        ecg_mne_from_meg = ecg_mne_from_meg[0]
    except Exception as e:
        print(f"  ERROR find_ecg_events deriving ECG from MEG: {e}")
        return None
    
    # -------------------------------------------------------------------------
    # 5) Decide the number of samples to analyze (length matching)
    # -------------------------------------------------------------------------
    if has_ecg:
        # In benchmark mode we need everything (real ECG, reference, baseline, ICA sources) to share length
        n = min(len(ecg_raw_real), len(ecg_ref_mne), len(ecg_mne_from_meg), raw_meg_only.n_times)
    else:
        # In MEG-only mode there is no real ECG or reference trace
        n = min(len(ecg_mne_from_meg), raw_meg_only.n_times)
    
    if n < int(round(5 * sfreq)):
        print("  Too short -> skipping")
        return None
    
    # Crop signals to n samples for consistent operations
    ecg_mne_from_meg = ecg_mne_from_meg[:n]
    if has_ecg:
        ecg_raw_real = ecg_raw_real[:n]
        ecg_ref_mne = ecg_ref_mne[:n]
    
    # -------------------------------------------------------------------------
    # 6) Fit ICA once (MEG-only), then select unsupervised (and supervised if ECG exists)
    # -------------------------------------------------------------------------
    try:
        ica, sources = fit_ica_and_get_sources(raw_meg_only)
    except Exception as e:
        print(f"  ERROR ICA fit: {e}")
        return None
    
    # Unsupervised selection is always possible (MEG-only)
    unsup_ic, unsup_details, _all_details_sorted = pick_best_ic_unsupervised(sources, sfreq)
    ecg_ica_unsup = sources[unsup_ic, :n]
    if unsup_details.get("flip_for_score", False):
        ecg_ica_unsup = -ecg_ica_unsup
    
    # Optional MEG-only sign stabilization (purely for plot readability):
    # Make the larger-magnitude side positive so spikes look consistent.
    if np.abs(np.min(ecg_ica_unsup)) > np.abs(np.max(ecg_ica_unsup)):
        ecg_ica_unsup = -ecg_ica_unsup
    
    # -------------------------------------------------------------------------
    # 7) MEG-only mode: plots + row + continue (skip supervised/ref computations)
    # -------------------------------------------------------------------------
    if not has_ecg:
        n_plot = min(n, int(round(PLOT_SECONDS * sfreq)))
        t = np.arange(n_plot) / sfreq
    
        stacked_png = os.path.join(OUTPUT_DIR, f"{subject_id}_MEGonly_stacked_MNEfromMEG_vs_ICAunsup.png")
        overlay_png = os.path.join(OUTPUT_DIR, f"{subject_id}_MEGonly_overlay_MNEfromMEG_vs_ICAunsup.png")
    
        plot_stacked_meg_only(
            out_png=stacked_png,
            t=t,
            ecg_mne_from_meg=ecg_mne_from_meg,
            ecg_ica_unsup=ecg_ica_unsup,
            subject_id=subject_id,
            sfreq=sfreq,
            unsup_ic=unsup_ic,
            unsup_details=unsup_details,
        )
    
        plot_overlay_meg_only(
            out_png=overlay_png,
            t=t,
            ecg_mne_from_meg=ecg_mne_from_meg,
            ecg_ica_unsup=ecg_ica_unsup,
            subject_id=subject_id,
            unsup_ic=unsup_ic,
            unsup_details=unsup_details,
        )
    
        # (Optional) detect ECG events on the IC trace, just as an informative field
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
    
                # Benchmark metrics not available
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
    
                # Unsupervised diagnostics available
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
    
                # Correlation to reference not applicable
                ica_unsup_corr_before=np.nan,
                ica_unsup_corr_after=np.nan,
                ica_unsup_lag_samples=np.nan,
                ica_unsup_lag_seconds=np.nan,
    
                stacked_plot=stacked_png,
                overlay_plot=overlay_png,
            )
    
        print(
            f"  Done (MEG-only) | ICA-unsup IC={unsup_ic}, score={unsup_details['score']:.3f}, "
            f"bpm≈{unsup_details['bpm_est']:.1f}, spikes/min={unsup_details['spike_rate_per_min']:.1f}, "
            f"events_detected={ica_unsup_n_events_detected}"
        )
        return None
    
    # -------------------------------------------------------------------------
    # 8) ECG-present mode (benchmarking): compute baseline alignment/corr
    # -------------------------------------------------------------------------
    lag_mne = best_lag_via_xcorr(ecg_ref_mne, ecg_mne_from_meg)
    ecg_mne_meg_aligned = shift_with_zeros(ecg_mne_from_meg, lag_mne)
    r_mne_before, _ = pearsonr(ecg_ref_mne, ecg_mne_from_meg)
    r_mne_after, _ = pearsonr(ecg_ref_mne, ecg_mne_meg_aligned)
    
    # -------------------------------------------------------------------------
    # 9) ICA SUPERVISED (benchmarking only): pick IC with max abs corr to reference
    # -------------------------------------------------------------------------
    sup_ic, sup_abs_corr = pick_best_ic_supervised(sources, ecg_ref_mne)
    ecg_ica_sup = sources[sup_ic, :n]
    
    # Sign stabilize to match reference for fair plotting/correlation
    if np.corrcoef(ecg_ica_sup, ecg_ref_mne)[0, 1] < 0:
        ecg_ica_sup = -ecg_ica_sup
    
    lag_sup = best_lag_via_xcorr(ecg_ref_mne, ecg_ica_sup)
    ecg_ica_sup_aligned = shift_with_zeros(ecg_ica_sup, lag_sup)
    r_sup_before, _ = pearsonr(ecg_ref_mne, ecg_ica_sup)
    r_sup_after, _ = pearsonr(ecg_ref_mne, ecg_ica_sup_aligned)
    
    # -------------------------------------------------------------------------
    # 10) ICA UNSUPERVISED (benchmarking overlay only):
    #     IMPORTANT: selection used NO ECG, but for plotting we can sign-stabilize
    #     using the reference (benchmark only).
    # -------------------------------------------------------------------------
    # NOTE: ecg_ica_unsup already cropped and optional spike-orientation above;
    # here we do a benchmark-only sign stabilization to reference.
    if np.corrcoef(ecg_ica_unsup, ecg_ref_mne)[0, 1] < 0:
        ecg_ica_unsup = -ecg_ica_unsup
    
    lag_unsup = best_lag_via_xcorr(ecg_ref_mne, ecg_ica_unsup)
    ecg_ica_unsup_aligned = shift_with_zeros(ecg_ica_unsup, lag_unsup)
    r_unsup_before, _ = pearsonr(ecg_ref_mne, ecg_ica_unsup)
    r_unsup_after, _ = pearsonr(ecg_ref_mne, ecg_ica_unsup_aligned)
    
    # -------------------------------------------------------------------------
    # 11) Plotting (benchmark mode)
    # -------------------------------------------------------------------------
    n_plot = min(n, int(round(PLOT_SECONDS * sfreq)))
    t = np.arange(n_plot) / sfreq
    
    stacked_png = os.path.join(OUTPUT_DIR, f"{subject_id}_stacked_ref_mne_sup_unsup.png")
    overlay_png = os.path.join(OUTPUT_DIR, f"{subject_id}_overlay_ref_mne_sup_unsup.png")
    
    plot_stacked_ecg_present(
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
    
    plot_overlay_ecg_present(
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
    
    # -------------------------------------------------------------------------
    # 12) Save results (benchmark mode)
    # -------------------------------------------------------------------------
    result = dict(
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
    
    print(
        f"  Done | ECG={ecg_ch_name} | "
        f"MNE r_after={r_mne_after:.3f} | "
        f"ICA-sup r_after={r_sup_after:.3f} (IC={sup_ic}) | "
        f"ICA-unsup r_after={r_unsup_after:.3f} (IC={unsup_ic}, score={unsup_details['score']:.3f}, bpm≈{unsup_details['bpm_est']:.1f}) | "
        f"unsup spikes/min={unsup_details['spike_rate_per_min']:.1f}"
    )
    return result


# Run processing (serial or parallel)
if N_JOBS == 1:
    results = [r for r in (process_file(p) for p in data_paths) if r is not None]
else:
    results = Parallel(n_jobs=N_JOBS)(delayed(process_file)(p) for p in data_paths)
    results = [r for r in results if r is not None]


# =============================================================================
# FINAL: SAVE CSV SUMMARY
# =============================================================================
df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "ecg_summary_supervised_unsupervised_ica_v6.csv")
df.to_csv(csv_path, index=False)

print("\n=== SUMMARY ===")
if len(df) > 0:
    cols = [
        "mne_meg_corr_after",
        "ica_sup_corr_after",
        "ica_unsup_corr_after",
        "ica_unsup_score",
        "ica_unsup_bpm_est",
        "ica_unsup_spike_rate_per_min",
        "ica_unsup_p_spike",
    ]
    # Some columns may be NaN for MEG-only rows; describe() will handle that.
    print(df[cols].describe())
    print(f"\nSaved summary CSV: {csv_path}")
else:
    print("No valid files processed.")

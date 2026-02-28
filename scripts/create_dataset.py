"""
create_dataset.py
-----------------
Reads raw sleep signals for all participants, applies bandpass filtering,
windows the signals (30 s, 50 % overlap), labels each window from the
flow_events annotation file, and saves the resulting dataset.

Usage:
    python scripts/create_dataset.py -in_dir Data -out_dir Dataset
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, sosfilt
from scipy.stats import skew, kurtosis

# ── constants ─────────────────────────────────────────────────────────────────
FS_RESP  = 32    # Hz for nasal airflow & thoracic movement
FS_SPO2  = 4     # Hz for SpO2

WINDOW_S  = 30   # window length in seconds
OVERLAP   = 0.50 # 50% overlap
STEP_S    = WINDOW_S * (1 - OVERLAP)

BP_LOW   = 0.17  # Hz  (lower bound for breathing band)
BP_HIGH  = 0.40  # Hz  (upper bound for breathing band)

LABEL_OVERLAP_THRESH = 0.50   # >50% overlap → event label


# ── filtering ─────────────────────────────────────────────────────────────────

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Butterworth bandpass filter using second-order sections (numerically stable)."""
    nyq = fs / 2
    lo  = lowcut  / nyq
    hi  = highcut / nyq
    # clamp to valid range
    lo = max(lo, 1e-6)
    hi = min(hi, 1 - 1e-6)
    sos = butter(order, [lo, hi], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)


def lowpass_filter(signal, cutoff, fs, order=4):
    """Butterworth lowpass for SpO2 smoothing."""
    nyq = fs / 2
    c   = min(cutoff / nyq, 1 - 1e-6)
    sos = butter(order, c, btype="low", output="sos")
    return sosfiltfilt(sos, signal)


# ── feature extraction ────────────────────────────────────────────────────────

def extract_features(nasal_w, thoracic_w, spo2_w):
    """
    Extract a compact feature vector from a single window.
    Returns a 1-D numpy array.
    """
    feats = []

    for sig, name in [(nasal_w, "nasal"), (thoracic_w, "thoracic")]:
        feats += [
            np.mean(sig),
            np.std(sig),
            np.max(np.abs(sig)),
            np.percentile(sig, 25),
            np.percentile(sig, 75),
            skew(sig),
            kurtosis(sig),
            np.mean(np.abs(np.diff(sig))),          # mean absolute difference
        ]
        # spectral power in breathing band using FFT
        fft_mag  = np.abs(np.fft.rfft(sig))
        freqs    = np.fft.rfftfreq(len(sig), d=1/FS_RESP)
        bp_mask  = (freqs >= BP_LOW) & (freqs <= BP_HIGH)
        total_pw = np.sum(fft_mag**2) + 1e-12
        bp_pw    = np.sum(fft_mag[bp_mask]**2)
        feats   += [bp_pw / total_pw,               # breathing band power ratio
                    bp_pw]                           # absolute breathing band power

    for sig, name in [(spo2_w, "spo2")]:
        feats += [
            np.mean(sig),
            np.std(sig),
            np.min(sig),
            np.max(sig) - np.min(sig),              # SpO2 range (desaturation depth proxy)
            np.mean(np.diff(sig)),                  # trend
        ]

    # cross-signal coherence proxy: correlation between nasal and thoracic
    if np.std(nasal_w) > 1e-6 and np.std(thoracic_w) > 1e-6:
        corr = np.corrcoef(
            nasal_w[:min(len(nasal_w), len(thoracic_w))],
            thoracic_w[:min(len(nasal_w), len(thoracic_w))]
        )[0, 1]
    else:
        corr = 0.0
    feats.append(corr)

    return np.array(feats, dtype=np.float32)


def feature_names():
    names = []
    for sig in ["nasal", "thoracic"]:
        names += [f"{sig}_{s}" for s in [
            "mean", "std", "abs_max", "p25", "p75",
            "skew", "kurt", "mad", "bp_ratio", "bp_power"]]
    for sig in ["spo2"]:
        names += [f"{sig}_{s}" for s in [
            "mean", "std", "min", "range", "trend"]]
    names.append("nasal_thoracic_corr")
    return names


# ── labelling ─────────────────────────────────────────────────────────────────

def label_window(w_start, w_end, events_df):
    """
    Return the label for a window [w_start, w_end].
    If >50% of the window overlaps with an event → that event label.
    Multiple events: pick the one with the most overlap.
    """
    window_len = w_end - w_start
    best_label = "Normal"
    best_overlap = 0.0

    for _, row in events_df.iterrows():
        overlap_start = max(w_start, row["start"])
        overlap_end   = min(w_end,   row["end"])
        overlap       = max(0.0, overlap_end - overlap_start)
        overlap_frac  = overlap / window_len

        if overlap_frac > LABEL_OVERLAP_THRESH and overlap_frac > best_overlap:
            best_overlap = overlap_frac
            best_label   = row["event"]

    return best_label


# ── per-participant processing ────────────────────────────────────────────────

def process_participant(folder, pid):
    nasal_path    = os.path.join(folder, "nasal_airflow.txt")
    thoracic_path = os.path.join(folder, "thoracic_movement.txt")
    spo2_path     = os.path.join(folder, "spo2.txt")
    events_path   = os.path.join(folder, "flow_events.csv")

    for p in [nasal_path, thoracic_path, spo2_path, events_path]:
        if not os.path.exists(p):
            print(f"  [WARN] Missing file: {p} – skipping {pid}")
            return pd.DataFrame()

    # ── load ──────────────────────────────────────────────────────────────────
    df_nasal    = pd.read_csv(nasal_path)
    df_thoracic = pd.read_csv(thoracic_path)
    df_spo2     = pd.read_csv(spo2_path)
    events_df   = pd.read_csv(events_path)

    t_nasal    = df_nasal["timestamp"].values.astype(float)
    nasal      = df_nasal["nasal_airflow"].values.astype(float)
    t_thoracic = df_thoracic["timestamp"].values.astype(float)
    thoracic   = df_thoracic["thoracic_movement"].values.astype(float)
    t_spo2     = df_spo2["timestamp"].values.astype(float)
    spo2       = df_spo2["spo2"].values.astype(float)

    # ── filter ────────────────────────────────────────────────────────────────
    nasal_filt    = bandpass_filter(nasal,    BP_LOW, BP_HIGH, FS_RESP)
    thoracic_filt = bandpass_filter(thoracic, BP_LOW, BP_HIGH, FS_RESP)
    spo2_filt     = lowpass_filter(spo2, 0.5, FS_SPO2)

    # ── windowing ─────────────────────────────────────────────────────────────
    win_resp  = int(WINDOW_S * FS_RESP)
    win_spo2  = int(WINDOW_S * FS_SPO2)
    step_resp = int(STEP_S   * FS_RESP)
    step_spo2 = int(STEP_S   * FS_SPO2)

    duration = min(t_nasal[-1], t_thoracic[-1], t_spo2[-1])
    n_windows = int((duration - WINDOW_S) / STEP_S) + 1

    rows = []
    for i in range(n_windows):
        w_start = i * STEP_S
        w_end   = w_start + WINDOW_S

        # index into arrays
        idx_r_s = int(w_start * FS_RESP)
        idx_r_e = idx_r_s + win_resp
        idx_s_s = int(w_start * FS_SPO2)
        idx_s_e = idx_s_s + win_spo2

        if idx_r_e > len(nasal_filt) or idx_s_e > len(spo2_filt):
            break

        nasal_w    = nasal_filt[idx_r_s:idx_r_e]
        thoracic_w = thoracic_filt[idx_r_s:idx_r_e]
        spo2_w     = spo2_filt[idx_s_s:idx_s_e]

        label = label_window(w_start, w_end, events_df)
        feats = extract_features(nasal_w, thoracic_w, spo2_w)

        row = {"participant": pid, "window_start": w_start, "window_end": w_end, "label": label}
        for fname, fval in zip(feature_names(), feats):
            row[fname] = fval
        rows.append(row)

    df = pd.DataFrame(rows)
    label_counts = df["label"].value_counts().to_dict()
    print(f"  [{pid}] {len(df)} windows  |  labels: {label_counts}")
    return df


# ── sleep stage dataset (bonus) ───────────────────────────────────────────────

def process_sleep_stages(folder, pid):
    sleep_path = os.path.join(folder, "sleep_profile.csv")
    if not os.path.exists(sleep_path):
        return pd.DataFrame()
    df = pd.read_csv(sleep_path)
    df["participant"] = pid
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build labeled window dataset from sleep signals.")
    parser.add_argument("-in_dir",  default="Data",    help="Root data directory")
    parser.add_argument("-out_dir", default="Dataset", help="Output directory for CSV files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    participant_dirs = sorted(
        [d for d in glob.glob(os.path.join(args.in_dir, "*"))
         if os.path.isdir(d)]
    )

    if not participant_dirs:
        print(f"[ERROR] No participant folders found in '{args.in_dir}'")
        return

    print(f"Found {len(participant_dirs)} participant(s). Processing ...")

    all_windows     = []
    all_sleep_stages = []

    for folder in participant_dirs:
        pid = os.path.basename(folder)
        print(f"\nProcessing {pid} ...")
        df_win   = process_participant(folder, pid)
        df_sleep = process_sleep_stages(folder, pid)
        if not df_win.empty:
            all_windows.append(df_win)
        if not df_sleep.empty:
            all_sleep_stages.append(df_sleep)

    if all_windows:
        df_all = pd.concat(all_windows, ignore_index=True)
        out_path = os.path.join(args.out_dir, "breathing_dataset.csv")
        df_all.to_csv(out_path, index=False)
        print(f"\n[OK] Breathing dataset saved → {out_path}")
        print(f"     Total windows: {len(df_all)}")
        print(f"     Label distribution:\n{df_all['label'].value_counts()}")

    if all_sleep_stages:
        df_sleep_all = pd.concat(all_sleep_stages, ignore_index=True)
        sleep_out = os.path.join(args.out_dir, "sleep_stage_dataset.csv")
        df_sleep_all.to_csv(sleep_out, index=False)
        print(f"\n[OK] Sleep stage dataset saved → {sleep_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()

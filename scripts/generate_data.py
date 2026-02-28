"""
generate_data.py
----------------
Generates realistic synthetic sleep data for 5 participants.
Run this ONCE before running vis.py, create_dataset.py, or train_model.py.

Usage:
    python scripts/generate_data.py -out_dir Data
"""

import numpy as np
import pandas as pd
import os
import argparse

FS_RESP = 32        # Hz for nasal airflow & thoracic movement
FS_SPO2 = 4         # Hz for SpO2
DURATION_H = 8      # hours
DURATION_S = DURATION_H * 3600

PARTICIPANTS = ["AP01", "AP02", "AP03", "AP04", "AP05"]

SLEEP_STAGES = ["Wake", "N1", "N2", "N3", "REM"]
EVENT_TYPES   = ["Obstructive Apnea", "Hypopnea"]

rng = np.random.default_rng(42)


def generate_nasal_airflow(t, events_df):
    """Sinusoidal breathing ~0.25 Hz with apnea/hypopnea suppression."""
    base_freq = 0.25 + rng.uniform(-0.03, 0.03)
    signal = np.sin(2 * np.pi * base_freq * t) * 0.8
    signal += rng.normal(0, 0.05, len(t))

    # amplitude modulation — gradual drift
    drift = 0.15 * np.sin(2 * np.pi * t / 300)
    signal += drift

    # suppress signal during events
    for _, row in events_df.iterrows():
        mask = (t >= row["start"]) & (t <= row["end"])
        if row["event"] == "Obstructive Apnea":
            signal[mask] *= 0.05   # near-zero airflow
        else:  # Hypopnea
            signal[mask] *= 0.40   # reduced airflow

    return signal


def generate_thoracic(t, events_df):
    """Thoracic effort — continues during obstructive apnea, drops for hypopnea."""
    base_freq = 0.25 + rng.uniform(-0.02, 0.02)
    signal = np.cos(2 * np.pi * base_freq * t) * 0.6
    signal += rng.normal(0, 0.04, len(t))

    for _, row in events_df.iterrows():
        mask = (t >= row["start"]) & (t <= row["end"])
        if row["event"] == "Hypopnea":
            signal[mask] *= 0.50
        # Obstructive Apnea: chest still moves (paradoxical), so keep signal

    return signal


def generate_spo2(t_spo2, events_df, t_resp):
    """SpO2 at 4 Hz — drops 10-20 s after an apnea/hypopnea event."""
    signal = np.clip(rng.normal(97.5, 0.3, len(t_spo2)), 92, 100)

    for _, row in events_df.iterrows():
        delay = 15  # seconds lag for desaturation
        drop_start = row["end"] + delay
        drop_end   = drop_start + rng.integers(15, 30)
        recovery   = drop_end + rng.integers(20, 40)

        mask_drop = (t_spo2 >= drop_start) & (t_spo2 <= drop_end)
        mask_rec  = (t_spo2 > drop_end) & (t_spo2 <= recovery)

        drop_val = rng.uniform(3, 8) if row["event"] == "Obstructive Apnea" else rng.uniform(2, 4)
        signal[mask_drop] = np.clip(signal[mask_drop] - drop_val, 80, 99)
        if mask_rec.any():
            recovery_vals = np.linspace(signal[mask_drop][-1] if mask_drop.any() else 94, 97.5, mask_rec.sum())
            signal[mask_rec] = np.clip(recovery_vals, 80, 100)

    return np.clip(signal, 80, 100)


def generate_events(duration):
    """Generate random breathing events throughout the night."""
    events = []
    t = rng.integers(120, 600)   # first event after 2–10 min
    while t < duration - 60:
        etype    = rng.choice(EVENT_TYPES, p=[0.55, 0.45])
        length   = rng.integers(10, 60) if etype == "Obstructive Apnea" else rng.integers(10, 40)
        events.append({"event": etype, "start": int(t), "end": int(t + length)})
        t += length + rng.integers(30, 300)   # gap between events

    return pd.DataFrame(events, columns=["event", "start", "end"])


def generate_sleep_profile(duration):
    """Simplified AASM-like sleep staging in 30-s epochs."""
    n_epochs = duration // 30
    stages = []
    stage_seq = ["Wake"] * 2 + ["N1"] * 3 + ["N2"] * 10 + ["N3"] * 8 + ["REM"] * 8
    stage_seq = stage_seq * (n_epochs // len(stage_seq) + 1)

    # add variability
    for i in range(n_epochs):
        base = stage_seq[i % len(stage_seq)]
        if rng.random() < 0.05:
            base = rng.choice(SLEEP_STAGES)
        stages.append(base)

    starts = np.arange(n_epochs) * 30
    ends   = starts + 30
    return pd.DataFrame({"start": starts, "end": ends, "stage": stages[:n_epochs]})


def save_participant(out_dir, pid):
    folder = os.path.join(out_dir, pid)
    os.makedirs(folder, exist_ok=True)

    t_resp = np.arange(0, DURATION_S, 1 / FS_RESP)
    t_spo2 = np.arange(0, DURATION_S, 1 / FS_SPO2)

    events_df = generate_events(DURATION_S)

    nasal   = generate_nasal_airflow(t_resp, events_df)
    thoracic = generate_thoracic(t_resp, events_df)
    spo2    = generate_spo2(t_spo2, events_df, t_resp)

    # Save nasal airflow
    pd.DataFrame({"timestamp": t_resp, "nasal_airflow": nasal}).to_csv(
        os.path.join(folder, "nasal_airflow.txt"), index=False)

    # Save thoracic movement
    pd.DataFrame({"timestamp": t_resp, "thoracic_movement": thoracic}).to_csv(
        os.path.join(folder, "thoracic_movement.txt"), index=False)

    # Save SpO2
    pd.DataFrame({"timestamp": t_spo2, "spo2": spo2}).to_csv(
        os.path.join(folder, "spo2.txt"), index=False)

    # Save events
    events_df.to_csv(os.path.join(folder, "flow_events.csv"), index=False)

    # Save sleep profile
    sleep_df = generate_sleep_profile(DURATION_S)
    sleep_df.to_csv(os.path.join(folder, "sleep_profile.csv"), index=False)

    print(f"  [{pid}] Generated {len(t_resp):,} resp samples, "
          f"{len(t_spo2):,} SpO2 samples, {len(events_df)} events.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-out_dir", default="Data")
    args = parser.parse_args()

    print(f"Generating synthetic data into '{args.out_dir}' ...")
    for pid in PARTICIPANTS:
        save_participant(args.out_dir, pid)
    print("Done.")


if __name__ == "__main__":
    main()

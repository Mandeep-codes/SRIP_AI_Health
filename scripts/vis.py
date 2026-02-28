"""
vis.py
------
Visualizes the 8-hour sleep recording for one participant and saves a PDF.

Usage:
    python scripts/vis.py -name "Data/AP01"
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# ── colour palette for events ────────────────────────────────────────────────
EVENT_COLORS = {
    "Obstructive Apnea": "#e74c3c",
    "Hypopnea":          "#f39c12",
    "Central Apnea":     "#9b59b6",
    "Mixed Apnea":       "#1abc9c",
}
STAGE_COLORS = {
    "Wake": "#e74c3c",
    "N1":   "#f39c12",
    "N2":   "#3498db",
    "N3":   "#2ecc71",
    "REM":  "#9b59b6",
}


def load_signal(path, time_col, val_col):
    df = pd.read_csv(path)
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[val_col]  = pd.to_numeric(df[val_col],  errors="coerce")
    df = df.dropna(subset=[time_col, val_col])
    return df[time_col].values, df[val_col].values


def load_events(path):
    df = pd.read_csv(path)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"]   = pd.to_numeric(df["end"],   errors="coerce")
    return df.dropna(subset=["start", "end"])


def load_sleep_profile(path):
    df = pd.read_csv(path)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"]   = pd.to_numeric(df["end"],   errors="coerce")
    return df.dropna(subset=["start", "end"])


def overlay_events(ax, events_df, ymin, ymax, alpha=0.25):
    for _, row in events_df.iterrows():
        color = EVENT_COLORS.get(row["event"], "#888888")
        ax.axvspan(row["start"] / 3600, row["end"] / 3600,
                   ymin=0, ymax=1, color=color, alpha=alpha, linewidth=0)


def add_event_legend(ax, events_df):
    present = events_df["event"].unique()
    patches = [mpatches.Patch(color=EVENT_COLORS.get(e, "#888888"),
                              alpha=0.6, label=e) for e in present]
    ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.7)


def plot_hypnogram(ax, sleep_df):
    stage_order = ["Wake", "REM", "N1", "N2", "N3"]
    stage_to_y  = {s: i for i, s in enumerate(reversed(stage_order))}

    for _, row in sleep_df.iterrows():
        s = row.get("stage", "N2")
        y = stage_to_y.get(s, 2)
        c = STAGE_COLORS.get(s, "#aaaaaa")
        ax.barh(y, (row["end"] - row["start"]) / 3600,
                left=row["start"] / 3600, height=0.8,
                color=c, alpha=0.85, linewidth=0)

    ax.set_yticks(list(stage_to_y.values()))
    ax.set_yticklabels(list(stage_to_y.keys()), fontsize=8)
    ax.set_ylabel("Sleep Stage", fontsize=9)
    ax.set_title("Hypnogram (Sleep Stages)", fontsize=10, fontweight="bold")


def main():
    parser = argparse.ArgumentParser(description="Visualise sleep data for one participant.")
    parser.add_argument("-name", required=True,
                        help='Path to participant folder, e.g. "Data/AP01"')
    args = parser.parse_args()

    folder = args.name
    pid    = os.path.basename(folder.rstrip("/\\"))

    # ── file paths ────────────────────────────────────────────────────────────
    nasal_path   = os.path.join(folder, "nasal_airflow.txt")
    thoracic_path = os.path.join(folder, "thoracic_movement.txt")
    spo2_path    = os.path.join(folder, "spo2.txt")
    events_path  = os.path.join(folder, "flow_events.csv")
    sleep_path   = os.path.join(folder, "sleep_profile.csv")

    for p in [nasal_path, thoracic_path, spo2_path, events_path]:
        if not os.path.exists(p):
            sys.exit(f"[ERROR] File not found: {p}")

    print(f"Loading data for {pid} ...")
    t_nasal,   nasal   = load_signal(nasal_path,   "timestamp", "nasal_airflow")
    t_thoracic, thoracic = load_signal(thoracic_path, "timestamp", "thoracic_movement")
    t_spo2,    spo2    = load_signal(spo2_path,    "timestamp", "spo2")
    events_df  = load_events(events_path)
    sleep_df   = load_sleep_profile(sleep_path) if os.path.exists(sleep_path) else pd.DataFrame()

    # ── downsample for plotting speed (plot at most 1 sample / 0.25 s) ───────
    def thin(t, s, target_fs=4):
        step = max(1, int(round(1 / (t[1] - t[0]) / target_fs)))
        return t[::step] / 3600, s[::step]   # convert to hours

    tn, sn  = thin(t_nasal,    nasal)
    tt, st  = thin(t_thoracic, thoracic)
    ts, ss  = t_spo2 / 3600,  spo2

    # ── output path ───────────────────────────────────────────────────────────
    os.makedirs("Visualizations", exist_ok=True)
    out_pdf = os.path.join("Visualizations", f"{pid}_visualization.pdf")

    print(f"Plotting and saving to {out_pdf} ...")

    with PdfPages(out_pdf) as pdf:
        # ── PAGE 1 : full-night overview ──────────────────────────────────────
        n_axes = 4 if not sleep_df.empty else 3
        fig, axes = plt.subplots(n_axes, 1,
                                 figsize=(16, 3.2 * n_axes),
                                 sharex=True,
                                 gridspec_kw={"hspace": 0.45})
        fig.suptitle(f"Sleep Study – Participant {pid}\n(Full 8-Hour Recording)",
                     fontsize=13, fontweight="bold", y=1.01)

        # Nasal Airflow
        ax = axes[0]
        ax.plot(tn, sn, color="#2980b9", linewidth=0.5, alpha=0.85)
        overlay_events(ax, events_df, sn.min(), sn.max())
        add_event_legend(ax, events_df)
        ax.set_ylabel("Nasal Airflow\n(a.u.)", fontsize=9)
        ax.set_title("Nasal Airflow", fontsize=10, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        # Thoracic Movement
        ax = axes[1]
        ax.plot(tt, st, color="#27ae60", linewidth=0.5, alpha=0.85)
        overlay_events(ax, events_df, st.min(), st.max())
        ax.set_ylabel("Thoracic Movement\n(a.u.)", fontsize=9)
        ax.set_title("Thoracic Movement", fontsize=10, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        # SpO2
        ax = axes[2]
        ax.plot(ts, ss, color="#8e44ad", linewidth=0.7, alpha=0.9)
        overlay_events(ax, events_df, 80, 100)
        ax.set_ylim(80, 102)
        ax.set_ylabel("SpO₂ (%)", fontsize=9)
        ax.set_title("Oxygen Saturation (SpO₂)", fontsize=10, fontweight="bold")
        ax.axhline(90, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="90% threshold")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        # Hypnogram (optional)
        if not sleep_df.empty:
            plot_hypnogram(axes[3], sleep_df)
            overlay_events(axes[3], events_df, 0, 1, alpha=0.15)
            axes[3].grid(axis="x", linestyle="--", alpha=0.3)

        axes[-1].set_xlabel("Time (hours)", fontsize=10)
        axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}h"))

        # global event summary box
        summary_lines = []
        for etype, grp in events_df.groupby("event"):
            durations = grp["end"] - grp["start"]
            summary_lines.append(
                f"{etype}: n={len(grp)},  "
                f"mean dur={durations.mean():.1f}s,  "
                f"total={durations.sum()/60:.1f} min"
            )
        fig.text(0.01, -0.02, "  |  ".join(summary_lines),
                 fontsize=8, color="#555555", style="italic")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── PAGE 2 : zoomed 30-minute window (first event + context) ─────────
        if not events_df.empty:
            first_event_start = events_df["start"].min()
            zoom_start = max(0, first_event_start - 300)
            zoom_end   = zoom_start + 1800   # 30 min

            fig2, axes2 = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                                       gridspec_kw={"hspace": 0.45})
            fig2.suptitle(f"Participant {pid} – Zoomed View "
                          f"({zoom_start/60:.0f}–{zoom_end/60:.0f} min)",
                          fontsize=12, fontweight="bold")

            mask_n = (t_nasal >= zoom_start) & (t_nasal <= zoom_end)
            mask_t = (t_thoracic >= zoom_start) & (t_thoracic <= zoom_end)
            mask_s = (t_spo2 >= zoom_start) & (t_spo2 <= zoom_end)
            ev_zoom = events_df[(events_df["start"] <= zoom_end) &
                                (events_df["end"] >= zoom_start)].copy()

            def zoom_time(t): return t / 60  # minutes

            axes2[0].plot(zoom_time(t_nasal[mask_n]), nasal[mask_n],
                          color="#2980b9", linewidth=0.8)
            axes2[1].plot(zoom_time(t_thoracic[mask_t]), thoracic[mask_t],
                          color="#27ae60", linewidth=0.8)
            axes2[2].plot(zoom_time(t_spo2[mask_s]), spo2[mask_s],
                          color="#8e44ad", linewidth=1.0)
            axes2[2].set_ylim(80, 102)
            axes2[2].axhline(90, color="red", linestyle="--", linewidth=0.8, alpha=0.6)

            titles2 = ["Nasal Airflow", "Thoracic Movement", "SpO₂ (%)"]
            ylabels2 = ["Airflow (a.u.)", "Movement (a.u.)", "SpO₂ (%)"]
            for ax, ttl, yl in zip(axes2, titles2, ylabels2):
                for _, row in ev_zoom.iterrows():
                    c = EVENT_COLORS.get(row["event"], "#888")
                    ax.axvspan(row["start"]/60, row["end"]/60,
                               color=c, alpha=0.3, linewidth=0)
                add_event_legend(ax, ev_zoom)
                ax.set_title(ttl, fontsize=10, fontweight="bold")
                ax.set_ylabel(yl, fontsize=9)
                ax.grid(axis="x", linestyle="--", alpha=0.3)

            axes2[-1].set_xlabel("Time (minutes)", fontsize=10)
            plt.tight_layout()
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

        # ── PAGE 3 : event statistics ─────────────────────────────────────────
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
        fig3.suptitle(f"Participant {pid} – Event Statistics", fontsize=12, fontweight="bold")

        # Count per type
        counts = events_df["event"].value_counts()
        axes3[0].bar(counts.index,
                     counts.values,
                     color=[EVENT_COLORS.get(e, "#888") for e in counts.index],
                     alpha=0.85, edgecolor="white")
        axes3[0].set_title("Event Counts by Type", fontsize=10, fontweight="bold")
        axes3[0].set_ylabel("Count")
        axes3[0].tick_params(axis="x", rotation=15)

        # Duration distribution
        events_df["duration"] = events_df["end"] - events_df["start"]
        for etype in events_df["event"].unique():
            subset = events_df[events_df["event"] == etype]["duration"]
            axes3[1].hist(subset, bins=15, alpha=0.65,
                          label=etype, color=EVENT_COLORS.get(etype, "#888"),
                          edgecolor="white")
        axes3[1].set_title("Event Duration Distribution", fontsize=10, fontweight="bold")
        axes3[1].set_xlabel("Duration (seconds)")
        axes3[1].set_ylabel("Count")
        axes3[1].legend(fontsize=8)

        plt.tight_layout()
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = f"Sleep Study Visualization – {pid}"
        d["Author"]  = "AI for Health – SRIP 2026"
        d["Subject"] = "Breathing Irregularity Detection"

    print(f"[OK] Saved {out_pdf}")


if __name__ == "__main__":
    main()

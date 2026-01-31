from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks


def _load_ecg(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        ecg = data.astype(float)
        fs = 300.0
        t = np.arange(len(ecg)) / fs
        return t, ecg, fs

    if data.shape[1] >= 2:
        t_col = data[:, 0].astype(float)
        ecg = data[:, 1].astype(float)
        dt = np.diff(t_col)
        if np.all(dt > 0):
            fs = float(1.0 / np.median(dt))
            t = t_col
            return t, ecg, fs

    ecg = data[:, 0].astype(float)
    fs = 300.0
    t = np.arange(len(ecg)) / fs
    return t, ecg, fs


def _bandpass(signal: np.ndarray, fs: float, lo: float = 1.0, hi: float = 150.0) -> np.ndarray:
    nyq = 0.5 * fs
    hi = min(hi, nyq - 1.0)
    lo = max(lo, 0.5)
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, signal)


def _envelope(signal: np.ndarray, fs: float, cutoff: float = 10.0) -> np.ndarray:
    rect = np.abs(signal)
    nyq = 0.5 * fs
    b, a = butter(2, cutoff / nyq, btype="low")
    return filtfilt(b, a, rect)


def _find_t_peaks(ecg_filt: np.ndarray, r_peaks: np.ndarray, fs: float) -> np.ndarray:
    t_peaks: list[int] = []
    start_offset = int(0.12 * fs)
    end_offset = int(0.4 * fs)
    for r_idx in r_peaks:
        start = r_idx + start_offset
        end = r_idx + end_offset
        if end >= len(ecg_filt):
            continue
        t_idx = int(np.argmax(ecg_filt[start:end]) + start)
        t_peaks.append(t_idx)
    return np.array(t_peaks, dtype=int)


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    candidates = [
        base / "ECGPCG.csv",
        base / "ECG_300Hz.csv",
        base / "ECG_150Hz.csv",
        base.parent.parent / "signals_2" / "files" / "ECGPCG.csv",
    ]
    ecg_path = next((p for p in candidates if p.exists()), None)
    if ecg_path is None:
        raise FileNotFoundError("No ECG file found. Place ECGPCG.csv in files/ or update the path.")

    t, ecg, fs = _load_ecg(ecg_path)
    ecg_filt = _bandpass(ecg, fs)
    env = _envelope(ecg_filt, fs)

    min_distance = int(0.25 * fs)
    prominence = 0.5 * np.std(env)
    r_peaks, _ = find_peaks(env, distance=min_distance, prominence=prominence)

    t_peaks = _find_t_peaks(ecg_filt, r_peaks, fs)

    print(f"Loaded: {ecg_path}")
    print(f"Estimated fs: {fs:.2f} Hz")
    print(f"R-peaks found: {len(r_peaks)}")
    print(f"T-peaks found: {len(t_peaks)}")

    plt.figure(figsize=(12, 6))
    plt.plot(t, ecg_filt, color="black", linewidth=1, label="ECG (bandpass)")
    plt.plot(t, env, color="tab:blue", alpha=0.8, label="Envelope")
    plt.scatter(t[r_peaks], ecg_filt[r_peaks], color="red", s=20, label="R-peaks")
    if len(t_peaks) > 0:
        plt.scatter(t[t_peaks], ecg_filt[t_peaks], color="green", s=20, label="T-peaks")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (a.u.)")
    plt.title("EKG med envelope, R-peaks og T-peaks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

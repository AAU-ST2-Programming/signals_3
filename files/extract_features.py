import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
from scipy.signal import find_peaks
filename = 'signals_3/files/b001.csv'
I,II,RESP,SCG,timestamp = np.loadtxt(filename, delimiter=',', unpack=True, skiprows=1)
fs = 1/(np.diff(timestamp).mean())

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def get_envelope(signal):
    """Compute the envelope of a signal using the analytic signal (Hilbert transform)"""
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope

def find_rr_peaks(ecg_signal, fs, prominence=None, distance=None):
    # Get envelope
    envelope = get_envelope(ecg_signal)
    
    # Set default parameters if not provided
    if distance is None:
        distance = int(0.4 * fs)  # minimum 0.4s between peaks (max 150 bpm)
    if prominence is None:
        prominence = np.max(envelope) * 0.3  # 30% of max envelope
    
    # Find peaks in the envelope
    peaks, _ = find_peaks(envelope, prominence=prominence, distance=distance)
    
    return peaks, envelope

def find_s1_s2_in_scg(scg_signal, r_peaks, fs, window_before=0.15, window_after=0.25):
    s1_peaks = []
    s2_peaks = []
    
    # Get envelope of SCG for better peak detection
    scg_envelope = get_envelope(scg_signal)

    
    # Define search windows relative to R peak (in samples)
    s1_start = int(0.01 * fs)  # Start 50 ms after R peak
    s1_end = int(window_before * fs)    # End 150 ms after R peak
    s2_start = int(window_after * fs)  # Start 150 ms after R peak
    s2_end = int(0.6 * fs)    # End 600 ms after R peak
    
    for r_idx in r_peaks:
        # Skip peaks near the edges
        if r_idx + s2_end >= len(scg_envelope):
            continue
        
        # Find S1 (first major deflection after R)
        s1_window = scg_envelope[r_idx + s1_start:r_idx + s1_end]
        if len(s1_window) > 0:
            s1_idx = np.argmax(np.abs(s1_window)) + r_idx + s1_start
            s1_peaks.append(s1_idx)
        
        # Find S2 (second major deflection after R)
        s2_window = scg_envelope[r_idx + s2_start:r_idx + s2_end]
        if len(s2_window) > 0:
            s2_idx = np.argmax(np.abs(s2_window)) + r_idx + s2_start
            s2_peaks.append(s2_idx)
    
    return np.array(s1_peaks), np.array(s2_peaks)

def find_t_waves(ecg_signal, r_peaks, fs, t_start=0.12, t_end=0.4):
    t_peaks = []
    t_start_samples = int(t_start * fs)
    t_end_samples = int(t_end * fs)

    for r_idx in r_peaks:
        if r_idx + t_end_samples >= len(ecg_signal):
            continue

        t_window = ecg_signal[r_idx + t_start_samples:r_idx + t_end_samples]
        if len(t_window) == 0:
            continue

        t_idx = np.argmax(t_window) + r_idx + t_start_samples
        t_peaks.append(t_idx)

    return np.array(t_peaks)

def _window_max_index(signal, start_idx, end_idx):
    start_idx = max(0, start_idx)
    end_idx = min(len(signal), end_idx)
    if end_idx <= start_idx:
        return None
    return int(np.argmax(signal[start_idx:end_idx]) + start_idx)

def build_feature_table(timestamp, ecg_signal, scg_signal, resp_signal, r_peaks, fs,
                        s1_window=(0.01, 0.15), s2_window=(0.25, 0.6), t_window=(0.12, 0.4)):
    scg_envelope = get_envelope(scg_signal)
    ecg_abs = np.abs(ecg_signal)

    r_times = timestamp[r_peaks]
    rr_intervals = np.concatenate(([np.nan], np.diff(r_times)))

    rows = []
    for i, r_idx in enumerate(r_peaks):
        s1_idx = _window_max_index(scg_envelope, r_idx + int(s1_window[0] * fs), r_idx + int(s1_window[1] * fs))
        s2_idx = _window_max_index(scg_envelope, r_idx + int(s2_window[0] * fs), r_idx + int(s2_window[1] * fs))
        t_idx = _window_max_index(ecg_abs, r_idx + int(t_window[0] * fs), r_idx + int(t_window[1] * fs))

        r_time = r_times[i]
        rr = rr_intervals[i]
        r_amp = ecg_signal[r_idx]

        r_to_t = (t_idx - r_idx) / fs if t_idx is not None else np.nan
        r_to_s1 = (s1_idx - r_idx) / fs if s1_idx is not None else np.nan
        r_to_s2 = (s2_idx - r_idx) / fs if s2_idx is not None else np.nan

        s1_amp = scg_envelope[s1_idx] if s1_idx is not None else np.nan
        s2_amp = scg_envelope[s2_idx] if s2_idx is not None else np.nan
        s1_to_s2 = (s2_idx - s1_idx) / fs if (s1_idx is not None and s2_idx is not None) else np.nan

        if i + 1 < len(r_peaks):
            rr_start = r_idx
            rr_end = r_peaks[i + 1]
            rr_resp = resp_signal[rr_start:rr_end]
            resp0 = resp_signal[rr_start]
            resp_mean = float(np.mean(rr_resp)) if len(rr_resp) > 0 else np.nan
            resp_std = float(np.std(rr_resp, ddof=1)) if len(rr_resp) > 1 else np.nan
        else:
            resp_mean = np.nan
            resp_std = np.nan

        rows.append([ rr, r_amp, r_to_t, r_to_s1, s1_amp, r_to_s2, s2_amp, s1_to_s2, resp0, resp_mean, resp_std])

    columns = [ 'RR', 'R_amp', 'RtoT', 'RtoS1', 'S1_amp', 'RtoS2', 'S2_amp', 'S1toS2', 'Resp_time', 'Resp_mean', 'Resp_std']
    return pd.DataFrame(rows, columns=columns)





# Find R peaks, S1/S2, and T waves
r_peaks, envelope = find_rr_peaks(I, fs)
s1_peaks, s2_peaks = find_s1_s2_in_scg(SCG, r_peaks, fs, window_before=0.15, window_after=0.25)
t_peaks = find_t_waves(I, r_peaks, fs)

# Feature table as DataFrame
feature_df = build_feature_table(timestamp, I, SCG, RESP, r_peaks, fs)
feature_df.to_csv('signals_3/files/features_table.csv', index=False)


print(feature_df)

# Correlation heatmap of features
corr = feature_df.corr(numeric_only=True)
fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
im = ax_corr.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax_corr.set_title('Feature Correlation Heatmap')
ax_corr.set_xticks(np.arange(len(corr.columns)))
ax_corr.set_yticks(np.arange(len(corr.columns)))
ax_corr.set_xticklabels(corr.columns, rotation=45, ha='right')
ax_corr.set_yticklabels(corr.columns)
fig_corr.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
fig_corr.tight_layout()

# Feature statistics plot (excluding R_time)
feature_df_stats = feature_df
feature_stats = feature_df_stats.describe().T
fig_stats, axs_stats = plt.subplots(2, 2, figsize=(12, 8))

# Mean values
axs_stats[0, 0].barh(feature_stats.index, feature_stats['mean'], color='steelblue')
axs_stats[0, 0].set_xlabel('Mean Value')
axs_stats[0, 0].set_title('Feature Mean Values')
axs_stats[0, 0].grid(axis='x', alpha=0.3)

# Standard deviation
axs_stats[0, 1].barh(feature_stats.index, feature_stats['std'], color='coral')
axs_stats[0, 1].set_xlabel('Standard Deviation')
axs_stats[0, 1].set_title('Feature Standard Deviation')
axs_stats[0, 1].grid(axis='x', alpha=0.3)

# Min and Max
x_pos = np.arange(len(feature_stats.index))
width = 0.35
axs_stats[1, 0].barh(x_pos - width/2, feature_stats['min'], width, label='Min', color='lightgreen')
axs_stats[1, 0].barh(x_pos + width/2, feature_stats['max'], width, label='Max', color='lightcoral')
axs_stats[1, 0].set_yticks(x_pos)
axs_stats[1, 0].set_yticklabels(feature_stats.index)
axs_stats[1, 0].set_xlabel('Value')
axs_stats[1, 0].set_title('Feature Min/Max Values')
axs_stats[1, 0].legend()
axs_stats[1, 0].grid(axis='x', alpha=0.3)

# Coefficient of variation (std/mean)
cv = (feature_stats['std'] / feature_stats['mean'].abs()) * 100
axs_stats[1, 1].barh(feature_stats.index, cv, color='mediumpurple')
axs_stats[1, 1].set_xlabel('Coefficient of Variation (%)')
axs_stats[1, 1].set_title('Feature Variability (CV)')
axs_stats[1, 1].grid(axis='x', alpha=0.3)

fig_stats.tight_layout()

print("\nFeature Statistics:")
print(feature_stats)

# Visualize
T = 10  # seconds

fig,axs = plt.subplots(5,1, figsize=(10,8))

axs: list[plt.Axes]
xlimits = (timestamp[0], timestamp[0]+T)

axs[0].plot(timestamp, SCG, label='SCG')
axs[0].set_xlim(xlimits)
axs[0].set_title('Seismocardiography Signal')

axs[1].plot(timestamp, RESP, label='RESP', color='orange')
axs[1].set_xlim(xlimits)
axs[1].set_title('Respiration Signal')

# Lead I with envelope and detected peaks
axs[2].plot(timestamp, I, label='Lead I', color='green', alpha=0.7)
axs[2].plot(timestamp, envelope, label='Envelope', color='red', linewidth=2)
axs[2].plot(timestamp[r_peaks], I[r_peaks], 'rx', markersize=8, label='R Peaks')
axs[2].plot(timestamp[t_peaks], I[t_peaks], 'mo', markersize=6, label='T Waves')
axs[2].set_xlim(xlimits)
axs[2].set_title('Lead I ECG Signal with Envelope, R Peaks, and T Waves')
axs[2].legend()

# SCG with S1 and S2 marked
scg_envelope = get_envelope(SCG)
axs[3].plot(timestamp, SCG, label='SCG', alpha=0.7, color='purple')
axs[3].plot(timestamp, scg_envelope, label='SCG Envelope', color='black', linewidth=2)
axs[3].plot(timestamp[r_peaks], SCG[r_peaks], 'g^', markersize=8, label='R Peaks (ref)')
axs[3].plot(timestamp[s1_peaks], SCG[s1_peaks], 'bs', markersize=8, label='S1')
axs[3].plot(timestamp[s2_peaks], SCG[s2_peaks], 'r^', markersize=8, label='S2')
axs[3].set_xlim(xlimits)
axs[3].set_title('SCG with S1 and S2 Detection')
axs[3].legend()

# SCG envelope detail
axs[4].plot(timestamp, scg_envelope, label='SCG Envelope', color='black', linewidth=2)
axs[4].plot(timestamp[r_peaks], scg_envelope[r_peaks], 'g^', markersize=8, label='R Peaks')
axs[4].plot(timestamp[s1_peaks], scg_envelope[s1_peaks], 'bs', markersize=8, label='S1')
axs[4].plot(timestamp[s2_peaks], scg_envelope[s2_peaks], 'r^', markersize=8, label='S2')
axs[4].set_xlim(xlimits)
axs[4].set_title('SCG Envelope Detail with S1/S2')
axs[4].legend()

axs[-1].set_xlabel('Time (s)')

plt.tight_layout()

plt.show()
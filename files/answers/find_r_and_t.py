import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from datetime import datetime, UTC


def envelope(x, window=50):
    """
    Beregn en envelope for et signal x.
    """
    # 1. Fjern gennemsnittet
    mean_x = np.mean(x)
    s0 = x - mean_x

    # 2. Rectify (absolut værdi)
    r = np.abs(s0)

    # 3. Moving Maximum
    mov_max = np.zeros_like(r)
    half_w = window // 2
    for i in range(len(r)):
        start = max(i - half_w, 0)
        end = min(i + half_w + 1, len(r))
        mov_max[i] = np.max(r[start:end])

    # 4. Tilføj gennemsnittet tilbage
    env = np.array(mov_max + mean_x)
    return env


def peak_detector(data, thr: float):
    value_record = 0
    time_record = 0
    indecies: list[float] = []
    for i, value in enumerate(data):
        if value > thr:
            if value > value_record:
                value_record = value
                time_record = i
        else:
            if value_record:
                indecies.append(time_record)
                time_record = 0
                value_record = 0

    return np.array(indecies)



filename = "/home/nurrix/Documents/ST2-AnvendtProgrammering/signals_3/files/ECGPCG.csv"
filename_out = "/home/nurrix/Documents/ST2-AnvendtProgrammering/signals_3/files/features_ECGPCG.csv"

time, ecg, pcg = np.genfromtxt(filename,delimiter=",", skip_header=1, unpack=True)
fs = round(1/np.diff(time).mean())
fc = [1,150]
b, a = butter(2, fc, fs=fs, btype='band',) # type: ignore
filtered_ecg = filtfilt(b, a, ecg)
env_ecg = envelope(filtered_ecg, window=round(fs/20))


R_indecies = peak_detector(env_ecg, thr=env_ecg.mean()+env_ecg.std())
R_time = time[R_indecies]
R_amp = env_ecg[R_indecies]
RR = np.diff(R_time)

T = []
RT = []
for i in range(len(RR)):
    R_loc = R_indecies[i]
    win = [R_loc+ round(fs*0.1),R_loc+round(fs*0.6)]
    my_window = filtered_ecg[win[0]:win[1]]
    T_ind = peak_detector(my_window,float(my_window.mean()+ my_window.std()))
    if len(T_ind)>0:
        T_loc = T_ind[0]+win[0]
        T.append(T_loc)
        RT.append((T_loc-R_loc)/fs)
T = np.array(T)
T_time = time[T]
T_amp = filtered_ecg[T]
RR = np.array(RR)
RT = np.array(RT)

# Save to file with meta data
metadata = {
    "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "source_file": filename,
    "sampling_frequency_hz": fs,
    "filter_type": "butterworth_bandpass",
    "filter_order": 2,
    "filter_low_hz": fc[0],
    "filter_high_hz": fc[1],
    "envelope_window_samples": round(fs / 20),
    "r_threshold": "mean(signal)+std(signal)",
    "t_search_window_s": "[0.1, 0.6] after R",
    "rr_count": len(RR),
    "rt_count": len(RT),
}

n_rows = max(len(RR), len(RT))
rr_pad = np.full(n_rows, np.nan)
rt_pad = np.full(n_rows, np.nan)
rr_pad[:len(RR)] = RR
rt_pad[:len(RT)] = RT
beat_index = np.arange(n_rows)

table = np.column_stack((beat_index, rr_pad, rt_pad))

metadata_header = "\n".join([f"# {key},{value}" for key, value in metadata.items()])
full_header = f"{metadata_header}\n# beat_index,RR_s,RT_s"

np.savetxt(
    filename_out,
    table,
    delimiter=",",
    fmt=["%d", "%.6f", "%.6f"],
    header=full_header,
    comments="",
)


plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.plot(time,ecg, label="raw",color="black", alpha= 0.5)
plt.plot(time,filtered_ecg, label="filtered",color="black")
plt.plot(time, env_ecg, label="envelope", linewidth=2)
plt.scatter(R_time, R_amp, label="R", color="blue")
plt.scatter(T_time,T_amp, label="T", color="red")
plt.legend()
plt.xlabel("Tid [s]")
plt.ylabel("Amplitude")
plt.title("EKG")

plt.subplot(3,1,2)
plt.hist(RR,25,range=(0,1), label="RR interval")
plt.hist(RT,25,range=(0,1), label="RT interval")
plt.title("Histograms")
plt.xlabel("interval [s]")
plt.xlim((0,1))
plt.legend()

plt.subplot(3,1,3)
plt.boxplot([RR,RT], vert=False, showmeans=True)
plt.yticks([1,2],["RR","RT"])
plt.xlabel("interval [s]")
plt.xlim((0,1))



plt.show()
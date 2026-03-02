import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

# Load file with NumPy and ignore missing values
RR_b, R_amp_b = np.genfromtxt('/home/nurrix/Documents/ST2-AnvendtProgrammering/signals_3/files/features_table_b001.csv', delimiter=',', skip_header=1, dtype=float, usecols=(0, 1), unpack=True)
# Header is RR,R_amp,RtoS2,S1toS2,Resp_amp,Resp_amp_RR_Rmean,Resp__amp_RR_std
mask = ~np.isnan(RR_b) & ~np.isnan(R_amp_b)
RR_b = RR_b[mask]
R_amp_b = R_amp_b[mask]

# Select features
RR_m, R_amp_m = np.genfromtxt('/home/nurrix/Documents/ST2-AnvendtProgrammering/signals_3/files/features_table_m001.csv', delimiter=',', skip_header=1, dtype=float, usecols=(0, 1), unpack=True)
# Header is RR,R_amp,RtoS2,S1toS2,Resp_amp,Resp_amp_RR_Rmean,Resp__amp_RR_std
mask = ~np.isnan(RR_m) & ~np.isnan(R_amp_m)
RR_m = RR_m[mask]
R_amp_m = R_amp_m[mask] 

labels = ["music", "baseline"]
RR_all = [RR_m, RR_b]
R_amp_all = [R_amp_m, R_amp_b]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Create plots for baseline and music with their own colors
for label, rr, r_amp in zip(labels, RR_all, R_amp_all):
    assert isinstance(rr, np.ndarray) and isinstance(r_amp, np.ndarray) , "Inputs must be numpy arrays, was: {}, {}".format(type(rr), type(r_amp))
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(rr, r_amp)
    x = np.array([min(rr), max(rr)])
    y = intercept + slope * x

    # visualization
    # scatter with regression line
    ax1.scatter(rr, r_amp, alpha=0.3*1000/(1000+len(rr)), edgecolor='w')
    ax1.plot(x, y, linewidth=2, label=f'{label} fit: y={slope:.2f}x+{intercept:.2f}, $R^2$={r_value**2:.2f}')  # type: ignore # Add the regression line   

    # # residuals
    ax2.scatter(rr, r_amp - (slope * rr + intercept), alpha=0.3*1000/(1000+len(rr)))

# Finalize plots
ax1.set_xlabel("RR Intervals")
ax1.set_ylabel("R Amplitude (a.u.)")
ax1.set_title("RR Intervals vs R Amplitude with Linear Regression")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.set_xlabel("RR Intervals")
ax2.set_ylabel("Residuals of R Amplitude")
ax2.set_title("Residuals of Linear Regression")
ax2.grid(True, alpha=0.3)


plt.show()
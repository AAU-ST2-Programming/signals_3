import matplotlib.pyplot as plt
import numpy as np


filename = "C:/Users/Martin/Documents/ST2_AP_all_lectures/signals_3/files/peaks_ECGPCG2.csv"


R,S1,S2 = np.genfromtxt(filename,delimiter=",", skip_header=1, unpack=True)
RR = np.diff(R)
RR_mean = np.mean(RR)
RR_std = np.std(RR)
plt.scatter([1]*len(RR),RR, label="R")
plt.errorbar(1,RR_mean,2*RR_std, label="errorbar")
plt.show()
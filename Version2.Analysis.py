# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21  19:57:32 2026

@author: Ldak
"""

# ============================================================
# PHY324 Data Analysis Project
# Written in the style of the provided StarterCode.py
# ============================================================

import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

# ------------------------------------------------------------
# Plot formatting (same idea as starter code)
# ------------------------------------------------------------

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 15}
rc('font', **font)

# ------------------------------------------------------------
# Fit functions (same as starter code)
# ------------------------------------------------------------

def myGauss(x, A, mean, width, base):
    """Gaussian peak + flat background"""
    return A * np.exp(-(x - mean)**2 / (2 * width**2)) + base

def pulse_shape(t_rise, t_fall):
    """Normalized pulse template"""
    xx = np.linspace(0, 4095, 4096)
    yy = -(np.exp(-(xx - 1000)/t_rise) - np.exp(-(xx - 1000)/t_fall))
    yy[:1000] = 0
    yy /= np.max(yy)
    return yy

def fit_pulse(x, A):
    """Used by curve_fit for pulse fitting"""
    _pulse_template = pulse_shape(20, 80)
    xx = np.linspace(0, 4095, 4096)
    return A * np.interp(x, xx, _pulse_template)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

with open("calibration.pkl", "rb") as file:
    calibration_data = pickle.load(file)

with open("noise.pkl", "rb") as file:
    noise_data = pickle.load(file)

with open("signal.pkl", "rb") as file:
    signal_data = pickle.load(file)

# ------------------------------------------------------------
# Plot 1 — Look at example pulses (overlapped)
# ------------------------------------------------------------

pulse_template = pulse_shape(20, 80)

plt.plot(pulse_template / 2000, color='r', label='Pulse Template')

for itrace in range(5):
    plt.plot(calibration_data['evt_%i' % itrace], alpha=0.4)

plt.xlabel('Sample Index')
plt.ylabel('Readout Voltage (V)')
plt.title('Example Calibration Pulses')
plt.legend()
plt.show()

# ------------------------------------------------------------
# Define energy estimator arrays
# ------------------------------------------------------------
amp0 = np.zeros(1000)
amp1 = np.zeros(1000)   # max - min
amp2 = np.zeros(1000)   # max - baseline
area1 = np.zeros(1000)  # sum full trace
area2 = np.zeros(1000)  # sum full trace (baseline sub)
area3 = np.zeros(1000)  # sum pulse only
pulse_fit = np.zeros(1000)  # template fit amplitude

# ------------------------------------------------------------
# Compute estimators for calibration data
# ------------------------------------------------------------

for ievt in range(1000):

    current_data = calibration_data['evt_%i' % ievt]

    baseline = np.mean(current_data[:1000])

    # 1. max - min
    amp1[ievt] = np.max(current_data) - np.min(current_data)

    # 2. max - baseline
    amp2[ievt] = np.max(current_data) - baseline

    # 3. sum full trace
    area1[ievt] = np.sum(current_data)

    # 4. sum full trace (baseline subtracted)
    area2[ievt] = np.sum(current_data - baseline)

    # 5. sum pulse only
    area3[ievt] = np.sum(current_data[1000:] - baseline)

    # 6. pulse template fit
    popt, _ = curve_fit(fit_pulse,
                        np.linspace(0, 4095, 4096),
                        current_data)
    pulse_fit[ievt] = popt[0]

# Convert from V to mV
amp1 *= 1000
amp2 *= 1000
area1 *= 1000
area2 *= 1000
area3 *= 1000
pulse_fit *= 1000

for ievt1 in range(1000):

    current_data1 = noise_data['evt_%i' % ievt1]

    baseline = np.mean(current_data1[:1000])

    # 1. max - min
    amp0[ievt1] = np.max(current_data1) - np.min(current_data1)
    amp0 *= 1000
# ------------------------------------------------------------
# Put estimators in a list (simple, explicit)
# ------------------------------------------------------------

estimator_names = [
    "Max - Min",
    "Max - Baseline",
    "Sum Full Trace",
    "Sum Full (Baseline Sub)",
    "Sum Pulse Only",
    "Template Fit"
]

estimators = [amp1, amp2, area1, area2, area3, pulse_fit]

calibration_factors = []
energy_resolutions = []


# ------------------------------------------------------------
# Calibration plots (Plots 2–7)
# ------------------------------------------------------------

for i in range(len(estimators)):
    data = estimators[i]
    real_data = []
    for j in range(len(data)):
        if data[j] > 0.1 and data[j] < 0.4:
            real_data.append(data[j])

    num_bins = 50
    n, bin_edges = np.histogram(real_data, bins=num_bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    

    sig = np.sqrt(n)

    popt, pcov = curve_fit(
        myGauss,
        bin_centers,
        n,
        sigma=sig,
        p0=(np.max(n), np.mean(real_data), np.std(real_data), 5),
        absolute_sigma=True
    )

    
    fit = myGauss(bin_centers, *popt)

    chisq = np.sum(((n - fit) / sig)**2)
    dof = num_bins - len(popt)
    chi_prob = 1 - chi2.cdf(chisq, dof)

    # calibration factor
    cal_factor = 10.0 / popt[1]
    calibration_factors.append(cal_factor)
    energy_resolutions.append(popt[2] * cal_factor)

    # ---- Plot ----
    plt.hist(real_data, bins=num_bins, histtype='step', color='k')
    plt.plot(bin_centers, fit, color='r')

    plt.xlabel('Energy Estimator (mV)')
    plt.ylabel('Counts (mV)')
    plt.title('Calibration: ' + estimator_names[i])

    plt.text(0.05, 0.85,
             r'$\mu$ = %3.3f mV' % popt[1],
             transform=plt.gca().transAxes)
    plt.text(0.05, 0.78,
             r'$\sigma$ = %3.3f mV' % popt[2],
             transform=plt.gca().transAxes)
    plt.text(0.05, 0.71,
             r'$\chi^2$ prob = %1.2f' % chi_prob,
             transform=plt.gca().transAxes)

    plt.show()

# ------------------------------------------------------------
# Choose best estimator
# ------------------------------------------------------------

best_index = np.argmin(energy_resolutions)
best_name = estimator_names[best_index]
best_factor = calibration_factors[best_index]

print("Best estimator:", best_name)
print("Energy resolution (keV):", energy_resolutions[best_index])

# ------------------------------------------------------------
# Plot 8 — Post-calibration spectrum
# ------------------------------------------------------------

energy_calib = estimators[best_index] * best_factor

plt.hist(energy_calib, bins=50, histtype='step', color='k')
plt.xlabel('Energy (keV)')
plt.ylabel('Counts')
plt.title('Post-Calibration Spectrum (' + best_name + ')')
plt.show()

# ------------------------------------------------------------
# Apply to signal data (Plot 9)
# ------------------------------------------------------------

signal_amp2 = np.zeros(1000)

for ievt in range(1000):
    trace = signal_data['evt_%i' % ievt]
    baseline = np.mean(trace[:1000])
    signal_amp2[ievt] = np.max(trace) - baseline

signal_amp2 *= 1000
signal_energy = signal_amp2 * best_factor

n, bins = np.histogram(signal_energy, bins=60)
centers = 0.5 * (bins[1:] + bins[:-1])

popt, _ = curve_fit(myGauss,
                    centers,
                    n,
                    p0=(np.max(n), 5, 1, 10))

plt.hist(signal_energy, bins=60, histtype='step', color='k')
plt.plot(centers, myGauss(centers, *popt), color='r')

plt.xlabel('Energy (keV)')
plt.ylabel('Counts')
plt.title('Signal Energy Spectrum')
plt.show()

# ------------------------------------------------------------
# Single pulse example
# ------------------------------------------------------------

evt_id = 2  # recommendation from project document(copy)

trace = calibration_data['evt_%i' % evt_id]
time = np.linspace(0, 4095/1e3, 4096)  # ms

plt.plot(time, trace, color='k', lw=1)

plt.xlabel('Time (ms)')
plt.ylabel('Readout Voltage (V)')
plt.title('Single Example Pulse (Calibration Data)')
plt.legend()
plt.show()


# ------------------------------------------------------------
# Plots 10–15 — Calibrated energy histograms (all estimators)
# ------------------------------------------------------------

for i in range(len(estimators)):

    # Apply calibration to this estimator
    energy_data = estimators[i] * calibration_factors[i]

    # Restrict to effective energy range around 10 keV
    energy_eff = []
    for val in energy_data:
        if val > 0.2 and val < 0.5:
            energy_eff.append(val)

    num_bins = 50
    n, bin_edges = np.histogram(energy_eff, bins=num_bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    sig = np.sqrt(n)

    # Gaussian + background fit
    popt, pcov = curve_fit(
        myGauss,
        bin_centers,
        n,
        sigma=sig,
        p0=(np.max(n), 10.0, 1.0, 5),
        absolute_sigma=True
    )

    fit = myGauss(bin_centers, *popt)

    chisq = np.sum(((n - fit) / sig)**2)
    dof = num_bins - len(popt)
    chi_prob = 1 - chi2.cdf(chisq, dof)

    # ---- Plot ----
    plt.hist(energy_eff, bins=num_bins, histtype='step', color='k')
    plt.plot(bin_centers, fit, color='r')

    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts (keV)')
    plt.title('Calibrated Energy Spectrum: ' + estimator_names[i])

    plt.text(0.05, 0.85,
             r'$\mu$ = %3.2f keV' % popt[1],
             transform=plt.gca().transAxes)
    plt.text(0.05, 0.78,
             r'$\sigma$ = %3.2f keV' % popt[2],
             transform=plt.gca().transAxes)
    plt.text(0.05, 0.71,
             r'$\chi^2$ prob = %1.2f' % chi_prob,
             transform=plt.gca().transAxes)

    plt.show()
    
    # ------------------------------------------------------------
# Plot 9 — Signal spectrum with improved functional form
# ------------------------------------------------------------







signal_amp = np.zeros(1000)

for ievt in range(1000):
    trace = signal_data['evt_%i' % ievt]
    baseline = np.mean(trace[:1000])
    signal_amp[ievt] = np.max(trace) - baseline

signal_amp *= 1000
signal_energy = signal_amp * best_factor

# Histogram
num_bins = 60
n, bins = np.histogram(signal_energy, bins=num_bins)
centers = 0.5 * (bins[1:] + bins[:-1])

# --- χ² validity: remove empty bins ---
mask = n > 0
n_fit = n[mask]
centers_fit = centers[mask]
sig = np.sqrt(n_fit)

# Double Gaussian + background
def doubleGauss(x, A1, mu1, sig1, A2, mu2, sig2, base):
    return (A1 * np.exp(-(x - mu1)**2 / (2 * sig1**2)) +
            A2 * np.exp(-(x - mu2)**2 / (2 * sig2**2)) +
            base)

p0 = (np.max(n_fit), 5.0, 0.8,
      np.max(n_fit)/2, 9.0, 1.0,
      5)


popt, pcov = curve_fit(doubleGauss,
                       centers_fit,
                       n_fit,
                       sigma=sig,
                       p0=p0,
                       absolute_sigma=True)

fit = doubleGauss(centers_fit, *popt)

chisq1 = np.sum(((n_fit - fit) / sig)**2)
chisq1_red = chisq1 / (len(n_fit) - 7) 
dof = len(n_fit) - len(popt)
chi_prob = 1 - chi2.cdf(chisq1_red, dof)

# ---- Plot ----
plt.hist(signal_energy, bins=num_bins, histtype='step', color='k')
plt.plot(centers_fit, fit, color='r')

plt.xlabel('Energy (keV)')
plt.ylabel('Counts (keV)')
plt.title('Signal Energy Spectrum (Double Gaussian Model)')

plt.text(0.05, 0.85,
         r'$\chi^2$ reduced = %1.2f' % chisq1_red,
         transform=plt.gca().transAxes)

plt.show()



# ------------------------------------------------------------
# Plot 9 — Signal spectrum (χ²-valid single-component model)
# ------------------------------------------------------------


def signal_model(x, A, mu, sig, base):
    return A * np.exp(-(x - mu)**2 / (2 * sig**2)) + base



# ------------------------------------------------------------
# Plot 9 — Signal spectrum (robust χ² fit)
# ------------------------------------------------------------

signal_amp = np.zeros(1000)

for ievt in range(1000):
    trace = signal_data['evt_%i' % ievt]
    baseline = np.mean(trace[:1000])
    signal_amp[ievt] = np.max(trace) - baseline

signal_amp *= 1000
signal_energy = signal_amp * best_factor

num_bins = 60
n, bins = np.histogram(signal_energy, bins=num_bins)
centers = 0.5 * (bins[1:] + bins[:-1])

# Mask empty bins
mask = n > 0
n_fit = n[mask]
centers_fit = centers[mask]
sig = np.sqrt(n_fit)

# --- BETTER initial guesses ---
A0 = np.max(n_fit)
mu0 = centers_fit[np.argmax(n_fit)]   # peak position
sig0 = 1.0
base0 = np.median(n_fit)

p0 = (A0, mu0, sig0, base0)

# --- Physical bounds ---
bounds = (
    (0, 0, 0.1, 0),      # lower bounds
    (np.inf, 20, 5, np.inf)  # upper bounds
)

popt, pcov = curve_fit(
    signal_model,
    centers_fit,
    n_fit,
    sigma=sig,
    p0=p0,
    bounds=bounds,
    absolute_sigma=True,
    maxfev=10000
)

fit = signal_model(centers_fit, *popt)

chisq = np.sum(((n_fit - fit) / sig)**2)
dof = len(n_fit) - len(popt)
chi_prob = 1 - chi2.cdf(chisq, dof)

# ---- Plot ----
plt.hist(signal_energy, bins=num_bins, histtype='step', color='k')
plt.plot(centers_fit, fit, color='r')

plt.xlabel('Energy (keV)')
plt.ylabel('Counts')
plt.title('Signal Energy Spectrum (Single Gaussian Model)')

plt.text(0.05, 0.85,
         r'$\chi^2$ prob = %1.2f' % chi_prob,
         transform=plt.gca().transAxes)

plt.show()
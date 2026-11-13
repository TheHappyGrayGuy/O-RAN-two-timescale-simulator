# Setup sanity report

PRB bandwidth 1.00 MHz, noise/PRB -114.0 dBm, P/PRB 26.0 dBm

## Per-PRB SNR percentiles (5/50/95) [dB]

- log_distance: 13.7 / 26.1 / 40.5
- 3gpp_umi: 9.3 / 26.8 / 54.0

## KPIs with reference policy (max-gain association, round-robin PRB, equal power)

| unit | U | scenario | D eMBB / URLLC / mMTC [ms] (median) | phi_bar | QoS sat. | feasible | max CPU util | thr [Mb/s] |
|---|---|---|---|---|---|---|---|---|
| byte | 5 | all_edge | 0.61 / 0.27 / 0.41 | 1.977 | 1.00 | 1.00 | 0.10 | 704.5 |
| byte | 5 | cu_regional_du_edge | 0.70 / 0.33 / 0.48 | 0.948 | 1.00 | 1.00 | 0.10 | 704.5 |
| byte | 5 | all_regional | 0.62 / 0.27 / 0.40 | 0.021 | 1.00 | 0.80 | 0.01 | 704.5 |
| byte | 10 | all_edge | 0.69 / 0.38 / 0.45 | 3.845 | 1.00 | 1.00 | 0.20 | 856.5 |
| byte | 10 | cu_regional_du_edge | 0.78 / 0.43 / 0.52 | 1.843 | 1.00 | 1.00 | 0.19 | 856.5 |
| byte | 10 | all_regional | 0.66 / 0.34 / 0.39 | 0.042 | 1.00 | 0.89 | 0.02 | 856.5 |
| byte | 15 | all_edge | 0.77 / 0.44 / 0.45 | 4.914 | 1.00 | 1.00 | 0.25 | 867.4 |
| byte | 15 | cu_regional_du_edge | 0.85 / 0.52 / 0.51 | 2.355 | 1.00 | 1.00 | 0.24 | 867.4 |
| byte | 15 | all_regional | 0.70 / 0.39 / 0.38 | 0.053 | 1.00 | 0.90 | 0.03 | 867.4 |
| byte | 20 | all_edge | 0.93 / 0.56 / 0.53 | 6.791 | 1.00 | 1.00 | 0.35 | 887.9 |
| byte | 20 | cu_regional_du_edge | 1.00 / 0.62 / 0.59 | 3.255 | 1.00 | 1.00 | 0.33 | 887.9 |
| byte | 20 | all_regional | 0.78 / 0.41 / 0.38 | 0.075 | 1.00 | 0.89 | 0.04 | 887.9 |
| kbit | 5 | all_edge | inf / inf / inf | 247.176 | 0.00 | 0.00 | 12.69 | 704.5 |
| kbit | 5 | cu_regional_du_edge | inf / inf / inf | 118.464 | 0.00 | 0.00 | 12.16 | 704.5 |
| kbit | 5 | all_regional | inf / inf / inf | 2.672 | 0.00 | 0.00 | 1.34 | 704.5 |
| kbit | 10 | all_edge | inf / inf / inf | 480.589 | 0.00 | 0.00 | 24.38 | 856.5 |
| kbit | 10 | cu_regional_du_edge | inf / inf / inf | 230.333 | 0.00 | 0.00 | 23.36 | 856.5 |
| kbit | 10 | all_regional | inf / inf / inf | 5.261 | 0.00 | 0.00 | 2.63 | 856.5 |
| kbit | 15 | all_edge | inf / inf / inf | 614.221 | 0.00 | 0.00 | 31.36 | 867.4 |
| kbit | 15 | cu_regional_du_edge | inf / inf / inf | 294.378 | 0.00 | 0.00 | 30.04 | 867.4 |
| kbit | 15 | all_regional | inf / inf / inf | 6.681 | 0.00 | 0.00 | 3.34 | 867.4 |
| kbit | 20 | all_edge | inf / inf / inf | 848.872 | 0.00 | 0.00 | 43.30 | 887.9 |
| kbit | 20 | cu_regional_du_edge | inf / inf / inf | 406.841 | 0.00 | 0.00 | 41.48 | 887.9 |
| kbit | 20 | all_regional | inf / inf / inf | 9.353 | 0.00 | 0.00 | 4.68 | 887.9 |
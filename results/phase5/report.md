# Phase 5: optimal solutions

## Small instances (B=2, N=2, U=3, M=3): certified global optimum by full enumeration

| seed | exact optimum | certified | scalable solver | difference |
|---|---|---|---|---|
| 0 | 0.095998 | yes | 0.095998 | +0.0000% |
| 1 | 0.032609 | yes | 0.032609 | +0.0000% |
| 2 | 0.065357 | no* | 0.065357 | +0.0000% |
| 3 | 0.030029 | yes | 0.030029 | +0.0000% |
| 4 | 0.033695 | yes | 0.033695 | +0.0000% |
| 5 | 0.094000 | yes | 0.094000 | +0.0000% |
| 6 | 0.031297 | yes | 0.031297 | +0.0000% |
| 7 | 0.031597 | yes | 0.031597 | +0.0000% |
| 8 | 0.067987 | no* | 0.067987 | +0.0000% |
| 9 | 0.068042 | no* | 0.068042 | +0.0000% |
| 10 | 0.097251 | yes | 0.097251 | +0.0000% |
| 11 | 0.093301 | yes | 0.093301 | +0.0000% |

Scalable solver reaches the exact optimum in **12/12** instances.
*no = the relaxation bound is not attained, because the fronthaul budget binds once fronthaul transmission time is included; the reported optimum is then exact only for V-optimal radio solutions.

## ch6 scale (B=4, N=6, M=25): optimum vs baselines (mean over seeds)

| U | random | all_edge | cu_regional_du_edge | all_regional | static_best | placement_search | **optimum** | gain vs static_best | cond. gap | global gap | solve time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 200.0977† | 0.1806 | 0.1186 | 200.0391† | 0.0561 | 0.0355 | **0.0310** | 33.2% | 0.001% | 4.5% | 350s |
| 10 | 400.1566† | 0.3416 | 0.2032 | 0.0432 | 0.0432 | 0.0413 | **0.0344** | 19.5% | 0.351% | 7.1% | 2823s |
| 15 | 200.1397† | 0.4305 | 0.2502 | 0.0487 | 0.0487 | 0.0479 | **0.0384** | 20.8% | 1.512% | 10.0% | 11967s |
| 20 | 200600.1650† | 0.5923 | 0.3350 | 0.0525 | 0.0525 | 0.0520 | **0.0426** | 18.5% | 0.257% | 11.3% | 108861s |

† at least one seed infeasible (penalised value includes 1e3 per violated constraint).
cond. gap: certified bound over all PRB / power / placement choices for the association found.
global gap: certified bound with association relaxed as well.

## Per-slice delay at the optimum [ms] and edge share

| U | eMBB | URLLC | mMTC | edge share | QoS sat. |
|---|---|---|---|---|---|
| 5 | 0.641 | 0.084 | 0.030 | 0.65 | 1.00 |
| 10 | 0.688 | 0.090 | 0.089 | 0.45 | 1.00 |
| 15 | 0.719 | 0.105 | 0.077 | 0.35 | 1.00 |
| 20 | 0.783 | 0.118 | 0.090 | 0.35 | 1.00 |
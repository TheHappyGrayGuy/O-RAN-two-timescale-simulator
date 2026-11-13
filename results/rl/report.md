# Learned policies vs the Phase-5 optimum

Objective on the Phase-5 snapshot (lower is better); gap = objective / optimum - 1, mean over seeds.

| U | optimum | static best | MARL (2 x TD3) (gap) | TD3 + convex (gap) | SAC + convex (gap) | single SAC (gap) |
|---|---|---|---|---|---|---|
| 5 | 0.0308 | 0.0385 | 0.0371 (+20.2%) | 0.0371 (+20.0%) | 0.0371 (+20.0%) | 0.0372 (+20.3%) |
| 10 | 0.0344 | 0.0424 | 0.0403 (+17.0%) | 0.0397 (+15.3%) | 0.0397 (+15.3%) | 0.0406 (+17.7%) |
| 15 | 0.0383 | 0.0476 | 0.0434 (+13.1%) | 0.0426 (+10.9%) | 0.0426 (+10.9%) | 0.0450 (+17.3%) |
| 20 | 0.0423 | 0.0510 | 0.0479 (+13.3%) | 0.0461 (+8.8%) | 0.0461 (+8.8%) | 0.0490 (+15.8%) |

Infeasible final policies count as inf. 'best' = best feasible evaluation during training.

# Why the association/placement agent did not learn

Grid run in `results/rl/`: 4 methods × U ∈ {5, 10, 15, 20} × seeds {0, 1} × 500 episodes, with
`--prior 2 --temp 1 --act-pen 0.1 --large-hidden 256 256 --large-updates 2`.
The two hybrid methods (TD3 + convex, SAC + convex) gave identical, flat results, and no method
improved its association or CU/DU placement. The scripts in `scripts/diagnostics/` test why.
Gaps are relative to the Phase 5 optimum of the same (U, seed). All tests use the fast PRB/power solver of the hybrids.

| File | Script | Question |
|---|---|---|
| `large_U*_s*.txt` | `diag_large.py U seed` | Can the action change the decision? How good are random decisions and single moves? |
| `optimal_assoc.txt` | `diag_opt.py` | Is the remaining gap in the PRB/power part or in association/placement? |
| `joint_moves.txt` | `diag_joint.py` | Do greedy local moves, with CU and DU moved together, reach the optimum? |
| `placement_only.txt` | `diag_place.py` | With the optimal placement, can a simple rule find the association? |

## Findings

1. **The prior β = 2 locks the decision (parameter error).** Actions lie in [−1, 1], so the static option always
   scores at least 1 and every other option at most 1. In 2000 random actions, the decision never changed (test A).
   Every method therefore ran with the best static association and placement.
2. **The full-decision action cannot be explored (design problem).** With β = 0, 0 of 100 random decisions beat
   static, and the median is 2.8–4.6 times worse (test B). With β = 1 and small noise, under 1 entry changes and
   at most 9% of tries improve (test C).
3. **The whole gap is association/placement.** Given the optimum's association and placement, the fast PRB/power
   solver is within 0.1–0.6% of the optimum, against 4.7–34% with the static choice (`optimal_assoc.txt`).
4. **Single local moves get stuck.** On seed 1 only 1% of single moves improve at U = 10 (test D). Moving a CU and
   its DU together helps (U = 5: 34% → 0.1%), but greedy search still stops at 10–17% at U = 10–20 (`joint_moves.txt`).
5. **Placement is the hard part.** With the optimal placement, user-by-user improvement over each user's 3
   strongest RUs reaches 0.1–0.2% in 5 of 6 cases (3.3% in the sixth) in under 7 s. The strongest-signal rule
   alone is 11–116% worse because it ignores load (`placement_only.txt`).

## Proposed redesign

- The RL agent chooses only the CU/DU placement (one node pair per RU).
- The association is computed for that placement by user-by-user improvement, warm-started from the previous association.
- PRB/power stays as it is (convex solver in the hybrids, TD3 agent in MARL).
- β is set below 1, or to 0, so the agent can leave the static choice.

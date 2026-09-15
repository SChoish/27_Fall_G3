# D2D dynamic co-design pilot results

Execution status: **gate_failed**. Source and assumptions were frozen before characterization.

All figures below use the new dynamic environment; no earlier sizing/SAC/IQL result is counted.

| Controller | Passing episodes | Mean / worst violation | Longest failure | TX / control energy (pJ/episode) | Measurements/episode | Mean / worst downtime | Throughput (Gb/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0/8 | 55.27% / 82.81% | 50 | 1750.23 / 0.00 | 0.00 | 0.00% / 0.00% | 1.789 |
| periodic | 5/8 | 4.30% / 12.50% | 3 | 1750.23 / 614.71 | 16.00 | 20.26% / 20.44% | 3.053 |
| event | 3/8 | 10.16% / 28.12% | 17 | 1750.23 / 441.96 | 11.50 | 15.46% / 17.22% | 3.040 |
| bang_bang | 0/8 | 0.59% / 1.56% | 1 | 1750.23 / 2193.82 | 64.00 | 50.13% / 50.19% | 1.983 |

## Interpretation and gate diagnosis

No classical method passed all eight development episodes under both fixed constraints (quality violation ≤5%, downtime ≤25%). The gate therefore stopped PPO training, circuit search and final testing. No thresholds were relaxed and no extra learning was added.
The post-hoc full-information dynamic program found minimum failure-block counts [0, 0, 0, 0, 0, 0, 0, 0] across the same eight trajectories while respecting ±4-code movement and phase bounds. This uses future drift/full phase labels and is **not a deployable controller or a fair energy baseline**.
Every-block paid sensing tracks quality, but its pilot overhead violates downtime. Periodic sensing respects downtime, but loses quality on some trajectories. This identifies a sensing/cadence tradeoff for the implemented baselines; it does not prove that all causal controllers are infeasible.
The first next step is a separately preregistered development study of causal prediction from paid measurement history and a physically supported monitor implementation. The present experiment does not establish an RL or co-design advantage.

## Executed versus unexecuted

- Executed: fresh nominal transistor characterization, four classical controllers, paired development trajectories, integrity audit, bounded privileged diagnosis and fixed-trajectory cost sensitivity.
- Implemented but not physically executed: three-seed PPO pilot, twelve-design alternating search, all four joint comparisons, final held-out inputs/drift, RNC option. PPO and design integration are checked separately with synthetic data, which are software checks, not physical performance.
- The locked test packets and drift seeds were not evaluated; there is no selected return design.

## Cost ledger

```json
{
  "spice_transients": 4,
  "cache_hits": 0,
  "design_characterizations": 1,
  "evaluation_blocks": 2048,
  "evaluation_calibration_measurements": 732.0,
  "evaluation_phase_bank_lookups": 4244,
  "training_blocks": 0,
  "training_gradient_steps": 0,
  "training_phase_bank_lookups": 0,
  "training_calibration_measurements": 0,
  "diagnostic_phase_bank_lookups": 9216,
  "diagnostic_spice_transients": 0,
  "numerical_grid_validation_lookups": 4
}
```

SPICE runs characterize fixed waveforms. A phase-bank lookup is a two-lane numerical query, not SPICE. Each data block uses one lookup and each paid monitor request uses three additional lookups. Raw interpolation validation and privileged diagnostics are recorded separately. Calibration energy/time describes modeled hardware operation, not machine training power.

## Model and sensitivity limits

PTM predictive 45 nm bulk devices at fixed 90 nm gate length, lumped coupled RLC channels, 2 Gb/s/lane and behavioral RX. Voltage/temperature are fixed within each waveform; correlated timing displacement is quasi-static, not an actual PVT transient. Phase interpolation is bounded and has no wrap. The finite 64-bit margin/error check is not a demonstrated BER specification.
The ideal noiseless early/center/late amplitude monitor uses one known pilot packet with 27 parallel probes per lane. Its readout/aggregation energy is an assumed monitor budget (2 fJ per sample), not characterized silicon power; exact analog margins are stronger information than a one-bit early/late detector. Pilot TX and phase updates are charged separately. A realistic monitor may cost more and reveal less.
Sensitivity covers monitor energy ×0.1/1/10 and pilot downtime ×0.5/1/2 on fixed trajectories. TX, pilot TX and phase-update energy are unchanged. None of these post-hoc accounting scenarios replaces the primary criteria. Drift advances by data block, so the model omits extra drift during interruption.

## Artifacts

- [Frozen protocol](protocol.json), [source hashes](input_hashes.json), [raw logs](execution.log), [costs](costs.json).
- [Verification](analysis/verification.json), [sensitivity](analysis/sensitivity.json), [plot PNG](analysis/tracking.png), [plot PDF](analysis/tracking.pdf).
- [Privileged diagnosis](analysis/oracle_diagnosis.json) is development-only and never used for policy learning or design selection.

See `d2d_joint/SPEC.md` and `REFERENCES.md` for observation order, cost assumptions, algorithm scope and accessible-source limitations.

## Software validation and delivery

All 11 tests passed, including the full four-comparison synthetic pipeline (three seeds, both policy stages, twelve mock design banks, locked selections). Synthetic PPO updates total 4,864 blocks including the unit smoke test; these are separate from the physical-pilot training count of zero. [Test log](analysis/software_tests.log).

The new-module patch and optional sizing dependency patch were applied successfully in a temporary directory and compared byte-for-byte with their source. Existing worktree changes and campaign artifacts were preserved; no remote push was performed.

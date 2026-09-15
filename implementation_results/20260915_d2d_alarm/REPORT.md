# D2D RC sizing: ALARM pilot

**ALARM did not improve on the common initial population in this basic-profile pilot.** ALARM and random search returned identical designs for every seed, and both passed 2/3 confirmation evaluations. Regression improved the development objective in two seeds, but only one of those returned designs passed confirmation.

ALARM, the existing regression+buffer+attention optimizer, and random search were compared on the same transistor-level RC circuit.
Three methods share 8 measured initial designs per seed, including nominal. Each receives 16 additional design queries. No external offline dataset is used.

| Method | Confirmation pass | Passed and improved | Mean energy pJ/bit (pass only) | Savings mean ± sample std % (pass only) | Initial-set returns |
|---|---:|---:|---:|---:|---:|
| alarm | 2/3 | 1/3 | 0.219389 | 7.63 ± 10.78 | 3/3 |
| regression | 1/3 | 1/3 | 0.178566 | 24.81 ± — | 1/3 |
| random | 2/3 | 1/3 | 0.219389 | 7.63 ± 10.78 | 3/3 |

## Seed-level results

| Seed | Method | All confirmation inputs pass | Savings % | Energy pJ/bit |
|---:|---|---|---:|---:|
| 0 | alarm | False | 24.47 | 0.179392 |
| 0 | regression | False | 24.47 | 0.179392 |
| 0 | random | False | 24.47 | 0.179392 |
| 1 | alarm | True | 0.00 | 0.237501 |
| 1 | regression | False | 23.18 | 0.182455 |
| 1 | random | True | 0.00 | 0.237501 |
| 2 | alarm | True | 15.25 | 0.201278 |
| 2 | regression | True | 24.81 | 0.178566 |
| 2 | random | True | 15.25 | 0.201278 |

## Scope and fairness

- Six sizing variables, two coupled RC lanes, 72 MOS and behavioral RX. PTM MOS model; no new PDK extraction.
- All algorithms use the same initial measured rows, logarithmic parameter bounds, four development cases, physical evaluator and feasible return rule. Failed simulator calls count against the budget.
- ALARM uses the pinned upstream basic Config with a custom scalarizer that exactly preserves circuit.cost, including margin, overshoot and error constraints. The original optimizer code is not edited. No Lu VCO weights or profile are used.
- Regression uses 100 initial epochs with the last checkpoint, then 20 epochs per update, learned attention and buffer. This is a current equal-data pilot; older sizing campaigns used different offline data and are not directly comparable.
- Returning a design from the common initial set is not an improvement produced by the additional search. Initial-set return counts are reported explicitly.
- This compares complete algorithms, not an isolated buffer, attention or line-search ablation. Neural architectures and training computation differ.
- The eight confirmation cases use predefined packets at the same two development channel contexts. All return designs are locked before any confirmation evaluation. New-channel or PVT generalization is not established.
- Three optimization seeds are not independent physical circuits. Means over passing designs are conditional; failures remain failures.

## ALARM execution

| Seed | Critic fits | Attention updates | Buffer pushes | Head transfers | Duplicate fallback |
|---:|---:|---:|---:|---:|---:|
| 0 | 16 | 128 | 5 | 15 | 0 |
| 1 | 16 | 128 | 7 | 14 | 0 |
| 2 | 16 | 128 | 5 | 13 | 0 |

## Cost

Each method has 96 logical case measurements per seed for initialization plus search. Shared initial measurements are physically collected only once per seed.
Actual total costs, including confirmation:

```json
{
  "design_queries": 178,
  "logical_measurements": 752,
  "spice_transient_attempts": 712,
  "cache_hits": 40,
  "failed_transients": 0,
  "completed_transients": 712
}
```

Cache hits remain logical requests and do not count as new SPICE transients. Surrogate backtracking does not query SPICE.

Protocol, locked return designs, aggregate and per-query results are retained. No hyperparameter sweep or post-result checkpoint selection was performed.

## Failure details

Seed 0 returns were identical across all three methods and missed the margin constraint on two confirmation packets. Regression seed 1 also missed the margin constraint. These failed cases had zero measured bit errors; the required margin was not met. The failed designs remain failures, regardless of their lower energy. This pilot does not establish robust superiority of either learned optimizer.

## Stored records

`summary.json` and `selections_locked.json` omit repeated per-case context payloads, which are stored once in `protocol.json`. `verification.json` distinguishes original lock hashes from compact publication hashes. Per-query measurements, configurations and search events are in `traces/`. Raw waveforms and transient cache directories are local campaign outputs, not included in this compact result record.

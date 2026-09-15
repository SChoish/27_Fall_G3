# ALARM for D2D RC sizing

The standalone [ALARM optimizer](https://github.com/SChoish/ALARM) is connected to the existing transistor-level D2D evaluator through `alarm_adapter.py`. Run the paired pilot from the repository root:

```bash
python experiments/run_d2d_alarm.py --output campaigns/d2d_alarm_rc \
  --seeds 0 1 2 --initial 8 --additional 16 --jobs 3 --workers 2
python experiments/summarize_d2d_alarm.py campaigns/d2d_alarm_rc
```

Use a new output directory. NumPy, PyTorch and ngspice are required; set `NGSPICE_BIN` when ngspice is not on PATH. A run may launch `jobs × workers` ngspice processes, so reduce these values on smaller machines. The experiment is CPU-based and does not require a GPU.

## What is compared

- ALARM: upstream basic Config, scalar objective critic, attention, three-step Armijo search and critic buffer. Initial population is imported from actually measured common designs; imported measurements still count in each method's budget.
- Existing regression: the current D2D multi-output surrogate with learned attention and buffer, initialized for 100 epochs on those same designs and updated for 20 epochs per query.
- Random: log-uniform independent proposals after the same initial population.

Every seed starts from nominal plus seven predeclared random designs, then each method gets 16 additional queries. One design evaluation runs four development packets. Each method therefore has 24 design evaluations / 96 logical case measurements including initialization. The common initial population is physically collected once per seed. There is no additional offline dataset or validation-driven hyperparameter selection in this pilot.

All return designs are locked before an eight-case confirmation evaluation. Confirmation uses predefined packet seeds and patterns at the same development channel contexts; it is not a test of unseen channel/PVT generalization. Reports distinguish actual SPICE calls, cache hits, search budget and confirmation cost.

## Interface

The six existing sizing variables and their logarithmic bounds are unchanged. `metrics(result)` supplies mean energy, the two worst lane margins, worst overshoot and total observed bit errors. `D2DObjective` implements exactly the existing `circuit.cost` and feasible predicate; no VCO objective weights or new success threshold are introduced. Invalid simulation results are rejected, not assigned invented critic targets.

This ALARM critic is a supervised objective surrogate, not a Bellman critic or an RL policy. The comparison tests complete optimizers with different networks and training computation; it does not isolate a causal attention or buffer contribution.

Seven upstream modules are vendored without modification under `third_party/alarm`, pinned to `e407e8f1c616dc3a645b0793cf87ecb19a97433e`. See its `UPSTREAM.md` for provenance. The specialized Lu VCO profile is not used.

The RC model keeps assumed channel R/C/Cc values and PTM MOS devices. No PDK extraction is claimed. Quality is the finite packet margin/error/overshoot check implemented by the evaluator, not a certified BER specification.

No early stopping goal is set in the pilot. Upstream `stats["success"]` therefore remains false; quality success is read from the measured feasible predicate, not that optional stop flag.

# Dynamic D2D circuit–control pilot

Separate module for a two-lane, 72-MOS sizing driver under correlated timing drift. Existing static calibration, ADC and sizing experiments remain separate. The controller applies phase settings to every data block; evaluation never returns a best incumbent.

The actual initial pilot stopped at the predefined nominal feasibility gate. See [results](../campaigns/20260915_d2d_joint_pilot_s0-2/RESULTS.md): periodic calibration misses the quality constraint on some trajectories, while every-block calibration violates the interruption constraint. No physical PPO/co-design improvement is claimed.

## Requirements and run

Run from the repository root. Requires Python, NumPy, SciPy, PyTorch, matplotlib and ngspice. The existing `d2d_sizing/{__init__,circuit,surrogate}.py`, `adc_driver/{attention,rnc}.py`, and D2D link/model files are dependencies. `d2d_sizing` exists in the present worktree; commit `8a75f60` is not locally available as a Git object, while `d2d_implementation_8a75f60.patch` is available. Preserve the existing worktree rather than resetting to that commit.

Commands for this machine (use a NEW output directory for another run):

```bash
cd /home/svcho/27_fall_g3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=/home/svcho/anaconda3/envs/27fallg3/bin/python
RUN=/home/svcho/27_fall_g3/campaigns/d2d_joint_new_run
"$PYTHON" -m unittest discover -s d2d_joint/tests -v
"$PYTHON" -m d2d_joint.run initialize --output "$RUN" \
  --binary /home/svcho/anaconda3/envs/27fallg3/bin/ngspice
(cd "$RUN/code" && "$PYTHON" -u -m d2d_joint.run execute --output "$RUN" > "$RUN/execution.log" 2>&1)
"$PYTHON" -m d2d_joint.report --output "$RUN"
```

`initialize` freezes protocol, dependency/source snapshots, simulator version and hashes before any transistor measurement. `execute` requires the frozen copy and refuses overwrite or silent resume. `report` reads the campaign and writes derived analysis; a failed feasibility gate triggers an explicitly privileged diagnostic, with no additional SPICE or learning. Reporting can regenerate derived analysis, never source logs or physics.

Use `--representation rnc` at initialization for the separate RNC option. The default is regression. This option is implemented; the physical RNC ablation was not executed. No gate-bypass option exists. Any subsequent development study must change assumptions transparently in a new protocol and output directory, preserving the failed pilot.

## Contents

- `physics.py`: fresh transistor characterization, receipts, raw waves, bounded phase bank.
- `environment.py`: persistent phase/drift, causal paid observations, block reward and physical metrics.
- `controllers.py`: none, fixed-period, event-driven and every-block bang-bang.
- `ppo.py`: categorical PPO, four-frame history, design-conditioned actor and value network.
- `design.py`: reused sizing regression/RNC backbone and four-entry parent-reuse buffer for operational responses.
- `run.py`: nominal gate, three seeds, frozen policy stages, three circuit-search arms and four comparisons.
- `report.py`: audit, cost sensitivity, privileged diagnosis and tracking plots.
- `tests/`: information boundary, state, cost, PPO updates, selected design integrity and synthetic end-to-end checks.

[Model specification](SPEC.md) · [References and access limits](REFERENCES.md)

All four comparisons are supported in code. Synthetic integration tests exercise them, but are **not circuit performance experiments**. The actual run evaluated one circuit, four development waveform contexts and four classical controllers; it did not evaluate the locked final test set.

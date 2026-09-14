# Capacitively coupled D2D calibration

Two fixed 36-MOS transmitters, two three-section RLC channels, and three
inter-lane coupling capacitors. This is an ngspice electrical simulation with
behavioral receiver measurements in Python. It is not a foundry-PDK or UCIe
compliance implementation.

Read `COUPLED_REPORT.md` for measured outcomes and limitations. Inspect
`results/coupled_v1/summary.json`, `episodes.csv`, and `comparison.png` for the
comparison; `coupling_physics.png` shows capacitive injection into a quiet lane.
The offline dataset and trained checkpoints are in the main archive; the raw
waveform cache is provided as a separate archive.

## Dependencies

```bash
python3 -m pip install -r requirements_coupled.txt
```

A Linux x86-64 ngspice runtime is included. If executable permissions are lost
when extracting, run `chmod +x runtime/usr/bin/ngspice`. macOS and Windows need
a native ngspice installation. `NGSPICE_BIN` may point to a custom executable.
System ngspice is preferred when present; numerical details may depend on its
version. Each simulation uses one ngspice model-evaluation thread while the
driver runs several independent simulations concurrently.

## Reproduce

All commands run from this project's root. They reuse identical cached
waveforms and overwrite the named results. Copy the project for a new protocol;
do not mix changed circuit/model/packet definitions with existing caches.

```bash
python3 coupled_audit.py physical
python3 coupled_experiment.py prepare --split train
python3 coupled_experiment.py prepare --split validation
python3 coupled_experiment.py collect
python3 coupled_audit.py data_audit
python3 coupled_train.py --all
python3 coupled_evaluate.py validate
python3 coupled_experiment.py prepare --split test
python3 coupled_evaluate.py calibration
python3 coupled_evaluate.py confirm
python3 coupled_evaluate.py oracle
python3 coupled_evaluate.py oracle_confirmation
python3 coupled_audit.py evaluation
python3 coupled_report.py
```

Validation selects a single common hyperparameter/checkpoint configuration per
method across seeds. Both learning and classical search receive validation
tuning. Every test endpoint is saved before independent-packet verification.
All endpoints, including calibration failures, are independently checked.

## Artifact consistency and regression checks

```bash
python3 -m unittest discover -s tests -v
```

New measurement tables, datasets and checkpoints are written with
`<artifact>.metadata.json` sidecars. Loaders check the artifact hash, the complete
configuration, circuit/model and measurement/task code, and recorded dependency
hashes. Keep the sidecars when copying artifacts. A changed configuration or
measurement implementation requires regeneration; even a tuning-only config
change is conservatively treated as a new contract.

The original bundled assets remain readable only if they match the original
`COUPLED_MANIFEST_SHA256.json` and the pinned `legacy_contract.json`. The original
manifest is a historical release inventory, not a checksum list for the edited
source tree. Do not refresh the legacy contract to bypass a mismatch. For a new
experiment, copy the project, archive/remove its old `results/coupled_v1/waves/`
cache when changing the circuit or packet definition, then run prepare, collect,
train, validation and test evaluation in order. New artifacts get new metadata.

Configured margin and guard values are passed through search and audit paths.
SPICE runs discard prior waveform outputs and reject error logs even when the
process returns zero; retained run folders preserve the failure log.

## Python interface

```python
from coupled_link import (
    CoupledContext, CoupledSetting, payloads, simulate, measure, link,
)

context = CoupledContext(link.Context(), link.Context(), coupling_pf=0.10)
setting = CoupledSetting(strength0=3, phase0=32, strength1=2, phase1=32)
bits = payloads(seed=123, n=128)
wave = simulate(context, setting.strengths(), bits)
measurement = measure(wave, context, setting, bits)
print(measurement)
```

Strength codes `0,1,2,3` activate `1,2,4,8` output cells respectively. Each lane
always has all eight physical cells. Phase codes `0..64` cover one UI following
the fixed coarse alignment. Vref is fixed at half the lane supply.

`coupled_experiment.Environment` supplies the 50-dimensional observation and
four-dimensional quantized action interface used for offline RL. This is a
finite measurement-budget calibration task; settings remain fixed inside each
packet. Full physical waveforms and hidden RLC/coupling parameters are not
policy inputs.

## Evaluation

All methods start with the same 8+8 reference measurement. They may make up to
32 further measurements of both lanes. Reported budgets 4/8/16/32 are snapshots
of that same calibration trajectory, not separately trained budget policies.
Each query has nine phase probes per lane on a 64-bit packet.

Quality requires zero observed errors and at least 0.08 VDD worst guard margin
on both lanes, across all three held-out 128-bit packets. This finite test does
not establish a low long-run BER. Energy ratios compare identical traffic on
the selected and reference circuits. Energy covers both TX/predrivers only.

Primary comparison: joint quality and at least 10% energy reduction after 16
additional measurements. Other budgets and 5%/15% targets are reported as well.
Infeasible cases remain in the denominator. Oracle feasibility is diagnostic
only and never supplies actions to policies or data collection.

The optional post-evaluation feasibility diagnostic additionally requires the
same setting to pass the calibration packet. All initial references in this
test set pass calibration, so the common incumbent rule always returns a
calibration-feasible setting. The diagnostic enumerates this entire eligible
output set; it is not an unrestricted oracle on only the three held-out packets.

The main archive contains the exact measured tables, data, models and physical
control examples. The separate waveform archive contains the large raw cache;
extract it alongside the main archive to resume SPICE preparation from raw
waveforms. Numerical results can be inspected without downloading that cache.

# D2D RC channel

The channel can now use distributed resistance, ground capacitance and inter-lane coupling capacitance without inductors. This is the RC approximation discussed for the short D2D link; the existing R/C/Cc values are assumptions, not PDK-extracted parameters.

## Usage

```bash
python examples/d2d_rc_nominal.py --output results/rc_nominal
```

The example defaults to RC and evaluates the nominal design on four development cases. Use a new output directory. Install ngspice on PATH or set NGSPICE_BIN.

For a new sizing campaign, select the suite explicitly:

```python
from d2d_sizing.circuit import default_suite
cases = default_suite('development', channel_model='rc')
```

`Context(inductance_nh=0.0)` emits no inductor and connects each channel resistor directly to its output node. Negative or nonfinite inductance is rejected. Positive inductance retains the existing RLC netlist. The argument-free `default_suite()` remains RLC for campaign compatibility; the new example defaults to RC. Serialized cases, suite hashes and generated netlists distinguish the two models.

## Completed verification

Same nominal design and four development cases per model, ngspice 42:

| Channel | Passing cases | Mean energy (pJ/bit) | Worst lane margin |
|---|---:|---:|---:|
| RLC | 4/4 | 0.313718 | 0.140495 |
| RC | 4/4 | 0.313519 | 0.140185 |

Eight SPICE transients completed, zero failures and zero cache hits. No new optimization, RL training or test evaluation was performed. The earlier optimized energy savings have not been revalidated on RC.

Six structural checks passed: unchanged positive-L netlists at 1/3/10 sections; rejection of invalid L; RC connectivity and total R/C preservation; 72 MOS, three coupling capacitors and series-resistor terminals preserved; case serialization and L-only suite change; distinct RC/RLC identities. The delivered patch was applied to the base sources and the resulting files matched the checked sources. Detailed measurements and accounting are in verification.json.

## Scope

The supplied slide shows distributed RC coupling; x=L denotes wire length. Its capacitive-driving circuit also includes series input capacitors and three lanes. This change retains the existing two-lane CMOS drive and does not claim to reproduce that complete circuit.

Actual PEX requires a specified metal layer, length, width, spacing and return structure. Keep inter-lane coupling capacitors explicit. A CMOS PDK's on-chip wiring model does not by itself characterize an interposer or package channel. No PDK extraction or RC discretization convergence study was performed here.

Reference: https://skywater-pdk.readthedocs.io/en/main/rules/rcx.html

Base commit: e60ada39d6e4bce978ad2d6aef3a916657aa8036.

# G3 simulation-to-target protocol

The primary cost is one executed SPICE evaluation (a transient analysis for one circuit, design and stimulus/corner). A single waveform yielding several metrics costs one. Separate transients, failed launches that enter simulation, and independent verification transients each count. Record cache hits and surrogate predictions separately at zero incremental SPICE cost. A suite evaluating four cases typically costs four. A simultaneous batch incurs all actually executed transients, including work in flight when a hit is discovered.

Specify a physical FoM threshold, direction, hard constraints, and verification suite before ranking algorithms. Share the design domain, initialization protocol, fidelity, hardware/parallelism where time is compared, and evaluation cases. When common initial observations are precomputed, disclose their physical generation cost and charge the same stand-alone information cost to all methods. For transfer, report the source-generation cost, additional target-circuit simulations and cumulative cost across a stated number of target circuits.

Report the fraction of independent runs with a verified feasible target hit as a function of cumulative SPICE calls. Keep unsuccessful runs as right-censored at a common cap. Report success at cap and restricted mean calls to that cap (and median if estimable). Plot feasible best-so-far against calls as a secondary result. Lock candidate, rules and verification triggers, never use a final test bank for acquisition. Development-only historical hits must be labelled separately.

## Earlier lane monitor pilot

The reference path in REVISION provides a frozen delay lookup and precomputed channel transfer functions. One local measurement every four blocks in 128 blocks is a runtime sensing budget. This experiment can study monitoring but does not charge SPICE during candidate search. Its lane-probe counts cannot establish simulation efficiency. Source data retain their original attribution. No SPICE was launched to generate the local scheduler pilot.

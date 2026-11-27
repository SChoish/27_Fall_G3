# Transferable sparse lane monitoring

The reference path is the supplied G3 server package identified in REVISION. The SciPy import in pilot.py is deferred to the function that needs it. The reference metrics use the original 128 blocks, 20 lanes, one common observation per block and one local probe on blocks 2,6,...,126. A block fails when any lane falls below 0.25. Use paired trajectories and a common packet bank across arms.

Development uses layout 0 and a separate packet bank. Lock the algorithm before evaluating layout 2 in three configurations: original 32 Gb/s, an additional receiver RC pole at 32 Gb/s, and an alternative load assignment with a second RC value at 24 Gb/s. For transfer, only static circuit characterization (channel, rate, load-delay curves) changes. No target labels or policy updates. These are circuit models sharing the same timing buffer and drift generator.

Compare the original mode greedy, frozen PPO, and uniform probes, then cross scheduler, state estimator and phase correction. Record per-trajectory failure rates. Primary criterion is equally weighted mean across the four conditions, with every condition reported. Positive improvement means baseline minus candidate. Confidence intervals are trajectory-paired, conditional on the fixed circuit and packet bank.

# Running the physical time-to-FoM benchmark

See [the FoM definition](FOM_KO.md) and [the evaluation protocol](../PROTOCOL.md). Run from the root of the Git repository with Python, NumPy and ngspice. The reference monitor's optional risk routine requires SciPy, but the sizing and link-audit commands below only import NumPy.

```bash
cd campaigns/transferable_lane_monitor
model=../../adc_driver/models/sky130_1v8_tt.spice
python sizing/campaign.py --model "$model" --out rerun
python sizing/extra.py --model "$model" --dir rerun
python sizing/summarize.py rerun
for stages in 2 4
do
    python sizing/link_check.py --model "$model" --dir rerun --stages "$stages"
    python sizing/link_check.py --model "$model" --dir rerun --stages "$stages" --verified-five
done
```

For the pilot, the Emscripten ngspice build was loaded with Node 24.22.1 because native ngspice and the choi SSH connection were unavailable on the execution host. The JS/embedded WebAssembly module is a third-party runtime; it is not vendored. Its exact Git blob is [07df5e5d6607628589c3034b214098bae60319cf](https://api.github.com/repos/z-wasm/ngspice-wasm/git/blobs/07df5e5d6607628589c3034b214098bae60319cf). Pass its local ESM file via `--wasm /path/to/ngspice.mjs --node /path/to/node` to repeat the same backend. The wrapper uses Node filesystem permissions, a virtual model include and a virtual proc/meminfo for ngspice's memory inquiry. All circuit equations execute in ngspice. The SKY130 model file's Git blob is 22e271d81947a3bf9cbac6c8695223f1fce2d043. The two-stage 30-point reference grid differed from the packaged native results by at most 0.00093 ps over rise and fall delays.

## Interpreting the saved traces

Each JSON file contains the parameters proposed, the physical measurements for every executed transient, the chronological cumulative count, and verified first hits. `summary.csv` uses the full six-run denominator for each method and target, and assigns 40 to unresolved runs in the restricted mean. The source-model transfer arms all receive the same cold kernel trace of the matched seed on the two-stage circuit. Source calls are recorded per run; the search-cost total counts physical source runs once, even if several methods consume them.

| Method | Proposal rule |
|---|---|
| uniform | Uniform random width multipliers |
| sweep | Fixed input/output-width grid order following a common initial point |
| kernel | Kernel surrogate over target measurements, penalized constraints and an uncertainty bonus |
| replay | Evaluate source candidates ranked by development feasibility and EDP |
| residual | Fit the same surrogate on the source and correct its prediction with target measurements |
| target_cold | Maximize estimated probability of meeting the 10% EDP target and all constraints |
| target_transfer | Use the source surrogate as the prior for the target-probability acquisition |

The probability rule multiplies marginal Gaussian feasibility probabilities; dependent electrical metrics make this an acquisition approximation. The reported target hits use measured transients. See [Gardner et al., ICML 2014](https://proceedings.mlr.press/v32/gardner14.html) for constrained Bayesian optimization.
For the separately labelled 10% transfer witness, use the link_check.py command with --stages 4 --target-ten.

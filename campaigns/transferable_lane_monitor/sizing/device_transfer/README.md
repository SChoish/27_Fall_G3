# Per-device optimization of branched and coupled CMOS buffers

The circuit-to-circuit transfer benchmark uses native-width vectors rather than the two endpoints from the earlier pilot. Each transistor receives its own multiplier. Forward models in this directory are trained offline on development SPICE and are queried against real transient measurements. See PROTOCOL.md for the test splits, benchmark commitments and exploratory additions.

## Reproduce

Use the SKY130 TT and GF180 typical transistor model files from the preceding domain-transfer campaign (the parent domain_transfer/prepare_gf.py extracts the GF180 model from the pinned official source). Use the pinned Emscripten ngspice module and Node version listed in ../README.md. Python requires NumPy. Run from this directory with absolute paths:

```bash
opts=(--sky /path/to/sky130_1v8_tt.spice --gf /path/to/gf.spice --wasm /path/to/ngspice.mjs --node /path/to/node --out runs)
python experiment.py source "${opts[@]}"
python experiment.py search "${opts[@]}"
python extend.py source "${opts[@]}"
python extend.py search "${opts[@]}"
python linesearch.py search "${opts[@]}"
python audit.py runs --invariants
python audit.py runs
python refinement.py search "${opts[@]}"
python package.py
```

Each source model learns five local electrical sensitivities from individual size reductions and learns a sixth, per-device log width from the four best of thirty jointly perturbed source designs. All devices share both heads. The topology input joins complementary device pairs, driving and driven stages, and capacitively coupled outputs. Mean aggregation and single-head scaled-dot-product attention use the same typed edge values. The input features shared with the MLP include transistor polarity, normalized width, stage depth, input/output flags, output load and fanout. The layer knows where a coupling capacitor exists; its magnitude and the PDK identity are not input features. The polarity-only model uses source means by transistor polarity. The target-side electrical residual is ridge-fitted on development measurements. The new-head control removes both learned output heads while retaining the GAT embeddings. Its first proposal uses the tie order of the shared candidate pool. The pure random comparator samples full-range widths; it is a broad exploration baseline.

Physical records use a SHA-256 key over six-decimal width multipliers and corner index within each circuit. Graph and simulator input remain fixed for each domain. The raw archive stores these as one JSONL file per domain. To reconstruct the cache, read each record and use the key formula in experiment.Meter.read; audit.py reconstructs individual costs directly from these records. Three seeds vary both source training initialization and candidate streams; they are not independent samples from a circuit population. Shared first proposals such as the sizing rule can be identical across seeds.

After the search run `python retry.py search "${opts[@]}"` to investigate simulator errors serially. These reruns use the recorded candidate designs and leave the original trajectory untouched. Include their calls and the finer-step runs as separate validation overhead. To audit an existing archive, run `python package.py --restore restored` followed by `python audit.py restored`.

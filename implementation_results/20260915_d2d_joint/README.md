# D2D joint pilot delivery

`d2d_joint.patch` adds the new module, tests and documentation. It does not modify existing experiments. Apply from a checkout where `d2d_joint/` is absent:

```bash
git apply --check /path/to/d2d_joint.patch
git apply /path/to/d2d_joint.patch
```

The current worktree already contains the required sizing files. On a clean checkout lacking them, first apply `sizing_prerequisites.patch`, which adds only the three reused sizing source files. Do not apply that optional patch over existing sizing files. The historical implementation patch in the repository remains untouched.

Actual results: `campaigns/20260915_d2d_joint_pilot_s0-2/RESULTS.md`. Raw SPICE netlists, logs, compressed waveforms, phase bank, fixed protocol and source snapshot are retained there. The pilot stopped at the nominal controllability gate; there are no physical PPO checkpoints or final co-design results.

No files were staged and no remote push was performed. Patches were checked/applied in a temporary directory. `manifest.json` records delivery and result hashes.

# Vendored ALARM optimizer

Source: https://github.com/SChoish/ALARM
Pinned commit: e407e8f1c616dc3a645b0793cf87ecb19a97433e

The seven Python modules in alarm_torch are imported from that revision without algorithm edits. The original project contains implementation notes, tests and separate VCO/RC examples. This copy provides a reproducible dependency for the D2D pilot; it does not replace the upstream project.

The D2D adapter uses the library's supported custom scalarizer and logarithmic parameter mapping. It preserves d2d_sizing.circuit.cost rather than borrowing VCO objective weights. The basic Config profile is used, not the specialized Lu VCO profile. Local integration: d2d_sizing/alarm_adapter.py. Entry point: experiments/run_d2d_alarm.py.

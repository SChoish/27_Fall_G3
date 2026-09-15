# Source reading and scope

Access checked 2026-09-15. These sources motivate separating real-time control from circuit sizing; neither circuit nor algorithm is claimed to be reproduced in full.

1. **A Variation-Robust 20-Gb/s Wireline Transceiver With Real-Time Calibration in 28-nm CMOS**, DOI [10.1109/TCSI.2024.3516127](https://doi.org/10.1109/TCSI.2024.3516127).

   The [authors' institution record and abstract](https://scholarworks.bwise.kr/hanyang/handle/2021.sw.hanyang/210667?mode=full) identify sampling threshold and clock phase as real-time calibration targets, alongside a four-tap FFE transceiver. The accessible abstract supports phase calibration as a relevant control target. The full paper was not accessible in this session (publisher access unavailable; author ResearchGate page says no full text). Therefore no claim is made about its exact detector, measurement sequence, observability or calibration cost. The present E/C/L margin monitor is an explicitly assumed model, not a reconstruction of that paper's receiver. Our fixed-strength 2 Gb/s/lane predictive-model driver is not its 28 nm, 20 Gb/s silicon implementation.

2. **Analog Circuit Design Automation via Sequential RL Agents and Gm/ID Methodology**, DOI [10.1109/ACCESS.2024.3435331](https://doi.org/10.1109/ACCESS.2024.3435331).

   Read the [accessible author manuscript](https://www.researchgate.net/publication/382688640_Analog_Circuit_Design_Automation_via_Sequential_RL_Agents_and_gmID_Methodology), particularly the problem formulation, state/observation discussion and sequential feasible-region procedure. The manuscript separates sizing parameters from measured circuit outputs and conditions the state on specifications and characteristics. Its gm/ID treatment preserves DC conditions while adjusting AC-related sizing, and its sequential RL procedure addresses circuit design. Here x is likewise explicit, but the operational policy acts on persistent sampling phase and paid sensing. We do not implement its gm/ID lookup, sequential feasible-region algorithm, DC-bias guarantee or reported amplifiers. This pilot's finite alternating co-design is a different model.

Literature-derived content is confined to these scoped summaries. No measured energy or success value from either paper supplies labels, monitor costs, or acceptance targets for the new experiment. ALARM is not implemented; the small design-surrogate buffer is the repository's existing parent-reuse idea with documented limitations.

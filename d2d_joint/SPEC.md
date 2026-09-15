# Circuit–control model and protocol

## Circuit and design variables

Reuse the existing two-lane 72-MOS driver with six physical variables. Gate length is fixed at 90 nm using the repository's predictive PTM 45 nm bulk model. This is not a foundry PDK, 28 nm silicon reproduction, UCIe compliance model, complete SerDes or package extraction. Channels are lumped coupled RLC networks with a behavioral receiver. Source excitation and receiver static circuit power are excluded from transistor TX energy; the added monitor budget is explicit below. Drive strength remains fixed within each episode; there is no PLL/CDR, FFE, supply-network or threshold-control extension.

| x variable | Lower | Upper | Nominal |
|---|---:|---:|---:|
| pre1_n_um | 0.4 | 2.5 | 1 |
| pre2_n_um | 1.5 | 10 | 4 |
| output_n_um | 0.25 | 4 | 2 |
| pn_ratio | 1.4 | 3 | 2 |
| enable_width_ratio | 1 | 4 | 2 |
| series_resistance_ohm | 1 | 80 | 5 |

Each x is fixed during an episode. Models receive its six log-normalized coordinates. New x values require new transistor simulations; operational rollouts use that design's own phase bank. Prior SAC/IQL rewards and old sizing outcomes do not label this environment.

## Transitions and sampling support

One step transmits 64 payload bits per lane at 2 Gb/s per lane, with a nominal 32 ns data interval. Each 64-block episode fixes its external voltage/temperature/channel setting and alternates two predetermined packets. Voltage and temperature are static during each SPICE waveform. Correlated timing displacement is a quasi-static resampling approximation; it does not implement actual temperature or voltage transients.

The hidden two-lane timing state obeys `v[t+1] = 0.96*v[t] + Normal(0, 0.004 UI)` and `d[t+1] = clip(d[t]+v[t+1], -0.32, 0.32 UI)`. Boundary hits reverse and halve velocity. Initial drift is uniform ±0.08 UI; initial velocity uniform ±0.008 UI/block. The independent lane processes are temporally correlated. The same seeded paths are reused across controllers/designs, never selected for favorable outcomes. Drift advances per data block, not continuously through calibration interruptions; this approximation matters for timing sensitivity.

Receiver effective phase is `phase_code/64 - d[t]`. Codes start at 32, persist, and clamp to [0,64]. The characterized phase range is [-0.5,1.5] UI with 513 points, an explicit coarse latency of 1 UI and nine aperture samples across ±0.1 UI. No wrap or outside-support interpolation is allowed. Even extreme control/drift/dither/aperture combinations remain inside this support. Raw-time support is checked before interpolation; nominal margins are checked against the original direct-waveform measurement.

The phase bank contains fixed packet waveforms with preamble/tail. Alternating precharacterized packets approximates blocks independently and does not simulate arbitrary long-stream inter-block electrical memory. Discrete phase changes themselves are not circuit transients; their energy and settling pause are modeled costs.

## Actions, measurement and causal observation

There are 18 actions: lane 0 increment × lane 1 increment × measurement request, with increments {-4,0,+4} codes and request {0,1}. An action is selected using history through the previous step. Then:

1. Apply/clamp phase increments and charge any phase-update energy/pause.
2. Transmit the current block at those settings and evaluate its physical quality.
3. If requested, execute a paid known-pilot measurement at the same block's quasi-static drift; save its result and the exact measured phase codes.
4. Increment time and return the new public observation. The measurement can first influence the NEXT action.

The idealized monitor reads signed voltage margins at early/center/late phase centers separated by 0.0625 UI. Each center uses nine aperture samples per bit. Signed margins use the known pilot bits, yielding a 2×3 matrix of minimum normalized margin. This is a noiseless analog margin monitor, stronger than a single binary early/late detector; its analog readout, aggregation and calibration are not transistor-characterized. One 64-bit/lane pilot is assumed with 27 parallel probes per lane. It reuses the current packet waveform as a known pilot proxy. It does not reveal the full phase curve, absolute timing error or optimal phase.

A frame has 24 fields: x (6), current phase (2), last measured phase (2), last E/C/L result (6), normalized age (1), measurement-exists mask (1), per-lane same-phase masks (2), time fraction (1), last applied phase increments and measurement flag (3). Four frames form a 96-vector, with zero padding initially. PPO actor AND value network use this same vector. No observation normalization is fit on hidden/test data.

Without a measurement, the last result and measured phase remain unchanged; age advances. Changing phase invalidates the corresponding same-phase mask, even though the historical observation remains available. Hidden drift, full curves, current unmeasured quality, reward and diagnostic `info` are never appended to policy history. Reward is used for gradient training, not as an inference input. Classical policies receive only this same public vector and never an environment handle.

## Costs, reward and physical evaluation

TX energy is the integrated driver supply energy per payload bit from that fixed SPICE waveform, times 128 transmitted bits. At fixed x/packet this term is independent of phase actions. A measurement adds one packet of TX energy, `2*64*3*9*0.002 = 6.912 pJ` of assumed monitor energy, and 64 UI interruption. Each changed lane adds 0.1 pJ; any phase change adds two UI interruption once. These are explicit modeling assumptions, not measured monitor or phase-shifter silicon power. Receiver baseline, leakage during interruptions, policy inference hardware, and monitor readout overhead beyond the assumed budget remain uncharacterized.

Sensitivity reaccounts monitor energy ×{0.1,1,10} and pilot time ×{0.5,1,2}. It retains fixed trajectories/actions and TX energy, and never changes primary success criteria or selects policies from sensitivity results. It cannot establish accuracy of the assumed costs.

Physical block success requires both lanes' minimum signed normalized margin ≥0.08, zero center sample errors over the finite packet, and static waveform overshoot fraction ≤0.2. The finite packet check is not a long-run BER guarantee. Report every applied block, not a returned incumbent. Episode constraints are block violation rate ≤5% AND interruption/(data+interruption) ≤25%; a method passes the gate only if every development episode passes both. These are pilot requirements, not claimed industry standards. There is no 10% energy-saving requirement.

Training reward is `-total_block_energy_pj/(128*1 pJ) - quality_signal`, where quality_signal is the maximum of mean clipped normalized margin deficit, sampled error fraction and overshoot-violation indicator. The fixed quality coefficient is 1. This bounded shaping signal is distinct from binary physical quality and does not guarantee satisfying constraints. Operational design scores rank constraint exceedance first and mean energy among feasible responses. Failed responses have score `1 + exceedance`; feasible ones `E/(8192+E)`. Overshoot and sampled error failures enter through the physical violation rate.

Metrics include violation rate, maximum consecutive failed blocks, TX/pilot/monitor/phase-update energy, paid measurements, interruption seconds/fraction, delivered bits and effective throughput. Delivered bits count the whole two-lane block only if full quality passes; no FEC or retransmission model is claimed. Compute/runtime training cost, phase-bank lookups and actual SPICE are separate from modeled operating energy.

## Controllers and learning

None never changes phase or measures. Periodic requests one measurement every four blocks. Event control requests initially, after age 7, or after age 3 if the last center margin is below 0.16. Every-block bang-bang measures on every block. Each paid observation is consumed once. If center margin <0.18 or an E/L margin <0.1, the controller moves by the sign of late-minus-early with a 0.01 deadband, only for a still-matching measured phase. Thus no controller uses true drift or future measurements.

PPO uses 96→64 tanh→64 tanh, categorical 18-way policy and scalar value. Four-frame history addresses partial observability approximately; it is not a sufficient-state or recurrent optimality claim. Training samples actions; evaluation uses argmax. Protocol fixes gamma 0.99, GAE lambda 0.95, Adam 3e-4, clip 0.2, entropy coefficient 0.01, value coefficient 0.5, rollout 256, minibatch 128, four epochs and gradient clipping 0.5. Three seeds each receive 7,680 blocks per stage; no validation-driven extra learning is added.

## Finite joint optimization

The nominal gate precedes learning/search. If it passes, V1 learns only on nominal x, then freezes. Initial six design points and two new proposals for each static/classical/RL arm use equal per-candidate development rollouts. Each arm has a pool of eight (six shared, two own); total twelve distinct characterizations. Candidate labels use the identical fixed V1 policies across all proposals, not checkpoint stages of different quality.

The design surrogate reuses the sizing Model encoder/regressor/RNC projector and Entry buffer mechanics. Regression is default; RNC is optional. Four operational targets are worst violation, worst downtime, log mean energy, and longest failure/64. The x-conditioned model and fixed normalization statistics are checkpointed together. Initial training is 200 gradient steps, each update 100; predictions are recorded before observing new labels. The parent-reuse buffer holds up to four full models, selected by softmax of negative premeasurement error score. New children are used without an independent adoption test; this is a heuristic buffer, not evidence of superior retention and not a SAC Bellman critic.

V2 then fine-tunes each policy once on ALL twelve designs for 7,680 blocks, exactly ten episodes/design/seed, balanced across two environments. Every design is rescored under each fixed V2. RL arm selection uses only its original eight-design pool under that V2. Static+RL and joint+RL use the exact same V2 checkpoint for each seed. Classical settings remain development-selected and unchanged. This finite alternating procedure does not solve an exact bilevel optimum.

Supported final comparisons are static design+classical, static design+RL, classical-operational design+classical, and RL-operational design+RL. Classical comparisons are deterministic rows, not duplicated into fake independent training seeds. The twelve-design pilot can demonstrate software feasibility, not establish statistically reliable optimizer superiority. Protocol separates development waveform seeds/drift from new test waveform seeds/drift; selections and checkpoint hashes lock before test characterization. Every returned x must exactly match its evaluation bank. No silent resume accepts changed traces.

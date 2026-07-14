# Fresh Live Mininet / SDN QoS Validation with Incremental Recovery

Methodology:
- Mininet version: 2.3.0 (VM guide baseline)
- OVS version: Open vSwitch 2.13.1 (VM guide baseline)
- Controller: Reward-Gated GNN-LPD Traffic Engineering (RG-GNN-LPD) (FIX1 strict-all)
- VM/OS: Ubuntu 20.04.1 LTS VirtualBox VM via SSH port 2222
- Number of runs: 5 per executed scenario
- Incremental update policy: controller stayed warm, unaffected flows were preserved, backup plans were precomputed, and only changed rules were reinstalled
- Batch installation policy: flow additions were sent per-switch in batched add-flows files instead of one ovs-ofctl add-flow call per rule
- Warm-up duration: 1 second pre-measure warm-up
- Failure timing: failure scenarios kept transient loss separated from post-recovery steady-state; mixed spike+failure applies spike after convergence and a short documented delay
- Mixed spike+failure delay: 400 ms after stable post-failure forwarding
- Startup time excluded: yes; Mininet, switch initialization, and controller boot were outside install and recovery timing

Validation:
- Fresh enhanced SDN CSVs created: PASS
- Raw logs exist: PASS
- Old SDN rerun CSVs preserved: FAIL
- All enhanced rows are live FIX1 reruns: PASS
- No historical SDN rows mixed: PASS
- Transient and post-recovery loss remain separated: PASS
- Offered UDP rates are recorded: PASS
- Rule diff / changed-rules counts are recorded: PASS
- No fabricated QoS values: PASS
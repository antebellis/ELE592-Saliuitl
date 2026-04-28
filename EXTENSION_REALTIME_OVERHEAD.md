# Real-Time Overhead Extension (APRICOT)

This extension measures runtime overhead for a streaming CV setting by comparing four operation modes on the same data:

1. `classification_only` (`--bypass_det --bypass`)
2. `detection_only` (`--bypass`)
3. `detection_plus_recovery` (default Saliuitl two-stage path)
4. `always_recovery` (`--bypass_det`)

## Script

- `realtime_mode_overhead.py`

## Full APRICOT Result (prepared set used in this project)

Saved summary:

- `realtime_mode_overhead_apricot_full/summary.json`

Key throughput results (574 effective frames):

- `classification_only`: 18.64 FPS
- `detection_only`: 0.89 FPS
- `detection_plus_recovery`: 0.84 FPS
- `always_recovery`: 0.76 FPS

Interpretation:

- Two-stage gating is faster than recovering every frame (~1.11x in this run).
- Most runtime overhead is introduced by the salience/detection pipeline relative to plain classification.

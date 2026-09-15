# Recording orchestration provenance

`lerobot_v2.py` derives its high-level episode/control-loop organization from
Hugging Face LeRobot `src/lerobot/scripts/lerobot_record.py` at commit
`4aaff99be4a1d81568c08c8f0296b41b40c99ec4`, licensed under Apache-2.0.

Project changes include schema-v2 synchronized observations, per-sensor host
timestamps, strict frame rejection, explicit operator acceptance, discarded
capture diagnostics, and recording `robot.send_action()`'s returned action.

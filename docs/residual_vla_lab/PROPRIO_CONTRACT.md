# Proprioception contract

RLinf `LiberoEnv._extract_image_and_state()` exposes an 8D robot state. The
actor receives exactly these robot-observable values through the adapter.

| Field | Source | Shape | Units | Normalized? | Real-robot observable? | Actor? |
|---|---|---:|---|---|---|---|
| `eef_pos` | `robot0_eef_pos` | 3 | environment position units | no extra residual normalization | yes | yes |
| `eef_quat_or_rot` | `robot0_eef_quat` converted by `quat2axisangle` | 3 | radians/axis-angle | no | kinematics | yes |
| `gripper` | `robot0_gripper_qpos` | 2 | joint position units | no | yes | yes |
| `joint_pos` | not present in wrapped RLinf observation | — | — | — | yes | no |
| `joint_vel` | not present in wrapped RLinf observation | — | — | — | yes | no |

Object pose, goal distance, contacts, and other simulator-oracle fields are not
read. Missing fields cause an error; they are never zero-filled or fabricated.


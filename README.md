<!-- SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# cuRobo

*CUDA Accelerated Robot Library*

**[Documentation](https://nvlabs.github.io/curobo) | [Paper](https://arxiv.org/abs/2603.05493)**

> [!NOTE]
> cuRoboV2 is a significant rewrite and the public API has changed from cuRobo v1.
> If you depend on the v1 API, pin to the [`v0.7.8`](https://github.com/NVlabs/curobo/tree/v0.7.8) tag.

> [!IMPORTANT]
> **This is a fork** ([`helpYoon/curobo`](https://github.com/helpYoon/curobo)) of
> [NVlabs/curobo](https://github.com/NVlabs/curobo), maintained for the
> [cuTAMP](https://github.com/helpYoon/cuTAMP) task-and-motion planner. It adds an
> opt-in **center-of-mass (CoM) safety term to inverse kinematics** for the Booster
> T1 humanoid. The fork is based on upstream
> [`ac5f931`](https://github.com/NVlabs/curobo/commit/ac5f931) (NVlabs PR #678).
> All additions default **off**, so an unconfigured solver is byte-identical to
> upstream. See [Fork changes](#fork-cutamp-com-aware-ik) below.

cuRobo is a CUDA-accelerated library for robot motion generation, built on
PyTorch, CUDA, and Warp. It provides GPU-parallel algorithms for
forward/inverse kinematics, collision checking, trajectory optimization,
geometric planning, GPU-native perception, and whole-body motion generation,
scaling from single-arm manipulators to high-DoF humanoids.

Key capabilities:
- **Dynamics-aware trajectory optimization** with B-spline representation enforcing smoothness and torque limits
- **GPU-native ESDF perception** that generates dense signed distance fields from depth images, up to 10x faster than state-of-the-art
- **Scalable whole-body computation** including topology-aware kinematics, differentiable inverse dynamics, and map-reduce self-collision for high-DoF robots
- **Collision-free motion generation** combining IK, geometric planning, and trajectory optimization.

## Fork: cuTAMP CoM-aware IK

This fork adds a **CoM-over-support-rectangle residual** to the inverse-kinematics
solver so that IK solutions for a humanoid keep the projected center of mass over
its support footprint (otherwise the robot tips). It is the load-bearing change
for cuTAMP's Booster T1 humanoid.

**Base:** upstream `ac5f931` (NVlabs PR #678). The fork carries **no CUDA/kernel
changes**. Every edit below is pure Python on the config/solver path.

### What it does

cuRobo's `Kinematics` can compute a per-link, mass-weighted CoM
(`KinematicsState.robot_com`), but only when built with `compute_com=True` (a
construction-time CUDA kernel-template flag; default off leaves `robot_com`
allocated but zero). This fork:

1. Plumbs `compute_com` through so the CoM can be turned on where it's needed, and
2. Adds a residual to the **seed-IK Levenberg–Marquardt** error calculator that
   penalizes the CoM leaving an axis-aligned rectangle in a chosen base frame
   (outside-quadratic + inside-margin barrier + optional pull-to-center). The
   seed-IK LM stage is the only IK injection point with a Python gradient path —
   the main IK solver is fused CUDA and bypasses Python costs — so the residual
   lives there and feeds the LM step via a single combined `autograd.backward`.

### Files changed (7)

| File | Change |
| --- | --- |
| `curobo/_src/solver/seed_ik/seed_ik_error_calculator.py` | Adds `com_support_penalty()` and wires the CoM residual into the seed-IK LM error/backward path (combined `autograd.backward([pose, com])`; CoM also folded into the LM acceptance norm, success stays pose-only). |
| `curobo/_src/solver/seed_ik/seed_ik_solver.py` | Builds the seed FK model with `compute_com=True` when the residual is enabled; raises if enabled without `use_backward=True` (the residual's gradient comes from the autograd path). |
| `curobo/_src/solver/seed_ik/seed_ik_solver_cfg.py` | Six `com_*` config fields consumed by the calculator. |
| `curobo/_src/solver/solver_ik.py` | Forwards `seed_com_*` → `com_*` into the seed solver; raises if `seed_com_support_weight > 0` without `use_lm_seed=True` (the residual only exists on the LM-seed path). |
| `curobo/_src/solver/solver_ik_cfg.py` | Six public `seed_com_*` fields + `create()` plumbing — the surface cuTAMP sets. |
| `curobo/_src/transition/robot_state_transition.py` | Passes `compute_com` to the planner's transition `Kinematics` (a separate instance from the seed-IK model). |
| `curobo/_src/transition/robot_state_transition_cfg.py` | Adds the `compute_com: bool = False` field. |

### Config knobs (all default to upstream-equivalent values)

Set on `IKSolverCfg` (forwarded to the seed solver):

| Field | Default | Meaning |
| --- | --- | --- |
| `seed_com_support_weight` | `0.0` | Master weight; **`0.0` disables the residual entirely** (byte-identical to upstream). |
| `seed_com_half_extents` | `None` | Support-rectangle half-extents `(x, y)` in meters, in the base frame. Required when the weight > 0. |
| `seed_com_inside_margin` | `0.02` | Width (m) of the inside-edge soft-barrier band. |
| `seed_com_inside_weight` | `1.0` | Inside-barrier scale vs. the outside quadratic. |
| `seed_com_center_weight` | `0.0` | Optional pull-to-center scale; `0.0` = pure hard constraint. |
| `seed_com_base_link_name` | `"mobile_base_link"` | Tool frame whose pose defines the rectangle; must be in `tool_frames`. |

For the planner path, set `RobotStateTransitionCfg.compute_com = True` (default
`False`) before constructing the motion planner.

## Citation

If you found this work useful, please cite cuRoboV2,

```
@misc{curobo_v2,
      title={cuRoboV2: Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots},
      author={Balakumar Sundaralingam and Adithyavairavan Murali and Stan Birchfield},
      year={2026},
      eprint={2603.05493},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## Contributing

Contributions are welcome. Bugs: [open an issue](https://github.com/NVlabs/curobo/issues). General usage questions: [GitHub Discussions](https://github.com/NVlabs/curobo/discussions). For pull requests, please read [`CONTRIBUTING.md`](CONTRIBUTING.md). All commits must include a DCO sign-off (`git commit -s`).

## License

cuRobo is released under the [Apache 2.0 license](LICENSE).

The example robot assets bundled in this repository are provided under their respective licenses. See [LICENSE_ASSETS](LICENSE_ASSETS) for details.

## Third-Party Software

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

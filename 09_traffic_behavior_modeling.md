# Traffic Behavior Modeling in Implementable Detail

`03_environment_design_space.md` introduced IDM, MOBIL, and the social force model at a survey level (what they are, roughly). This file goes to the level of actual formulas, actual default parameter values (read directly from source code where possible), and — the genuinely useful part for implementation planning — precisely which simulators ship each model natively versus which would require you to implement it yourself.

## The Intelligent Driver Model (IDM) — car-following

IDM computes a single vehicle's longitudinal acceleration as a function of its own speed, its desired speed, and the gap to the vehicle ahead:

```
a = a_max · [ 1 − (v / v0)^δ − (s*(v, Δv) / s)² ]

s*(v, Δv) = s0 + max(0, v·T + (v·Δv) / (2·√(a_max·b)))
```

where `v` is current speed, `v0` is desired (free-flow) speed, `s` is the actual gap to the leading vehicle, `Δv` is the closing speed (`v_ego − v_lead`), and `s*` is the *desired* minimum gap given current conditions.

**The six parameters, with concrete defaults** (read directly from `highway-env`'s `IDMVehicle` class, `highway_env/vehicle/behavior.py` — a genuine textbook-faithful implementation):

| Symbol | Name | highway-env default | Meaning |
|---|---|---|---|
| `a` (`a_max`) | Maximum acceleration | `6.0` m/s² (used as `ACC_MAX`) / `3.0` m/s² comfort ceiling (`COMFORT_ACC_MAX`) | How hard the vehicle accelerates toward `v0` when the road ahead is clear |
| `b` | Comfortable/desired deceleration | `5.0` m/s² (`COMFORT_ACC_MIN`, stored as negative) | How hard the vehicle brakes to maintain the desired gap |
| `v0` | Desired speed | vehicle/lane-specific, clipped to the lane speed limit | The speed the vehicle converges to in free flow |
| `δ` | Acceleration exponent | `4.0` (typical range 3.5-4.5 across implementations) | Higher = acceleration stays near maximum for longer as the vehicle approaches `v0`, then drops off sharply |
| `s0` | Minimum (jam) gap | `5.0 m` + vehicle length | The bumper-to-bumper gap maintained even at a dead stop |
| `T` | Desired time headway | `1.5 s` | The following-distance safety margin, in seconds, at the current speed |

**Which simulators ship IDM natively vs. require custom work:**
- **SUMO**: ships IDM as one of several selectable car-following models (`carFollowModel="IDM"` in vehicle-type XML), alongside its own default Krauss model — native, configurable per vehicle type.
- **highway-env**: the `IDMVehicle` class *is* a literal, textbook-faithful implementation (the formula above, verbatim) — the most directly hackable IDM implementation on this list if you want to read or modify the actual acceleration code.
- **MetaDrive**: ships an `IDMPolicy` traffic-vehicle behavior, configurable, used by default for background traffic.
- **CARLA's Traffic Manager**: implements its own custom C++ car-following/gap-keeping logic that is *conceptually* IDM-like (accelerate toward a target speed, decelerate to maintain a safe gap) but is **not a literal IDM implementation** — its parameters (`distance_to_leading_vehicle`, `vehicle_percentage_speed_difference`) are exposed at a higher, less mathematically transparent level than IDM's six named constants. If literal IDM fidelity matters to you, CARLA's Traffic Manager is not a drop-in match — you'd need a custom controller driving CARLA's vehicles directly (bypassing Traffic Manager) to get the textbook formula.
- **nuPlan**: uses IDM as its default "closed-loop reactive" agent behavior, later superseded (in the nuPlan-R extension) by learned diffusion-based reactive agents (per `01`/`03`).

Sources: [Twenty-Five Years of the Intelligent Driver Model (2506.05909)](https://arxiv.org/html/2506.05909v1), [highway-env `behavior.py` source](https://github.com/Farama-Foundation/HighwayEnv/blob/master/highway_env/vehicle/behavior.py), [traffic-simulation.de IDM reference](https://traffic-simulation.de/info/info_IDM.html), [CARLA Traffic Manager docs](https://carla.readthedocs.io/en/latest/adv_traffic_manager/).

## MOBIL — lane-changing decisions

MOBIL ("Minimizing Overall Braking Induced by Lane Changes," Kesting/Treiber/Helbing 2007) layers a lane-change *decision rule* on top of any car-following model (originally paired with IDM, but general). Two criteria must both be satisfied for a lane change to execute:

**Safety criterion**: the new following vehicle (the car that would end up behind you in the target lane) must not be forced to decelerate harder than a safety threshold `b_safe`:
```
a_new_follower ≥ −b_safe
```

**Incentive criterion**: your own acceleration gain from changing lanes must exceed a minimum threshold, *after* accounting — via a politeness factor `p` — for the (dis)advantage imposed on other affected drivers:
```
(a_new_self − a_old_self) + p · [(a_new_follower − a_old_follower) + (a_new_old_lane_follower − a_old_old_lane_follower)] > Δa_th
```

**Parameters, with concrete defaults** (again read directly from highway-env's `IDMVehicle.mobil()` implementation):

| Symbol | Name | highway-env default | Meaning |
|---|---|---|---|
| `p` | Politeness factor | `0.0` | `0` = purely selfish (only your own gain matters); `1` = fully altruistic (others' disadvantage weighted equally to your own gain). **highway-env's default is non-courteous** — worth knowing if you want more realistic, less aggressive scripted traffic, since you'd need to raise this yourself |
| `Δa_th` | Minimum acceleration gain to bother changing | `0.2` m/s² (`LANE_CHANGE_MIN_ACC_GAIN`) | Prevents lane-changing for negligible benefit (avoids unrealistic lane-change "flickering") |
| `b_safe` | Maximum safe braking imposed on the new follower | `2.0` m/s² (`LANE_CHANGE_MAX_BRAKING_IMPOSED`) | The hard safety gate — if satisfying the lane change would force someone behind you to brake harder than this, the change is vetoed regardless of incentive |
| — | Lane-change delay | `1.0 s` (`LANE_CHANGE_DELAY`) | Minimum time between successive lane changes, preventing unrealistic rapid weaving |

**Which simulators ship MOBIL natively vs. require custom work:**
- **highway-env**: literal, exact MOBIL — the `mobil()` method implements precisely the safety + incentive criteria above.
- **SUMO**: does **not** ship literal MOBIL by name. Its default lane-change model is **LC2013** (instantaneous, discrete lane changes), with **SL2015** as an alternative enabling continuous sub-lane lateral movement (used automatically once the sublane model is enabled). Both are SUMO's own bespoke gap-acceptance-and-incentive logic — conceptually parallel to MOBIL (both are "does this help me enough, and is it safe for whoever's affected" decision rules) but not the same equations, and SUMO does not expose a `laneChangeModel="MOBIL"` option. If literal MOBIL fidelity in SUMO specifically matters, you'd implement it yourself against SUMO's TraCI API rather than relying on the built-in models.
- **CARLA's Traffic Manager**: custom heuristic lane-change logic (gap-based, not MOBIL), again exposed via coarser parameters than MOBIL's four named constants.
- **MetaDrive**: its background-traffic policies handle lane selection as part of the broader `IDMPolicy`/route-following logic rather than a separately named MOBIL implementation.

Sources: [MOBIL: General Lane-Changing Model (Kesting, Treiber, Helbing)](https://www.mtreiber.de/publications/MOBIL_TRB.pdf), [traffic-simulation.de MOBIL reference](https://traffic-simulation.de/info/info_MOBIL.html), [highway-env `behavior.py` source](https://github.com/Farama-Foundation/HighwayEnv/blob/master/highway_env/vehicle/behavior.py), [SUMO SublaneModel docs](https://sumo.dlr.de/docs/Simulation/SublaneModel.html), [SUMO Lane-Changing Model developer docs](https://sumo.dlr.de/docs/Developer/How_To/Lane-Changing_Model.html).

## The social force model — pedestrians

Helbing & Molnár's 1995 social force model treats each pedestrian `i` as subject to a sum of forces, analogous to Newtonian mechanics:

```
m_i · dv_i/dt = f_i^0 + Σ_j≠i f_ij + Σ_w f_iw
```

- **`f_i^0`** (self-driving/motivation force): pulls pedestrian `i` toward their destination at their preferred speed, with a relaxation time constant governing how quickly they accelerate toward it.
- **`f_ij`** (pedestrian-pedestrian interaction force): a repulsive term keeping pedestrians from colliding with or crowding each other, typically modeled as an exponentially-decaying potential (parameters `A` = interaction strength, `B` = interaction range) plus a body-compression term active only at very close range.
- **`f_iw`** (pedestrian-obstacle/wall force): repels pedestrians from static obstacles (curbs, walls, parked cars), same functional family as `f_ij`.

This produces reasonably naturalistic emergent crowd behavior (lane formation in bidirectional flows, oscillation at doorways, avoidance) with zero learning — it's a genuinely durable model, still the default in commercial microscopic traffic tools like VISSIM's pedestrian module, over 30 years after publication.

**Which driving simulators ship this natively vs. require custom work:**
- **None of the driving-RL-focused simulators surveyed in `01` ship a literal social-force pedestrian model out of the box.** CARLA's pedestrian AI uses simpler waypoint-following with basic collision avoidance, not SFM. MetaDrive's default scenarios are vehicle-centric and don't include a first-class pedestrian-agent population comparable to its vehicle traffic. SUMO does have a native pedestrian model ("striping model" for sidewalk movement), which is a lane-discretized model conceptually related to but distinct from continuous social-force dynamics.
- **A real, usable open-source implementation exists if you want to add SFM yourself**: [`pysocialforce`](https://github.com/yuxiang-gao/PySocialForce) (yuxiang-gao) is a pure-Python, NumPy-based implementation of the *extended* social force model (including pedestrian social groups and inter-group interactions), configurable via a simple TOML file, with a built-in animation/visualization tool. The practical integration pattern for a driving simulator without native pedestrian SFM: run `pysocialforce` as an external per-tick controller that computes pedestrian accelerations each simulation step and feeds the resulting positions/velocities into the simulator's pedestrian actors (CARLA's Python API, or a custom pedestrian-actor abstraction in MetaDrive/highway-env), rather than expecting the simulator to do this internally.
- Recent research (the "personality-driven jaywalking" multi-agent RL pedestrian work cited in `03`) explicitly frames itself as going *beyond* social-force/scripted approaches specifically because SFM and fixed crossing scripts under-represent real behavioral heterogeneity (rule-breaking, personality variation) — worth knowing as the honest state of the art if pedestrian realism specifically is a stated goal rather than a nice-to-have.

Sources: [Social force model for pedestrian dynamics (Helbing & Molnár, Phys. Rev. E 1995)](https://link.aps.org/doi/10.1103/PhysRevE.51.4282), [pysocialforce GitHub](https://github.com/yuxiang-gao/PySocialForce), [Social force model for pedestrian-AV interaction](https://www.sciencedirect.com/science/article/abs/pii/S1569190X24000157), [SUMO pedestrian model docs](https://sumo.dlr.de/docs/Simulation/Pedestrians.html).

## Summary table: model → native support by simulator

| Model | Governs | highway-env | SUMO | CARLA (Traffic Manager) | MetaDrive | nuPlan |
|---|---|---|---|---|---|---|
| IDM (literal) | Car-following | Native, exact | Native (selectable) | Custom-only (TM's own logic is IDM-*like*, not literal) | Native (`IDMPolicy`) | Native (default reactive-agent behavior) |
| MOBIL (literal) | Lane-changing | Native, exact | Custom-only (ships LC2013/SL2015 instead) | Custom-only | Bundled into route/IDM policy, not separately named | Not applicable (planning benchmark, not a from-scratch traffic generator) |
| Social force model | Pedestrians | Not applicable (no pedestrian agents) | Partial (different "striping" model, not SFM) | Custom-only (`pysocialforce` + Python API bridge) | Custom-only, and pedestrians aren't a first-class agent type by default | Not applicable |

The practical upshot for environment design (cross-referencing `03`'s "other agents" axis): if literal-textbook parameter transparency matters to you — e.g., you want to *tune* `T`, `s0`, `p`, `Δa_th` directly and reason about the consequences — highway-env is the only platform on this list where you're editing the actual named constants from the traffic-modeling literature rather than a simulator-specific abstraction layer sitting one level removed from it.

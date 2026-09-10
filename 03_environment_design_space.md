# Environment Design Space

This is a genuine design space with real tradeoffs on each axis — not a recommendation of one setup. Think of each subsection as an independent dial; `06_recommendations.md` shows how these dials bundle sensibly at different project-scope tiers.

## Axis 1: Road network topology

| Topology | What it is | Where it's used | Tradeoffs |
|---|---|---|---|
| Single loop/ring road | One continuous loop, no intersections | Flow's classic "ring" scenario for studying stop-and-go traffic waves and mixed-autonomy platooning | Simplest possible; only tests longitudinal control/car-following, no navigation/decision-making at all |
| Grid | Regular city-block grid | Common in synthetic SUMO benchmarks and toy "gridworld city" environments | Easy to procedurally generate and reason about; unrealistically regular, doesn't stress irregular real-world topology |
| Small hand-built town | A few blocks, several intersection types, authored by hand | CARLA's Town01–Town07/Town10 maps, highway-env's dedicated scenario families, MetaDrive's block-based procedural generation | Controllable and reproducible; limited scenario diversity unless you author many variants or use procedural generation |
| Real-world OSM-derived maps | Actual street networks pulled from OpenStreetMap | CARLA's Digital Twin Tool (0.9.15+, combines OSM with OpenDRIVE for lane topology), nuPlan/Waymax (real logged cities: Boston, Pittsburgh, Las Vegas, Singapore, SF, Phoenix), Flow (OSM import) | Maximally realistic topology; OSM data "usually lacks the lane connectivity information necessary for most simulation scenarios," requiring hand-authoring to fix dangling roads and add lane-level detail |
| Procedural generation for diversity | Randomized composition of road-network primitives | MetaDrive's core mechanism — composes randomized block primitives into unique towns each episode | Directly targets generalization: MetaDrive's own experiments found that **increasing the diversity and size of the procedurally generated training set directly improves generalization to unseen scenes** — this is one of the few empirically validated "generalization levers" in the field (see also `05_challenges_and_pitfalls.md`) |

Sources: [MetaDrive paper](https://arxiv.org/abs/2109.12674), [CARLA 0.9.15 release](https://carla.org/2023/11/10/release-0.9.15/), [Procedural generation of urban roads from OSM](https://www.researchgate.net/publication/363625340_A_Procedural_Generation_Method_of_Urban_Roads_Based_on_OSM), [OpenTwinMap](https://arxiv.org/pdf/2511.21925).

## Axis 2: Traffic control elements (stop signs, traffic lights) and how they reach the agent

Whatever elements you include, there's a real design choice in *how* the agent perceives them, independent of whether they exist in the world at all:

1. **As part of a structured route/navigation command** — e.g., a "turn left / go straight / turn right at the next junction" high-level instruction fed alongside sensor observations, common in CARLA-style end-to-end setups.
2. **As an explicit feature in the observation vector** — e.g., a boolean or one-hot "upcoming signal phase" feature, as seen in many traffic-signal-control RL state representations (queue length, vehicle presence, current signal phase are the standard ingredients there) and in nuPlan/BEV-style vector observations.
3. **Rendered visually inside a camera/BEV image**, left for the network to infer from pixels — the highest-fidelity but hardest-to-learn-from option, and the one that actually stresses perception rather than assuming it away.
4. **As privileged ground-truth used only for reward/evaluation computation**, regardless of what the agent's own sensors show — this is literally how CARLA's Leaderboard scores infractions (running a red light, c=0.40; running a stop sign, c=0.25 — see `05_challenges_and_pitfalls.md`) even for agents whose actual sensor observation doesn't explicitly include "signal state" as a feature.

Traffic-signal-control-specific RL (a distinct sub-problem: the light itself is the RL agent, not the car) typically uses either vector state (vehicle counts/queue lengths/waiting times per approach plus current phase) or image-like grid encodings (vehicle positions mapped onto a pixel grid, fed to a CNN). This is a genuinely separate research direction (well-studied via sumo-rl) from "train a car to drive," worth knowing is orthogonal rather than assuming it's the same problem.

Sources: [Traffic Light Control with RL (2308.14295)](https://arxiv.org/pdf/2308.14295), [CARLA Leaderboard evaluation criteria](https://leaderboard.carla.org/evaluation_v2_0/).

## Axis 3: Other agents

### Scripted rule-based NPC vehicles — the default and by far most common approach
Almost every platform defaults here for tractability. The workhorse model is the **Intelligent Driver Model (IDM)** — a continuous car-following model where a vehicle accelerates toward a desired speed when the gap to the vehicle ahead is large, and decelerates to maintain a safe following distance when it shrinks. IDM has a 25-year track record (see ["Twenty-Five Years of the Intelligent Driver Model"](https://arxiv.org/html/2506.05909v1)) with extensions for lane-changing (MOBIL-style rules layered on top). Concretely: SUMO/Flow use IDM natively, CARLA's **Traffic Manager** runs autopilot NPCs on similar principles, and highway-env has a built-in `IDMVehicle` class. Recent work also **hybridizes IDM with RL/deep learning** — e.g. extending IDM with deep RL to better predict/identify leading-vehicle behavior and optimize trajectories in dense traffic, or combining TD3-based RL with IDM for improved longitudinal control.

### Logged/replayed real trajectories ("non-reactive" simulation)
Other agents simply replay their real recorded trajectories regardless of what your ego agent does. This is what nuPlan calls "closed-loop non-reactive" simulation, and what Nocturne/Waymax/GPUDrive do by default when using WOMD data. Maximally realistic *individual* agent behavior (it's real human driving), but non-reactive — the other agents won't actually respond to your car's presence, which is a meaningful realism gap if your ego agent does anything the real driver in the log didn't encounter.

### Learned reactive agents
The most sophisticated and most expensive tier. **nuPlan-R** replaces IDM reactive agents with **learned diffusion-based reactive agents** for more realistic responsiveness. The **Waymo Open Sim Agents Challenge (WOSAC)** benchmarks exactly this capability, scoring simulated agents by how closely their *behavior distribution* (not any single trajectory) matches real logged human distributions via negative log-likelihood across kinematic, interactive, and map-based metric groups. If your other cars are themselves RL agents (see multi-agent RL in `02_rl_algorithms.md`), you're implicitly building this tier yourself — and the self-play literature's finding that **naive self-play drifts non-human without anchoring to logged human data** is the central lesson to carry into this choice.

### Pedestrians
The classical approach is the **social force model**: each pedestrian is modeled as subject to a combination of forces — a "driving force" pulling them toward their destination, an "agent interaction force" repelling them from other agents/pedestrians, and a "wall interaction force" repelling them from static obstacles/curbs. This produces reasonably naturalistic emergent crowd flow without any learning at all, and is still the default in most AV-pedestrian-interaction simulation work. More recent work moves toward deep-learning pedestrian trajectory prediction/generation models, and toward **RL-based pedestrian agents** — including a notable "personality-driven jaywalking" approach that **co-trains pedestrians and the ego car with multi-agent RL** specifically to get more realistic, less-scripted crossing behavior (including occasional rule-breaking) than either fixed crossing scripts or pure social-force models produce. A real caution flagged in the literature: **most simulation-based AV testing still relies on scripted pedestrian motion or simplified crossing rules that don't reflect real human behavioral heterogeneity** — this is a known, acknowledged gap in the field, not solely a limitation of hobbyist projects.

Sources: [Social force model for pedestrian-AV interaction](https://www.sciencedirect.com/science/article/abs/pii/S1569190X24000157), [MARL for safe driving under pedestrian uncertainty](https://arxiv.org/html/2605.20255), [IDM survey (2506.05909)](https://arxiv.org/html/2506.05909v1).

## Axis 4: Sensing modalities — realism/sim-to-real tradeoffs

| Modality | What it gives you | Sim-to-real tradeoff |
|---|---|---|
| Raw RGB camera | Full photorealistic pixel input, the most "real deployment"-like sensor | Hardest sim-to-real gap of any modality — photorealism/appearance differences between sim and real cameras are large enough that dedicated tools exist just to shrink this gap (e.g. [CARLA2Real](https://arxiv.org/pdf/2410.18238)). Also the hardest to learn from directly (highest-dimensional). |
| Semantic segmentation | Per-pixel class labels (road, vehicle, pedestrian, etc.), removing appearance/texture as a nuisance variable | Meaningfully easier sim-to-real than raw RGB, since a real-time segmentation model can be run on real cameras to produce a similar-looking intermediate representation on both sides — but now depends on that segmentation model's own accuracy |
| LiDAR point clouds | Geometric ground truth of surroundings, largely lighting/weather-invariant | Real LiDAR is sparser/noisier than simulated LiDAR by default; requires point-cloud-specific architectures (voxelization, PointNet-style encoders) adding engineering complexity |
| BEV rasterized occupancy grid | A top-down, spatially structured image showing lanes/vehicles/signals in a consistent frame | Currently the most popular "sweet spot" in the RL-for-driving literature — described in Kiran et al.'s survey as **"sensor agnostic but still close to the spatial organization"** of the scene. Multiple studies report strong lane-keeping/collision-avoidance results from small (e.g. 64×64) color-coded BEV rasters, faster and easier to learn from than raw pixels. The catch: BEV typically assumes something else (privileged simulator state, or a separate perception pipeline) already produced the rasterization — it partially *sidesteps* the perception problem rather than solving it end-to-end. |
| Privileged ground-truth state vectors | Exact position/velocity/heading of ego + all nearby agents, lane-graph features, etc. | Fastest and cheapest to train on by far — this is highway-env's default Kinematics observation and MetaDrive's default vector+LiDAR-distance observation. But it has the **largest sim-to-real gap of all**: real cars don't have magical ground-truth knowledge of every other agent's exact state — that's an entire perception subsystem in reality. Excellent for prototyping decision/control algorithms in isolation; not usable as-is for eventual real-world transfer without separately building a perception stack. |

The honest framing: this axis is really a decision about **whether this project is about perception, decision-making, or both.** If the interesting part to you is control/decision policy learning, privileged state or BEV is the right choice and lets you iterate fast. If part of the appeal is an end-to-end pixels-to-actions pipeline (closer in spirit to camera-only Duckietown/Donkeycar projects), raw RGB is unavoidable and you should budget for a much harder, slower training process and a real sim-to-real gap to eventually confront.

Sources: [Comprehensive Review of RL for AD in CARLA (BEV/sensor discussion)](https://arxiv.org/pdf/2509.08221), [BEV occupancy grids survey](https://arxiv.org/pdf/2405.05173), [CARLA2Real](https://arxiv.org/pdf/2410.18238).

## Axis 5: Action space design

| Action space | Description | Tradeoffs |
|---|---|---|
| Discrete high-level maneuvers | e.g. highway-env's `{LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER}` — the environment's own lower-level kinematics execute the chosen maneuver | Simplifies exploration and credit assignment; pairs naturally with DQN/Rainbow; **"lacks flexibility... velocity can only change in fixed increments"** and produces jerkier control unless paired with a smoothing lower-level controller |
| Continuous throttle/steer/brake | Direct continuous control, typically normalized to e.g. `[-1, 1]` | Smoother, more realistic, matches how a real vehicle is actually actuated; harder to learn (higher-variance policy gradients), needs SAC/PPO-continuous/DDPG rather than vanilla DQN |
| Hybrid/parameterized (discrete + continuous) | Pick a discrete maneuver, then a continuous parameter for executing it (e.g. "accelerate," with a continuous magnitude) | A genuine middle ground used in several published lane-change papers (e.g. PASAC); adds its own architectural complexity (parameterized action-space algorithms aren't as off-the-shelf as either pure discrete or pure continuous RL) |
| Hierarchical (two-timescale) | A high-level discrete/continuous decision every N steps, a low-level continuous controller (often a classical PID or pure-pursuit controller, not itself learned) every step | Directly addresses long-horizon credit assignment (see `05_challenges_and_pitfalls.md`) by shortening the effective decision horizon; adds architectural complexity and a hard interface boundary between the learned and classical-control parts of the system |

Independent of the discrete/continuous choice, **action-space size itself is its own lever** — a recent dedicated study (["Action Space Reduction Strategies for RL in AD"](https://arxiv.org/pdf/2507.05251)) found that larger, less-structured action spaces slow training regardless of whether they're discrete or continuous, meaning "how big/expressive should the action space be" is worth treating as a distinct question from "discrete vs. continuous."

Sources: [Highway-env Actions documentation](https://highway-env.farama.org/actions/index.html), [Action space reduction strategies (2507.05251)](https://arxiv.org/pdf/2507.05251), [Overview of Action Space for Deep RL](https://dl.acm.org/doi/fullHtml/10.1145/3508546.3508598).

## How these axes interact

None of these choices are independent in practice — they cluster into recognizable bundles (which is exactly why the platform survey in `01_simulation_platforms.md` reads like a menu of pre-packaged combinations rather than a fully general toolkit):

- **highway-env's default bundle**: small/synthetic topology + privileged Kinematics state + discrete meta-actions or continuous — optimized entirely for fast iteration, zero sim-to-real ambition.
- **MetaDrive's default bundle**: procedurally diverse topology + vector/LiDAR-distance state + continuous action + native safety-cost tracking — optimized for generalization and safe-RL research specifically.
- **CARLA's default bundle**: hand-built or OSM-derived topology + full sensor suite (camera/LiDAR/semantic/BEV all available) + continuous action + Traffic-Manager-scripted NPCs — optimized for photorealism and closed-loop leaderboard-style evaluation.
- **Waymax/GPUDrive's default bundle**: real logged topology + abstracted (non-photorealistic) state + continuous action + log-replay or learned-reactive NPCs — optimized for large-scale, realistic multi-agent throughput.

`06_recommendations.md` uses exactly this bundling logic to propose which combination of choices is actually practical at each project-scope tier.

# Simulation Platforms

A survey of the platforms actually used in autonomous-driving RL research and hobbyist projects, organized roughly from "heavyweight, photorealistic, research-grade" through "lightweight, 2D/kinematic, hobbyist-friendly" to "data-driven from real logs." For each: what it actually simulates, how RL-friendly it is, maturity, licensing, and realistic hardware/compute needs.

## Quick comparison table

| Platform | Fidelity | Traffic modeling | RL-friendliness | Maturity/community | License | Hardware to run |
|---|---|---|---|---|---|---|
| CARLA | Photorealistic 3D (Unreal Engine) | Traffic Manager autopilot NPCs, procedural OSM import | Good via `gym-carla`/`CARLA-GymDrive` wrappers, not native Gym | Very high — the de facto AD research standard, active leaderboard | MIT | 6–8GB+ dedicated GPU minimum, 20GB+ disk (binary), 130GB+ if building from source |
| SUMO + sumo-rl / Flow | Microscopic, no 3D rendering, kinematic/point-mass | IDM car-following, MOBIL lane-changing | Native Gymnasium + PettingZoo (sumo-rl) | High, long-running (SUMO since ~2001), huge traffic-eng community | Open source (EPL) | CPU-only, very lightweight, fast |
| MetaDrive | Lightweight 3D (Panda3D), procedurally generated | Configurable scripted traffic + native multi-agent | Native Gym-style API, built for RL research (safe-RL cost signal built in) | High in RL-safety research niche | Apache 2.0 | Runs well on a single consumer GPU or even CPU; thousands of FPS achievable |
| SMARTS | Mid 3D, scenario-driven | "Zoo" of diverse pretrained NPC policies + SUMO traffic engine option | Purpose-built for multi-agent RL, used in NeurIPS driving competitions | High in MARL-for-driving niche (Huawei Noah's Ark Lab, CoRL 2020 best system paper) | MIT | Moderate — CPU-heavy for SUMO backend, light GPU need |
| highway-env | 2D top-down kinematic bicycle model | Configurable IDM-based scripted vehicles | Native Gymnasium (Farama Foundation), extremely SB3-friendly | High — widely used as a teaching/prototyping environment | MIT | CPU-only, trains in minutes on a laptop |
| Duckietown / gym-duckietown | Real hardware + simulator, monocular camera only | Minimal (small-town-scale, stop signs, other Duckiebots) | Gym-compatible; strong sim2real focus | High in education (edX MOOC), moderate in research | Apache/GPL mix | Lightweight sim; real hardware optional (~hundreds of dollars) |
| gym-torcs / TORCS | 3D racing, historical | None (racing loop, not urban) | Custom OpenAI-gym-like wrapper | Legacy — largely superseded, historically important (used in original DDPG paper) | GPL | Moderate, runs on modest hardware |
| F1TENTH / RoboRacer | 1/10-scale physical hardware + Gym simulator | None (racing, not urban) | Native Gym environment, active offline-RL research | High in racing/education niche, real competitions | Open source (BSD-ish) | Lightweight sim; real hardware ~$3,000+ |
| nuPlan | Real logged city data (Boston/Pittsburgh/Las Vegas/Singapore), closed-loop | Log-replay, IDM reactive, or learned reactive (nuPlan-R) | Planning-benchmark oriented, not a from-scratch RL playground | High in industry/academic planning research | Non-commercial research license (large dataset) | Substantial — large dataset download, GPU for learned components |
| CommonRoad-RL | 2D/kinematic, configurable MDPs | Real highway datasets (highD) | Purpose-built to compare RL formulations (reward/action/state) side by side | Moderate, academic (TU Munich) | Open source | CPU-only, lightweight |
| gym-carla / CARLA-GymDrive / carla-gym | Wraps CARLA (see above) | Inherits CARLA's Traffic Manager | Adds proper Gym/Gymnasium/PettingZoo API on top of CARLA | Moderate — several competing community wrappers | MIT-style | Same as CARLA |
| Waymax | Data-driven from Waymo Open Motion Dataset, JAX-native | Log-replay or reactive sim agents | Provides dm-env/RL adapters; V-Max extends it for RL benchmarking | High and growing (Waymo Research + DeepMind) | Requires WOMD data license | GPU/TPU accelerated, no 3D rendering — very fast but needs JAX fluency |
| Nocturne | Lightweight 2D, C++, built on WOMD | Partial observability focus, occlusion modeling | Designed explicitly for multi-agent RL research | Moderate (Meta/FAIR), niche | Requires WOMD license | CPU, extremely fast (2000+ steps/sec) |
| GPUDrive | Data-driven from WOMD, GPU-native (Madrona engine) | Real logged agents + custom C++ behaviors | Gymnasium-compliant for PyTorch/JAX, tuned RL baselines included | New (2024), growing fast in large-scale MARL niche | Open source; requires WOMD data | GPU required, but extraordinarily fast (1M+ steps/sec, 30M steps/hour on consumer GPU) |
| AirSim / Project AirSim | Photorealistic 3D (Unreal Engine), drones + cars | Basic | CNTK/Python RL examples exist, less car-specific tooling than CARLA | Moderate, evolved into "Project AirSim" (2024, DARPA-backed) | MIT | Similar to CARLA (Unreal Engine-based) |
| LGSVL / SVL Simulator | High-fidelity 3D (Unity) | Integrated with Autoware/Apollo stacks | Was usable but no native RL-first tooling | **Discontinued since 2022** — no new releases/PRs | Apache 2.0, unmaintained | N/A — do not start a new project on this |
| Vista / Vista 2.0 | Data-driven photorealistic (built from real video/LiDAR/event-camera data, not a game engine) | Can incorporate simulated other vehicles | Research-grade, sim2real-focused | Moderate (MIT CSAIL), research-grade | Open source | Needs real sensor data to build environments; moderate GPU |
| BeamNG.tech | High-fidelity soft-body vehicle dynamics, camera/LiDAR/ultrasonic/IMU sensors | Basic | Python API (BeamNGpy); more ADAS/testing-oriented than RL-training-oriented | Moderate, niche in vehicle-dynamics/testing research | BeamNG.tech requires a research license (not free like the BeamNG.drive game) | Moderate-high GPU (game-engine physics) |
| Donkeycar / gym-donkeycar | Unity-based hobbyist simulator, monocular camera | None (track-based) | OpenAI Gym wrapper, SB3-compatible | High in the DIY/hobbyist RC-car racing community (DIYRobocars) | Open source | Lightweight; real hardware ~$250+ |
| CarRacing-v2 (Gymnasium/Box2D) | Minimal top-down pixel racing | None | Native Gymnasium, the simplest possible sanity-check env | Very high as a toy benchmark (used in the original 2018 "World Models" paper) | MIT | Trivial — CPU, seconds per episode |
| Apollo / ApolloRL | Full industrial AV stack + RL platform | Real-world-scenario replay, 20–30s scenario slices | Purpose-built distributed RL platform, but tightly coupled to Apollo's production stack | High as an industrial reference, low as a hobbyist starting point | Apache 2.0 | Heavy — production-grade infrastructure |

## Detailed notes

### CARLA
The de facto standard for photorealistic, sensor-rich AD research. Built on Unreal Engine, it provides RGB/depth/semantic-segmentation cameras, LiDAR, radar, GPS/IMU, and a scripted "Traffic Manager" that runs autopilot NPC vehicles. Since version 0.9.15, CARLA ships a **Digital Twin Tool** that procedurally generates 3D maps directly from real-world road networks pulled from OpenStreetMap, augmented with OpenDRIVE for lane-level topology. Native gym support doesn't exist, so the community has produced several wrappers: [`gym-carla`](https://github.com/cjy1992/gym-carla) (front-camera + LiDAR + BEV semantic mask observations, configurable weighted reward combining speed with collision/out-of-lane/steering penalties), [`CARLA-GymDrive`](https://www.sciencedirect.com/science/article/pii/S235271102400325X) (a more modern Gymnasium-style wrapper), and [`carla-gym`](https://github.com/johnMinelli/carla-gym) (adds PettingZoo-compatible multi-agent support). The official **CARLA Leaderboard** (v1.0 and v2.0) is the community's standard closed-loop evaluation harness, with v2.0 (Nov 2022) adding 39 curated real-world corner cases on top of v1.0's 10. Hardware: at least a 6GB GPU (8GB recommended), ~20GB disk for the pre-built binary or 130GB+ if building from source with Unreal Engine. Training runs reported in the literature range from ~22 hours (simpler experiments, RTX 2080) to multiple consecutive days for ~600k iterations. One review found **>80% of CARLA-based RL studies still use plain model-free methods (DQN/PPO/SAC)**.

Sources: [Comprehensive Review of RL for AD in CARLA](https://arxiv.org/abs/2509.08221), [CARLA 0.9.15 release notes](https://carla.org/2023/11/10/release-0.9.15/), [gym-carla](https://github.com/cjy1992/gym-carla), [CARLA build FAQ](https://carla.readthedocs.io/en/latest/build_faq/), [CARLA Leaderboard evaluation criteria](https://leaderboard.carla.org/evaluation_v2_0/).

### SUMO + sumo-rl / Flow
SUMO (Simulation of Urban Mobility) is a mature, open-source **microscopic traffic simulator** — it models individual vehicle trajectories precisely but has no 3D rendering or camera-level sensing at all; vehicles are point/kinematic objects following the **Intelligent Driver Model (IDM)** for car-following and MOBIL-style rules for lane changes. [`sumo-rl`](https://github.com/LucasAlegre/sumo-rl) wraps it with a clean Gymnasium/PettingZoo interface, but its primary use case is **traffic signal control** (an RL agent controls light phases to minimize cumulative vehicle waiting time), not driving a single ego car through traffic. [Flow](https://github.com/flow-project/flow) (UC Berkeley) is a related, older framework integrating SUMO with RL libraries specifically for **mixed-autonomy traffic control** (a fraction of vehicles are RL-controlled AVs influencing overall traffic flow, e.g. smoothing stop-and-go waves on a ring road), and can import real road networks from OpenStreetMap. Both are extremely lightweight (CPU-only, real-time-or-faster). If your interest is specifically "train an RL policy that drives one car around a city with sensors," SUMO/Flow are the wrong primary tool — but they are the right tool if part of your interest shifts toward traffic-signal control or fleet-level/mixed-autonomy effects.

Sources: [sumo-rl GitHub](https://github.com/LucasAlegre/sumo-rl), [Flow: A Modular Learning Framework for Mixed Autonomy Traffic](https://arxiv.org/pdf/1710.05465).

### MetaDrive
Purpose-built for **generalizable RL research**, and arguably the best-fit "middle ground" platform for a solo project that wants more than 2D toy environments but doesn't want CARLA's weight. Built on the lightweight Panda3D engine, MetaDrive's headline feature is **procedural composition**: it generates an effectively unlimited number of diverse road-network "blocks" (intersections, ramps, roundabouts) that can be strung together into unique towns, and can also import real scenario data (via the related [ScenarioNet](https://github.com/metadriverse/scenarionet) project, compatible with nuPlan/Waymo data). It has native **single-agent and multi-agent** modes, and a first-class **safe-RL cost signal** baked in (`cost = 1` on collision or going off-road, tracked separately from reward) — this made it the go-to environment for a large chunk of the safe-RL-for-driving literature (e.g. SafeRL-Kit, various constrained-RL papers). Default observation is a compact vector (ego state + navigation info + 2D-LiDAR-style distance readings to nearby objects, e.g. a 30-dimensional distance vector out of a 49-dim total observation), though first-person image observations are also supported. Action space is continuous (steering, acceleration). Runs comfortably on a single consumer GPU or even CPU, at very high step-rates.

Sources: [MetaDrive paper (arXiv 2109.12674)](https://arxiv.org/abs/2109.12674), [MetaDrive project site](https://metadriverse.github.io/metadrive/), [MetaDrive GitHub](https://github.com/decisionforce/metadrive).

### SMARTS
Developed by Huawei Noah's Ark Lab with Shanghai Jiao Tong University and UCL (CoRL 2020 Best System Paper), SMARTS is specifically oriented toward **multi-agent** realism: it maintains a "zoo" of diverse pretrained NPC behavior policies (rather than either pure hand-scripting or pure self-play) to create richer, more varied traffic interactions, and has been the basis of NeurIPS "Driving SMARTS" competitions. It can use SUMO as a traffic backend. Good fit if your central research interest is specifically the multi-agent interaction problem (other cars as adaptive/diverse agents) rather than single-ego sensor-to-control learning.

Sources: [SMARTS GitHub](https://github.com/huawei-noah/SMARTS), [SMARTS paper (arXiv 2010.09776)](https://arxiv.org/pdf/2010.09776).

### highway-env
A **Farama Foundation** project (the org that maintains Gymnasium itself), highway-env is purely 2D/top-down and kinematic — no 3D rendering, no camera realism — but it is the single easiest platform to get an RL loop running end-to-end on a laptop. It ships 10 built-in scenario families (highway, intersection, exit, lane-keeping/racetrack, merge, parking, roundabout, two-way, u-turn), each configurable. Three action-space options are supported natively: **Discrete Action**, **Discrete Meta-Action** (`{LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER}` — genuinely maneuver-level, letting the environment itself handle low-level kinematics), and **Continuous Action**. Observation options include a **Kinematics** vector (a V×F array of nearby-vehicle features like position/velocity — e.g. a 5×5 array for 5 nearby vehicles), BEV-style occupancy grid, grayscale/RGB rendering, and time-to-collision encodings. It registers itself automatically with Gymnasium on import and works out of the box with Stable-Baselines3. This is the best "does my RL pipeline even work" first environment for almost any driving-RL project.

Sources: [highway-env GitHub](https://github.com/Farama-Foundation/HighwayEnv), [highway-env documentation](https://highway-env.farama.org/), [highway-env Actions docs](https://highway-env.farama.org/actions/index.html).

### Duckietown / gym-duckietown
An education-first platform (the first hardware-based MOOC in AI/robotics, on edX) built around small, cheap three-wheeled "Duckiebot" robots with a **single forward-facing monocular camera** as their only sensor, operating in small physical or simulated "Duckietowns" with lane markings, stop signs, intersections, and other Duckiebots. The simulator (`gym-duckietown`, evolving into "Duckiematrix") is Gym-compatible and has published results training DDPG/TD3/SAC agents for steering control directly from camera pixels, with a strong and explicit focus on **sim2real transfer** (there's a whole "AI Driving Olympics" competition series built around exactly this). This is the platform to consider if the appeal of the SO-101 project's sim-to-real, hands-on-hardware character is something you'd like to carry into this new project too — it's the only platform on this list combining "real cheap physical car" with "genuine small-town road network" out of the box.

Sources: [gym-duckietown GitHub](https://github.com/duckietown/gym-duckietown), [Duckietown platform page](https://duckietown.com/platform/), [Real-Time RL in Duckiematrix](https://duckietown.com/real-time-reinforcement-learning-in-duckiematrix/).

### TORCS / gym-torcs
Historically important (the original 2015 DDPG paper, Lillicrap et al., used TORCS as its driving benchmark) but now largely of historical interest — it's a **racing** simulator (89 sensors, no intersections/traffic/pedestrians) and the community has mostly moved on to CARLA or F1TENTH for anything urban. [`gym-torcs`](https://github.com/ugo-nama-kun/gym_torcs) provides an OpenAI-gym-like Python wrapper.

Sources: [gym-torcs GitHub](https://github.com/ugo-nama-kun/gym_torcs).

### F1TENTH / RoboRacer
A **1/10-scale physical-hardware racing platform** with a matching Gym simulator ("F1TENTH Gym," maps of 20+ real race tracks scaled down). Recently rebranded "RoboRacer." Like Duckietown, this is a racing (not urban traffic) context, but with active recent research specifically applying **offline RL** methods to real-hardware racing. Relevant if physical deployment on real (if small-scale) hardware is an eventual goal, but racing-specific, not city-driving-specific.

Sources: [F1TENTH paper](https://proceedings.mlr.press/v123/o-kelly20a.html), [RoboRacer GitHub org](https://github.com/f1tenth), [F1tenth Autonomous Racing With Offline RL (arXiv 2408.04198)](https://arxiv.org/abs/2408.04198).

### nuPlan
Not really a simulator you'd use to train an RL agent from scratch — it's a **closed-loop planning benchmark and dataset**, built from 1,500 hours of real logged human driving across Boston, Pittsburgh, Las Vegas, and Singapore. Its core value is the **evaluation protocol**: it defines three simulation modes — open-loop (just check predicted vs. logged trajectory), closed-loop non-reactive (integrate your planner's trajectory via a bicycle model, but other agents just replay their logged trajectories regardless of what you do), and closed-loop reactive (other agents respond to you, historically via IDM; the newer **nuPlan-R** extension replaces IDM with learned diffusion-based reactive agents for more realistic interaction). Best fit if your goal shifts toward benchmarking a trained planner against real-city data rather than training end-to-end from pixels/sensors.

Sources: [nuPlan paper](https://arxiv.org/abs/2106.11810), [nuPlan-R](https://arxiv.org/abs/2511.10403).

### CommonRoad-RL
An academic (TU Munich) toolbox whose specific selling point is letting you **hold everything else constant and vary just the reward function, action space, or state representation** to compare their effects — addressing the problem that most driving-RL papers each invent their own bespoke MDP formulation and are never compared apples-to-apples. 2D/kinematic, uses real highway datasets (highD). Useful less as a "platform to build your project on" and more as a methodological example of how to run controlled reward/action-space ablations, which is directly relevant to the design-space tradeoffs in `03_environment_design_space.md`.

Sources: [CommonRoad-RL paper](https://mediatum.ub.tum.de/doc/1616584/pbazfbwsvmisz7qko1odv8d8f.pdf), [CommonRoad project](https://commonroad.in.tum.de/).

### Waymax, Nocturne, GPUDrive — the "data-driven, hardware-accelerated" family
These three share a philosophy: instead of hand-building or procedurally generating a city, **simulate directly from real logged driving data** (overwhelmingly the Waymo Open Motion Dataset), at very high throughput, without photorealistic rendering (agents are typically bounding boxes/polylines, not textured 3D models).

- **Waymax** (Waymo Research + Google DeepMind): entirely JAX-based, GPU/TPU-accelerated, differentiable, provides `dm-env`-style RL adapters. The [V-Max](https://github.com/valeoai/V-Max) framework (Valeo) extends it into a fuller RL benchmarking toolkit with observation/reward function libraries and transformer-based encoders.
- **Nocturne** (Meta/FAIR): a lightweight 2D C++ simulator built on WOMD, explicitly designed to study **multi-agent coordination under partial observability** (vehicles occlude each other) rather than raw driving performance — runs at 2000+ steps/second.
- **GPUDrive**: the newest and fastest (2024), built on the Madrona game engine, reports **over 1 million simulation steps/second** and 30 million steps/hour on a single consumer GPU, with Gymnasium-compliant PyTorch/JAX bindings and configurable LiDAR/human-like-view-cone sensing.

All three require a Waymo Open Motion Dataset license/download and meaningfully more infrastructure sophistication (JAX, CUDA/Madrona ECS concepts) than CARLA/MetaDrive/highway-env — they're the right tool if you specifically want large-scale, realistic *multi-agent* traffic learned/evaluated against real data, and are willing to accept a steeper setup cost for that.

Sources: [Waymax paper](https://arxiv.org/abs/2310.08710), [Waymax GitHub](https://github.com/waymo-research/waymax), [Nocturne paper](https://arxiv.org/abs/2206.09889), [Nocturne GitHub](https://github.com/facebookresearch/nocturne), [GPUDrive paper](https://arxiv.org/abs/2408.01584 ), [V-Max GitHub](https://github.com/valeoai/V-Max).

### AirSim / Project AirSim
Microsoft's Unreal-Engine-based simulator, originally for both drones and cars, with RL examples (DQN via CNTK) in its docs. Less car-specific tooling and community momentum than CARLA at this point; evolved in 2024 into "Project AirSim," a DARPA-supported collaboration with IAMAI, released under MIT license. Consider only if a combined drone+car research angle specifically appeals to you.

Sources: [AirSim GitHub](https://github.com/microsoft/AirSim), [AirSim RL docs](https://github.com/microsoft/AirSim/blob/main/docs/reinforcement_learning.md).

### LGSVL / SVL Simulator — discontinued, avoid for new projects
LG's Unity-based simulator, historically an integration point for the Autoware and Baidu Apollo stacks. **No new releases, bugfixes, or PR reviews since 2022.** Mentioned here only so you don't waste time evaluating it as an option.

Sources: [LGSVL GitHub (obsolete)](https://github.com/lgsvl/simulator-2019.05-obsolete).

### Vista / Vista 2.0
MIT CSAIL's **data-driven photorealistic simulator** — rather than a game engine, it synthesizes photorealistic views (and LiDAR, and event-camera data) directly from real recorded driving footage, letting an agent "recover" from perturbed viewpoints (e.g., simulated near-crash) that were never actually driven in the source data. Genuinely different in kind from CARLA/MetaDrive: you need real sensor data to build a Vista environment in the first place, rather than authoring a map by hand. Research-grade, open source.

Sources: [MIT News: photorealistic simulator release](https://news.mit.edu/2022/researchers-release-open-source-photorealistic-simulator-autonomous-driving-0621).

### BeamNG.tech
Distinguished by genuinely high-fidelity **soft-body vehicle dynamics** (crash/suspension physics far beyond a bicycle model), with a sensor suite (camera, LiDAR, ultrasonic, IMU) and a Python API (`BeamNGpy`). Used more for ADAS/vehicle-dynamics validation and fuzz/verification testing of driving software than as a from-scratch RL training ground — worth knowing about if vehicle-dynamics realism specifically matters to you, but it's a niche choice for this project. Note the research edition (BeamNG.tech) requires a separate license from the consumer BeamNG.drive game.

Sources: [BeamNG.tech docs](https://documentation.beamng.com/beamng_tech/).

### Donkeycar / gym-donkeycar and CarRacing-v2
Two "smallest reasonable first step" options. `gym-donkeycar` wraps a Unity-based hobbyist track simulator in an OpenAI Gym interface, with an active real-world DIY racing community (DIYRobocars) if you ever want to build the $250-ish physical car. `CarRacing-v2` (built into Gymnasium/Box2D) is the absolute simplest top-down pixel-based racing toy, historically used as a testbed in foundational work like the 2018 "World Models" paper (Ha & Schmidhuber) — useful only as a first sanity check that your RL training loop works at all, not as a serious project environment (no traffic, no intersections).

Sources: [gym-donkeycar GitHub](https://github.com/tawnkramer/gym-donkeycar).

### Apollo / ApolloRL
Baidu's full industrial AV stack includes its own RL research platform, ApolloRL, which pretrains agents via supervised learning on expert driver logs and then RL-refines them on short (20–30 second) scenario slices. Useful as a case study in industrial practice (see `02_rl_algorithms.md`), not a realistic starting point for a solo hobbyist project given its coupling to Apollo's full production stack.

Sources: [ApolloRL paper (arXiv 2201.12609)](https://arxiv.org/pdf/2201.12609), [Apollo GitHub](https://github.com/ApolloAuto/apollo).

## Bottom line on this axis

There is a real fidelity/complexity ladder here, and it roughly is: **highway-env → MetaDrive → CARLA → (Waymax/GPUDrive/nuPlan, if working from real logged data instead of a hand-built world)**. See `06_recommendations.md` for how these map onto minimal/medium/full-city project tiers.

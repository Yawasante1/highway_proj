# Double DQN vs DQN for Highway Driving

### A controlled reinforcement learning experiment: does fixing Q-value overestimation make a better driver?

A team project by **Kwame Ofori-Gyau, Opoku Asante Emmanuel and Julius Aban Odai**.

On a motorway the costs are lopsided: driving slightly slower costs seconds, while hitting another car is catastrophic and irreversible. We trained agents in the [`highway-env`](https://github.com/Farama-Foundation/HighwayEnv) simulator and asked a specific question: does **Double DQN**, which is designed to reduce the upward bias in standard DQN's value estimates, actually produce better driving?

**Short answer:** the bias reduction is clearly visible in the value estimates, but we found **no detectable difference in driving performance** between the two algorithms. Both learned to drive far better than a random policy, and neither had converged by the end of training.

## Headline results

**1. Double DQN lowers the value estimates, as the theory predicts.** On a fixed set of 512 probe states, DQN's estimated value ended at 8.74 ± 0.18 against 6.71 ± 0.42 for Double DQN (p = 0.006). Comparing estimates with the return actually achieved, both algorithms overestimate, but Double DQN's gap is about 40% smaller (+2.62 vs +4.41; marginal at p = 0.050, so we call it suggestive, not established).

**2. The driving metrics show no significant difference.** Over 50 held-out evaluation episodes per policy (mean ± SD across 3 training seeds):

| Metric | Random | DQN | Double DQN | p |
|---|---|---|---|---|
| Cumulative reward | -8.71 | 7.96 ± 1.87 | 7.45 ± 1.24 | 0.72 |
| Collision rate | 0.98 | 0.54 ± 0.09 | 0.61 ± 0.04 | 0.30 |
| Mean speed (m/s) | 23.28 | 26.37 ± 0.30 | 26.79 ± 0.49 | 0.28 |
| Distance travelled (m) | 245 | 727 ± 53 | 688 ± 25 | 0.34 |
| Successful lane changes | 2.84 | 7.22 ± 0.78 | 6.21 ± 1.28 | 0.32 |
| Episode length (steps) | 10.5 | 27.3 ± 2.08 | 25.4 ± 0.45 | 0.26 |

Both learned agents roughly tripled the distance travelled and nearly halved the collision rate compared with the random policy. The gaps between the two algorithms are smaller than the variation between training runs of the same algorithm.

**3. The value estimates and the driving can both be true.** A greedy policy depends on the *ordering* of action values, not their size. If overestimation inflates all five action values similarly, the best action does not change. With only five actions and a reward dominated by speed, there is little room for the correction to reorder them.

## What makes the experiment careful

- **A controlled comparison.** DQN and Double DQN use configs that inherit from one shared base file and differ only in the algorithm name. Every hyperparameter, environment setting and reward weight is identical, so any difference can only come from the target computation.
- **A random-policy floor.** A collision rate alone says nothing, so a uniform-random policy runs through the same code path to show whether the agents learned anything.
- **Paired evaluation.** Episode `i` uses seed `1000 + i` for every policy, so all arms face identical traffic. Evaluation seeds are disjoint from training seeds.
- **The training run is the unit of replication.** Statistics use a Welch t-test on per-seed means. Our first analysis pooled all 150 episodes per algorithm and found two apparently significant differences, but that was wrong: episodes from one run share one trained network, so pooling inflates the sample size from 3 to 150. Re-running per training run removed both effects.
- **Reward components logged separately**, so we can say specifically what the reward design did, rather than report one number.
- **A tested pipeline.** A verification suite checks, among other things, that episode truncation is handled correctly and not treated as a terminal state.

## How the problem is set up

**State.** A 96-number vector: 4 stacked frames x 24 features. Each frame describes the ego vehicle and the 4 nearest vehicles (presence, position, velocity), with other vehicles expressed relative to the ego vehicle and all values scaled to [-1, 1]. Stacking frames lets the network see how positions and speeds change over time.

**Actions.** Five high-level commands: `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, `SLOWER`. The agent decides once per second, and the simulator's controller handles the steering. We did not mask impossible actions (for example, moving left from the leftmost lane). Instead we penalise them, so the agent learns the road's boundaries itself.

**Reward.** We replaced the environment's built-in reward with a weighted sum of six terms:

| Term | Meaning | Weight |
|---|---|---|
| Speed | Scaled between 20 and 30 m/s | 1.0 |
| Headway | Penalty for short time gap to the car ahead | 0.4 |
| Lane change | Small cost for each lane change, to stop weaving | 0.1 |
| Invalid lane change | Penalty for a lane change that cannot happen | 0.2 |
| Collision | Penalty on the crash step | 10.0 |
| Off-road | Penalty for leaving the road | 10.0 |

Headway uses time (gap divided by speed) instead of raw distance, because a 20 m gap is generous at 5 m/s and dangerous at 30 m/s.

**Termination vs truncation.** A crash or leaving the road ends the episode (the value is zero). Reaching the 40-step limit only stops the recording, so the value estimate must keep bootstrapping. Mixing these up would teach the agent that the world ends at 40 seconds.

**Discount factor.** 0.95, an effective horizon of about 20 seconds. That covers the timescales of a rear-end collision (2 to 5 s), a lane change (1 to 2 s) and an overtake (5 to 10 s).

**Training.** 3 seeds per algorithm, 60,000 steps each (about 1,500 episodes), CPU only, about 38 minutes per run. Network: two hidden layers of 256 units, Huber loss. Built on Stable-Baselines3, with `DoubleDQN` implemented as a subclass that overrides only the training step.

## What went wrong, and what we would change

We think the honest findings are as useful as the positive ones.

- **The reward design did not enforce safe following distances.** The headway penalty got *worse* with training (-1.58 for random, -2.97 for DQN, -2.67 for Double DQN), and the median minimum time headway was only about 0.08 s. The agents learned to drive fast and tailgate, paying the penalty to collect more speed reward. With a headway weight of 0.4 against a speed reward of up to 1.0, that trade was rational. Raising the headway weight, or tying headway to the speed term, would be the first change.
- **Neither algorithm converged.** Return was still rising and collision rate still falling at 60,000 steps, so these are partially trained agents and the comparison is of learning, not of final policies.
- **Traffic was reduced for compute reasons.** At the default of 50 vehicles the experiment would have taken about 45 hours. We simulated fewer vehicles at higher density. Comparisons within the study are valid, but absolute collision rates cannot be compared with results published at 50 vehicles.
- **A collision rate of 54% is nowhere near deployable.** A learned policy gives a statistical rate, not a worst-case guarantee, and real use would need a separately verified safety layer.
- **Simulation limits.** Other cars follow a rule-based driver model (IDM) and behave predictably, and the agent picks lane-level commands rather than controlling steering and throttle.
- **Low statistical power.** Three seeds and fifty episodes detect only large effects. The null result describes this experiment, not proof that no difference exists.

## Further work

In priority order:

1. Sweep `target_update_interval`. Double DQN and DQN are identical whenever the two networks are synced and diverge only as the online network drifts between syncs, so this most directly tests our explanation.
2. Train for longer, since neither algorithm converged.
3. Raise the headway weight or couple it to speed.
4. Replace the kinematic observation with an occupancy grid and a CNN.
5. Add more seeds first. Run-to-run spread, not training budget, was the real limit on what we could conclude.


## My contribution

I was responsible for three parts of the experiment:

- **Network architecture.** I set up the Q-network used by both agents (96 inputs, two hidden layers of 256 units, 5 Q-value outputs), sized for the 96-dimensional observation.
- **Training.** I ran the training pipeline for both algorithms across the three seeds, with hyperparameters frozen across all six runs.
- **Baseline control.** I managed the baselines, the unmodified Stable-Baselines3 DQN and the uniform-random policy, so that every arm went through the same code path and any difference could be traced to the target computation alone.

The MDP formulation, reward design, evaluation and write-up were shared team work.

## Credits and attribution

- DQN implementation: [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). `DoubleDQN` adapts the library's `DQN.train` method, and the point of change is noted in the source.
- Highway simulator and kinematic observation: [highway-env](https://github.com/Farama-Foundation/HighwayEnv). Environment API: [Gymnasium](https://gymnasium.farama.org/).
- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529-533.
- van Hasselt, H., Guez, A. and Silver, D. (2016). Deep reinforcement learning with double Q-learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 30(1).

## Team

- Kwame Ofori-Gyau
- Opoku Asante Emmanuel (vkpoasante@gmail.com)
- Julius Aban Odai



# Ideas

Things worth trying after the current changes have been trained on, one at a time.

## Exploration

- **Temporally extended epsilon-greedy** (Dabney et al. 2020): repeat each random action for a sampled duration, so exploration commits to a direction long enough to reach a side room or a different path.
- **NoisyNets or a curiosity bonus (RND)** if side rooms still never get found.

## Long horizon

At 15 decisions per game second, `gamma = 0.995` looks ~200 steps (~13 s) ahead. A level takes ~600 steps, so from its top the agent can't see that level's bonus, let alone a shop's hp paying off two levels later. The ideas below are ordered cheapest first.

- **Higher gamma with longer n-step returns**: `gamma = 0.998` (~500 steps, ~33 s) or `0.999` (~1000 steps, ~67 s) with `n_steps` 5-10. Config only, but Q values grow with the horizon, so watch loss and mean Q for divergence, and consider a slower `target_update_interval`.
- **Value function rescaling** (Pohlen et al. 2018, used in R2D2): train on `h(x) = sign(x)(sqrt(|x| + 1) - 1) + eps*x` of the target. It keeps `gamma >= 0.997` stable without clipping rewards, which makes it the usual companion to a higher gamma.
- **Distributional head** (QR-DQN or C51): learns the spread of returns, not just their mean, which tends to help when returns are long and noisy.
- **Discount by game time, not by step**: the upgrade menu and the transitions freeze the game, but each step there still costs a factor of gamma. Discounting by frames elapsed makes those steps free.
- **Separate upgrade picker**: picks are ~0.2% of the buffer and pay off levels later. If they stay tied to the slot rather than the upgrade even with the one-step menu, try a contextual bandit scoring each offer from the run state (hp, max hp, max ammo, owned upgrades, level), learned from how far the episode got after the pick, with UCB or epsilon exploration. Start with a table of mean outcome per upgrade, split by hp; it can be seeded from old `training.log` picks (~6k per 5M run). The env plays the menu, the DQN never sees it.
- **Shops stay with the DQN**: reaching one is movement, and buying hp pays now through `heal_reward`. If runs show shops entered but nothing bought, script the purchase like the upgrade menu. Unlike that menu, the shop cursor wraps around its four slots, and slot 0 (price 0) is the way back out.
- **PPO with GAE** (`gamma = 0.999`, `lambda = 0.95`): copes better with long horizons in practice, but wants several environments in parallel and there is one game process. This is a large change.

## Training

- **Save more than the best weights**: `best.pt` (current `model.pt`), `final.pt` at the end, and a weights-only checkpoint every 100k steps (~1.1 MB each) to compare versions with `downwell-play`.
- **Lightweight resume**: Weights, target network, optimizer, step and RNG state, without the ~400 MB replay buffer. Only if crashes keep happening; a resumed run starts with an empty buffer and behaves differently.
- **Pick the best model over several evaluation episodes**: One episode decides `model.pt` today, which is noisy.
- **CUDA graphs for acting and learning**: the network is small enough that python launching its kernels, not the gpu, is most of the cost. Measured on the 3080, a graph replay cut action selection from 0.76 to 0.20 ms and a gradient step from ~6 to ~2.5 ms, about 1 ms per env step in all. It needs fused capturable Adam, fixed input tensors copied into before each replay, and the warmup optimizer steps undone after capture so the first replay is the first real step.
- **Learn while the game simulates, via rtgym**: a step spends ~1.3 ms polling for the game's 4 frames. rtgym runs the environment step in the background, so the gradient step and replay sampling would happen while the game simulates instead of after it. Deferred because it is a larger change.

# Ideas

Things worth trying after the current changes have been trained on.

## Next, in order

1. **Motion history, fresh `full_run`** -- does hits per 1,000 px get below the ~0.4 floor all earlier runs hit around 2M steps?
2. **Longer horizon** -- `gamma = 0.998`, `n_steps = 5`, nothing else changed
   - 0.995 sees ~200 steps (~13 s), a level is ~600 steps
   - Q values grow ~2.5x, watch loss and mean Q
   - if it diverges, add value rescaling (Pohlen et al. 2018): train on `h(x) = sign(x)(sqrt(|x| + 1) - 1) + eps*x`

## Backlog

Roughly cheapest first.

- **Overlap learning with the game** -- split `step()` into send and wait, learn while the game simulates its 4 frames; profile the env step first to see how much of it is the game
- **Bigger batch, less often** -- 256 every 8 steps instead of 128 every 4, half the kernel launches at the same replay ratio; changes learning, needs its own run
- **Pick the best model over several evaluations** -- one episode decides `best.pt` today
- **Discount by game time** -- the menu and transitions freeze the game but still cost a factor of gamma per step
- **Temporally extended epsilon** (Dabney et al. 2020) -- repeat each random action for a sampled duration
- **NoisyNets or RND** -- if side rooms or shops still go unfound
- **Distributional head** (QR-DQN or C51) -- long, noisy returns
- **Lightweight resume** -- weights, target, optimizer, step and RNG without the buffer; only if crashes keep happening
- **CUDA graphs** -- launches are still most of the learn step; graphs measured ~2.5 ms per update against ~4.2 now, but need capturable Adam, fixed input tensors and undoing the warmup steps
- **PPO with GAE** -- better with long horizons, but wants parallel environments and there is one game process; a large change

## Difficult, to assess later

1. **Upgrade picker**
   - baseline first: fixed ranking, apple first
   - from 4,604 near-random picks: apple +0.36 ± 0.04 levels over the average pick on the same level, everything else within ±0.1, safety jetpack last at -0.12
   - then a learned one, only worth it if it beats the ranking through context (apple when hurt, gem upgrades with gems in hand)
   - small MLP (2x64, ~10k params) scoring one option at a time from run state + option, target = main reward from the pick to the episode end, epsilon exploration, trained per episode, saved with the weights
   - env plays the menu through it, the DQN never sees it
   - log the full context (hp, gems, owned) at each pick first
2. **Shops**
   - export shop item ids (`layout.py`, re-inject, `gen_ptrs.py`) -- only prices today
   - same picker with a "leave" option, purchase scripted like the upgrade menu
   - cursor wraps over 4 slots, slot 0 (price 0) is the way out
   - entered in 0.6-3.6% of episodes and nothing ever bought, so entering may need a small reward while holding gems

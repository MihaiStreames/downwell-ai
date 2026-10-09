# Configs

Train with `uv run downwell-train configs/default.toml`. There are no defaults in code: every field below must be set, and a missing or unknown one fails before training starts. `configs/long_run.toml` is the same setup over 2.5M steps and `configs/full_run.toml` over 5M, all three decaying exploration over the same 400k steps: how long the agent needs random actions depends on the game, not on the run's length. Copy one to try other values. The `--headless/--rendered` flag overrides the file.

## Speed and episodes

| field                   | default | meaning                                                                                                                         |
| ----------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `headless`              | true    | skip drawing, vsync and audio while training; `false` to watch the agent play                                                   |
| `frame_skip`            | 4       | game frames per decision, 15 decisions per game second                                                                          |
| `max_episode_steps`     | 30000   | about 33 game minutes before the episode is truncated                                                                           |
| `stall_timeout_seconds` | 30.0    | game seconds without new depth before the episode is truncated, paused in side rooms, shops and the upgrade menu; 0 disables it |

The game runs in lockstep with the agent, as fast as the agent can step it, so speed never changes what the agent sees.

## Learning

| field                    | default | meaning                                                             |
| ------------------------ | ------- | ------------------------------------------------------------------- |
| `learning_rate`          | 1e-4    | Adam step size                                                      |
| `gamma`                  | 0.995   | discount, a horizon of ~200 steps or ~13 game seconds               |
| `n_steps`                | 3       | rewards summed into each stored transition, 1 is plain one-step dqn |
| `batch_size`             | 128     | transitions per gradient step                                       |
| `buf_cap`                | 500000  | replay buffer size, ~3.5 KB per transition so ~1.8 GB of RAM        |
| `learning_starts`        | 20000   | random steps collected before the first gradient step               |
| `train_frequency`        | 4       | env steps per gradient step                                         |
| `target_update_interval` | 4000    | env steps between target network syncs                              |
| `max_grad_norm`          | 10.0    | gradient clipping                                                   |

## Exploration and evaluation

| field                    | default | meaning                                                                                                |
| ------------------------ | ------- | ------------------------------------------------------------------------------------------------------ |
| `eps_start`              | 1.0     | random action probability at the start                                                                 |
| `eps_end`                | 0.05    | floor it decays to                                                                                     |
| `resume_eps_start`       | 0.3     | where epsilon starts instead when continuing with `--resume`, the weights already play                 |
| `exploration_steps`      | 400000  | steps the linear decay takes, the same whatever `total_steps` is                                       |
| `eval_interval_episodes` | 25      | one evaluation episode every n episodes, saved as a gif to spot reward hacking                         |
| `eval_eps`               | 0.02    | random action rate in evaluation episodes, breaks deterministic loops like jumping in place on a ledge |

## Reward

An episode is one run: only death ends it. Clearing a level pays a bonus and play goes on, with depth tracking restarting at the new level. Getting stuck is handled by `stall_timeout_seconds`, not a penalty.

| field           | default | meaning                                                                     |
| --------------- | ------- | --------------------------------------------------------------------------- |
| `depth_reward`  | 0.003   | per px of new best depth on the current level, ~10 for a ~3600 px level     |
| `gem_reward`    | 0.02    | per gem picked up, spending them costs nothing                              |
| `hit_penalty`   | -2.5    | per non-fatal hit                                                           |
| `heal_reward`   | 2.5     | per hp gained, from shops, upgrades or pickups; mirrors `hit_penalty`       |
| `ammo_reward`   | 1.0     | per point of max ammo gained, losing it costs nothing                       |
| `death_penalty` | -5.0    | on death, ends the episode                                                  |
| `level_reward`  | 10.0    | per level cleared                                                           |
| `kill_reward`   | 0.5     | per kill, counted as the combo's increases (worth ~25 gems)                 |
| `combo_reward`  | 0.05    | times N² when a combo of N ends alive, so long chains pay off               |
| `scale`         | 1.0     | multiplies every term, used instead of clipping so their ratios stay intact |

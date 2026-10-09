// TODO: the whole paper is a draft; sections hold notes to expand, not finished prose

#set document(title: "Learning to play Downwell from game memory", author: "Mihai")
#set page(paper: "a4", margin: 2.5cm, numbering: "1")
#set text(size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.1")

#let todo(body) = block(
  width: 100%,
  inset: 8pt,
  radius: 4pt,
  stroke: orange,
  fill: orange.lighten(85%),
  [*TODO:* #body],
)

#align(center)[
  #text(size: 17pt, weight: "bold")[Learning to play Downwell from game memory]

  Mihai

  #text(fill: gray)[Draft]
]

#heading(numbering: none)[Abstract]

#todo[one paragraph: a double DQN agent plays the live game in lockstep through its memory, what it reaches, what limits it]

= Introduction

#todo[why Downwell: a fast vertical roguelite with stomping, ammo, combos and run-long upgrade choices; why read memory instead of pixels]

= Environment

== Stepping

The game runs in lockstep with the agent. A patch injected into `data.win` exports a snapshot of the game state into a buffer each frame and reads the agent's buttons back from memory. Each step holds the chosen buttons for `frame_skip` frames, after which the game blocks until the next step, so no keyboard input or screen capture is involved and the speed of training never changes what the agent sees.

#todo[the patch: `layout.py` as the single source of truth, generated GML, pointer resolution with `gen_ptrs.py`; measured cost per step]

== Episodes

An episode is one run, starting fresh at 1-1. Death terminates it; the step limit or 30 game seconds without new depth truncate it. The stall clock pauses in side rooms (shops included) and in the upgrade menu. Clearing a level pays a bonus and play goes on.

== Observation

A multi-channel grid over exactly what the game shows, rasterized from every active instance's collision box: solid terrain, platforms, spikes, enemies split into stompable now, stompable by landing and hurting, hazards, gems, pickups, the player's shots, the player, water and side rooms. The moving channels (both stompable kinds, hurting and hazards) are repeated from the previous two steps, newest first, so the network sees where things move and not only where they are; a step is 4 game frames and a cell 8 px, so a slow enemy needs two past steps to cross a cell. Alongside the grid, a vector of player motion, combat and progress, the owned upgrades, and the level-end menu and the shop when open.

#todo[grid size and cell size, the vector layout, a figure of one rendered observation]

== Actions

None, jump, left, right, left and jump, right and jump. In the level-end upgrade menu, action $n$ picks offer $n$ (wrapping past the last offer) in a single step. In the shop the same buttons move the cursor and buy.

== Stomping

Which enemies are safe to land on comes from each enemy's own collision code, kept in `downwell.toml` and evaluated live each frame; some are only stompable while the player falls, some only in a given state.

== Reward

New best depth, paid in proportion so bouncing earns nothing; gems picked up; kills; combos, paying the length squared when one ends; healing and max ammo gained; penalties for hits and death; a bonus per level cleared. The coefficients live in the config's `[reward]` table.

#todo[table of coefficients and the reasoning behind each; reward components over training]

= Learning

Double DQN with a hard-synced target network, uniform replay and 3-step returns, so hits and level clears reach the actions behind them sooner. Epsilon decays linearly over a fixed 400k steps whatever the run length.

#todo[network architecture, hyperparameter table, training throughput]

= Experiments

#todo[runs compared: 5M baseline, 2.5M with heal and ammo rewards and the paused stall clock, the one-step upgrade menu; metrics: level reached, depth, hits, detours]

= Results

#todo[plateau around 2M steps; side rooms learned once the stall clock paused there; upgrade picks tied to the menu slot rather than the upgrade; shops passed but never entered]

= Limitations and future work

#todo[horizon of gamma 0.995 against a level of ~600 steps, shops, a separate upgrade picker; see `docs/ideas.md`]

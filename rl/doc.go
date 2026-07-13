// Package rl provides the reinforcement-learning building blocks: episodic
// environments, return/advantage estimation, and two canonical agents.
//
// # For AI practitioners
//
// The pieces are:
//
//   - [Env] — a minimal episodic environment interface (Reset/Step/NumActions/
//     ObsDim), with [Chain] as a small, exactly-analyzable N-state MDP whose
//     optimal policy is "always right".
//   - [DiscountedReturns] — G_t = Σ γ^k r_{t+k}, the Monte-Carlo target.
//   - [GAE] — Generalized Advantage Estimation (Schulman et al. 2016,
//     arXiv:1506.02438): turns rewards and value estimates into the advantages a
//     clipped policy-gradient loss trains on. λ=0 is one-step TD, λ=1 is
//     Monte-Carlo; it produces exactly the advantages nn.PPOClipLoss consumes.
//   - [Reinforce] — the REINFORCE policy-gradient agent (Williams 1992) with a
//     cross-episode EMA baseline.
//   - [DQN] — a minimal deep Q-learner (Mnih et al. 2015): replay buffer, target
//     network, ε-greedy decay, and the replaced-entry MSE update.
//
// This is the same policy-optimization machinery that underlies RLHF for language
// models: PPO with GAE advantages is the standard RLHF optimizer, and the
// nn package's alignment losses (PPO/DPO/KTO/IPO) plug into the same training
// pattern. Each agent is deterministic given a seed; the exactly-computable pieces
// (returns, GAE, the chain dynamics) are verified against hand-derived values, and
// GAE is confirmed against its defining paper (§V16, §R).
//
// # For everyone else
//
// Reinforcement learning is learning by trial and error: an agent takes actions in
// an environment, receives rewards, and gradually shifts toward the actions that
// pay off. This package contains a tiny practice environment (a line of squares
// where stepping to the right end wins), the bookkeeping that decides how much
// credit each action deserves for a later reward, and two classic learning
// algorithms — one that directly nudges the action probabilities (REINFORCE) and
// one that learns the value of each move and acts greedily (DQN). The same credit-
// assignment math (GAE + PPO) is what is used to fine-tune large language models
// from human feedback.
//
// The runnable Example functions below compute returns and advantages by hand and
// train a DQN agent to solve the chain.
//
// Further reading: Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed., MIT Press 2018) — the canonical textbook every algorithm in this package cites into.
//
// In plain terms: reinforcement learning — programs that learn by trying actions in an environment and preferring what earned reward — with the canonical small algorithms and environments to study them.
package rl

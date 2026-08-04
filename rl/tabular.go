package rl

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// Tabular temporal-difference (TD) control: Q-learning and SARSA.
//
// # For AI practitioners
//
// These are the two canonical value-based TD-control algorithms from Sutton &
// Barto ("Reinforcement Learning: An Introduction", 2nd ed., MIT Press 2018,
// chapter 6). Both maintain an explicit Q-table Q[state][action] rather than a
// function approximator, so they apply to environments with a small discrete
// state space. Each step nudges one Q entry toward a bootstrapped TD target:
//
//	Q-learning (off-policy):  Q(s,a) ← Q(s,a) + α·[r + γ·max_a' Q(s',a') − Q(s,a)]
//	SARSA      (on-policy):   Q(s,a) ← Q(s,a) + α·[r + γ·Q(s',a')       − Q(s,a)]
//
// The only difference is the bootstrap: Q-learning takes the greedy max over the
// next state (learning the value of the optimal policy regardless of how it
// behaves), while SARSA uses Q(s',a') for the action a' actually taken next by the
// ε-greedy behaviour policy (learning the value of the policy it follows, including
// its exploration). On a task where the optimal path is risky under exploration
// (a cliff), Q-learning learns the optimal risky policy while SARSA learns a safer
// one — the classic Sutton-Barto cliff-walking result.
//
// Both agents map an observation to a discrete state index. [Chain] and other
// one-hot environments are handled by the default argmax indexer; supply
// [WithStateIndexFunc] for a custom mapping (e.g. tile coding, or a hand-built
// grid encoder). Determinism is fixed by the seed. α (learning rate), γ (discount)
// and ε (exploration) must lie in [0,1]; ε can decay per episode via
// [WithEpsilonDecay].
//
// # For everyone else
//
// These are the textbook "learn a table of move values by trial and error"
// algorithms. The agent keeps a score for every (situation, move) pair, tries
// moves, and each time it sees a reward it slides the score for the move it made a
// little toward "reward now plus the best score reachable next". Do that enough and
// the table tells you the best move in every situation. Q-learning is the
// optimistic version (it assumes it will always pick the best next move); SARSA is
// the cautious version (it accounts for the random exploring it actually does),
// which is why SARSA prefers safer routes near a cliff.
//
// Further reading: Watkins, "Learning from Delayed Rewards" (PhD thesis, 1989) for
// Q-learning; Rummery & Niranjan, "On-Line Q-Learning Using Connectionist Systems"
// (CUED tech report, 1994) for SARSA; Sutton & Barto (2nd ed., 2018) chapter 6.

// TabularOption configures a tabular TD-control agent ([QLearning] or [SARSA]).
// Options are supplied variadically to the constructors.
type TabularOption func(*tabularAgent)

// WithStateIndexFunc supplies a custom observation→state-index mapping for
// environments whose observations are not one-hot (the default indexer requires a
// one-hot observation and returns its argmax). The function receives a raw
// observation and returns a state index in [0, numStates) or an error; a returned
// error aborts the episode. Use this for tile coding, discretised continuous
// observations, or a hand-built grid encoder.
func WithStateIndexFunc(f func(obs []float64) (int, error)) TabularOption {
	return func(a *tabularAgent) { a.stateIndex = f }
}

// WithNumStates overrides the number of Q-table rows (states). By default the
// agent infers it from the environment's ObsDim (one state per one-hot slot); set
// this when a custom state-index function maps into a different-sized space.
func WithNumStates(n int) TabularOption {
	return func(a *tabularAgent) { a.numStates = n }
}

// WithEpsilonDecay enables per-episode multiplicative ε decay: after each episode
// ε ← max(εMin, ε·decay). decay must lie in (0,1] and εMin in [0,1]; the default
// (no option) keeps ε constant. This anneals exploration toward exploitation as
// learning progresses.
func WithEpsilonDecay(decay, epsMin float64) TabularOption {
	return func(a *tabularAgent) { a.epsDecay, a.epsMin = decay, epsMin }
}

// tabularAgent holds the state shared by [QLearning] and [SARSA]: the Q-table, the
// hyper-parameters, the observation→state indexer, and the RNG. The two agents
// differ only in their Episode TD update (max vs. actual next action), so they
// embed this and share GreedyAction/Policy/QTable and ε-greedy behaviour.
type tabularAgent struct {
	Q       [][]float64 // Q[state][action] action-value table (live; see QTable)
	Alpha   float64     // learning rate α ∈ [0,1]
	Gamma   float64     // discount factor γ ∈ [0,1]
	Epsilon float64     // current ε-greedy exploration rate ∈ [0,1]

	numStates  int
	numActions int
	stateIndex func(obs []float64) (int, error)
	epsDecay   float64 // per-episode multiplicative ε decay (1 = no decay)
	epsMin     float64 // ε floor under decay
	rng        *rand.Rand
}

// newTabularAgent builds the shared state and applies options, validating ranges.
func newTabularAgent(env Env, alpha, gamma, epsilon float64, seed uint64, opts ...TabularOption) tabularAgent {
	if alpha < 0 || alpha > 1 {
		panic(fmt.Sprintf("rl: tabular alpha (learning rate) must be in [0,1], got %v", alpha))
	}
	if gamma < 0 || gamma > 1 {
		panic(fmt.Sprintf("rl: tabular gamma (discount) must be in [0,1], got %v", gamma))
	}
	if epsilon < 0 || epsilon > 1 {
		panic(fmt.Sprintf("rl: tabular epsilon (exploration) must be in [0,1], got %v", epsilon))
	}
	a := tabularAgent{
		Alpha: alpha, Gamma: gamma, Epsilon: epsilon,
		numStates:  env.ObsDim(),
		numActions: env.NumActions(),
		epsDecay:   1.0, // no decay by default
		epsMin:     epsilon,
		rng:        rand.New(rand.NewPCG(seed, 0x7ab)),
	}
	for _, opt := range opts {
		opt(&a)
	}
	if a.stateIndex == nil {
		a.stateIndex = oneHotStateIndex(a.numStates)
	}
	if a.epsDecay <= 0 || a.epsDecay > 1 {
		panic(fmt.Sprintf("rl: tabular epsilon decay must be in (0,1], got %v", a.epsDecay))
	}
	if a.epsMin < 0 || a.epsMin > 1 {
		panic(fmt.Sprintf("rl: tabular epsilon floor must be in [0,1], got %v", a.epsMin))
	}
	a.Q = make([][]float64, a.numStates)
	for s := range a.Q {
		a.Q[s] = make([]float64, a.numActions)
	}
	return a
}

// oneHotStateIndex returns the default indexer: argmax of a one-hot observation,
// with validation that the observation really is one-hot over numStates slots.
func oneHotStateIndex(numStates int) func(obs []float64) (int, error) {
	return func(obs []float64) (int, error) {
		if len(obs) != numStates {
			return 0, fmt.Errorf("rl: tabular observation has length %d, want %d (one state per one-hot slot); supply WithStateIndexFunc for non-one-hot observations", len(obs), numStates)
		}
		idx := -1
		for i, v := range obs {
			switch {
			case math.Abs(v-1) < 1e-9:
				if idx >= 0 {
					return 0, fmt.Errorf("rl: tabular observation is not one-hot (multiple 1s at %d and %d); supply WithStateIndexFunc for non-one-hot observations", idx, i)
				}
				idx = i
			case math.Abs(v) > 1e-9:
				return 0, fmt.Errorf("rl: tabular observation is not one-hot (value %v at index %d is neither 0 nor 1); supply WithStateIndexFunc for non-one-hot observations", v, i)
			}
		}
		if idx < 0 {
			return 0, fmt.Errorf("rl: tabular observation is not one-hot (no 1 found); supply WithStateIndexFunc for non-one-hot observations")
		}
		return idx, nil
	}
}

// greedyAction returns argmax_a Q[s][a], breaking ties toward the lowest index for
// determinism.
func (a *tabularAgent) greedyAction(s int) int {
	best, bv := 0, a.Q[s][0]
	//perfscan:ignore PS3068 argmax over small numActions, low trip count
	for act := 1; act < a.numActions; act++ {
		if v := a.Q[s][act]; v > bv {
			bv, best = v, act
		}
	}
	return best
}

// maxQ returns max_a Q[s][a], the greedy value of state s.
func (a *tabularAgent) maxQ(s int) float64 {
	m := a.Q[s][0]
	for act := 1; act < a.numActions; act++ {
		if v := a.Q[s][act]; v > m {
			m = v
		}
	}
	return m
}

// epsilonGreedy picks a uniform-random action with probability ε and the greedy
// action otherwise.
func (a *tabularAgent) epsilonGreedy(s int) int {
	if a.rng.Float64() < a.Epsilon {
		return a.rng.IntN(a.numActions)
	}
	return a.greedyAction(s)
}

// decayEpsilon applies one step of the configured per-episode ε decay.
func (a *tabularAgent) decayEpsilon() {
	if a.epsDecay < 1 && a.Epsilon > a.epsMin {
		a.Epsilon *= a.epsDecay
		if a.Epsilon < a.epsMin {
			a.Epsilon = a.epsMin
		}
	}
}

// GreedyAction returns the learned greedy action argmax_a Q(stateIdx, a) — the
// extracted deterministic policy at that state. Ties break toward the lowest
// action index. stateIdx must lie in [0, numStates).
func (a *tabularAgent) GreedyAction(stateIdx int) int { return a.greedyAction(stateIdx) }

// Policy returns the greedy action for every state, policy[s] = argmax_a Q(s,a) —
// the full deterministic policy extracted from the learned Q-table.
func (a *tabularAgent) Policy() []int {
	p := make([]int, a.numStates)
	//perfscan:ignore PS3065 one-time policy extraction, small per-state work
	for s := range p {
		p[s] = a.greedyAction(s)
	}
	return p
}

// QTable returns the live Q-value table Q[state][action]. The slice is the agent's
// own backing store (not a copy); treat it as read-only during training.
func (a *tabularAgent) QTable() [][]float64 { return a.Q }

// QLearning is the off-policy tabular TD-control agent (Watkins 1989). It behaves
// ε-greedily but bootstraps each update from the greedy value max_a' Q(s',a') of
// the next state, so its Q-table converges to the optimal action-value function Q*
// regardless of the exploratory behaviour policy.
type QLearning struct {
	tabularAgent
}

// NewQLearning builds a Q-learning agent for env with learning rate alpha,
// discount gamma and exploration rate epsilon (all in [0,1]), deterministic via
// seed. By default the Q-table has env.ObsDim() states (one-hot observations,
// argmax indexing) and env.NumActions() actions; override with [WithNumStates],
// [WithStateIndexFunc] and [WithEpsilonDecay]. Panics on out-of-range hyper-
// parameters (a programmer error).
func NewQLearning(env Env, alpha, gamma, epsilon float64, seed uint64, opts ...TabularOption) *QLearning {
	return &QLearning{newTabularAgent(env, alpha, gamma, epsilon, seed, opts...)}
}

// Episode runs one full episode on env, applying the off-policy Q-learning update
// online at every step, and returns the undiscounted episode reward. It decays ε
// once at the end if decay is configured. An error from the state-index function
// aborts the episode.
func (q *QLearning) Episode(env Env) (float64, error) {
	s, err := q.stateIndex(env.Reset())
	if err != nil {
		return 0, err
	}
	var total float64
	for {
		a := q.epsilonGreedy(s)
		next, reward, done := env.Step(a)
		total += reward
		target := reward
		var s2 int
		if !done {
			if s2, err = q.stateIndex(next); err != nil {
				return 0, err
			}
			target += q.Gamma * q.maxQ(s2) // off-policy: greedy bootstrap
		}
		//perfscan:ignore PS6007 inherently sequential MDP rollout scalar TD update
		q.Q[s][a] += q.Alpha * (target - q.Q[s][a])
		if done {
			break
		}
		s = s2
	}
	q.decayEpsilon()
	return total, nil
}

// SARSA is the on-policy tabular TD-control agent (Rummery & Niranjan 1994). It
// behaves ε-greedily and bootstraps each update from Q(s',a') for the action a'
// its own ε-greedy policy actually takes next, so its Q-table reflects the value of
// the policy it follows — including the cost of exploration. Named for the
// (State, Action, Reward, State', Action') quintuple each update consumes.
type SARSA struct {
	tabularAgent
}

// NewSARSA builds a SARSA agent for env with learning rate alpha, discount gamma
// and exploration rate epsilon (all in [0,1]), deterministic via seed. Defaults
// and options match [NewQLearning]. Panics on out-of-range hyper-parameters.
func NewSARSA(env Env, alpha, gamma, epsilon float64, seed uint64, opts ...TabularOption) *SARSA {
	return &SARSA{newTabularAgent(env, alpha, gamma, epsilon, seed, opts...)}
}

// Episode runs one full episode on env, applying the on-policy SARSA update online
// at every step, and returns the undiscounted episode reward. Unlike Q-learning it
// bootstraps from the actual next ε-greedy action rather than the greedy max. It
// decays ε once at the end if decay is configured.
func (s *SARSA) Episode(env Env) (float64, error) {
	st, err := s.stateIndex(env.Reset())
	if err != nil {
		return 0, err
	}
	a := s.epsilonGreedy(st)
	var total float64
	for {
		next, reward, done := env.Step(a)
		total += reward
		target := reward
		var st2, a2 int
		if !done {
			if st2, err = s.stateIndex(next); err != nil {
				return 0, err
			}
			a2 = s.epsilonGreedy(st2)
			target += s.Gamma * s.Q[st2][a2] // on-policy: actual next action
		}
		s.Q[st][a] += s.Alpha * (target - s.Q[st][a])
		if done {
			break
		}
		st, a = st2, a2
	}
	s.decayEpsilon()
	return total, nil
}

# ERSAC

Epistemic Risk-Sensitive Actor-Critic for Offline Reinforcement Learning.

## Installation

```bash
pip install torch numpy minari gymnasium
```

## Usage

```bash
# Train with different methods
python benchmark_comparison.py cql           # Conservative Q-Learning
python benchmark_comparison.py iql           # Implicit Q-Learning
python benchmark_comparison.py brac          # BRAC with KL penalty
python benchmark_comparison.py box           # ERSAC Box uncertainty
python benchmark_comparison.py ellipsoidal   # ERSAC Ellipsoidal uncertainty
python benchmark_comparison.py convex_hull   # ERSAC Convex Hull
python benchmark_comparison.py epinet        # ERSAC with EpiNet
```

## Options

```bash
--dataset       Minari dataset name (default: atari/breakout/expert-v0)
--steps         Training steps (default: 100000)
--batch_size    Batch size (default: 256)
```

## License

MIT


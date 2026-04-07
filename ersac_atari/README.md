
# ERSAC for Atari (Minari)

This repo contains a minimal, research-friendly PyTorch implementation of the models from
**"Epistemic Robust Offline Reinforcement Learning (ERSAC)"** targeting **discrete-action Atari**
benchmarks using **Minari** offline datasets.

Implements:
- **SAC-N** (ensemble, box uncertainty set — Proposition 1 equivalence)
- **ERSAC-Epi** (Epinet + closed-form ellipsoidal uncertainty set)

> Paper reference for algorithms and formulas (losses, uncertainty sets, Epinet structure):
> *Epistemic Robust Offline Reinforcement Learning*, ICLR 2026 under review.

## Quickstart

### 1) Install
```bash
pip install torch torchvision minari gymnasium gymnasium[atari]
# if needed: AutoROM to install Atari ROMs for evaluation
# pip install autorom && AutoROM --accept-license
```

### 2) Train on a Minari Atari dataset
```bash
python -m ersac_atari.run_ersac_atari --dataset atari-breakout-expert-v0 --algo ersac_epi --max_steps 300000
# or SAC-N (ensemble size configurable)
python -m ersac_atari.run_ersac_atari --dataset atari-breakout-expert-v0 --algo sac_n --n_ensemble 20
```

### 3) Notes
- `--ell_radius` controls ellipsoidal conservativeness in ERSAC-Epi. In the paper, this
  corresponds to `sqrt(chi2_inv(|A|, υ))`; tuning directly works well in practice.
- This code uses a **fixed α** for entropy regularization; you can add SAC auto-tuning easily.
- For exact ENN training as per the appendix (bootstrap noise term), extend the critic loss in
  `ersac_atari/train_ersac.py`.

## Structure
```
ersac_atari/
  __init__.py
  models.py           # CNN encoder, policy, ensemble Q, Epinet Q (mu, Σ)
  uncertainty.py      # Box & ellipsoidal uncertainty utilities (worst-case q*)
  data_minari.py      # Minari dataset adapter -> PyTorch DataLoader
  train_ersac.py      # Training steps for SAC-N and ERSAC-Epi
  run_ersac_atari.py  # CLI entrypoint
```

## Disclaimer
- Minari dataset field names may vary slightly; adjust keys in `data_minari.py` if needed.
- This is a minimal research codebase meant for clarity and reproducibility.

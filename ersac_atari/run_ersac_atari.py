# ersac_atari/run_ersac_atari.py (only the CLI argument part and dataloader call change)

import argparse
from pathlib import Path

import torch
import torch.optim as optim

from .data_minari import make_dataloader
from .train_ersac import (TrainCfg, SACNAgent, ERSACEpiAgent,
                          sacn_step, ersac_epi_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None,
                        help="Minari dataset string. Accepts full ids like "
                             "'atari/breakout/expert-v0' or loose aliases like 'breakout expert'.")
    parser.add_argument('--game', type=str, default=None,
                        help="Game name (e.g., breakout, pong). Overrides --dataset if provided.")
    parser.add_argument('--quality', type=str, default=None,
                        help="Quality string (e.g., expert-v0, medium-replay-v2).")
    # ... (rest of your args unchanged)
    # ...
    args = parser.parse_args()

    # Data — robust resolver is inside make_dataloader
    ds, dl = make_dataloader(args.dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=8,
                             game=args.game,
                             quality=args.quality)

    # ... remainder unchanged ...
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Minari dataset name, e.g., atari-breakout-expert-v0')
    parser.add_argument('--algo', type=str, default='ersac_epi', choices=['ersac_epi', 'sac_n'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_steps', type=int, default=500_000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--target_tau', type=float, default=0.005)
    parser.add_argument('--ell_radius', type=float, default=3.0)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--n_ensemble', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=25_000)
    args = parser.parse_args()

    # Data
    ds, dl = make_dataloader(args.dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    obs_shape = ds.obs.shape[1:]  # (C, H, W)
    n_actions = ds.n_actions

    cfg = TrainCfg(
        algo=args.algo,
        n_ensemble=args.n_ensemble,
        z_dim=args.z_dim,
        ellipsoid_radius=args.ell_radius,
        alpha=args.alpha,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
    )

    device = cfg.device

    # Build agent + optimizers
    if cfg.algo == 'sac_n':
        agent = SACNAgent(obs_channels=obs_shape[0], n_actions=n_actions, cfg=cfg).to(device)
        opt_q = optim.Adam(agent.qens.parameters(), lr=cfg.lr)
        opt_pi = optim.Adam(agent.policy.parameters(), lr=cfg.lr)
        train_step = sacn_step
    else:
        agent = ERSACEpiAgent(obs_channels=obs_shape[0], n_actions=n_actions, cfg=cfg).to(device)
        opt_q = optim.Adam(agent.qepi.parameters(), lr=cfg.lr)
        opt_pi = optim.Adam(agent.policy.parameters(), lr=cfg.lr)
        train_step = ersac_epi_step

    step = 0
    while step < cfg.max_steps:
        for batch in dl:
            logs = train_step(agent, opt_pi, opt_q, batch, cfg, step)
            if logs and (step % cfg.log_interval == 0):
                print(f"step={step} | " + " ".join([f"{k}:{v:.4f}" for k, v in logs.items()]))
            step += 1
            if step >= cfg.max_steps:
                break

    outdir = Path('checkpoints') / args.dataset / args.algo
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), outdir / 'final.pt')
    print(f'Saved to {outdir / "final.pt"}')


if __name__ == '__main__':
    main()

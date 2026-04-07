
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import minari
except ImportError:  # pragma: no cover
    minari = None


@dataclass
class TransitionBatch:
    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor
    trunc: torch.Tensor


class MinariAtariBuffer(Dataset):
    """
    Adapter for Minari Atari datasets with image observations.
    Converts observations to float32 in [0,1] and channels-first (N,C,H,W).
    """
    def __init__(self, dataset_name: str):
        assert minari is not None, "Please `pip install minari`."
        self.ds = minari.load_dataset(dataset_name)
        rb = self.ds.replay_buffer

        obs = rb.observations
        nxt = rb.next_observations
        acts = rb.actions
        rews = rb.rewards
        # terminals / terminations
        if hasattr(rb, 'terminations'):
            dones = rb.terminations
        elif hasattr(rb, 'terminals'):
            dones = rb.terminals
        else:
            raise AttributeError('Minari replay buffer missing terminals/terminations')
        # truncations / timeouts
        if hasattr(rb, 'truncations'):
            truncs = rb.truncations
        elif hasattr(rb, 'timeouts'):
            truncs = rb.timeouts
        else:
            truncs = np.zeros_like(dones, dtype=dones.dtype)

        self.obs = self._to_float_chw(obs)
        self.next_obs = self._to_float_chw(nxt)
        self.acts = torch.as_tensor(acts, dtype=torch.long)
        self.rews = torch.as_tensor(rews, dtype=torch.float32)
        self.dones = torch.as_tensor(dones.astype(np.uint8), dtype=torch.float32)
        self.truncs = torch.as_tensor(truncs.astype(np.uint8), dtype=torch.float32)

        # environment to retrieve action space size
        try:
            self.env = self.ds.recover_environment()
            self.n_actions = self.env.action_space.n
        except Exception:
            # fallback if env recovery requires ROMs; infer from data if available
            self.env = None
            # Best-effort inference: assume actions are 0..K-1
            self.n_actions = int(self.acts.max().item()) + 1

    @staticmethod
    def _to_float_chw(x: np.ndarray) -> torch.Tensor:
        # (N,H,W,C) -> (N,C,H,W) if needed
        if x.ndim == 4 and x.shape[-1] in (1,3,4):
            x = np.transpose(x, (0,3,1,2))
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        return torch.from_numpy(x)

    def __len__(self):
        return self.acts.shape[0]

    def __getitem__(self, idx: int) -> TransitionBatch:
        return TransitionBatch(
            obs=self.obs[idx],
            act=self.acts[idx],
            rew=self.rews[idx],
            next_obs=self.next_obs[idx],
            done=self.dones[idx],
            trunc=self.truncs[idx],
        )


def make_dataloader(dataset_name: str, batch_size: int, shuffle: bool = True, num_workers: int = 4):
    ds = MinariAtariBuffer(dataset_name)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return ds, dl

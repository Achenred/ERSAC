import gym
import d4rl_atari
import pybullet
import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer import ReplayBuffer
import glob
from utils import save, collect_random
import random
from agent import SACN
import dill


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="Robust-SACN-discrete", help="Run name, default: CQL-SAC")
    parser.add_argument("--env", type=str, default="breakout-mixed-v0", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--trainer_type", type=str, default="random",
                        help="Gym environment name, default: risk-neutral")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes, default: 200")
    parser.add_argument("--buffer_size", type=int, default=100_000,
                        help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0,
                        help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")

    args = parser.parse_args()
    return args


def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = gym.make(config.env)

    env.seed(config.seed)
    env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0

    with wandb.init(project="CQL", name=config.run_name, config=config):

        agent = SACN(state_size=env.observation_space.shape[0],
                     action_size=env.action_space.n,
                     device=device)

        wandb.watch(agent, log="gradients", log_freq=10)

        if config.trainer_type == 'random':
            buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)
            collect_random(env=env, dataset=buffer, num_samples=1000)
        else:
            file_name = str("/Users/abhilashchenreddy/PycharmProjects/CQL_AC/datasets/") + str(config.env) + "-" + str(
                config.trainer_type) + ".pkl"

            with open(file_name, 'rb') as file:
                buffer = dill.load(file)

        if config.log_video:
            env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x % 10 == 0, force=True)

        for i in range(1, config.episodes + 1):
            state = env.reset()
            episode_steps = 0
            rewards = 0
            while True:
                action = agent.get_action(state)
                steps += 1
                next_state, reward, done, _ = env.step(action)
                buffer.add(state, action, reward, next_state, done)

                alpha_loss, critic_loss, policy_loss, actor_batch_entropy, current_alpha, q_policy_std, q_random_std = agent.learn(
                    steps, buffer.sample(), gamma=0.99)

                # policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = agent.learn(steps, buffer.sample(), gamma=0.99)
                state = next_state
                rewards += reward
                episode_steps += 1
                if done:
                    break

            average10.append(rewards)
            total_steps += episode_steps
            print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps, ))

            wandb.log({"Reward": rewards,
                       "Average10": np.mean(average10),
                       "Steps": total_steps,
                       "Policy Loss": policy_loss,
                       "Alpha Loss": alpha_loss,
                       "Critic Loss": critic_loss,
                       "Actor Batch Entropy": actor_batch_entropy,
                       "Q policy std": q_policy_std,
                       "Q random rnd": q_random_std,
                       "Alpha": current_alpha,
                       "Steps": steps,
                       "Episode": i,
                       "Buffer size": buffer.__len__()})

            if (i % 10 == 0) and config.log_video:
                mp4list = glob.glob('video/*.mp4')
                if len(mp4list) > 1:
                    mp4 = mp4list[-2]
                    wandb.log({"gameplays": wandb.Video(mp4, caption='episode: ' + str(i - 10), fps=4, format="gif"),
                               "Episode": i})

            if i % config.save_every == 0:
                save(config, save_name="Robust-SACN-discrete", model=agent.actor_local, wandb=wandb, ep=0)


if __name__ == "__main__":
    config = get_config()
    train(config)

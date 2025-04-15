import gym
import torch
import torch.nn as nn
import numpy as np
import time
import argparse

FPS = 25
NUM_EPISODES = 4
ENV_NAME = 'MountainCar-v0'


class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_action(state: np.ndarray, policy_net: DQN) -> int:
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state_tensor)
    return int(torch.argmax(q_values).item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='Model file to load')
    args = parser.parse_args()

    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    net = DQN(state_dim, action_dim)
    net.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))
    net.eval()

    for ep in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            starts_ts = time.time()
            env.render()

            action = select_action(state, net)
            next_state, reward, done, _ = env.step(action)
            state = next_state

            total_reward += reward

            delta = 1/FPS - (time.time() - starts_ts)
            if delta > 0:
                time.sleep(delta)

        print(f"Episode {ep + 1}: Total Reward = {total_reward}")

    env.close()

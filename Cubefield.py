import os
import gym
import torch
import random
import numpy as np
import pygame
from gym import spaces
from collections import deque
from torch import nn
from torch.distributions import Categorical
import torch.optim as optim
import matplotlib.pyplot as plt

class CubefieldEnv(gym.Env):
    def __init__(self):
        super(CubefieldEnv, self).__init__()
        self.screen_width = 600
        self.screen_height = 400
        self.player_size = 20
        self.obstacle_width = 30
        self.obstacle_height = 30
        self.obstacle_speed = 5
        self.clock_tick = 30

        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.reset_env()

    def reset_env(self):
        self.player_x = self.screen_width // 2
        self.obstacles = []
        self.time = 0
        self.done = False
        self.total_reward = 0

    def reset(self):
        self.reset_env()
        return self._get_obs()

    def _get_obs(self):
        nearest = sorted(self.obstacles, key=lambda o: o[1])[:1]
        if nearest:
            dx = (self.player_x - nearest[0][0]) / self.screen_width
            dy = (self.screen_height - nearest[0][1]) / self.screen_height
        else:
            dx, dy = 0, 1
        return np.array([self.player_x / self.screen_width, dx, dy, len(self.obstacles)/10], dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.player_x -= 10
        elif action == 2:
            self.player_x += 10

        self.player_x = np.clip(self.player_x, 0, self.screen_width - self.player_size)

        new_obs = []
        if random.random() < 0.2:
            new_obs.append((random.randint(0, self.screen_width - self.obstacle_width), 0))
        self.obstacles.extend(new_obs)
        self.obstacles = [(x, y + self.obstacle_speed) for x, y in self.obstacles if y < self.screen_height]

        reward = 1
        self.done = False
        for x, y in self.obstacles:
            if abs(x - self.player_x) < self.obstacle_width and y + self.obstacle_height >= self.screen_height - self.player_size:
                reward = -100
                self.done = True
                break

        self.total_reward += reward
        return self._get_obs(), reward, self.done, {}

    def render(self, mode='human'):
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (0, 255, 0), (self.player_x, self.screen_height - self.player_size, self.player_size, self.player_size))
        for x, y in self.obstacles:
            pygame.draw.rect(self.screen, (255, 0, 0), (x, y, self.obstacle_width, self.obstacle_height))
        pygame.display.flip()
        pygame.time.Clock().tick(self.clock_tick)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean()


class AttentionDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = x.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        x = torch.relu(self.fc2(attn_out.squeeze(1)))
        return self.out(x)


class QuantileRegressionDQN(nn.Module):
    def __init__(self, input_dim, output_dim, num_quantiles=51):
        super(QuantileRegressionDQN, self).__init__()
        self.num_quantiles = num_quantiles
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_dim * num_quantiles)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x.view(-1, self.output_dim, self.num_quantiles)


class A3CNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(A3CNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)


class AgentQuantileDQN:
    def __init__(self, env, num_quantiles=51):
        self.env = env
        self.num_quantiles = num_quantiles
        self.model = QuantileRegressionDQN(env.observation_space.shape[0], env.action_space.n, num_quantiles)
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def act(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state).mean(dim=2)
        return torch.argmax(q_values).item()

    def remember(self, s, a, r, s_, d):
        self.memory.append((s, a, r, s_, d))

    def quantile_huber_loss(self, preds, targets, tau):
        u = targets.unsqueeze(1) - preds.unsqueeze(2)
        huber_loss = torch.where(u.abs() < 1.0, 0.5 * u.pow(2), u.abs() - 0.5)
        loss = (tau.unsqueeze(1) - (u < 0).float()).abs() * huber_loss
        return loss.mean()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_, d = zip(*batch)
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        s_ = torch.FloatTensor(s_)
        d = torch.FloatTensor(d).unsqueeze(1)

        q_dist = self.model(s)
        next_q_dist = self.model(s_).detach()
        next_q = next_q_dist.mean(dim=2)
        best_actions = next_q.max(1)[1].unsqueeze(1).unsqueeze(2).expand(-1, 1, self.num_quantiles)
        q_next = next_q_dist.gather(1, best_actions).squeeze(1)

        q_target = r + self.gamma * q_next * (1 - d)
        q_pred = q_dist.gather(1, a.unsqueeze(2).expand(-1, 1, self.num_quantiles)).squeeze(1)

        tau = torch.linspace(0.0, 1.0, self.num_quantiles + 1)[1:] - 0.5 / self.num_quantiles
        tau = tau.to(q_pred.device).unsqueeze(0).expand(self.batch_size, -1)

        loss = self.quantile_huber_loss(q_pred, q_target, tau)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, episodes=1000):
        rewards = []
        losses = []
        for ep in range(episodes):
            state = self.env.reset()
            total = 0
            for _ in range(1000):
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total += reward
                if done:
                    break
            loss = self.replay()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards.append(total)
            losses.append(loss)
            print(f"[QR-DQN] Episode {ep+1}, Reward: {total}, Loss: {loss:.4f}")
        return rewards, losses

class AgentA3C:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99
        self.model = A3CNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, episodes=1000):
        all_rewards = []
        all_losses = []
        for ep in range(episodes):
            state = self.env.reset()
            log_probs = []
            values = []
            rewards = []
            for _ in range(1000):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                probs, value = self.model(state_tensor)
                dist = Categorical(probs)
                action = dist.sample()
                next_state, reward, done, _ = self.env.step(action.item())

                log_probs.append(dist.log_prob(action))
                values.append(value)
                rewards.append(reward)
                state = next_state
                if done:
                    break

            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)

            returns = torch.FloatTensor(returns).unsqueeze(1)
            values = torch.cat(values)
            log_probs = torch.stack(log_probs)

            advantage = returns - values.detach()
            actor_loss = -(log_probs * advantage).mean()
            critic_loss = nn.functional.mse_loss(values, returns)
            total_loss = actor_loss + critic_loss

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            total_reward = sum(rewards)
            all_rewards.append(total_reward)
            all_losses.append(total_loss.item())
            print(f"[A3C] Episode {ep+1}, Reward: {total_reward}, Loss: {total_loss.item():.4f}")

        return all_rewards, all_losses

class PPODuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPODuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        return self.advantage(x), self.value(x)


class AgentPPODuelingDQN:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.model = PPODuelingDQN(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, episodes=1000):
        all_rewards = []
        smoothed_rewards = []
        for ep in range(episodes):
            state = self.env.reset()
            log_probs = []
            values = []
            actions = []
            rewards = []
            states = []
            for _ in range(1000):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                probs, value = self.model(state_tensor)
                dist = Categorical(probs)
                action = dist.sample()
                next_state, reward, done, _ = self.env.step(action.item())

                log_probs.append(dist.log_prob(action))
                values.append(value)
                actions.append(action)
                rewards.append(reward)
                states.append(state_tensor)

                state = next_state
                if done:
                    break

            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)

            returns = torch.FloatTensor(returns).unsqueeze(1)
            values = torch.cat(values)
            log_probs = torch.stack(log_probs)
            states = torch.cat(states)
            actions = torch.stack(actions)
            advantage = returns - values.detach()

            for _ in range(4):
                new_probs, _ = self.model(states)
                dist = Categorical(new_probs)
                new_log_probs = dist.log_prob(actions)

                ratio = torch.exp(new_log_probs - log_probs.detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                loss = -torch.min(surr1, surr2).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_reward = sum(rewards)
            all_rewards.append(total_reward)
            avg_reward = np.mean(all_rewards[-10:])
            smoothed_rewards.append(avg_reward)
            print(f"[PPO-Dueling-DQN] Episode {ep+1}, Reward: {total_reward}, Smoothed: {avg_reward:.2f}")
        return smoothed_rewards

class AgentAttentionDQN:
    def __init__(self, env):
        self.env = env
        self.model = AttentionDQN(env.observation_space.shape[0], env.action_space.n)
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def act(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0)
        return torch.argmax(self.model(state)).item()

    def remember(self, s, a, r, s_, d):
        self.memory.append((s, a, r, s_, d))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_, d = zip(*batch)
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        s_ = torch.FloatTensor(s_)
        d = torch.FloatTensor(d).unsqueeze(1)

        q_vals = self.model(s).gather(1, a)
        q_next = self.model(s_).max(1)[0].detach().unsqueeze(1)
        q_target = r + self.gamma * q_next * (1 - d)

        loss = nn.functional.mse_loss(q_vals, q_target)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, episodes=1000):
        rewards = []
        losses = []
        for ep in range(episodes):
            state = self.env.reset()
            total = 0
            for _ in range(1000):
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total += reward
                if done:
                    break
            loss = self.replay()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards.append(total)
            losses.append(loss)
            print(f"[Attention-DQN] Episode {ep+1}, Reward: {total}, Loss: {loss:.4f}")
        return rewards, losses


def smooth(data, window_size=25):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

if __name__ == '__main__':
    env = CubefieldEnv()

    agents = {
        #"DQN": AgentDQN(env),
       # "REINFORCE": REINFORCEAgent(env),
        "PPO-Dueling-DQN": AgentPPODuelingDQN(env),
        "A3C": AgentA3C(env),
        "QR-DQN": AgentQuantileDQN(env),
        "Attention-DQN": AgentAttentionDQN(env),
    }

    rewards = {}
    losses = {}
    for name, agent in agents.items():
        print(f"\n=== Training {name} ===")
        result = agent.train(episodes=1500)
        if isinstance(result, tuple):
            rewards[name], losses[name] = result
        else:
            rewards[name] = result

    plt.figure(figsize=(12, 6))
    for name, r in rewards.items():
        plt.plot(smooth(r), label=name)

    plt.title("Learning Curve Comparison on Cubefield")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cubefield_rl_comparison_smoothed.png")
    plt.show()

    if losses:
        plt.figure(figsize=(12, 6))
        for name, l in losses.items():
            plt.plot(smooth(l), label=name)
        plt.title("Loss Curve Comparison on Cubefield")
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("cubefield_rl_loss_comparison_smoothed.png")
        plt.show()

env.close()

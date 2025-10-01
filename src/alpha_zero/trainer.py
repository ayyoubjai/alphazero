import torch
import torch.optim as optim
from multiprocessing import Pool
import numpy as np
from typing import List
from tqdm import tqdm
from .config import Config
from .games.abstract_game import AbstractGame
from .networks.base_net import BaseNet
from .mcts.mcts import MCTS
from .replay_buffer import ReplayBuffer

class Trainer:
    def __init__(self, game: AbstractGame, net: BaseNet, config: Config):
        self.game = game
        self.net = net
        self.config = config
        self.mcts = MCTS(game, net, config)
        self.buffer = ReplayBuffer(config.buffer_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.learning_rate)
        self.device = net.device

    def self_play_game(self, _) -> List[tuple]:
        state = self.game.get_initial_state()
        player = 1  # player to move at 'state'
        trajectory = []  # (state, policy, player_at_state)

        while True:
            policy, _ = self.mcts.search(state, player)
            temp = self.config.temperature
            if temp != 1.0:
                action_probs = policy ** (1.0 / temp)
                action_probs /= np.sum(action_probs)
            else:
                action_probs = policy

            action = np.random.choice(range(self.game.action_size), p=action_probs)
            trajectory.append((state.copy(), action_probs, player))

            # Environment step
            state = self.game.get_next_state(state, action, player)
            terminal, env_value = self.game.is_terminal(state)
            if terminal:
                # env_value: +1 if player 1 has a win in terminal state, -1 if player -1, 0 draw
                # Label from each state's current player's perspective:
                data = []
                for s, p, pl in trajectory:
                    z = env_value if pl == 1 else -env_value
                    data.append((s, p, z))
                return data

            player = -player  # switch turns

    def generate_self_play(self, num_games: int):
        with Pool(self.config.num_workers) as pool:
            trajectories = pool.map(self.self_play_game, range(num_games))
        for traj in trajectories:
            for s, p, v in traj:
                self.buffer.add(s, p, v)
                for aug_s, aug_p in self.game.get_symmetries(s, p):
                    self.buffer.add(aug_s, aug_p, v)

    def train(self, num_epochs: int):
        self.net.train()  # ensure training mode
        for _ in tqdm(range(num_epochs), desc="Training epochs"):
            if len(self.buffer) < self.config.batch_size:
                continue
            states, policies, values = self.buffer.sample(self.config.batch_size)
            encoded = np.array([self.game.get_encoded_state(s) for s in states])
            encoded = torch.tensor(encoded, device=self.device).float()
            target_policies = torch.tensor(np.array(policies), device=self.device).float()
            target_values = torch.tensor(np.array(values), device=self.device).float().unsqueeze(1)

            pred_policies, pred_values = self.net(encoded)

            # Cross-entropy with log probs (your net outputs probs already; if you switch to logits, use log_softmax)
            policy_loss = -torch.mean(torch.sum(target_policies * torch.log(pred_policies + 1e-8), dim=1))
            value_loss = torch.mean((pred_values - target_values) ** 2)
            l2_loss = self.config.l2_reg * sum(p.pow(2).sum() for p in self.net.parameters() if p.dim() > 1)
            loss = policy_loss + value_loss + l2_loss

            print(f"Value Loss {value_loss} Policy Loss {policy_loss} Total Loss {loss}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def checkpoint(self, path: str):
        print(f"Saving checkpoint to {path}")
        torch.save(self.net.state_dict(), path)
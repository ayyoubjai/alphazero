import os
import time
import torch
import torch.multiprocessing as mp
import numpy as np
import copy
from multiprocessing import Queue, Event
from typing import List
from .config import Config
from .games.abstract_game import AbstractGame
from .networks.base_net import BaseNet
from .mcts.mcts import MCTS
from .replay_buffer import ReplayBuffer
import signal

# -------------------------
# Actor process (self-play)
# -------------------------
def actor_process(actor_id: int,
                  game_cls,
                  net_constructor_args: dict,
                  config: Config,
                  traj_queue: Queue,
                  ckpt_path: str,
                  stop_event: Event):
    """
    Actor process function.
    - game_cls: class (not instance) of game, so child can instantiate safely.
    - net_constructor_args: dict used to construct BaseNet (input_channels, board_size, action_size, etc.)
    - traj_queue: multiprocessing.Queue to put trajectories into
    - ckpt_path: path to latest checkpoint file to read
    - stop_event: Event to exit loop
    """
    # Build local CPU-only game and network
    device = 'cpu'
    game = game_cls()
    net = BaseNet(**net_constructor_args, device=device)
    mcts = MCTS(game, net, config)

    last_ckpt_mtime = None

    def maybe_reload_checkpoint():
        nonlocal last_ckpt_mtime
        try:
            if not os.path.exists(ckpt_path):
                return
            mtime = os.path.getmtime(ckpt_path)
            if last_ckpt_mtime is None or mtime > last_ckpt_mtime:
                # safe load on CPU
                state = torch.load(ckpt_path, map_location='cpu',weights_only=True)
                net.load_state_dict(state)
                last_ckpt_mtime = mtime
                print(f"[actor {actor_id}] loaded checkpoint {ckpt_path} (mtime={mtime})")
        except Exception as e:
            print(f"[actor {actor_id}] failed to reload checkpoint: {e}")

    # Best-effort graceful handling of termination from main process
    signal.signal(signal.SIGINT, lambda sig, frame: stop_event.set())

    # Main loop
    while not stop_event.is_set():
        # Reload new weights if available
        maybe_reload_checkpoint()

        # Self-play one game
        traj = []
        state = game.get_initial_state()
        player = 1
        while True:
            policy, _ = mcts.search(state, player)
            # temperature handling
            temp = config.temperature
            if temp != 1.0:
                action_probs = np.maximum(policy, 1e-12) ** (1.0 / temp)
                action_probs = action_probs / np.sum(action_probs)
            else:
                action_probs = policy / np.sum(policy) if np.sum(policy) > 0 else np.ones_like(policy) / len(policy)

            action = np.random.choice(range(game.action_size), p=action_probs)
            traj.append((copy.deepcopy(state), action_probs.astype(np.float32), player))
            state, next_player = game.get_next_state(state, action, player)
            terminal, value = game.is_terminal(state, next_player)
            if terminal:
                # For each position, z is outcome from perspective of the player who moved at that position
                out = []
                for s, p, pl in traj:
                    # value is from the terminal player's (next_player) view; adjust to pl's view
                    z = value if pl == next_player else -value
                    out.append((s, p, float(z), pl))
                try:
                    traj_queue.put(out, block=True, timeout=5)
                except Exception as e:
                    print(f"[actor {actor_id}] failed to enqueue trajectory: {e}")
                break
            player = next_player

    print(f"[actor {actor_id}] stopping.")


# -------------------------
# Learner controller
# -------------------------
class ParallelTrainer:
    def __init__(self, game_cls, net_constructor_args: dict, config: Config, ckpt_dir: str = "models", num_actors: int = 4):
        self.game_cls = game_cls
        self.net_constructor_args = net_constructor_args
        self.config = config
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_path = os.path.join(ckpt_dir, "latest.pth")

        # Learner's net on GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = BaseNet(**net_constructor_args, device=device)
        self.mcts = MCTS(game_cls(), self.net, config)  # not used for training, but handy
        self.replay = ReplayBuffer(config.buffer_size)  # Assume updated to handle (s, p, z, pl)

        self.traj_queue: Queue = mp.Queue(maxsize=512)
        self.stop_event = mp.Event()
        self.num_actors = num_actors
        self.actors = []
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.learning_rate)

    def start_actors(self):
        for i in range(self.num_actors):
            p = mp.Process(target=actor_process, args=(
                i,
                self.game_cls,
                self.net_constructor_args,
                self.config,
                self.traj_queue,
                self.ckpt_path,
                self.stop_event
            ))
            p.daemon = True
            p.start()
            self.actors.append(p)
            print(f"[learner] started actor {i} (pid={p.pid})")

    def stop_actors(self):
        self.stop_event.set()
        # allow actors to exit gracefully
        for p in self.actors:
            p.join(timeout=5)
            if p.is_alive():
                print(f"[learner] terminating actor pid={p.pid}")
                p.terminate()

    def save_checkpoint(self):
        # Save CPU-state dict so actors can reload quickly
        cpu_state = {k: v.cpu() for k, v in self.net.state_dict().items()}
        torch.save(cpu_state, self.ckpt_path)
        print(f"[learner] saved checkpoint to {self.ckpt_path}")

    def collect_trajectories(self, target_num: int, timeout: float = 1.0):
        """
        Pull up to target_num trajectories from the queue (blocking with timeout).
        Returns number of trajs collected.
        """
        collected = 0
        t0 = time.time()
        while collected < target_num and (time.time() - t0) < (timeout * target_num + 1.0):
            try:
                traj = self.traj_queue.get(block=True, timeout=timeout)
                # traj is a list of (state, policy, z, player). Add and augment
                for s, p, z, pl in traj:
                    self.replay.add(s, p, z, pl)
                    for aug_s, aug_p in self.game_cls().get_symmetries(s, p):
                        self.replay.add(aug_s, aug_p, z, pl)
                collected += 1
            except Exception:
                # timeout or empty; break early if nothing to collect
                break
        return collected

    def train_loop(self, total_iters: int = 1000, games_per_iter: int = 10, epochs_per_iter: int = 10, checkpoint_interval: int = 5):
        """
        High-level training driver:
          - ensure actors are running
          - repeatedly collect games, then train for several epochs
        """
        self.start_actors()
        try:
            for it in range(total_iters):
                # collect self-play games
                needed = games_per_iter
                print(f"[learner] collecting {needed} self-play games (iter {it})")
                collected = 0
                while collected < needed:
                    got = self.collect_trajectories(needed - collected, timeout=20.0)
                    if got == 0:
                        print("[learner] waiting for actors to produce more trajectories...")
                        time.sleep(1.0)
                    collected += got
                    print(f"[learner] collected {collected}/{needed} games, buffer size={len(self.replay)}")

                # train
                print(f"[learner] training for {epochs_per_iter} epochs")
                self._train_epochs(epochs_per_iter)

                # checkpoint
                if it % checkpoint_interval == 0:
                    self.save_checkpoint()
        finally:
            self.stop_actors()

    def _train_epochs(self, epochs: int):
        self.net.train()
        for _ in range(epochs):
            if len(self.replay) < self.config.batch_size:
                break
            states, policies, values, players = self.replay.sample(self.config.batch_size)
            encoded = np.array([self.game_cls().get_encoded_state(s, pl) for s, pl in zip(states, players)])
            encoded = torch.tensor(encoded, device=self.net.device).float()
            target_policies = torch.tensor(np.array(policies), device=self.net.device).float()
            target_values = torch.tensor(np.array(values), device=self.net.device).float().unsqueeze(1)

            pred_policies, pred_values = self.net(encoded)
            # assume pred_policies are probabilities; if changed to logits, adjust
            policy_loss = -torch.mean(torch.sum(target_policies * torch.log(pred_policies + 1e-8), dim=1))
            value_loss = torch.mean((pred_values - target_values) ** 2)
            l2_loss = self.config.l2_reg * sum(p.pow(2).sum() for p in self.net.parameters() if p.dim() > 1)
            loss = policy_loss + value_loss + l2_loss
            print(f"Loss {loss} Policy loss {policy_loss} Value loss {value_loss}")
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            self.optimizer.step()
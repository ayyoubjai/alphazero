
import math
import random
from typing import Tuple, Any

import numpy as np
import torch
from graphviz import Digraph

from .node import Node
from ..config import Config
from ..games.abstract_game import AbstractGame
from ..networks.base_net import BaseNet
from ..utils import add_dirichlet_noise


class MCTS:
    def __init__(self, game: AbstractGame, net: BaseNet, config: Config):
        self.game = game
        self.net = net
        self.config = config
        self.device = net.device
        self.root: Node = None  # keep latest root

    def search(self, state: Any, player: int) -> Tuple[np.ndarray, float]:
        """Run MCTS from (state, player_to_move). Returns (policy, root_value).

        policy is a vector of probabilities across all actions (summing to 1).
        The returned root_value is the root node's estimated value (average) from the root player's perspective.
        """
        root = Node(state=state, prior=1.0, parent=None, action=None, player=player)
        self._expand_root(root)

        for _ in range(self.config.num_simulations):
            node = root
            # SELECTION: descend until you find an unexpanded node
            while node.is_expanded() and node.children:
                node = self._select_child(node)
            # EXPANSION + EVAL
            value = self._expand_and_evaluate(node)
            # BACKUP
            self._backup(node, value)

        self.root = root  # save the root so we can inspect/visualize later
        policy = self._get_policy(root)
        return policy, root.value

    def _encode(self, state: Any, player: int) -> torch.Tensor:
        """
        Encode (state, player) for the network.

        Preferred: game.get_encoded_state(state, player). If not available, call
        game.get_encoded_state(state) and append a player plane so the network
        always receives the player-to-move info.
        """
        try:
            enc = self.game.get_encoded_state(state, player)
        except TypeError:
            enc = self.game.get_encoded_state(state)

        enc = np.asarray(enc)

        # Ensure channel-first shape (C, H, W)
        if enc.ndim == 2:
            enc = enc[np.newaxis, ...]
        elif enc.ndim == 3:
            # assume it's already (C, H, W)
            pass
        else:
            raise ValueError(f"Unexpected encoding shape: {enc.shape}")

        expected_channels = getattr(self.net, "conv1", None)
        if expected_channels is not None:
            expected_channels = getattr(self.net.conv1, "in_channels", enc.shape[0])
        else:
            expected_channels = enc.shape[0]

        # If encoding doesn't include a player plane, append one and zeros for remaining
        if enc.shape[0] < expected_channels:
            extra = expected_channels - enc.shape[0]
            player_plane = np.full((1, enc.shape[1], enc.shape[2]), 1.0 if player == 1 else -1.0, dtype=enc.dtype)
            extras = [player_plane]
            for _ in range(extra - 1):
                extras.append(np.zeros_like(player_plane))
            enc = np.concatenate([enc] + extras, axis=0)

        return torch.tensor(enc[np.newaxis, ...], device=self.device).float()

    def _expand_root(self, root: Node):
        encoded = self._encode(root.state, root.player)
        policy_tensor, _ = self.net(encoded)
        policy = policy_tensor.detach().cpu().numpy()[0]

        # mask illegal actions and normalize
        legal_actions = self.game.get_legal_actions(root.state, root.player)
        mask = np.zeros_like(policy, dtype=bool)
        mask[legal_actions] = True
        policy = np.where(mask, policy, 0.0)

        # safe normalization
        policy_sum = np.sum(policy)
        if policy_sum <= 0 or not np.isfinite(policy_sum):
            # fallback to uniform priors across legal moves
            for action in legal_actions:
                child_prior = 1.0 / len(legal_actions)
                child_state, child_player = self.game.get_next_state(root.state, action, root.player)
                child = Node(state=child_state, prior=child_prior, parent=root, action=action, player=child_player)
                root.children.append(child)
            return

        policy = policy / policy_sum

        # Dirichlet noise at root for exploration. We apply it only over legal actions.
        try:
            policy = add_dirichlet_noise(policy, self.config.dirichlet_alpha, self.config.dirichlet_epsilon)
            # ensure illegal remain zero after noise (defensive)
            policy = np.where(mask, policy, 0.0)
            policy_sum = np.sum(policy)
            if policy_sum > 0 and np.isfinite(policy_sum):
                policy = policy / policy_sum
        except Exception:
            # if add_dirichlet_noise not behaving as expected, continue with pre-noise policy
            pass

        for action in legal_actions:
            child_prior = float(policy[action])
            # if numerical issues lead to tiny negative/NaN priors, clip
            if not np.isfinite(child_prior) or child_prior < 0:
                child_prior = 0.0
            child_state, child_player = self.game.get_next_state(root.state, action, root.player)
            child = Node(state=child_state, prior=child_prior, parent=root, action=action, player=child_player)
            root.children.append(child)

    def _select_child(self, node: Node) -> Node:
        best_score = -np.inf
        best_child = None
        # compute parent visit sum explicitly (safer)
        parent_visits = sum(child.visits for child in node.children)
        sqrt_parent = math.sqrt(max(1.0, parent_visits))

        for child in node.children:
            # q should be value from parent's perspective (parent -> child: parent's Q = - child.value)
            q = -child.value
            # PUCT exploration term
            u = self.config.c_puct * child.prior * (sqrt_parent / (1.0 + child.visits))
            # tiny random tie-breaker so identical scores don't always pick the same child
            tie_break = 1e-6 * random.random()
            score = q + u + tie_break
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand_and_evaluate(self, node: Node) -> float:
        terminal, value_from_env = self.game.is_terminal(node.state, node.player)
        if terminal:
            return value_from_env

        legal_actions = self.game.get_legal_actions(node.state, node.player)
        if not legal_actions and self.game.is_continuation_end(node.state):
            # Special case for end of capture sequence: evaluate from opponent's perspective
            encoded = self._encode(node.state, -node.player)
            _, value_tensor = self.net(encoded)
            value = -float(value_tensor.item())
            return value

        encoded = self._encode(node.state, node.player)
        policy_tensor, value_tensor = self.net(encoded)

        # detach and convert
        policy = policy_tensor.detach().cpu().numpy()[0]
        value = float(value_tensor.item())

        # mask illegal actions and normalize
        mask = np.zeros_like(policy, dtype=bool)
        mask[legal_actions] = True
        policy = np.where(mask, policy, 0.0)

        policy_sum = np.sum(policy)
        if policy_sum <= 0 or not np.isfinite(policy_sum):
            # fallback uniform priors
            for action in legal_actions:
                child_prior = 1.0 / len(legal_actions)
                child_state, child_player = self.game.get_next_state(node.state, action, node.player)
                child = Node(state=child_state, prior=child_prior, parent=node, action=action, player=child_player)
                node.children.append(child)
        else:
            policy = policy / policy_sum
            for action in legal_actions:
                child_prior = float(policy[action])
                if not np.isfinite(child_prior) or child_prior < 0:
                    child_prior = 0.0
                child_state, child_player = self.game.get_next_state(node.state, action, node.player)
                child = Node(state=child_state, prior=child_prior, parent=node, action=action, player=child_player)
                node.children.append(child)

        return value

    def _backup(self, node: Node, value: float):
        current = node
        # value is from node.player's perspective
        while current is not None:
            current.visits += 1
            current.value_sum += value
            # flip value for parent (opponent perspective)
            value = -value
            current = current.parent

    def _get_policy(self, root: Node) -> np.ndarray:
        """Return visit-count policy over ALL actions (0 if never visited).
        If no visits were made (sum==0), try to return network policy at root as a fallback.
        """
        policy = np.zeros(self.game.action_size, dtype=np.float32)
        for child in root.children:
            policy[child.action] = child.visits
        s = policy.sum()
        if s > 0:
            policy = policy / s
            return policy

        # fallback: use network policy if available
        try:
            enc = self._encode(root.state, root.player)
            p_tensor, _ = self.net(enc)
            p = p_tensor.detach().cpu().numpy()[0]
            legal_actions = self.game.get_legal_actions(root.state, root.player)
            out = np.zeros_like(p, dtype=np.float32)
            p[~np.isfinite(p)] = 0.0
            mask = np.zeros_like(p, dtype=bool)
            mask[legal_actions] = True
            p = np.where(mask, p, 0.0)
            p_sum = np.sum(p[legal_actions])
            if p_sum <= 0 or not np.isfinite(p_sum):
                for a in legal_actions:
                    out[a] = 1.0 / len(legal_actions)
                return out
            else:
                for a in legal_actions:
                    out[a] = float(p[a] / p_sum)
                return out
        except Exception:
            # last resort uniform
            out = np.zeros(self.game.action_size, dtype=np.float32)
            legal = self.game.get_legal_actions(root.state, root.player)
            for a in legal:
                out[a] = 1.0 / len(legal)
            return out

    # === NEW: export tree with Graphviz ===
    def export_tree(self, max_depth: int = 2) -> Digraph:
        if self.root is None:
            raise RuntimeError("No root available. Run search() first.")
        dot = Digraph(comment="MCTS Tree")

        def add_node(node: Node, name: str, depth: int):
            label = f"Player={node.player}\\nV={node.value:.2f}\\nN={node.visits}\\nP={node.prior:.2f}"
            dot.node(name, label=label, shape="box")
            if depth < max_depth:
                for i, child in enumerate(node.children):
                    child_name = f"{name}_{i}"
                    add_node(child, child_name, depth + 1)
                    dot.edge(name, child_name, label=str(child.action))

        add_node(self.root, "root", 0)
        return dot

    # === NEW: debug printing of root child stats ===
    def print_root_stats(self):
        """Print (action, prior, visits, q, u) for each child of the last root.

        q is reported from the root player's perspective (i.e. -child.value).
        """
        if self.root is None:
            print("No root available. Run search() first.")
            return
        total_visits = max(1.0, sum(child.visits for child in self.root.children))
        sqrt_parent = math.sqrt(max(1.0, self.root.visits))
        print(f"Root: player={self.root.player}, root.visits={self.root.visits}, total_child_visits={total_visits}")
        rows = []
        for child in self.root.children:
            q = -child.value  # value from root player's perspective
            u = self.config.c_puct * child.prior * (sqrt_parent / (1.0 + child.visits))
            rows.append((child.action, child.prior, child.visits, q, u))
        rows.sort(key=lambda r: -r[2])  # sort by visits desc
        for action, prior, visits, q, u in rows:
            print(f"Action {action}: prior={prior:.4f}, visits={visits}, q={q:.4f}, u={u:.4f}")
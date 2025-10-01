from .config import Config
from .games.abstract_game import AbstractGame
from .networks.base_net import BaseNet
from .mcts.mcts import MCTS
from .trainer import Trainer
import torch
import numpy as np
from typing import Optional, Any

# Try to import the parallel trainer if available
try:
    from .parallel_trainer import ParallelTrainer
except Exception:
    ParallelTrainer = None


class AlphaZeroAgent:
    def __init__(
        self,
        game: AbstractGame,
        config: Config = Config(),
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_parallel: bool = False,
        parallel_kwargs: Optional[dict] = None,
    ):
        """AlphaZero agent wrapper.

        Args:
            game: an instance of AbstractGame
            config: training/config dataclass
            device: 'cuda' or 'cpu' for the main network
            use_parallel: if True, you'll use ParallelTrainer (actors/learner). If False, uses legacy Trainer.
            parallel_kwargs: forwarded to ParallelTrainer (num_actors, ckpt_dir, ...)
        """
        self.game = game
        self.config = config
        self.device = device
        self.use_parallel = use_parallel and (ParallelTrainer is not None)
        self.parallel_kwargs = parallel_kwargs or {}

        # Network: use game's input_channels if available
        input_channels = getattr(game, 'input_channels', 6)
        board_size = getattr(game, 'board_size', 8)
        action_size = getattr(game, 'action_size')

        # Use smaller net for TTT, larger for Dama/others
        if board_size <= 3:
            num_res_blocks = 2
            hidden_channels = 64
            value_hidden = 32
        else:
            num_res_blocks = 10
            hidden_channels = 256
            value_hidden = 128

        self.net = BaseNet(
            input_channels=input_channels,
            board_size=board_size,
            action_size=action_size,
            num_res_blocks=num_res_blocks,
            hidden_channels=hidden_channels,
            value_hidden=value_hidden,
            device=self.device
        )
        # Legacy trainer (single-process)
        if not self.use_parallel:
            self.trainer = Trainer(game, self.net, config)
        else:
            # Parallel trainer will be constructed on demand by start_parallel_training
            self.parallel_trainer = None

        # MCTS instance for inference (uses self.net and self.config)
        self.mcts = MCTS(game, self.net, config)

    # ---------------------- training API ----------------------
    def train(self, num_games: int, num_epochs: int):
        """Legacy training wrapper that keeps behaviour similar to your previous API.

        If the agent was created with use_parallel=True you should instead call
        `start_parallel_training(...)` which gives more control over actors/iters.
        """
        if self.use_parallel:
            raise RuntimeError("Agent constructed for parallel training — call start_parallel_training() instead")

        games_per_iteration = max(1, self.config.num_self_play_games)
        iterations = max(1, num_games // games_per_iteration)

        for i in range(iterations):
            # generate self-play games
            self.trainer.generate_self_play(games_per_iteration)
            # train epochs
            self.trainer.train(num_epochs)
            if i % self.config.checkpoint_interval == 0:
                # use trainer.checkpoint which saves self.net.state_dict()
                try:
                    self.trainer.checkpoint(f"models/checkpoint_{i}.pth")
                except Exception:
                    # fallback to direct torch.save
                    torch.save(self.net.state_dict(), f"models/checkpoint_{i}.pth")

    def start_parallel_training(
        self,
        total_iters: int = 200,
        games_per_iter: int = 20,
        epochs_per_iter: int = 10,
        checkpoint_interval: int = 5,
        num_actors: int = 2,
        ckpt_dir: str = "models",
    ):
        """Start the parallel actor/learner training loop using ParallelTrainer.

        This will create a ParallelTrainer object (learner) and run train_loop(...) which
        starts actor processes and performs the full training lifecycle.
        """
        if ParallelTrainer is None:
            raise RuntimeError("ParallelTrainer not available. Make sure alpha_zero.parallel_trainer is importable.")

        board_size = getattr(self.game, 'board_size', 8)
        if board_size <= 3:
            num_res_blocks = 2
            hidden_channels = 64
            value_hidden = 32
        else:
            num_res_blocks = 10
            hidden_channels = 256
            value_hidden = 128

        net_args = {
            "input_channels": getattr(self.game, 'input_channels', 6),
            "board_size": board_size,
            "action_size": getattr(self.game, 'action_size'),
            "num_res_blocks": num_res_blocks,
            "hidden_channels": hidden_channels,
            "value_hidden": value_hidden,
        }

        # merge user-supplied kwargs
        pt_kwargs = dict(self.parallel_kwargs)
        pt_kwargs.update({
            "ckpt_dir": ckpt_dir,
            "num_actors": num_actors,
        })

        self.parallel_trainer = ParallelTrainer(
            game_cls=self.game.__class__,
            net_constructor_args=net_args,
            config=self.config,
            ckpt_dir=ckpt_dir,
            num_actors=num_actors,
        )

        # run the training loop (blocking). You can also call trainer.train_loop in a thread/process
        self.parallel_trainer.train_loop(
            total_iters=total_iters,
            games_per_iter=games_per_iter,
            epochs_per_iter=epochs_per_iter,
            checkpoint_interval=checkpoint_interval,
        )

    # ---------------------- inference / utilities ----------------------
    def infer(self, state: Any, num_sims: int, player: int, mcts_tree: bool = False, deterministic: bool = True) -> int:
        """Run MCTS from (state, player) and return selected action.

        - Uses `num_sims` by updating self.config.num_simulations (MCTS reads config).
        - `deterministic`: if True choose argmax of visit-count policy; else sample from it.
        - `mcts_tree`: if True export a visualization (tries graphviz then falls back to text)
        """
        # ensure config & mcts agree
        self.config.num_simulations = num_sims
        self.mcts.config = self.config

        policy, _ = self.mcts.search(state, player)

        # Optionally render tree (Graphviz) and fallback to ASCII printing
        if mcts_tree:
            try:
                dot = self.mcts.export_tree(max_depth=3)
                # try to render to mounted trees folder if available
                out_path = "/app/trees/mcts_tree"
                dot.render(out_path, format="png")
            except Exception as e:
                # fallback to text printing instead of crashing
                print("Graphviz render failed (falling back to ascii). Reason:", e)
                if hasattr(self.mcts, 'root') and self.mcts.root is not None:
                    self._print_tree_ascii(self.mcts.root, max_depth=3)

        # Use visit-count policy (returned by _get_policy) — it's already normalized
        visit_policy = policy
        if deterministic:
            return int(np.argmax(visit_policy))
        else:
            # sample according to visit counts distribution
            p = np.array(visit_policy, dtype=np.float64)
            s = p.sum()
            if s <= 0:
                p = np.ones_like(p) / len(p)
            else:
                p = p / s
            return int(np.random.choice(len(p), p=p))

    def _print_tree_ascii(self, node, depth: int = 0, max_depth: int = 3):
        """Simple ASCII dump of an MCTS tree rooted at `node`."""
        indent = "  " * depth
        try:
            val = node.value
        except Exception:
            val = 0.0
        print(f"{indent}Node(action={node.action}, player={node.player}, visits={node.visits}, value={val:.3f}, prior={node.prior:.3f})")
        if depth >= max_depth:
            return
        for child in sorted(node.children, key=lambda c: -c.visits):
            self._print_tree_ascii(child, depth + 1, max_depth)

    # ---------------------- model I/O ----------------------
    def load(self, path: str):
        """Load a checkpoint to the agent's network. Works with CPU or CUDA checkpoints."""
        state = torch.load(path, map_location=self.net.device, weights_only=True)
        # If the saved checkpoint contains additional metadata, attempt to extract state_dict
        if isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()):
            # likely a state_dict already
            self.net.load_state_dict(state)
        elif isinstance(state, dict) and 'model_state_dict' in state:
            self.net.load_state_dict(state['model_state_dict'])
        else:
            # last resort: try to load directly
            try:
                self.net.load_state_dict(state)
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")

    def save(self, path: str):
        """Save current network state_dict to path (CPU-friendly)."""
        cpu_state = {k: v.cpu() for k, v in self.net.state_dict().items()}
        torch.save(cpu_state, path)

    # Convenience: if the agent was configured to use ParallelTrainer, expose the object
    def get_parallel_trainer(self):
        return getattr(self, 'parallel_trainer', None)
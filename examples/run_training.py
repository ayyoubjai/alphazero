import torch
import torch.multiprocessing as mp
from alpha_zero.parallel_trainer import ParallelTrainer
from alpha_zero.games.tic_tac_toe import TicTacToe
from alpha_zero.config import Config

if __name__ == "__main__":
    # spawn is required on many platforms when using CUDA and multiprocessing
    mp.set_start_method("spawn", force=True)

    cfg = Config()
    # Adjust these for faster local debugging
    cfg.num_simulations = 100        # MCTS sims per move (increase when stable)
    cfg.num_self_play_games = 20     # used only if you reuse earlier logic; here we control via games_per_iter
    cfg.temperature = 1.0
    cfg.batch_size = 64
    cfg.learning_rate = 1e-3

    net_args = {
        "input_channels": 3,
        "board_size": TicTacToe().board_size,
        "action_size": TicTacToe().action_size,
        "hidden_size": 64,
    }

    # Create trainer: learner uses GPU if available; actors will run on CPU
    trainer = ParallelTrainer(
        game_cls = TicTacToe,
        net_constructor_args = net_args,
        config = cfg,
        ckpt_dir = "models",
        num_actors = 2,           # start small locally, raise when comfortable
    )

    # Training driver: total_iters is number of learner iterations
    trainer.train_loop(
        total_iters = 200,        # number of outer iterations
        games_per_iter = 20,      # how many self-play games to collect per iter
        epochs_per_iter = 10,     # training epochs per iter
        checkpoint_interval = 5,  # how often (iters) to save latest.pth
    )

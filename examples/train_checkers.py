import multiprocessing as mp

from alpha_zero.agent import AlphaZeroAgent
from alpha_zero.games.checkers import Checkers

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Set to 'spawn' for CUDA compatibility
    game = Checkers()
    agent = AlphaZeroAgent(game)
    #agent.train(100, 10)  # Example: 100 games, 10 epochs each batch
    agent.start_parallel_training()
    agent.trainer.checkpoint("models/ckr_final.pth")
import multiprocessing as mp

from alpha_zero.agent import AlphaZeroAgent
from alpha_zero.games.tic_tac_toe import TicTacToe

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Set to 'spawn' for CUDA compatibility
    game = TicTacToe()
    agent = AlphaZeroAgent(game)
    agent.start_parallel_training()
    #agent.train(100, 10)  # Example: 100 games, 10 epochs each batch
    agent.trainer.checkpoint("models/ttt_latest.pth")
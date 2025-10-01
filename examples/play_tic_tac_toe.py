# play_tic_tac_toe.py (updated printing + input validation)
import numpy as np
from alpha_zero.agent import AlphaZeroAgent
from alpha_zero.games.tic_tac_toe import TicTacToe
import os
import sys

game = TicTacToe()
agent = AlphaZeroAgent(game)
# Assume trained, load if exists
model_path = "models/ttt_latest.pth"
if os.path.exists(model_path):
    agent.load(model_path)

def human_readable_cell(v):
    """Map internal cell value to human marker."""
    if v == 1:
        return "X"
    if v == -1:
        return "O"
    return "."

def board_to_grid(board):
    """Convert a 3x3 numeric board (numpy array) into a 3x3 of markers."""
    return [[human_readable_cell(int(board[r, c])) for c in range(3)] for r in range(3)]

def print_board(board, show_indices=False, clear_screen=False):
    """
    Nicely print the board.
    - board: numpy array shape (3,3) with values -1,0,1
    - show_indices: if True, print a second grid showing indices 0..8 for player reference
    - clear_screen: if True, attempt to clear terminal before printing (nice for interactive play)
    """
    if clear_screen:
        # best-effort clear (works on most terminals)
        os.system('cls' if os.name == 'nt' else 'clear')

    grid = board_to_grid(board)
    # draw with borders
    print(" ┌───┬───┬───┐")
    for r in range(3):
        print(" │ " + " │ ".join(grid[r]) + " │")
        if r < 2:
            print(" ├───┼───┼───┤")
    print(" └───┴───┴───┘")

    if show_indices:
        # show index map under the board so players can pick 0..8
        indices = [[str(r*3 + c) for c in range(3)] for r in range(3)]
        print("\nIndex map:")
        print(" ┌───┬───┬───┐")
        for r in range(3):
            print(" │ " + " │ ".join(indices[r]) + " │")
            if r < 2:
                print(" ├───┼───┼───┤")
        print(" └───┴───┴───┘")

def valid_human_move(board, action):
    """Return True if action is in 0..8 and the cell is empty."""
    if not (0 <= action <= 8):
        return False
    r, c = divmod(action, 3)
    return board[r, c] == 0

# initial state
state = game.get_initial_state()
player = 1  # Human starts as 1

# show indices on first turn so the player knows the mapping
first_turn = True

while True:
    # print board (clear screen for nicer UX)
    print_board(state['board'], show_indices=first_turn, clear_screen=True)
    first_turn = False

    if player == 1:
        # Human move; validate input
        while True:
            try:
                raw = input("Your action (0-8): ").strip()
                action = int(raw)
            except (ValueError, TypeError):
                print("Please enter an integer between 0 and 8 (the index of the cell).")
                continue

            if not valid_human_move(state['board'], action):
                print("Invalid move: either out of range or cell already occupied. Try again.")
                continue

            break

        state, next_player = game.get_next_state(state, action, player)
        done, result = game.is_terminal(state, next_player)
        if done:
            print_board(state['board'], clear_screen=False)
            if result == 0:
                print("Game over: Draw")
            else:
                print(f"Game over: Winner is the player with {player} ({human_readable_cell(player)})")
            break
        player = next_player
    else:
        # AI move
        print("AI is thinking...", file=sys.stderr)
        ai_action = agent.infer(state, num_sims=100, player=player, mcts_tree=True)
        state, next_player = game.get_next_state(state, ai_action, player)
        done, result = game.is_terminal(state, next_player)
        if done:
            print_board(state['board'], clear_screen=False)
            if result == 0:
                print("Game over: Draw")
            else:
                print(f"Game over: Winner is the player with {player} ({human_readable_cell(player)})")
            break
        player = next_player

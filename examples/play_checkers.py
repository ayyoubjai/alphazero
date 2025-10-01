import numpy as np
from alpha_zero.agent import AlphaZeroAgent
from alpha_zero.games.checkers import Checkers
import os
import sys

game = Checkers()
agent = AlphaZeroAgent(game)
# If you have a trained model, load it:
# agent.load("models/ckr_latest.pth")

def human_readable_cell(v):
    """Map numeric cell value to human marker."""
    if v == 1:
        return "W"  # White
    if v == -1:
        return "B"  # Black
    return "."

def print_board(board, show_coords=True, clear_screen=False):
    """
    Nicely print an 8x8 checkers board.
    - board: numpy array shape (8,8)
    - show_coords: if True, display row/column indices
    - clear_screen: clear terminal for interactive play
    """
    if clear_screen:
        os.system('cls' if os.name == 'nt' else 'clear')

    rows, cols = board.shape
    grid = [[human_readable_cell(int(board[r, c])) for c in range(cols)] for r in range(rows)]

    if show_coords:
        # column headers
        print("    " + " ".join(str(c) for c in range(cols)))
        print("   " + "--"*cols)

    for r in range(rows):
        line = " ".join(grid[r])
        if show_coords:
            print(f"{r} | {line}")
        else:
            print(line)
    print()

def parse_coord_item(item: str):
    """Parse a single coordinate input, either int (0-63) or 'r,c'."""
    item = item.strip()
    if not item:
        raise ValueError("Empty coordinate item")
    if ',' in item:
        parts = item.split(',')
        if len(parts) != 2:
            raise ValueError(f"Bad coordinate format: {item}")
        r = int(parts[0].strip())
        c = int(parts[1].strip())
        if not (0 <= r < 8 and 0 <= c < 8):
            raise ValueError(f"Coordinates out of bounds: {item}")
        return (r, c)
    # else assume single index 0..63
    idx = int(item)
    if not (0 <= idx < 64):
        raise ValueError(f"Index out of bounds: {idx}")
    r, c = divmod(idx, 8)
    return (r, c)

def parse_list_input(prompt: str):
    """
    Prompt user for semicolon-separated items.
    Each item can be an int 0..63 or 'r,c' coordinates.
    Returns a list of parsed coordinates.
    """
    s = input(prompt).strip()
    if s == "":
        return []
    items = [x.strip() for x in s.split(';') if x.strip() != ""]
    return [parse_coord_item(it) for it in items]

# Initial game state
state = game.get_initial_state()
current_player = 1  # start with white; adjust if desired

first_turn = True

while True:
    # Print board nicely
    print_board(state["board"], show_coords=first_turn, clear_screen=True)
    first_turn = False
    print(f"Player to move: {current_player} ({'white' if current_player==1 else 'black'})\n")

    # Human input for moves
    try:
        from_list = parse_list_input("Enter FROM squares (semicolon-separated, as index 0..63 or r,c): ")
        to_list   = parse_list_input("Enter TO   squares (semicolon-separated, same length): ")

        if len(from_list) == 0 or len(to_list) == 0:
            print("No move entered — please enter at least one from/to pair.")
            continue
        if len(from_list) != len(to_list):
            print("FROM and TO lists must have the same length.")
            continue

        # Apply human move(s)
        state, next_player = game.get_next_state_human(state, from_list, to_list, current_player)

    except ValueError as e:
        print(f"Invalid move or input: {e}. Please try again.")
        continue

    # Check terminal state for next player
    is_term, score = game.is_terminal(state, next_player)
    if is_term:
        print_board(state["board"], show_coords=True, clear_screen=False)
        if score == 0.0:
            print("Game over: Draw.")
        elif score == -1.0:
            winner = -next_player
            print(f"Game over: Winner is player {winner} ({'white' if winner==1 else 'black'})")
        else:
            print(f"Game over: terminal score {score}")
        break

    current_player = next_player

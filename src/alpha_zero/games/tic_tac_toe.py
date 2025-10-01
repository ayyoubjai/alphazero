import numpy as np
from typing import List, Tuple, Any
from collections import Counter

from .abstract_game import AbstractGame

class TicTacToe(AbstractGame):
    def __init__(self):
        self.board_size = 3
        self.history_length = 8  # Like AlphaZero, stack 8 positions (7 past + current)

    @property
    def action_size(self) -> int:
        return self.board_size ** 2

    @property
    def input_channels(self) -> int:
        # 2 piece planes * history_length + 1 constant plane (player)
        return 2 * self.history_length + 1

    def get_initial_state(self) -> Any:
        board = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        return {'board': board, 'history': []}

    def get_legal_actions(self, state: Any, player: int) -> List[int]:
        board = state['board']
        return [i for i in range(self.action_size) if board.flatten()[i] == 0]

    def get_next_state(self, state: Any, action: int, player: int) -> Tuple[Any, int]:
        board = state['board'].copy()
        row, col = divmod(action, self.board_size)
        board[row, col] = player
        prev_state = (state['board'].copy(), player)
        history = state['history'] + [prev_state]
        history = history[-(self.history_length - 1):]  # Keep last (history_length - 1) past states
        new_state = {'board': board, 'history': history}
        return new_state, -player

    def is_terminal(self, state: Any, player: int) -> Tuple[bool, float]:
        board = state['board']
        lines = [board[i] for i in range(3)] + [board[:, i] for i in range(3)] + [np.diag(board), np.diag(np.fliplr(board))]
        for line in lines:
            if np.all(line == line[0]) and line[0] != 0:
                return True, -1.0  # Loss from current player's perspective (winner was previous player)
        if np.all(board != 0):
            return True, 0.0  # Draw
        # Check for repetition draw
        current_board_tuple = tuple(tuple(row) for row in board)
        current_full = (current_board_tuple, player)
        past_fulls = [(tuple(tuple(row) for row in h_board), h_pl) for h_board, h_pl in state['history']]
        all_fulls = past_fulls + [current_full]
        counts = Counter(all_fulls)
        if any(count >= 3 for count in counts.values()):
            return True, 0.0  # Draw by 3-fold repetition
        return False, 0.0

    def get_encoded_state(self, state: Any, player: int) -> np.ndarray:
        board = state['board']
        # Current piece planes
        player_one = (board == 1).astype(np.float32)
        player_two = (board == -1).astype(np.float32)
        encs = [player_one, player_two]

        # History piece planes
        history_boards = [h[0] for h in state['history']]  # h[0] is past board
        for past_board in history_boards[-(self.history_length - 1):]:
            p1 = (past_board == 1).astype(np.float32)
            p2 = (past_board == -1).astype(np.float32)
            encs.extend([p1, p2])

        # Pad missing history with zeros
        missing = self.history_length - (1 + len(history_boards))
        for _ in range(missing):
            zero_plane = np.zeros_like(player_one)
            encs.extend([zero_plane] * 2)

        # Constant plane: player (binary: 1 for white, 0 for black)
        player_plane = np.full_like(player_one, 1.0 if player == 1 else 0.0)
        encs.append(player_plane)

        return np.stack(encs, axis=0)

    def get_opponent_perspective(self, state: Any) -> Any:
        board = -state['board']
        history = []
        for h_board, h_pl in state['history']:
            flipped_h_board = -h_board
            flipped_h_pl = -h_pl
            history.append((flipped_h_board, flipped_h_pl))
        return {'board': board, 'history': history}

    def is_continuation_end(self, state: Any) -> bool:
        return False

    def get_symmetries(self, state: Any, policy: np.ndarray) -> List[Tuple[Any, np.ndarray]]:
        symmetries = []
        board = state['board']
        policy_shaped = policy.reshape(self.board_size, self.board_size)
        # Rotations
        for i in range(4):
            rot_board = np.rot90(board, i)
            rot_policy = np.rot90(policy_shaped, i).flatten()
            rot_history = []
            for h_board, h_pl in state['history']:
                rot_h_board = np.rot90(h_board, i)
                rot_history.append((rot_h_board, h_pl))
            rot_state = {'board': rot_board, 'history': rot_history}
            symmetries.append((rot_state, rot_policy))
            # Flips
            flip_board = np.fliplr(rot_board)
            flip_policy = np.fliplr(np.rot90(policy_shaped, i)).flatten()
            flip_history = []
            for h_board, h_pl in state['history']:
                flip_h_board = np.fliplr(np.rot90(h_board, i))
                flip_history.append((flip_h_board, h_pl))
            flip_state = {'board': flip_board, 'history': flip_history}
            symmetries.append((flip_state, flip_policy))
        return symmetries
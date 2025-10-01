import numpy as np
from typing import List, Tuple, Any
import copy
from collections import Counter

from .abstract_game import AbstractGame

class Checkers(AbstractGame):
    def __init__(self):
        self.board_size = 8
        self._action_size = self.board_size**2 * self.board_size**2  # from_idx * 64 + to_idx = 4096
        self.history_length = 8  # Like AlphaZero, stack 8 positions (7 past + current)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def input_channels(self) -> int:
        # 4 piece planes * history_length + 2 constant planes (continuation, player)
        return 4 * self.history_length + 2

    def get_initial_state(self) -> Any:
        board = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        # Pieces on dark squares ((r + c) % 2 == 0)
        # Row 0-2: black (-1 men), row 5-7: white (1 men)
        for r in range(3):
            for c in range(self.board_size):
                if (r + c) % 2 == 0:
                    board[r, c] = -1
        for r in range(5, 8):
            for c in range(self.board_size):
                if (r + c) % 2 == 0:
                    board[r, c] = 1
        return {'board': board, 'jumping_pos': None, 'history': []}

    def get_legal_actions(self, state: Any, player: int) -> List[int]:
        board = state['board']
        jumping_pos = state['jumping_pos']
        actions = []
        has_capture = self.has_any_capture(board, player, jumping_pos)
        pieces = self._get_player_pieces(board, player)
        if jumping_pos is not None:
            pieces = [jumping_pos]
        for r, c in pieces:
            is_king = abs(board[r, c]) == 2
            if not has_capture and jumping_pos is None:
                moves = self._get_possible_moves(board, r, c, player, is_king)
                actions += [self._encode_action(r, c, tr, tc) for tr, tc in moves]
            captures = self._get_possible_captures(board, r, c, player, is_king)
            actions += [self._encode_action(r, c, tr, tc) for tr, tc in captures]
        return list(set(actions))  # dedup if any

    def get_next_state(self, state: Any, action: int, player: int) -> Tuple[Any, int]:
        board = state['board'].copy()
        r, c = divmod(action // 64, 8)
        tr, tc = divmod(action % 64, 8)
        piece = board[r, c]
        is_capture = self._perform_move(board, r, c, tr, tc, player)
        # Promotion
        if abs(piece) == 1 and self._is_promotion_row(tr, player):
            board[tr, tc] = player * 2
        # Determine continuation
        next_jumping_pos = None
        next_player = -player
        if is_capture:
            can_continue = self._has_captures_from(board, tr, tc, player)
            if can_continue:
                next_jumping_pos = (tr, tc)
                next_player = player
        # Update history: append (previous board, previous jumping_pos, previous player)
        prev_state = (state['board'].copy(), state['jumping_pos'], player)
        history = state['history'] + [prev_state]
        history = history[-(self.history_length - 1):]  # Keep last (history_length - 1) past states
        new_state = {'board': board, 'jumping_pos': next_jumping_pos, 'history': history}
        return new_state, next_player

    def get_next_state_human(self, state: Any, from_list: List[Any], to_list: List[Any], player: int) -> Tuple[Any, int]:
        """Apply a sequence of human-provided moves to the given state.

        Args:
            state: the current game state (dictionary as produced by this class)
            from_list: list of source squares. Each item can be either an (r, c) tuple or an int 0..63.
            to_list:   list of destination squares. Each item can be either an (r, c) tuple or an int 0..63.
            player: current player (1 for white, -1 for black)

        Returns:
            (new_state, next_player) after applying the sequence.

        The function validates each step against get_legal_actions for the current sub-state and player.
        If an invalid or illegal move is encountered, a ValueError is raised indicating which step failed.

        Typical usage:
            new_state, next_player = game.get_next_state_human(state, [(5,2)], [(3,0)], 1)
        or with flattened indices:
            new_state, next_player = game.get_next_state_human(state, [42], [26], -1)
        """
        if len(from_list) != len(to_list):
            raise ValueError("from_list and to_list must have the same length")

        # Helpers
        def _normalize_coord(coord):
            # Accept either an int in [0,63] or a (r,c) pair with 0<=r,c<board_size
            if isinstance(coord, int):
                if 0 <= coord < self.board_size * self.board_size:
                    return divmod(coord, self.board_size)
                raise ValueError(f"Invalid square index: {coord}")
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                r, c = coord
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    return (int(r), int(c))
                raise ValueError(f"Invalid square coordinates: {coord}")
            raise ValueError(f"Invalid coordinate format: {coord}")

        # Work on a deep copy to avoid mutating the provided state
        cur_state = copy.deepcopy(state)
        cur_player = player

        for step, (f_raw, t_raw) in enumerate(zip(from_list, to_list), start=1):
            fr, fc = _normalize_coord(f_raw)
            tr, tc = _normalize_coord(t_raw)
            action = self._encode_action(fr, fc, tr, tc)

            legal = self.get_legal_actions(cur_state, cur_player)
            if action not in legal:
                # Provide a helpful error with step number and readable coords
                raise ValueError(f"Illegal action at step {step}: from {(fr,fc)} to {(tr,tc)} for player {cur_player}")

            cur_state, next_player = self.get_next_state(cur_state, action, cur_player)
            cur_player = next_player

        return cur_state, cur_player

    def is_terminal(self, state: Any, player: int) -> Tuple[bool, float]:
        legal = self.get_legal_actions(state, player)
        if legal:
            # Check for repetition draw
            current_board_tuple = tuple(tuple(row) for row in state['board'])
            current_jp = state['jumping_pos'] or (-1, -1)
            current_full = (current_board_tuple, current_jp, player)
            past_fulls = [(tuple(tuple(row) for row in h_board), h_jp or (-1, -1), h_pl) for h_board, h_jp, h_pl in state['history']]
            all_fulls = past_fulls + [current_full]
            counts = Counter(all_fulls)
            if any(count >= 3 for count in counts.values()):
                return True, 0.0  # Draw by 3-fold repetition
            return False, 0.0
        # No moves: loss if not in continuation, else handled in eval
        if state['jumping_pos'] is None:
            return True, -1.0
        return False, 0.0

    def get_encoded_state(self, state: Any, player: int) -> np.ndarray:
        board = state['board']
        # Current piece planes
        white_men = (board == 1).astype(np.float32)
        white_kings = (board == 2).astype(np.float32)
        black_men = (board == -1).astype(np.float32)
        black_kings = (board == -2).astype(np.float32)
        encs = [white_men, white_kings, black_men, black_kings]

        # History piece planes (only boards, ignore past jp/player)
        history_boards = [h[0] for h in state['history']]  # h[0] is past board
        for past_board in history_boards[-(self.history_length - 1):]:
            wm = (past_board == 1).astype(np.float32)
            wk = (past_board == 2).astype(np.float32)
            bm = (past_board == -1).astype(np.float32)
            bk = (past_board == -2).astype(np.float32)
            encs.extend([wm, wk, bm, bk])

        # Pad missing history with zeros
        missing = self.history_length - (1 + len(history_boards))
        for _ in range(missing):
            zero_plane = np.zeros_like(white_men)
            encs.extend([zero_plane] * 4)

        # Constant planes: continuation and player (binary: 1 for white, 0 for black)
        continuation = np.zeros_like(white_men)
        if state['jumping_pos'] is not None:
            r, c = state['jumping_pos']
            continuation[r, c] = 1.0
        player_plane = np.full_like(white_men, 1.0 if player == 1 else 0.0)
        encs.extend([continuation, player_plane])

        return np.stack(encs, axis=0)

    def get_opponent_perspective(self, state: Any) -> Any:
        board = -state['board'][::-1, ::-1]  # flip signs and board vertically and horizontally
        jumping_pos = state['jumping_pos']
        if jumping_pos is not None:
            r, c = jumping_pos
            jumping_pos = (self.board_size - 1 - r, self.board_size - 1 - c)
        history = []
        for h_board, h_jp, h_pl in state['history']:
            flipped_h_board = -h_board[::-1, ::-1]
            flipped_h_jp = None
            if h_jp is not None:
                hr, hc = h_jp
                flipped_h_jp = (self.board_size - 1 - hr, self.board_size - 1 - hc)
            flipped_h_pl = -h_pl
            history.append((flipped_h_board, flipped_h_jp, flipped_h_pl))
        return {'board': board, 'jumping_pos': jumping_pos, 'history': history}

    def is_continuation_end(self, state: Any) -> bool:
        return state['jumping_pos'] is not None

    def get_symmetries(self, state: Any, policy: np.ndarray) -> List[Tuple[Any, np.ndarray]]:
        # Left-right flip for checkers symmetry
        symmetries = []
        # Original
        symmetries.append((state, policy))
        # Flip left-right
        flipped_board = state['board'][:, ::-1]
        flipped_jp = None
        if state['jumping_pos'] is not None:
            r, c = state['jumping_pos']
            flipped_jp = (r, self.board_size - 1 - c)
        flipped_history = []
        for h_board, h_jp, h_pl in state['history']:
            flipped_h_board = h_board[:, ::-1]
            flipped_h_jp = None
            if h_jp is not None:
                hr, hc = h_jp
                flipped_h_jp = (hr, self.board_size - 1 - hc)
            flipped_history.append((flipped_h_board, flipped_h_jp, h_pl))  # player unchanged for symmetry
        flipped_state = {'board': flipped_board, 'jumping_pos': flipped_jp, 'history': flipped_history}
        flipped_policy = np.zeros_like(policy)
        for a in range(len(policy)):
            fr, fc = divmod(a // 64, 8)
            tr, tc = divmod(a % 64, 8)
            flipped_a = self._encode_action(fr, self.board_size - 1 - fc, tr, self.board_size - 1 - tc)
            flipped_policy[flipped_a] = policy[a]
        symmetries.append((flipped_state, flipped_policy))
        return symmetries

    # Private helpers
    def _encode_action(self, fr: int, fc: int, tr: int, tc: int) -> int:
        return ((fr * self.board_size + fc) * self.board_size**2) + (tr * self.board_size + tc)

    def _get_player_pieces(self, board: np.ndarray, player: int) -> List[Tuple[int, int]]:
        return list(zip(*np.where((board * player > 0))))

    def has_any_capture(self, board: np.ndarray, player: int, jumping_pos) -> bool:
        pieces = self._get_player_pieces(board, player)
        if jumping_pos is not None:
            pieces = [jumping_pos]
        for r, c in pieces:
            is_king = abs(board[r, c]) == 2
            if len(self._get_possible_captures(board, r, c, player, is_king)) > 0:
                return True
        return False

    def _get_possible_moves(self, board: np.ndarray, r: int, c: int, player: int, is_king: bool) -> List[Tuple[int, int]]:
        moves = []
        directions = self._get_directions(player, is_king)
        for dr, dc in directions:
            tr, tc = r + dr, c + dc
            if is_king:
                # Flying: advance until edge or occupied
                while 0 <= tr < self.board_size and 0 <= tc < self.board_size and board[tr, tc] == 0:
                    moves.append((tr, tc))
                    tr += dr
                    tc += dc
            else:
                if 0 <= tr < self.board_size and 0 <= tc < self.board_size and board[tr, tc] == 0:
                    moves.append((tr, tc))
        return moves

    def _get_possible_captures(self, board: np.ndarray, r: int, c: int, player: int, is_king: bool) -> List[Tuple[int, int]]:
        captures = []
        directions = self._get_directions(player, is_king)
        for dr, dc in directions:
            dist = 1
            tr, tc = r + dr * dist, c + dc * dist
            found_opponent = False
            while 0 <= tr < self.board_size and 0 <= tc < self.board_size:
                if board[tr, tc] * player > 0:  # own piece
                    break
                if board[tr, tc] * player < 0:  # opponent
                    if found_opponent:
                        break  # multiple opponents in line not allowed
                    found_opponent = True
                    dist += 1
                    tr, tc = r + dr * dist, c + dc * dist
                    continue
                if found_opponent:
                    # Empty after opponent
                    captures.append((tr, tc))
                    if not is_king:
                        break  # men can't fly
                elif not found_opponent and not is_king:
                    break  # men can't skip empties without opponent
                dist += 1
                tr, tc = r + dr * dist, c + dc * dist
        return captures

    def _get_directions(self, player: int, is_king: bool) -> List[Tuple[int, int]]:
        forward = -player  # white=1 moves -row (up), black=-1 moves +row (down)
        dirs = [(forward, -1), (forward, 1)]  # forward left/right
        if is_king:
            dirs += [(-forward, -1), (-forward, 1)]
        return dirs

    def _perform_move(self, board: np.ndarray, r: int, c: int, tr: int, tc: int, player: int) -> bool:
        is_capture = False
        board[tr, tc] = board[r, c]
        board[r, c] = 0
        # Find and remove captured if any
        dr = 1 if tr > r else -1 if tr < r else 0
        dc = 1 if tc > c else -1 if tc < c else 0
        if abs(tr - r) != abs(tc - c) or dr == 0 or dc == 0:
            raise ValueError("Invalid move: not diagonal")
        pos_r, pos_c = r + dr, c + dc
        while pos_r != tr or pos_c != tc:
            if board[pos_r, pos_c] * player < 0:
                board[pos_r, pos_c] = 0
                is_capture = True
            pos_r += dr
            pos_c += dc
        return is_capture

    def _is_promotion_row(self, row: int, player: int) -> bool:
        return (player == 1 and row == 0) or (player == -1 and row == self.board_size - 1)

    def _has_captures_from(self, board: np.ndarray, r: int, c: int, player: int) -> bool:
        is_king = abs(board[r, c]) == 2
        return len(self._get_possible_captures(board, r, c, player, is_king)) > 0

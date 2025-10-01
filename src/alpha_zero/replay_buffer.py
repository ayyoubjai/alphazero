from collections import deque
import numpy as np
from typing import List, Tuple, Any

class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def add(self, state: Any, policy: np.ndarray, value: float, player: int):
        self.buffer.append((state, policy, value, player))

    def sample(self, batch_size: int) -> Tuple[List[Any], List[np.ndarray], List[float], List[int]]:
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        states, policies, values, players = [], [], [], []
        for i in indices:
            s, p, v, pl = self.buffer[i]
            states.append(s)
            policies.append(p)
            values.append(v)
            players.append(pl)
        return states, policies, values, players

    def __len__(self) -> int:
        return len(self.buffer)
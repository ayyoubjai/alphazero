
import numpy as np
from typing import List, Any

class Node:
    def __init__(self, state: Any, prior=0.0, parent=None, action=None, player=1):
        self.state = state
        self.prior = float(prior)
        self.parent = parent
        self.action = action               # action taken from parent -> this node
        self.player = player               # player to move at THIS node
        self.children: List["Node"] = []
        self.visits = 0
        self.value_sum = 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        """Mean value at this node from this node's player perspective.

        Note: when selecting from the parent, use -child.value to get the value
        in the parent's player's perspective (players alternate).
        """
        return 0.0 if self.visits == 0 else float(self.value_sum / self.visits)
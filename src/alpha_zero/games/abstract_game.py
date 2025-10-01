from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Any

class AbstractGame(ABC):
    @abstractmethod
    def get_initial_state(self) -> Any:
        pass

    @abstractmethod
    def get_legal_actions(self, state: Any, player: int) -> List[int]:
        pass

    @abstractmethod
    def get_next_state(self, state: Any, action: int, player: int) -> Tuple[Any, int]:
        pass

    @abstractmethod
    def is_terminal(self, state: Any, player: int) -> Tuple[bool, float]:
        pass  # Returns (terminal, value from current player's view)

    @abstractmethod
    def get_encoded_state(self, state: Any, player: int) -> np.ndarray:
        pass  # For NN input

    @abstractmethod
    def get_opponent_perspective(self, state: Any) -> Any:
        pass

    @abstractmethod
    def is_continuation_end(self, state: Any) -> bool:
        pass  # For games with continuations; return False for others

    @property
    @abstractmethod
    def action_size(self) -> int:
        pass

    @abstractmethod
    def get_symmetries(self, state: Any, policy: np.ndarray) -> List[Tuple[Any, np.ndarray]]:
        pass  # For data augmentation
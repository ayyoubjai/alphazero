import numpy as np
import logging
from typing import List

logging.basicConfig(level=logging.INFO)

def add_dirichlet_noise(policy: np.ndarray, alpha: float, epsilon: float) -> np.ndarray:
    noise = np.random.dirichlet([alpha] * len(policy))
    return (1 - epsilon) * policy + epsilon * noise

def augment_data(states: List[np.ndarray], policies: List[np.ndarray], values: List[float], symmetries) -> tuple:
    # Example for Tic-Tac-Toe: rotations and flips
    augmented_states, augmented_policies, augmented_values = [], [], []
    for s, p, v in zip(states, policies, values):
        for sym in symmetries:
            aug_s = sym(s)
            aug_p = sym(p)  # Remap actions
            augmented_states.append(aug_s)
            augmented_policies.append(aug_p)
            augmented_values.append(v)
    return augmented_states, augmented_policies, augmented_values


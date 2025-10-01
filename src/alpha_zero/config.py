from dataclasses import dataclass

@dataclass
class Config:
    num_simulations: int = 10
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.5
    buffer_size: int = 10000
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 10
    num_self_play_games: int = 100
    num_workers: int = 4
    checkpoint_interval: int = 50
    l2_reg: float = 1e-4 
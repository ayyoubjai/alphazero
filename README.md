# AlphaZero Training and Playing Pipeline

This project implements an AlphaZero-style reinforcement learning pipeline for training and playing board games like Tic-Tac-Toe (ttt) and Checkers (ckr). It leverages Monte Carlo Tree Search (MCTS), self-play, and neural network training, with two training implementations: `trainer.py` (synchronous) and `parallel_trainer.py` (asynchronous actor-learner). The project runs inside a Docker container, with Poetry managing Python dependencies.

## Project Structure

- `alpha_zero/`
  - `main.py`: Defines the CLI entrypoint (`alpha_zero.cli:main`) for building, training, and playing.
- `examples/`: Example scripts for training and playing Tic-Tac-Toe and Checkers.
- `models/`: Stores `.pth` checkpoint files (mounted to Docker container).
- `trees/`: Stores `.png` and Graphviz files visualizing MCTS trees during gameplay (mounted to Docker container).
- `src/alpha_zero/`
  - `games/`: Game implementations (e.g., Tic-Tac-Toe, Checkers).
  - `mcts/`: Monte Carlo Tree Search logic.
  - `networks/`: Neural network definitions (BaseNet).
  - `agent.py`: Agent logic for playing games.
  - `config.py`: Training hyperparameters.
  - `parallel_trainer.py`: Asynchronous actor-learner training pipeline.
  - `replay_buffer.py`: Replay buffer for storing trajectories.
  - `trainer.py`: Synchronous training pipeline.
  - `utils.py`: Utility functions.
- `Dockerfile`: Defines the Docker container setup.
- `poetry.lock`, `pyproject.toml`: Poetry dependency management files.
- `requirements.txt`: Exported dependencies for reference.
- `README.md`: This file.

## Prerequisites

- **Docker**: Ensure Docker Engine is installed and running.
- **Poetry**: Install Poetry for dependency management (`pip install poetry`).
- **Python**: Version 3.8+ (managed via Poetry inside the container).

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Poetry dependencies**:
   ```bash
   poetry install
   ```

3. **Build the Docker container** (Docker Engine must be running):
   ```bash
   poetry run alpha_zero --build
   ```

   This builds the container defined in `Dockerfile`, mounting `models/` and `trees/` directories for persistent storage of checkpoints and MCTS visualizations.

## Usage

The project is run via the CLI defined in `alpha_zero/main.py`, with the entrypoint `poetry run alpha_zero`. Available commands:

- **`--build`**: Builds the Docker container. Run this first to set up the environment.
  ```bash
  poetry run alpha_zero --build
  ```

- **`--train <game>`**: Trains a model for the specified game (`ttt` for Tic-Tac-Toe, `ckr` for Checkers). Outputs checkpoints to `models/`.
  ```bash
  poetry run alpha_zero --train ttt
  ```

- **`--play <game>`**: Plays the specified game (`ttt` or `ckr`) using a trained model. Saves MCTS visualizations to `trees/`.
  ```bash
  poetry run alpha_zero --play ckr
  ```

**Note**: The `models/` and `trees/` directories are mounted to the Docker container, ensuring persistence across runs.

## Differences Between `trainer.py` and `parallel_trainer.py`

This section explains the key differences between the two training implementations for the AlphaZero-style pipeline. Both scripts handle self-play game generation using MCTS, data augmentation via symmetries, replay buffer management, and neural network training with policy (cross-entropy) and value (MSE) losses plus L2 regularization. However, they differ significantly in architecture, parallelism, and execution flow, impacting scalability and efficiency.

### Overview of `trainer.py`

The `trainer.py` implementation follows a **synchronous, single-process design** with batched parallel self-play. It generates a fixed number of self-play games in parallel using a `multiprocessing.Pool`, adds the trajectories to the replay buffer (including augmentations), and then performs training epochs sequentially. The model remains fixed during each batch of self-play, and the process repeats in a loop. This is straightforward for prototyping but can be less efficient for large-scale training due to idle times during data collection or training phases.
```mermaid
graph TD
  A[Start] --> B[Self-Play (Parallel via Pool)]
  B --> C[Add Trajectories to Replay Buffer (with Symmetries)]
  C --> D[Train Neural Network (Epochs)]
  D --> E[Checkpoint Model]
  E -->|Repeat| B
```
*Diagram: Synchronous pipeline with sequential self-play, training, and evaluation stages, as used in `trainer.py`.*

### Overview of `parallel_trainer.py`

In contrast, `parallel_trainer.py` uses an **asynchronous actor-learner architecture**. Multiple actor processes run continuously on CPU, generating self-play trajectories independently and enqueueing them via a shared queue. The central learner (typically on GPU) periodically collects trajectories, adds them to the buffer with augmentations, trains for several epochs, and saves checkpoints. Actors poll for updated checkpoints and reload the model dynamically, allowing self-play to improve over time without restarting. This design enables better resource utilization, live model updates, and scalability with more actors, making it suitable for long-running distributed training.
```mermaid
graph TD
  A[Start] --> B[Spawn Actors]
  B --> C[Actors: Continuous Self-Play on CPU]
  C --> D[Enqueue Trajectories]
  D --> E[Learner: Collect Trajectories]
  E --> F[Add to Replay Buffer (with Symmetries)]
  F --> G[Train Neural Network (Epochs, GPU)]
  G --> H[Save Checkpoint]
  H --> I[Actors Reload Checkpoint]
  I --> C
  H -->|Repeat| E
```
*Diagram: Asynchronous actor-learner setup with a global learner and multiple actors, as used in `parallel_trainer.py`.*

### Key Differences

| Aspect | `trainer.py` (Synchronous) | `parallel_trainer.py` (Asynchronous) |
| --- | --- | --- |
| **Parallelism** | Uses `multiprocessing.Pool` for batched self-play games; waits for completion. | Persistent actor processes generate games continuously; learner collects via queue. |
| **Model Updates** | Model fixed per self-play batch; updated only during training. | Actors reload latest checkpoint periodically for improved self-play. |
| **Device Usage** | Shared device (e.g., GPU) for net; single process. | Actors on CPU; learner on GPU if available. |
| **Training Loop** | Generate games → Train epochs → Repeat; skips if buffer small. | Collect games (with timeouts) → Train epochs → Checkpoint; actors always running. |
| **Trajectory Labeling** | Values labeled from initial player's perspective. | Values adjusted based on terminal player's view; includes player in buffer. |
| **Robustness** | Basic; no dynamic handling. | Graceful shutdown, timeouts, error handling for queues/checkpoints. |
| **Scalability** | Limited by pool size; good for small setups. | Scales with num_actors; ideal for large, long trainings. |

### When to Choose Each

- **Use** `trainer.py` for quick experiments, small-scale training, or resource-constrained environments.
- **Use** `parallel_trainer.py` for efficient, scalable training with ongoing model improvements during self-play.

**Note**: Ensure consistency in game interface methods (e.g., `is_terminal` value conventions) to avoid bugs, as the two implementations handle player perspectives differently.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, new game implementations, or enhancements.

## License

This project is licensed under the MIT License.
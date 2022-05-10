from typing import Tuple

import numpy as np


def dense_reward(target: np.ndarray, state: np.ndarray) -> float:
    """Generates dense rewards as euclidean error norm of state and target vector

    Args:
        target (np.ndarray): target state vector of dimension (n)
        state (np.ndarray): state vector of dimension (m)

    Returns:
        float: reward
    """

    return -np.linalg.norm(target - state)

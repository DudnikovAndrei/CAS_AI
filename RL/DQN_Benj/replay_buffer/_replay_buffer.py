from collections import deque
from typing import Tuple, List

import numpy as np

from experience import Experience


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer.
        """
        self.buffer.append(experience)

    def extend(self, experiences: List[Experience]) -> None:
        """
        Extend buffer with experience
        """
        self.buffer.extend(experiences)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool),
                np.array(next_states))

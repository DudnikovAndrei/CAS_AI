from typing import List, Tuple

import numpy as np


class Experience:
    state: np.ndarray
    action: int
    reward: float
    done: bool
    next_state: np.ndarray
    _iterator_index: int

    def __init__(self, state: np.ndarray, action: int, reward: float, done: bool, next_state: np.ndarray):
        self.action = action
        self.reward = reward
        self.done = done

        if state.shape != (1, 4):
            state = np.expand_dims(state, axis=0)
        if next_state.shape != (1, 4):
            next_state = np.expand_dims(next_state, axis=0)

        self.state = state
        self.next_state = next_state

        self._properties = [self.state, self.action, self.reward, self.next_state, self.done]

    def __iter__(self):
        self._iterator_index = 0
        return self

    def __next__(self):
        if self._iterator_index <= 4:
            property = self._properties[self._iterator_index]
            self._iterator_index += 1
            return property
        else:
            raise StopIteration


def experiences_to_numpy(experiences: List[Experience]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _states = []
    _actions = []
    _rewards = []
    _next_states = []
    _dones = []
    for experience in experiences:
        _states.append(experience.state)
        _actions.append(experience.action)
        _rewards.append(experience.reward)
        _next_states.append(experience.next_state)
        _dones.append(experience.done)
    try:

        states = np.squeeze(np.array(_states).astype(np.float32))
        actions = np.array(_actions).astype(np.int32)
        rewards = np.array(_rewards).astype(np.float32)
        next_states = np.squeeze(np.array(_next_states).astype(np.float32))
        dones = np.array(_dones)
    except ValueError:
        a = 1
        raise
    return states, actions, rewards, next_states, dones

import gymnasium as gym
from gymnasium.spaces import Space

class MultiAgentActionSpace(Space):
    def __init__(self, array):
        super().__init__()
        for x in array:
            assert isinstance(x, Space), f"Expected gymnasium Space instance, got {type(x)}"
        self.agents_action_space = array

    def sample(self):
        return [agent_space.sample() for agent_space in self.agents_action_space]

    def __len__(self):
        return len(self.agents_action_space)

    def __getitem__(self, idx):
        return self.agents_action_space[idx]

    def __repr__(self):
        return "MultiAgentActionSpace(" + ", ".join([str(s) for s in self.agents_action_space]) + ")"

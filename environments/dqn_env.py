from environments.envwrapper import EnvWrapper


class DQNGym(EnvWrapper):
    def __init__(self, env_name):
        super().__init__()
        self.env = gym.make(env_name)
        self.action_dim = self.env.action_space.n

    def step(self, action: np.ndarray) -> tuple:
        return self.env.step(action)

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()

        return obs
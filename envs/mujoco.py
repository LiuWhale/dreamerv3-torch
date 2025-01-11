import gym
import numpy as np

class Mujoco:
    metadata = {}

    def __init__(self, env_name="hopper", action_repeat=1, size=(64, 64), seed=0):
        if env_name == "hopper":
            env_name = "Hopper-v3"
        elif env_name == "cheetah":
            env_name = "HalfCheetah-v3"
        self._env = gym.make(env_name, render_mode="rgb_array")
        self.seed = seed
        # self._env.seed(seed)
        self._action_repeat = action_repeat
        self._size = size
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        # 与 dmc.py 类似，添加一个包含 'image' 的字典空间
        original_space = gym.spaces.Box(
            low=self._env.observation_space.low,
            high=self._env.observation_space.high,
            dtype=np.float64
        )
        spaces = {"state": original_space,
                "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)}
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        reward = 0
        done = False
        info = {}
        for _ in range(self._action_repeat):
            obs, rew, done, trunc, info = self._env.step(action)
            reward += rew
            if done:
                break
        # 构造与 dmc 风格相似的 obs 字典
        obs_dict = {"state": obs, "image": self.render()}
        obs_dict["is_first"] = False  # 仅示例，可自行修改
        obs_dict["is_terminal"] = done
        return obs_dict, reward, done, info

    def reset(self):
        obs = self._env.reset(seed=self.seed)
        obs_dict = {"state": obs[0], "image": self.render()}
        # obs_dict = {"image": self.render()}
        obs_dict["is_first"] = True
        obs_dict["is_terminal"] = False
        return obs_dict

    def render(self):
        return self._env.render()


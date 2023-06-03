"""
This is a test Gym environment that reads to a file. After testing here it was moved to a fork
of https://github.com/eloialonso/iris:

https://github.com/proger/iris/tree/twist-rollout
"""
import gym
from gym import spaces
import numpy as np
import gzip
import cv2

class CustomEnvironment(gym.Env):
    def __init__(self):
        self.current_position = None
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.reward_range = (0, 1)

        self.frames = np.fromfile('image.bin', dtype=np.uint8).reshape(-1, 64, 64, 3)
        print(len(self.frames)/(64*64*3), 'total')
        self.num_frames = self.frames.shape[0]

    def reset(self):
        self.current_position = np.random.randint(0, self.num_frames)
        return self._get_observation()

    def step(self, action):
        # Ignore agent actions and just return the next frame
        self.current_position += 1

        # Check if we reached the end of the file
        done = self.current_position >= self.num_frames

        # Calculate the reward
        reward = 1.0

        if done:
            # Reset if we reached the end of the file
            obs = self.reset()
        else:
            # Return the next frame
            obs = self._get_observation()

        return obs, reward, done, {}

    def render(self, mode='rgb_array'):
        return self._get_observation()

    def _get_observation(self):
        return self.frames[self.current_position]


if __name__ == '__main__':
    env = CustomEnvironment()

    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        image = env.render()

        cv2.imshow('Frame', image)
        cv2.waitKey(100)

    cv2.destroyAllWindows()


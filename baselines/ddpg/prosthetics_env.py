import numpy as np
import gym
from pdb import set_trace

import osim.env

prosthetics_env_observation_len = None

class Wrapper(osim.env.ProstheticsEnv):
    def __init__(self, osim_env, frameskip):
        global prosthetics_env_observation_len

        self.__dict__.update(osim_env.__dict__)
        self.env = osim_env
        self.frameskip = frameskip

        o_low = self.env.observation_space.low
        o_high = self.env.observation_space.high
        o_shape = self.env.observation_space.shape
        o_dtype = self.env.observation_space.dtype

        spread = o_high - o_low
        self.observation_mid = (o_low + o_high) / 2.
        self.observation_half_spread = spread / 2.

        # Call our own reset() method here. It gets the dict from the
        # osim.env.reset() method and does its own projection onto a vector.
        # Unlike a call to osim.env.reset(project=True), our reset() method
        # preserves all of the data from the dict.
        # Cache the prosthetics_env_observation_len to avoid this RuntimeError:
        #   "Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)"
        if not prosthetics_env_observation_len:
            observation = self.reset()
            prosthetics_env_observation_len = len(observation)

        # Sanity check to make sure we can reshape the observation space using
        # just the first low and high values from the original observation space.
        for x in list(self.env.observation_space.low):
            assert(self.env.observation_space.low[0] == x)
        for x in list(self.env.observation_space.high):
            assert(self.env.observation_space.high[0] == x)

        # Now use the observation from our reset() method to create a reshaped
        # observation_space.
        # Note also that we can embellish our observation with features we
        # deem important (e.g., joint angles, body lean, ground forces, etc.)
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            shape=(prosthetics_env_observation_len,),
            dtype=np.float32)

        a_low = self.env.action_space.low
        a_high = self.env.action_space.high
        spread = a_high - a_low
        self.action_mid = (a_low + a_high) / 2.
        self.action_half_spread = spread / 2.

        assert(np.shape(self.env.action_space.low) == np.shape(self.env.action_space.high))
        self.action_space = gym.spaces.Box(
            -0.5 + np.zeros(np.shape(self.env.action_space.low)),
            0.5 + np.zeros(np.shape(self.env.action_space.high)))

    def change_model(self, **kwargs):
        self.env.change_model(**kwargs)

    def reset(self, project = True):
        observation = self.env.reset(project=False)
        self.embellish_features(observation)
        if project:
            projection = []
            self._project(observation, projection)
            observation = projection
        return observation

    def step(self, action, project=True):
        # Okay, here's a hack, but it works. There are quite a few envs.
        # - osim.env.ProstheticsEnv
        # - this Wrapper
        # - the aliased EvaluationWrapper defined at the bottom of this file
        # - the envs returned by bench.Monitor()
        # The envs returned by bench.Monitor() don't support the "project"
        # keyword arg to step().
        # You might think that we should use the value of project that's passed
        # into this method when we call ProstheticsEnv.step() but that's not the
        # case. We always want to get the dictionary back from the ProstheticsEnv
        # so we can do the projection ourselves.
        typename = type(self.env).__name__
        if typename == "ProstheticsEnv":  # osim.env.osim.ProstheticsEnv
            reward = 0.
            for _ in range(self.frameskip):
                observation, tmp_reward, done, info = self.env.step(self._openai_to_opensim_action(action), project=False)
                reward += tmp_reward
                if done:
                    break

            self.embellish_features(observation)
            reward = self.shaped_reward(observation, reward, done)
        elif "Monitor":  # baselines.bench.monitor.Monitor
            reward = 0.
            for _ in range(self.frameskip):
                observation, tmp_reward, done, info = self.env.step(action)
                reward += tmp_reward
                if done:
                    break
        else:
            raise RuntimeError("WTF", typename)

        if project:
            projection = []
            self._project(observation, projection)
            observation = projection
        return observation, reward, done, info

    def shaped_reward(self, observation_dict, reward, done):
        return reward

    def embellish_features(self, observation_dict):
        pass

    def _openai_to_opensim_action(self, action):
        return action + 0.5

    def _project(self, obj, accumulator = []):
        if type(obj).__name__ == "list":
            [self._project(item, accumulator) for item in obj]
        elif type(obj).__name__ == "dict":
            [self._project(obj[key], accumulator) for key in sorted(obj.keys())]
        else:
            accumulator.append(obj)


class EvaluationWrapper(Wrapper):
    def step(self, action, project=True):
        # Okay, here's a hack, but it works. There are quite a few envs.
        # - osim.env.ProstheticsEnv
        # - this Wrapper
        # - the aliased EvaluationWrapper defined at the bottom of this file
        # - the envs returned by bench.Monitor()
        # The envs returned by bench.Monitor() don't support the "project"
        # keyword arg to step().
        # You might think that we should use the value of project that's passed
        # into this method when we call ProstheticsEnv.step() but that's not the
        # case. We always want to get the dictionary back from the ProstheticsEnv
        # so we can do the projection ourselves.
        typename = type(self.env).__name__
        if typename == "ProstheticsEnv":  # osim.env.osim.ProstheticsEnv
            reward = 0.
            for _ in range(self.frameskip):
                observation, tmp_reward, done, info = self.env.step(self._openai_to_opensim_action(action), project=False)
                reward += tmp_reward
                if done:
                    break
            self.embellish_features(observation)
            if done:
                print(" eval: reward:{:>6.1f}".format(reward))
        elif "Monitor":  # baselines.bench.monitor.Monitor
            reward = 0.
            for _ in range(self.frameskip):
                observation, tmp_reward, done, info = self.env.step(action)
                reward += tmp_reward
                if done:
                    break
            observation, reward, done, info = self.env.step(action)
        else:
            raise RuntimeError("WTF", typename)

        if project:
            projection = []
            self._project(observation, projection)
            observation = projection
        return observation, reward, done, info

import numpy as np
import gym
from pdb import set_trace

import osim.env

prosthetics_env_observation_len = None

class Wrapper(osim.env.ProstheticsEnv):
    def __init__(self, osim_env, frameskip, reward_shaping, feature_embellishment):
        global prosthetics_env_observation_len
        assert(type(osim_env).__name__) == "ProstheticsEnv"

        self.__dict__.update(osim_env.__dict__)
        self.env = osim_env
        self.reward_shaping = reward_shaping
        self.feature_embellishment = feature_embellishment
        self.frameskip = frameskip
        self.step_num = 0

        o_low = self.env.observation_space.low
        o_high = self.env.observation_space.high
        o_shape = self.env.observation_space.shape
        o_dtype = self.env.observation_space.dtype

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
        for x in list(o_low):
            assert(o_low[0] == x)
        for x in list(o_high):
            assert(o_high[0] == x)

        # Now use the observation from our reset() method to create a reshaped
        # observation_space.
        # Note also that we can embellish our observation with features we
        # deem important (e.g., joint angles, body lean, ground forces, etc.)
        self.observation_space = gym.spaces.Box(
            low=o_low[0],
            high=o_high[0],
            shape=(prosthetics_env_observation_len,),
            dtype=np.float32)
        print("observation_space shape is ({:d},)".format(prosthetics_env_observation_len))

        a_low = self.env.action_space.low
        a_high = self.env.action_space.high
        assert(np.shape(a_low) == np.shape(a_high))
        self.action_space = gym.spaces.Box(
            -0.5 + np.zeros(np.shape(a_low)),
            0.5 + np.zeros(np.shape(a_high)))

    def change_model(self, **kwargs):
        self.env.change_model(**kwargs)

    def reset(self, project = True):
        observation = self.env.reset(project=False)  # never project=True when calling the ProstheticsEnv
        if self.reward_shaping or self.feature_embellishment:
            self.embellish_features(observation)
        if project:
            projection = []
            self._project(observation, projection)
            observation = projection
        return observation

    def step(self, action, project=True):
        if self.step_num % self.frameskip == 0:
            observation, reward, done, info = self.env.step(self._openai_to_opensim_action(action), project=False)
            if self.reward_shaping or self.feature_embellishment:
                self.embellish_features(observation)
            if self.reward_shaping:
                reward = self.shaped_reward(observation, reward, done)
            if project:
                projection = []
                self._project(observation, projection)
                observation = projection
            self.prev_step = observation, reward, done, info
        else:
            observation, reward, done, info = self.prev_step
        self.step_num += 1
        return observation, reward, done, info

    def _openai_to_opensim_action(self, action):
        return action + 0.5

    def _project(self, obj, accumulator = []):
        if type(obj).__name__ == "list":
            [self._project(item, accumulator) for item in obj]
        elif type(obj).__name__ == "dict":
            [self._project(obj[key], accumulator) for key in sorted(obj.keys())]
        else:
            accumulator.append(obj)

    # The head and pelvis entries contain 3 numbers, the x, y, and z coordinates.
    # Lean is defined as:
    #   Standing straight up = 0
    #   Fallen on face -> +inf
    #   Fallen on back -> -inf
    def torso_lean(self, observation_dict):
        body_pos = observation_dict["body_pos"]
        head = body_pos["head"]
        pelvis = body_pos["pelvis"]
        return (head[0] - pelvis[0]) / (head[1] - pelvis[1])

    # Only generate negative rewards for undesired states so that "successful"
    # observations reflect actual rewards.
    def torso_lean_reward(self, observation_dict):
        lean = observation_dict["z_torso_lean"]
        reward = 0
        if lean < 0 and lean >= -0.1:
            reward = -2
        elif lean < -0.1 and lean >= -0.2:
            reward = -3
        elif lean < -0.3:
            reward = -20
        return reward

    # The femur_l and femur_r entries contain 3 numbers, the x, y, and z coordinates.
    # It's unclear what part of the femur (center?) is referred to, but the idea
    # here is to average the femur positions and determine where this position
    # leans relative to the femur.
    # Lean is defined as:
    #   Standing straight up = 0
    #   Fallen on face -> +inf
    #   Fallen on back -> -inf
    def legs_lean(self, observation_dict):
        body_pos = observation_dict["body_pos"]
        pelvis = body_pos["pelvis"]
        femur_l, femur_r = body_pos["femur_l"], body_pos["femur_r"]
        return [
            (pelvis[0] - femur_l[0]) / (pelvis[1] - femur_l[1]),
            (pelvis[0] - femur_r[0]) / (pelvis[1] - femur_r[1])
        ]

    # Only generate negative rewards for undesired states so that "successful"
    # observations reflect actual rewards.
    def legs_lean_reward(self, observation_dict):
        femur_l = observation_dict["z_femur_l_lean"]
        femur_r = observation_dict["z_femur_r_lean"]
        reward = 0
        if femur_l < 0 and femur_l >= -0.1 and femur_r < 0 and femur_r >= -0.1:
            reward = -2
        elif femur_l < -0.1 and femur_l >= -0.2 and femur_r < -0.1 and femur_r >= -0.2:
            reward = -3
        elif femur_l < -0.3 and femur_r < -0.3:
            reward = -20
        return reward

    # The knee_l and knee_r entries contain just one number, the joint flexion.
    # A positive flexion number means (hyper)extension. Typically the largest
    # positive flexion is about 0.2 or 0.3.
    # Negative flexion means, well, flexion. These numbers can reach ~ -5.0 based
    # on empirical observation.
    # The only documentation I could find says that actual physical flexion can
    # reach ~ 100 degrees.
    def knees_flexion(self, observation_dict):
        joint_pos = observation_dict["joint_pos"]
        return joint_pos["knee_l"][0] + joint_pos["knee_r"][0]

    def knees_flexion_reward(self, observation_dict):
        flexion = observation_dict["z_knees_flexion"]
        reward = 0
        if flexion > 0:
            reward = -2
        return reward

    # Modifies the observation_dict in place.
    def embellish_features(self, observation_dict):
        observation_dict["z_torso_lean"] = self.torso_lean(observation_dict)
        legs_lean = self.legs_lean(observation_dict)
        observation_dict["z_femur_l_lean"] = legs_lean[0]
        observation_dict["z_femur_r_lean"] = legs_lean[1]
        observation_dict["z_knees_flexion"] = self.knees_flexion(observation_dict)

    def shaped_reward(self, observation_dict, reward, done):
        torso_r = self.torso_lean_reward(observation_dict)
        legs_r = self.legs_lean_reward(observation_dict)
        knees_r = self.knees_flexion_reward(observation_dict)

        torso = observation_dict["z_torso_lean"]
        z_femur_l_lean = observation_dict["z_femur_l_lean"]
        z_femur_r_lean = observation_dict["z_femur_r_lean"]
        knees_flexion = observation_dict["z_knees_flexion"]

        shaped_reward = reward + torso_r + legs_r + knees_r

        if done:
            print("train: reward:{:>6.1f} shaped reward:{:>6.1f} torso:{:>6.1f} ({:>8.3f}) legs:{:>6.1f} ({:>8.3f}, {:>8.3f}) knee flex:{:>6.1f} ({:>8.3f})".format(
                reward, shaped_reward, torso_r, torso, legs_r, z_femur_l_lean, z_femur_r_lean, knees_r, knees_flexion))

        return shaped_reward

class EvaluationWrapper(Wrapper):
    def step(self, action, project=True):
        if self.step_num % self.frameskip == 0:
            observation, reward, done, info = self.env.step(self._openai_to_opensim_action(action), project=False)
            if self.reward_shaping or self.feature_embellishment:
                self.embellish_features(observation)
            if done:
                print(" eval: reward:{:>6.1f}".format(reward))
            if project:
                projection = []
                self._project(observation, projection)
                observation = projection
            self.prev_step = observation, reward, done, info
        else:
            observation, reward, done, info = self.prev_step
        self.step_num += 1
        return observation, reward, done, info

import numpy as np
import gym
import copy
from pdb import set_trace
from gym.utils import seeding
import osim.env
from baselines import logger

prosthetics_env_observation_len = None

def _clip_to_action_space(action):
    clipped_action = np.clip(action, 0., 1.)
    actions_equal = clipped_action == action
    if not np.all(actions_equal):
        logger.warn("Had to clip action since it wasn't constrained to the [0,1] action space:", action)
    return clipped_action

def openai_to_opensim_action(action):
    #return _clip_to_action_space(action + 0.5)
    return _clip_to_action_space((action + 1.0) * 0.5)

def openai_to_crowdai_submit_action(action):
    return _clip_to_action_space((action + 1.0) * 0.5)

def project_values(obj, accumulator=None):
    if accumulator is None:
        accumulator = []
    _project_values(obj, accumulator)
    return accumulator

def _project_values(obj, accumulator):
    if type(obj).__name__ == "list":
        [_project_values(item, accumulator) for item in obj]
    elif type(obj).__name__ == "dict":
        [_project_values(obj[key], accumulator) for key in sorted(obj.keys())]
    else:
        accumulator.append(obj)

def _embellish_features_inplace(observation_dict):
    observation_dict["z_torso_xaxis_lean"] = torso_xaxis_lean(observation_dict)
    observation_dict["z_torso_zaxis_lean"] = torso_zaxis_lean(observation_dict)
    legs_xaxis_lean = femurs_xaxis_lean(observation_dict)
    observation_dict["z_femur_l_xaxis_lean"] = legs_xaxis_lean[0]
    observation_dict["z_femur_r_xaxis_lean"] = legs_xaxis_lean[1]
    legs_zaxis_lean = femurs_zaxis_lean(observation_dict)
    observation_dict["z_femur_l_zaxis_lean"] = legs_zaxis_lean[0]
    observation_dict["z_femur_r_zaxis_lean"] = legs_zaxis_lean[1]
    observation_dict["z_knees_flexion"] = knees_flexion(observation_dict)

# The head and pelvis entries contain 3 numbers, the x, y, and z coordinates.
# Lean is defined as:
#   Standing straight up = 0
#   Fallen on face -> +inf
#   Fallen on back -> -inf
def torso_xaxis_lean(observation_dict):
    xaxis = 0
    yaxis = 1
    body_pos = observation_dict["body_pos"]
    head = body_pos["head"]
    pelvis = body_pos["pelvis"]
    #return (head[xaxis] - pelvis[xaxis]) / (head[yaxis] - pelvis[yaxis])
    # We've already normalized all the body_pos xaxis values wrt the pelvis.
    return head[xaxis] / (head[yaxis] - pelvis[yaxis])

# Only generate negative rewards for undesired states so that "successful"
# observations reflect actual rewards.
def torso_xaxis_lean_reward(observation_dict):
    lean = observation_dict["z_torso_xaxis_lean"]
    reward = 0
    if lean < 0 and lean >= -0.1:
        reward = -1
    elif lean < -0.1 and lean >= -0.2:
        reward = -2
    elif lean < -0.3:
        reward = -3
    return reward

# The head and pelvis entries contain 3 numbers, the x, y, and z coordinates.
# Lean is defined as:
#   Standing straight up = 0
#   Fallen on side -> +inf
#   Fallen on other side -> -inf
def torso_zaxis_lean(observation_dict):
    zindex = 2
    yindex = 1
    body_pos = observation_dict["body_pos"]
    head = body_pos["head"]
    pelvis = body_pos["pelvis"]
    return (head[zindex] - pelvis[zindex]) / (head[yindex] - pelvis[yindex])

# Only generate negative rewards for undesired states so that "successful"
# observations reflect actual rewards.
def torso_zaxis_lean_reward(observation_dict):
    lean = abs(observation_dict["z_torso_zaxis_lean"])
    reward = 0
    if lean > 0.1 and lean <= 0.2:
        reward = -1
    elif lean > 0.3:
        reward = -2
    return reward

# The femur_l and femur_r entries contain 3 numbers, the x, y, and z coordinates.
# It's unclear what part of the femur (center?) is referred to, but the idea
# here is to average the femur positions and determine where this position
# leans relative to the femur.
# Lean is defined as:
#   Standing straight up = 0
#   Fallen on face -> +inf
#   Fallen on back -> -inf
def femurs_xaxis_lean(observation_dict):
    xindex = 0
    yindex = 1
    body_pos = observation_dict["body_pos"]
    pelvis = body_pos["pelvis"]
    # Yes, use the tibias here. They're at the *bases* of the femurs.
    femur_l, femur_r = body_pos["tibia_l"], body_pos["pros_tibia_r"]
    return [
        #(pelvis[xindex] - femur_l[xindex]) / (pelvis[yindex] - femur_l[yindex]),
        #(pelvis[xindex] - femur_r[xindex]) / (pelvis[yindex] - femur_r[yindex])
        # We've already normalized all the body_pos xaxis values wrt the pelvis.
        -femur_l[xindex] / (pelvis[yindex] - femur_l[yindex]),
        -femur_r[xindex] / (pelvis[yindex] - femur_r[yindex])
    ]

def femurs_zaxis_lean(observation_dict):
    zindex = 2
    yindex = 1
    body_pos = observation_dict["body_pos"]
    pelvis = body_pos["pelvis"]
    # Yes, use the tibias here. They're at the *bases* of the femurs.
    femur_l, femur_r = body_pos["tibia_l"], body_pos["pros_tibia_r"]
    return [
        (pelvis[zindex] - femur_l[zindex]) / (pelvis[yindex] - femur_l[yindex]),
        (pelvis[zindex] - femur_r[zindex]) / (pelvis[yindex] - femur_r[yindex])
    ]

# Only generate negative rewards for undesired states so that "successful"
# observations reflect actual rewards.
def femurs_xaxis_lean_reward(observation_dict):
    femur_l = observation_dict["z_femur_l_xaxis_lean"]
    femur_r = observation_dict["z_femur_r_xaxis_lean"]
    reward = 0
    if femur_l < 0 and femur_l >= -0.1 and femur_r < 0 and femur_r >= -0.1:
        reward = -1
    elif femur_l < -0.1 and femur_l >= -0.2 and femur_r < -0.1 and femur_r >= -0.2:
        reward = -2
    elif femur_l < -0.3 and femur_r < -0.3:
        reward = -3
    return reward

# Only generate negative rewards for undesired states so that "successful"
# observations reflect actual rewards.
def femurs_zaxis_lean_reward(observation_dict):
    femur_l = observation_dict["z_femur_l_zaxis_lean"]
    femur_r = observation_dict["z_femur_r_zaxis_lean"]
    reward = 0
    avg_lean = abs((femur_l + femur_r) / 2)
    if avg_lean > 0.1 and avg_lean <= 0.2:
        reward = -1
    elif avg_lean > 0.2:
        reward = -2
    return reward

# The knee_l and knee_r entries contain just one number, the joint flexion.
# A positive flexion number means (hyper)extension. Typically the largest
# positive flexion is about 0.2 or 0.3.
# Negative flexion means, well, flexion. These numbers can reach ~ -5.0 based
# on empirical observation.
# The only documentation I could find says that actual physical flexion can
# reach ~ 100 degrees.
def knees_flexion(observation_dict):
    joint_pos = observation_dict["joint_pos"]
    return joint_pos["knee_l"][0] + joint_pos["knee_r"][0]

def knees_flexion_reward(observation_dict):
    flexion = observation_dict["z_knees_flexion"]
    reward = 0
    if flexion > 0.3:
        reward = -3
    elif flexion > 0.2:
        reward = -2
    elif flexion > 0.1:
        reward = -1
    return reward * 0.5

def tibias_pos_reward(observation_dict):
    yindex = 1
    zindex = 2
    body_pos = observation_dict["body_pos"]
    pelvis_pos = body_pos["pelvis"]
    tibia_l_pos = body_pos["tibia_l"]
    tibia_r_pos = body_pos["pros_tibia_r"]
    reward = 0
    # Don't lift the knees above the pelvis.
    #logger.info("obs")
    #logger.info("  pelvis_pos", pelvis_pos)
    #logger.info("  tibia_l_pos", tibia_l_pos)
    if tibia_l_pos[yindex] > pelvis_pos[yindex] * 0.9:
        reward += -1
    elif tibia_r_pos[yindex] > pelvis_pos[yindex] * 0.9:
        reward += -1
    # Don't cross over the center line.
    elif tibia_l_pos[zindex] > 0:
        reward += -1
    elif tibia_r_pos[zindex] < 0:
        reward += -1
    #logger.info("obs:")
    #logger.info("  tibia_l pos", body_pos["tibia_l"])
    #logger.info("  pros_tibia_r pos", body_pos["pros_tibia_r"])
    #body_pos_rot = observation_dict["body_pos_rot"]
    #logger.info("  tibia_l rot", body_pos_rot["tibia_l"])
    #logger.info("  pros_tibia_r pos", body_pos_rot["pros_tibia_r"])
    return reward * 0.

# Modifies the observation_dict in place.
def _adjust_relative_x_pos_inplace(observation_dict):
    xindex = 0
    #ground_pelvis_pos = observation_dict["joint_pos"]["ground_pelvis"]
    #logger.info("ground_pelvis_pos", ground_pelvis_pos)
    body_pos = observation_dict["body_pos"]
    pelvis_pos = body_pos["pelvis"]
    #logger.info("  pelvis", body_pos["pelvis"])
    #logger.info("  before head", body_pos["head"])
    # This code demonstrates that:
    #   observation_dict["body_pos"]["pelvis"] != observation_dict["joint_pos"]["ground_pelvis"]
    # However, I don't know what the difference is.
    #if body_pos["pelvis"][xindex] != ground_pelvis_pos[xindex]:
    #    logger.warn('observation_dict["body_pos"]["pelvis"] != observation_dict["joint_pos"]["ground_pelvis"]',
    #        body_pos["pelvis"], observation_dict["joint_pos"]["ground_pelvis"])
    for body_part in ["calcn_l", "talus_l", "tibia_l", "toes_l", "femur_l", "femur_r", "head", "torso", "pros_foot_r", "pros_tibia_r"]:
        body_pos[body_part][xindex] -= pelvis_pos[xindex]
    #logger.info("  after head", body_pos["head"])
    # documentation says mass_center_pos has x,y,z coords but observation shows only 2, let's leave the x coord alone
    #observation_dict["misc"]["mass_center_pos"][xindex] -= ground_pelvis_pos[xindex]

# Modifies the observation_dict in place.
def _adjust_relative_z_pos_inplace(observation_dict):
    zindex = 2
    #ground_pelvis_pos = observation_dict["joint_pos"]["ground_pelvis"]
    body_pos = observation_dict["body_pos"]
    pelvis_pos = body_pos["pelvis"]
    # This code demonstrates that:
    #   observation_dict["body_pos"]["pelvis"] != observation_dict["joint_pos"]["ground_pelvis"]
    # However, I don't know what the difference is.
    #if body_pos["pelvis"][zindex] != ground_pelvis_pos[zindex]:
    #    logger.warn('observation_dict["body_pos"]["pelvis"] != observation_dict["joint_pos"]["ground_pelvis"]',
    #        body_pos["pelvis"], observation_dict["joint_pos"]["ground_pelvis"])
    for body_part in ["calcn_l", "talus_l", "tibia_l", "toes_l", "femur_l", "femur_r", "head", "torso", "pros_foot_r", "pros_tibia_r"]:
        body_pos[body_part][zindex] -= pelvis_pos[zindex]
    # documentation says mass_center_pos has x,y,z coords but observation shows only 2
    #observation_dict["misc"]["mass_center_pos"][zindex] -= ground_pelvis_pos[zindex]

# Transform the observation dictionary returned by the opensim environment
# step by applying various transformations such as embellishing the feature set
# with additional derived features, and changing absolute coordinate positions
# to relative ones.
# Finally, if project=True, transform the dictionary into a vector.
def transform_observation(observation_dict, reward_shaping, reward_shaping_x, feature_embellishment, relative_x_pos, relative_z_pos):
    observation_dict_copy= copy.deepcopy(observation_dict)
    # observation_dict_copy = {}.update(observation_dict)
    if reward_shaping or feature_embellishment:
        _embellish_features_inplace(observation_dict_copy)
    if relative_x_pos:  # adjust the relative_x_pos *after* embellish_features please
        _adjust_relative_x_pos_inplace(observation_dict_copy)
    if relative_z_pos:  # adjust the relative_z_pos *after* embellish_features please
        _adjust_relative_z_pos_inplace(observation_dict_copy)
    return observation_dict_copy, project_values(observation_dict_copy)

# Must not have any side effects (do *not* modify observation_dict in place).
def shaped_reward(observation_dict, reward, done, reward_shaping_x):
    torso_xaxis_rwd = torso_xaxis_lean_reward(observation_dict)*reward_shaping_x
    torso_zaxis_rwd = torso_zaxis_lean_reward(observation_dict)*reward_shaping_x
    legs_xaxis_rwd = femurs_xaxis_lean_reward(observation_dict)*reward_shaping_x
    legs_zaxis_rwd = femurs_zaxis_lean_reward(observation_dict)*reward_shaping_x
    knees_rwd = knees_flexion_reward(observation_dict)*reward_shaping_x

    torso_xaxis_lean = observation_dict["z_torso_xaxis_lean"]
    torso_zaxis_lean = observation_dict["z_torso_zaxis_lean"]
    z_femur_l_xaxis_lean = observation_dict["z_femur_l_xaxis_lean"]
    z_femur_l_zaxis_lean = observation_dict["z_femur_l_zaxis_lean"]
    z_femur_r_xaxis_lean = observation_dict["z_femur_r_xaxis_lean"]
    z_femur_r_zaxis_lean = observation_dict["z_femur_r_zaxis_lean"]
    knees_flexion = observation_dict["z_knees_flexion"]

    shaped_reward = reward + torso_xaxis_rwd + torso_zaxis_rwd + legs_xaxis_rwd + legs_zaxis_rwd + knees_rwd

    if done:
        logger.debug("train: reward:{:>6.1f} shaped reward:{:>6.1f} torso:{:>6.1f} ({:>8.3f}) legs:{:>6.1f} ({:>8.3f}, {:>8.3f}) knee flex:{:>6.1f} ({:>8.3f})".format(
            reward, shaped_reward, torso_xaxis_rwd, torso_xaxis_lean, legs_xaxis_rwd, z_femur_l_xaxis_lean, z_femur_r_xaxis_lean, knees_rwd, knees_flexion))

    return shaped_reward

class Wrapper(osim.env.ProstheticsEnv):
    def __init__(self, osim_env, frameskip, reward_shaping, reward_shaping_x, feature_embellishment, relative_x_pos, relative_z_pos):
        global prosthetics_env_observation_len
        assert(type(osim_env).__name__) == "ProstheticsEnv"

        self.__dict__.update(osim_env.__dict__)
        self.env = osim_env
        self.reward_shaping = reward_shaping
        self.reward_shaping_x = reward_shaping_x
        self.feature_embellishment = feature_embellishment
        self.relative_x_pos = relative_x_pos
        self.relative_z_pos = relative_z_pos
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

        a_low = self.env.action_space.low
        a_high = self.env.action_space.high
        assert(np.shape(a_low) == np.shape(a_high))
        self.action_space = gym.spaces.Box(
            -1. + np.zeros(np.shape(a_low)),
            1. + np.zeros(np.shape(a_high)))

    def change_model(self, **kwargs):
        self.env.change_model(**kwargs)

    def reset(self, project=True):
        observation_dict = self.env.reset(project=False)  # never project=True when calling the osim ProstheticsEnv
        observation_dict, observation_projection = transform_observation(
            observation_dict,
            reward_shaping=self.reward_shaping,
            reward_shaping_x=self.reward_shaping_x,
            feature_embellishment=self.feature_embellishment,
            relative_x_pos=self.relative_x_pos,
            relative_z_pos=self.relative_z_pos)
        if project:
            return observation_projection
        else:
            return observation_dict

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, project=True):
        if self.step_num % self.frameskip == 0:
            opensim_action = openai_to_opensim_action(action)
            observation_dict, reward, done, info = self.env.step(opensim_action, project=False)
            observation_dict, observation_projection = transform_observation(
                observation_dict,
                reward_shaping=self.reward_shaping,
                reward_shaping_x=self.reward_shaping_x,
                feature_embellishment=self.feature_embellishment,
                relative_x_pos=self.relative_x_pos,
                relative_z_pos=self.relative_z_pos)
            if self.reward_shaping:
                reward = shaped_reward(observation_dict, reward, done, self.reward_shaping_x)
            self.prev_step = observation_dict, observation_projection, reward, done, info
        else:
            observation_dict, observation_projection, reward, done, info = self.prev_step
        self.step_num += 1
        if project:
            return observation_projection, reward, done, info
        else:
            return observation_dict, reward, done, info

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


class EvaluationWrapper(Wrapper):
    def step(self, action, project=True):
        if self.step_num % self.frameskip == 0:
            opensim_action = openai_to_opensim_action(action)
            observation_dict, reward, done, info = self.env.step(opensim_action, project=False)
            observation_dict, observation_projection = transform_observation(
                observation_dict,
                reward_shaping=self.reward_shaping,
                reward_shaping_x=0,
                feature_embellishment=self.feature_embellishment,
                relative_x_pos=self.relative_x_pos,
                relative_z_pos=self.relative_z_pos)
            if done:
                logger.debug(" eval: reward:{:>6.1f}".format(reward))
            self.prev_step = observation_dict, observation_projection, reward, done, info
        else:
            observation_dict, observation_projection, reward, done, info = self.prev_step
        self.step_num += 1
        if project:
            return observation_projection, reward, done, info
        else:
            return observation_dict, reward, done, info

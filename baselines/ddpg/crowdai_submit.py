import opensim as osim
from osim.http.client import Client
import osim.env
import numpy as np
from baselines.ddpg import prosthetics_env

# Settings
remote_base = "http://grader.crowdai.org:1729"
crowdai_token = "9c48765358e511504cf7731614afac30"

client = Client(remote_base)

# Create environment
observation = client.env_create(crowdai_token, env_id="ProstheticsEnv")
#env = osim.env.ProstheticsEnv()
env = prosthetics_env.Wrapper(osim.env.ProstheticsEnv(visualize=False))

# IMPLEMENTATION OF YOUR CONTROLLER
# my_controller = ... (for example the one trained in keras_rl)

while True:
    print(observation)
    #[observation, reward, done, info] = client.env_step(my_controller(observation), True)
    [observation, reward, done, info] = client.env_step(env.action_space.sample().tolist())
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()

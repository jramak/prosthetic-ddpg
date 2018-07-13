import opensim as osim
from osim.http.client import Client
import osim.env
import numpy as np
from baselines.ddpg import prosthetics_env


def evaluate_model(submit=False):
    client = Client(remote_base)

    # Create environment
    observation = client.env_create(args.token, env_id="ProstheticsEnv")
    #env = osim.env.ProstheticsEnv()
    env = prosthetics_env.Wrapper(osim.env.ProstheticsEnv(visualize=False),
                                                      frameskip=1,
                                                      reward_shaping=False,
                                                      feature_embellishment=False,
                                                      relative_x_pos=False)

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

    if submit:
        client.submit()

# Settings
remote_base = "http://grader.crowdai.org:1729"

# Command line parameters
parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
parser.add_argument('--token', dest='token', action='store',
                    default='9c48765358e511504cf7731614afac30')
parser.add_argument('--restore-model-name', type=str, default=None)
parser.add_argument('--submit', default=False, action='store_true')
args = parser.parse_args()

evaluate_model(submit=args.submit)

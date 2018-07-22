from datetime import datetime
from os import getpid
import osim.env as osim_env
from baselines.ddpg import prosthetics_env
from psutil import Process


def time():
    return datetime.now().strftime('%H:%M:%S')


def memory_used():
    process = Process(getpid())
    return process.memory_info().rss  # https://pythonhosted.org/psutil/#psutil.Process.memory_info


env = prosthetics_env.Wrapper(osim_env.ProstheticsEnv(visualize=False),
                                                      frameskip=1,
                                                      reward_shaping=True,
                                                      reward_shaping_x=2.,
                                                      feature_embellishment=True,
                                                      relative_x_pos=True,
                                                      relative_z_pos=True)
env.reset()
step = 0
episode = 0
while True:
    observation, reward, done, info = env.step(env.action_space.sample())
    step += 1
    if done:
        episode += 1
        env.reset()
        print("%s Episode %s, steps %s, memory %s" % (time(), episode, step, memory_used()))
        step = 0

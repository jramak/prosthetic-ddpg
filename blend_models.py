import argparse
from baselines.ddpg.noise import AdaptiveParamNoiseSpec
from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines import logger, bench
from baselines.ddpg import prosthetics_env
import baselines.common.tf_util as U
import osim.env as osim_env
import tensorflow as tf
import numpy as np
import pickle
import os
from pdb import set_trace


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-files', type=str, nargs='+')
    parser.add_argument('--layer-sizes', type=int, nargs='+')
    args = parser.parse_args()
    return args


def evaluate_one_episode(env, ddpg_agents, sess_list, graph_list, nb_eval_steps, render):
    if nb_eval_steps <= 0:
        print('evaluate_one_episode nb_eval_steps must be > 0')
    reward = 0.
    qs = []
    obs = env.reset()
    num_agents = len(ddpg_agents)
    for step in range(nb_eval_steps):
        pre_reward = [0.] * num_agents
        possible_actions = []
        for i in range(num_agents):
            actor_agent = ddpg_agents[i]
            # for each actor use a different possible critic
            with sess_list[i].as_default():
                with graph_list[i].as_default():
                    action, _ = actor_agent.pi(obs, apply_noise=False, compute_Q=True)
            possible_actions.append(action)
            for j in range(num_agents):
                critic_agent = ddpg_agents[j]
                with sess_list[j].as_default():
                    with graph_list[j].as_default():
                        q = critic_agent.compute_Q_for_action(obs, action)
                        pre_reward[i] += np.asscalar(q[0])

        best_action_idx = np.argmax(pre_reward)
        obs, r, done, info = env.step(possible_actions[best_action_idx])
        if render:
            env.render()
        reward += r
        qs.append(pre_reward[best_action_idx])
        print("Eval step " + str(step))
        if done:
            #obs = env.reset()
            break  # the original baseline code didn't have this break statement, so would average multiple evaluation episodes
        elif step >= nb_eval_steps:
            logger.warn('evaluate_one_episode step', step, 'exceeded nb_eval_steps', nb_eval_steps, 'but done is False')
            #obs = env.reset()
            break
    return reward, np.mean(qs), step+1


def main():
    args = parse_args()
    logger.configure()
    gamma = 0.99
    tau = 0.01
    normalize_returns = False
    normalize_observations = True
    batch_size = 64
    action_noise = None
    stddev = 0.2
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev),
                                         desired_action_stddev=float(stddev))
    critic_l2_reg = 1e-2
    actor_lr = 1e-4
    critic_lr = 1e-3
    popart = False
    clip_norm = None
    reward_scale = 1.

    env = prosthetics_env.Wrapper(osim_env.ProstheticsEnv(visualize=False),
                                  frameskip=4,
                                  reward_shaping=True,
                                  reward_shaping_x=1,
                                  feature_embellishment=True,
                                  relative_x_pos=True,
                                  relative_z_pos=True)

    top_model_dir = 'top-models/'

    # create tf sessions and graphs
    sess_list = []
    graph_list = []
    for i in range(len(args.model_files)):
        graph_list.append(tf.Graph())
        sess_list.append(tf.Session(graph=graph_list[i]))
    ddpg_agents = []
    for i in range(len(args.model_files)):
        model_name = args.model_files[i]
        sess = sess_list[i]
        graph = graph_list[i]
        l_size = args.layer_sizes[i]
        with sess.as_default():
        #with U.make_session(num_cpu=1, graph=g) as sess:
            with graph.as_default():
                #tf.global_variables_initializer()

                # restore agents from model files and store in ddpg_agents
                print("Restoring from..." + model_name)

                # Configure components.
                memory = Memory(limit=int(1e6), action_shape=env.action_space.shape,
                                observation_shape=env.observation_space.shape)
                critic = Critic(layer_norm=True, activation='relu', layer_sizes=[l_size, l_size])
                actor = Actor(env.action_space.shape[-1], layer_norm=True,
                              activation='relu', layer_sizes=[256, 256])
                agent = DDPG(actor, critic, memory, env.observation_space.shape,
                             env.action_space.shape, gamma=gamma, tau=tau,
                             normalize_returns=normalize_returns,
                             normalize_observations=normalize_observations,
                             batch_size=batch_size, action_noise=action_noise,
                             param_noise=param_noise, critic_l2_reg=critic_l2_reg,
                             actor_lr=actor_lr, critic_lr=critic_lr,
                             enable_popart=popart, clip_norm=clip_norm,
                             reward_scale=reward_scale)

                # restore adam state and param noise
                restore_model_path = top_model_dir + model_name
                saver = tf.train.Saver(max_to_keep=500)

                # restore network weights
                saver.restore(sess, restore_model_path)

                adam_optimizer_store = pickle.load(open(restore_model_path
                                                        + ".pkl", "rb"))
                agent.actor_optimizer.m = adam_optimizer_store['actor_optimizer']['m']
                agent.actor_optimizer.v = adam_optimizer_store['actor_optimizer']['v']
                agent.actor_optimizer.t = adam_optimizer_store['actor_optimizer']['t']
                agent.critic_optimizer.m = adam_optimizer_store['critic_optimizer']['m']
                agent.critic_optimizer.v = adam_optimizer_store['critic_optimizer']['v']
                agent.critic_optimizer.t = adam_optimizer_store['critic_optimizer']['t']
                if 'param_noise' in adam_optimizer_store:
                    agent.param_noise = adam_optimizer_store['param_noise']

                # intialize and prepare agent session.
                agent.initialize(sess)
                #sess.graph.finalize()
                agent.reset()

                ddpg_agents.append(agent)

    # setup eval env
    eval_env = prosthetics_env.EvaluationWrapper(osim_env.ProstheticsEnv(visualize=False),
                                                 frameskip=4,
                                                 reward_shaping=True,
                                                 reward_shaping_x=1,
                                                 feature_embellishment=True,
                                                 relative_x_pos=True,
                                                 relative_z_pos=True)
    eval_env.change_model(model=('3D').upper(), prosthetic=True, difficulty=0, seed=0)
    eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))

    nb_eval_steps = 1000
    reward, mean_q, final_steps = evaluate_one_episode(eval_env, ddpg_agents, sess_list, graph_list,
                                                       nb_eval_steps=nb_eval_steps,
                                                       render=False)

    for i in range(len(sess_list)):
        sess_list[i].close()

    print("Reward: " + str(reward))
    print("Mean Q: " + str(mean_q))
    print("Final num steps: " + str(final_steps))


if __name__ == "__main__":
    main()

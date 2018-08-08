import argparse
from baselines.ddpg.noise import AdaptiveParamNoiseSpec
from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines import logger, bench
from baselines.ddpg import prosthetics_env
import baselines.common.tf_util as U
from baselines.common.misc_util import boolean_flag
import osim.env as osim_env
from osim.http.client import Client
import tensorflow as tf
import numpy as np
import pickle
import os
from pdb import set_trace
from baselines.ddpg.training import evaluate_one_episode


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-files', type=str, nargs='+')
    parser.add_argument('--layer-sizes', type=int, nargs='+')
    boolean_flag(parser, 'crowdai-submit', default=False)
    parser.add_argument('--crowdai-token', default='9c48765358e511504cf7731614afac30')
    boolean_flag(parser, 'evaluation', default=False)
    args = parser.parse_args()
    return args


class BlendedAgent():
    def __init__(self, ddpg_agents, sess_list, graph_list):
        self.ddpg_agents = ddpg_agents
        self.sess_list = sess_list
        self.graph_list = graph_list
        self.num_agents = len(ddpg_agents)

    def pi(self, obs, apply_noise=True, compute_Q=True):
        pre_reward = [0.] * self.num_agents
        possible_actions = []
        for i in range(self.num_agents):
            actor_agent = self.ddpg_agents[i]
            # for each actor use a different possible critic
            with self.sess_list[i].as_default():
                with self.graph_list[i].as_default():
                    action, _ = actor_agent.pi(obs, apply_noise=apply_noise, compute_Q=True)
            possible_actions.append(action)
            for j in range(self.num_agents):
                critic_agent = self.ddpg_agents[j]
                with self.sess_list[j].as_default():
                    with self.graph_list[j].as_default():
                        q = critic_agent.compute_Q_for_action(obs, action)
                        pre_reward[i] += np.asscalar(q[0])

        best_action_idx = np.argmax(pre_reward)
        action_out = possible_actions[best_action_idx]
        q_out = pre_reward[best_action_idx]
        return action_out, q_out


# def evaluate_one_episode(env, ddpg_agents, sess_list, graph_list, nb_eval_steps, render):
#    if nb_eval_steps <= 0:
#        print('evaluate_one_episode nb_eval_steps must be > 0')
#    reward = 0.
#    qs = []
#    obs = env.reset()
#    num_agents = len(ddpg_agents)
#    for step in range(nb_eval_steps):
#        pre_reward = [0.] * num_agents
#        possible_actions = []
#        for i in range(num_agents):
#            actor_agent = ddpg_agents[i]
#            # for each actor use a different possible critic
#            with sess_list[i].as_default():
#                with graph_list[i].as_default():
#                    action, _ = actor_agent.pi(obs, apply_noise=False, compute_Q=True)
#            possible_actions.append(action)
#            for j in range(num_agents):
#                critic_agent = ddpg_agents[j]
#                with sess_list[j].as_default():
#                    with graph_list[j].as_default():
#                        q = critic_agent.compute_Q_for_action(obs, action)
#                        pre_reward[i] += np.asscalar(q[0])
#
#         best_action_idx = np.argmax(pre_reward)
#         obs, r, done, info = env.step(possible_actions[best_action_idx])
#         if render:
#             env.render()
#         reward += r
#         qs.append(pre_reward[best_action_idx])
#         print("Eval step " + str(step))
#         if done:
#             #obs = env.reset()
#             break  # the original baseline code didn't have this break statement, so would average multiple evaluation episodes
#         elif step >= nb_eval_steps:
#             logger.warn('evaluate_one_episode step', step, 'exceeded nb_eval_steps', nb_eval_steps, 'but done is False')
#             #obs = env.reset()
#             break
#     return reward, np.mean(qs), step+1


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
                              activation='relu', layer_sizes=[l_size, l_size])
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

    agent = BlendedAgent(ddpg_agents, sess_list, graph_list)

    if args.evaluation:
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
        # reward, mean_q, final_steps = evaluate_one_episode(eval_env, ddpg_agents, sess_list, graph_list,
        #                                                    nb_eval_steps=nb_eval_steps,
        #                                                    render=False)
        reward, mean_q, final_steps = evaluate_one_episode(eval_env, agent, nb_eval_steps, render=False)

    # Submit to crowdai competition. What a hack. :)
    # if crowdai_client is not None and crowdai_token is not None and eval_env is not None:
    crowdai_submit_count = 0
    if args.crowdai_submit:
        remote_base = "http://grader.crowdai.org:1729"
        crowdai_client = Client(remote_base)
        eval_obs_dict = crowdai_client.env_create(args.crowdai_token, env_id="ProstheticsEnv")
        eval_obs_dict, eval_obs_projection = prosthetics_env.transform_observation(
            eval_obs_dict,
            reward_shaping=True,
            reward_shaping_x=1.,
            feature_embellishment=True,
            relative_x_pos=True,
            relative_z_pos=True)
        while True:
            action, _ = agent.pi(eval_obs_projection, apply_noise=False, compute_Q=False)
            submit_action = prosthetics_env.openai_to_crowdai_submit_action(action)
            clipped_submit_action = np.clip(submit_action, 0., 1.)
            actions_equal = clipped_submit_action == submit_action
            if not np.all(actions_equal):
                logger.debug("crowdai_submit_count:", crowdai_submit_count)
                logger.debug("  openai-action:", action)
                logger.debug("  submit-action:", submit_action)
            crowdai_submit_count += 1
            [eval_obs_dict, reward, done, info] = crowdai_client.env_step(clipped_submit_action.tolist(), True)
            # [eval_obs_dict, reward, done, info] = crowdai_client.env_step(agent.pi(eval_obs_projection, apply_noise=False, compute_Q=False), True)
            eval_obs_dict, eval_obs_projection = prosthetics_env.transform_observation(
                eval_obs_dict,
                reward_shaping=True,
                reward_shaping_x=1.,
                feature_embellishment=True,
                relative_x_pos=True,
                relative_z_pos=True)
            if done:
                logger.debug("done: crowdai_submit_count:", crowdai_submit_count)
                eval_obs_dict = crowdai_client.env_reset()
                if not eval_obs_dict:
                    break
                logger.debug("done: eval_obs_dict exists after reset")
                eval_obs_dict, eval_obs_projection = prosthetics_env.transform_observation(
                    eval_obs_dict,
                    reward_shaping=True,
                    reward_shaping_x=1.,
                    feature_embellishment=True,
                    relative_x_pos=True,
                    relative_z_pos=True)
        crowdai_client.submit()

    for i in range(len(sess_list)):
        sess_list[i].close()

    print("Reward: " + str(reward))
    print("Mean Q: " + str(mean_q))
    print("Final num steps: " + str(final_steps))


if __name__ == "__main__":
    main()

import os
import sys
import time
from collections import deque
import pickle
import random
import string
from tensorflow.python.framework.errors import InvalidArgumentError

from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U
from baselines.ddpg import prosthetics_env

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI


def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    saved_model_basename, restore_model_name, crowdai_client, crowdai_token,
    reward_shaping, feature_embellishment, relative_x_pos,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50):
    rank = MPI.COMM_WORLD.Get_rank()

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    saved_model_dir = 'saved-models/'
    if saved_model_basename is None:
        saved_model_basename = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    saved_model_path = saved_model_dir + saved_model_basename
    if restore_model_name:
        restore_model_path = saved_model_dir + restore_model_name
    if rank == 0:
        saver = tf.train.Saver(max_to_keep=100)
    else:
        saver = None

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        try:
            if restore_model_name:
                logger.info("Restoring from model at", restore_model_path)
                #saver.restore(sess, tf.train.latest_checkpoint(model_path))
                saver.restore(sess, restore_model_path)
            else:
                logger.info("Creating new model")
                sess.run(tf.global_variables_initializer()) # this should happen here and not in the agent right?
        except InvalidArgumentError as exc:
            if "Assign requires shapes of both tensors to match." in str(exc):
                print("Unable to restore model from {:s}.".format(restore_model_path))
                print("Chances are you're trying to restore a model with reward embellishment into an environment without reward embellishment (or vice versa). Unfortunately this isn't supported (yet).")
                print(exc.message)
                sys.exit()
            else:
                raise exc

        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()
        obs = env.reset()
        if eval_env is not None:
            eval_obs = eval_env.reset()
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    assert max_action.shape == action.shape
                    new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()
                        obs = env.reset()

                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                # Submit to crowdai competition. What a hack. :)
                #if crowdai_client is not None and crowdai_token is not None and eval_env is not None:
                crowdai_submit_count = 0
                if crowdai_client is not None and crowdai_token is not None:
                    eval_obs_dict = crowdai_client.env_create(crowdai_token, env_id="ProstheticsEnv")
                    eval_obs_dict, eval_obs_projection = prosthetics_env.transform_observation(
                        eval_obs_dict,
                        reward_shaping=reward_shaping,
                        feature_embellishment=feature_embellishment,
                        relative_x_pos=relative_x_pos)
                    while True:
                        action, _ = agent.pi(eval_obs_projection, apply_noise=False, compute_Q=False)
                        submit_action = prosthetics_env.openai_to_crowdai_submit_action(action)
                        clipped_submit_action = np.clip(submit_action, 0., 1.)
                        actions_equal = clipped_submit_action == submit_action
                        if not np.all(actions_equal):
                            logger.info("crowdai_submit_count:", crowdai_submit_count)
                            logger.info("  openai-action:", action)
                            logger.info("  submit-action:", submit_action)
                        crowdai_submit_count += 1
                        [eval_obs_dict, reward, done, info] = crowdai_client.env_step(clipped_submit_action.tolist(), True)
                        #[eval_obs_dict, reward, done, info] = crowdai_client.env_step(agent.pi(eval_obs_projection, apply_noise=False, compute_Q=False), True)
                        eval_obs_dict, eval_obs_projection = prosthetics_env.transform_observation(
                            eval_obs_dict,
                            reward_shaping=reward_shaping,
                            feature_embellishment=feature_embellishment,
                            relative_x_pos=relative_x_pos)
                        if done:
                            logger.debug("done: crowdai_submit_count:", crowdai_submit_count)
                            eval_obs_dict = crowdai_client.env_reset()
                            if not eval_obs_dict:
                                break
                            logger.debug("done: eval_obs_dict exists after reset")
                            eval_obs_dict, eval_obs_projection = prosthetics_env.transform_observation(
                                eval_obs_dict,
                                reward_shaping=reward_shaping,
                                feature_embellishment=feature_embellishment,
                                relative_x_pos=relative_x_pos)
                    crowdai_client.submit()
                    return  # kids, don't try any of these (expedient hacks) at home!

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                eval_steps = []
                if eval_env is not None:
                    eval_episode_reward = 0.
                    for t_rollout in range(nb_eval_steps):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_obs = eval_env.reset()
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.
                            eval_steps.append(t_rollout+1)
                            break  # the original baseline code didn't have this break statement, so would average multiple evaluation episodes

            mpi_size = MPI.COMM_WORLD.Get_size()
            # Log stats.
            # XXX shouldn't call np.mean on variable length lists
            duration = time.time() - start_time
            if nb_epochs and nb_epoch_cycles and nb_train_steps > 0:
                stats = agent.get_stats()
                combined_stats = stats.copy()
                combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
                combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
                combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
                combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
                combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
                combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
                combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
                combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
                combined_stats['total/duration'] = duration
                combined_stats['total/steps_per_second'] = float(t) / float(duration)
                combined_stats['total/episodes'] = episodes
                combined_stats['rollout/episodes'] = epoch_episodes
                combined_stats['rollout/actions_std'] = np.std(epoch_actions)
            else:
                combined_stats = {}
            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = np.mean(eval_episode_rewards)
                combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                combined_stats['eval/Q'] = np.mean(eval_qs)
                combined_stats['eval/episodes'] = len(eval_episode_rewards)
                combined_stats['eval/steps'] = np.mean(eval_steps)
            def as_scalar(x):
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s'%x)
            combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
            combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()

            if nb_epochs and nb_epoch_cycles and nb_train_steps > 0:
                logger.info('Saving model to', saved_model_dir + saved_model_basename)
                saver.save(sess, saved_model_path, global_step=epoch, write_meta_graph=False)

            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)

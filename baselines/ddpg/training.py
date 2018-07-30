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
# https://bitbucket.org/mpi4py/mpi4py/issues/54/example-mpi4py-code-not-working
import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI
import pickle
from pdb import set_trace
import pathlib
import resource

NB_EVAL_EPOCHS=1
NB_EVAL_EPOCH_CYCLES=0
NB_EVAL_STEPS=1001

def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    saved_model_basename, restore_model_name, crowdai_client, crowdai_token,
    reward_shaping, feature_embellishment, relative_x_pos, relative_z_pos,
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
        restore_model_path = restore_model_name
        if not pathlib.Path(restore_model_path+'.index').is_file():
            restore_model_path = saved_model_dir + restore_model_name
    max_to_keep = 500
    eval_reward_threshold_to_keep = 300
    saver = tf.train.Saver(max_to_keep=max_to_keep)
    adam_optimizer_store = dict()
    adam_optimizer_store['actor_optimizer'] = dict()
    adam_optimizer_store['critic_optimizer'] = dict()

    #eval_episode_rewards_history = deque(maxlen=100)
    #episode_rewards_history = deque(maxlen=100)
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

        # restore adam optimizer
        try:
            if restore_model_name:
                logger.info("Restoring pkl file with adam state", restore_model_path)
                #saver.restore(sess, tf.train.latest_checkpoint(model_path))
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
        except:
            print("Unable to restore adam state from {:s}.".format(restore_model_path))

        obs = env.reset()
        done = False
        episode_reward = 0.
        #episode_step = 0
        #episodes = 0
        #t = 0

        #epoch_episode_steps = []
        #epoch_episode_eval_rewards = []
        #epoch_episode_eval_steps = []
        #epoch_start_time = time.time()
        #epoch_actions = []
        #epoch_episodes = 0
        for epoch in range(nb_epochs):
            start_time = time.time()
            epoch_episode_rewards = []
            epoch_qs = []
            eval_episode_rewards = []
            eval_qs = []
            eval_steps = []
            epoch_actor_losses = []
            epoch_critic_losses = []
            worth_keeping = False
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
                    #new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    new_obs, r, done, info = env.step(action)
                    #t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    #episode_step += 1

                    # Book-keeping.
                    #epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        #episode_rewards_history.append(episode_reward)
                        #epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        #episode_step = 0
                        #epoch_episodes += 1
                        #episodes += 1

                        agent.reset()
                        obs = env.reset()

                # Train.
                #epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        #epoch_adaptive_distances.append(distance)

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
                        reward_shaping_x=1.,
                        feature_embellishment=feature_embellishment,
                        relative_x_pos=relative_x_pos,
                        relative_z_pos=relative_z_pos)
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
                        #[eval_obs_dict, reward, done, info] = crowdai_client.env_step(agent.pi(eval_obs_projection, apply_noise=False, compute_Q=False), True)
                        eval_obs_dict, eval_obs_projection = prosthetics_env.transform_observation(
                            eval_obs_dict,
                            reward_shaping=reward_shaping,
                            reward_shaping_x=1.,
                            feature_embellishment=feature_embellishment,
                            relative_x_pos=relative_x_pos,
                            relative_z_pos=relative_z_pos)
                        if done:
                            logger.debug("done: crowdai_submit_count:", crowdai_submit_count)
                            eval_obs_dict = crowdai_client.env_reset()
                            if not eval_obs_dict:
                                break
                            logger.debug("done: eval_obs_dict exists after reset")
                            eval_obs_dict, eval_obs_projection = prosthetics_env.transform_observation(
                                eval_obs_dict,
                                reward_shaping=reward_shaping,
                                reward_shaping_x=1.,
                                feature_embellishment=feature_embellishment,
                                relative_x_pos=relative_x_pos,
                                relative_z_pos=relative_z_pos)
                    crowdai_client.submit()
                    return  # kids, don't try any of these (expedient hacks) at home!

            if eval_env:
                eval_episode_reward_mean, eval_q_mean, eval_step_mean = evaluate_n_episodes(3, eval_env, agent, nb_eval_steps, render_eval)
                if eval_episode_reward_mean >= eval_reward_threshold_to_keep:
                    worth_keeping = True

            mpi_size = MPI.COMM_WORLD.Get_size()
            # Log stats.
            # XXX shouldn't call np.mean on variable length lists
            duration = time.time() - start_time
            if nb_epochs and nb_epoch_cycles and nb_train_steps > 0:
                #stats = agent.get_stats()
                #combined_stats = stats.copy()
                combined_stats = {}
                combined_stats['train/epoch_episode_reward_mean'] = np.mean(epoch_episode_rewards)
                #combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
                #combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
                #combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
                combined_stats['train/epoch_Q_mean'] = np.mean(epoch_qs)
                combined_stats['train/epoch_loss_actor'] = np.mean(epoch_actor_losses)
                combined_stats['train/epoch_loss_critic'] = np.mean(epoch_critic_losses)
                #combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
                combined_stats['train/epoch_duration'] = duration
                #combined_stats['epoch/steps_per_second'] = float(t) / float(duration)
                #combined_stats['total/episodes'] = episodes
                #combined_stats['rollout/episodes'] = epoch_episodes
                #combined_stats['rollout/actions_std'] = np.std(epoch_actions)
                #combined_stats['memory/rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            else:
                combined_stats = {}
            # Evaluation statistics.
            if eval_env:
                combined_stats['eval/epoch_episode_reward_mean'] = eval_episode_reward_mean # np.mean(eval_episode_rewards)
                #combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                #combined_stats['eval/epoch_episode_reward_std'] = np.std(eval_episode_rewards)
                combined_stats['eval/epoch_Q_mean'] = eval_q_mean # np.mean(eval_qs)
                #combined_stats['eval/episodes'] = len(eval_episode_rewards)
                combined_stats['eval/steps_mean'] = eval_step_mean # np.mean(eval_steps)
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
            #combined_stats['total/epochs'] = epoch + 1
            #combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.info('')
            logger.info('Epoch', epoch)
            logger.dump_tabular()
            logdir = logger.get_dir()

            if worth_keeping and rank == 0 and nb_epochs and nb_epoch_cycles and nb_train_steps and nb_rollout_steps:
                logger.info('Saving model to', saved_model_dir + saved_model_basename + '-' + str(epoch))
                saver.save(sess, saved_model_path, global_step=epoch, write_meta_graph=False)
                adam_optimizer_store['actor_optimizer']['m'] = agent.actor_optimizer.m
                adam_optimizer_store['actor_optimizer']['v'] = agent.actor_optimizer.v
                adam_optimizer_store['actor_optimizer']['t'] = agent.actor_optimizer.t

                adam_optimizer_store['critic_optimizer']['m'] = agent.critic_optimizer.m
                adam_optimizer_store['critic_optimizer']['v'] = agent.critic_optimizer.v
                adam_optimizer_store['critic_optimizer']['t'] = agent.critic_optimizer.t

                adam_optimizer_store['param_noise'] = agent.param_noise

                pickle.dump(adam_optimizer_store, open((saved_model_path +
                                                        "-" + str(epoch) +
                                                        ".pkl"), "wb"))
                old_epoch = epoch - max_to_keep
                if old_epoch >= 0:
                    try:
                        os.remove(saved_model_path + "-" + str(old_epoch)
                                  + ".pkl")
                    except OSError:
                        pass

            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)

def evaluate_n_episodes(n, eval_env, agent, nb_eval_steps, render):
    results = []
    for i in range(n):
        results.append(evaluate_one_episode(eval_env, agent, nb_eval_steps, render))
    rewards = [r[0] for r in results]
    q_means = [r[1] for r in results]
    steps   = [r[2] for r in results]
    return np.mean(rewards), np.average(q_means, weights=steps), np.mean(steps)


def evaluate_one_episode(env, agent, nb_eval_steps, render):
    if nb_eval_steps <= 0:
        logger.error('evaluate_one_episode nb_eval_steps must be > 0')
    reward = 0.
    qs = []
    obs = env.reset()
    for step in range(nb_eval_steps):
        action, q = agent.pi(obs, apply_noise=False, compute_Q=True)
        obs, r, done, info = env.step(action)
        if render:
            env.render()
        reward += r
        qs.append(q)
        if done:
            #obs = env.reset()
            break  # the original baseline code didn't have this break statement, so would average multiple evaluation episodes
        elif step >= nb_eval_steps:
            logger.warn('evaluate_one_episode step', step, 'exceeded nb_eval_steps', nb_eval_steps, 'but done is False')
            #obs = env.reset()
            break
    return reward, np.mean(qs), step+1

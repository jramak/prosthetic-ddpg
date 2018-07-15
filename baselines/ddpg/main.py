import argparse
import time
import os
import sys
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.ddpg.training as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *

import gym
import tensorflow as tf
from mpi4py import MPI
import osim.env as osim_env
import opensim as osim
from osim.http.client import Client
from baselines.ddpg import prosthetics_env
from pdb import set_trace
import gc
gc.enable()

def dispatch(seed, noise_type=None, layer_norm=None, evaluation=False, **kwargs):
    kwargs['crowdai_client'] = None
    if kwargs['crowdai_submit']:
        crowdai_submit(seed, noise_type, layer_norm, evaluation, **kwargs)
    elif kwargs['eval_only']:
        evaluate(seed, noise_type, layer_norm, evaluation, **kwargs)
    else:
        run(seed, noise_type, layer_norm, evaluation, **kwargs)

def ignoring_int(k, v, **kwargs):
    if kwargs[k] != v:
        logger.warn('Ignoring {:s}={:d}, using {:d} instead'.format(k, kwargs[k], v))

def evaluate(seed, noise_type, layer_norm, evaluation, **kwargs):
    ignoring_int('nb_epochs', 1, **kwargs)
    kwargs['nb_epochs'] = 1
    ignoring_int('nb_epoch_cycles', 1, **kwargs)
    kwargs['nb_epoch_cycles'] = 1
    ignoring_int('nb_rollout_steps', 0, **kwargs)
    kwargs['nb_rollout_steps'] = 0
    ignoring_int('nb_train_steps', 0, **kwargs)
    kwargs['nb_train_steps'] = 0
    run(seed, noise_type, layer_norm, evaluation, **kwargs)

def crowdai_submit(seed, noise_type, layer_norm, evaluation, **kwargs):
    if 'restore_model_name' not in kwargs:
        logger.error('You must specify the --restore-model-name in order to submit')
        sys.exit()
    remote_base = "http://grader.crowdai.org:1729"
    crowdai_token = kwargs['crowdai_token']
    crowdai_client = Client(remote_base)
    kwargs['crowdai_client'] = crowdai_client
    evaluate(seed, noise_type, layer_norm, evaluation=True, **kwargs)

def run(seed, noise_type, layer_norm, evaluation, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create the opensim env.
    train_env = prosthetics_env.Wrapper(osim_env.ProstheticsEnv(visualize=kwargs['render']),
        frameskip=kwargs['frameskip'],
        reward_shaping=kwargs['reward_shaping'],
        reward_shaping_x=kwargs['reward_shaping_x'],
        feature_embellishment=kwargs['feature_embellishment'],
        relative_x_pos=kwargs['relative_x_pos'],
        relative_z_pos=kwargs['relative_z_pos'])
    train_env.change_model(model=kwargs['model'].upper(), prosthetic=kwargs['prosthetic'], difficulty=kwargs['difficulty'], seed=seed)

    if evaluation and rank==0:
        train_env = bench.Monitor(train_env, None)
        eval_env = prosthetics_env.EvaluationWrapper(osim_env.ProstheticsEnv(visualize=kwargs['render_eval']),
            frameskip=kwargs['eval_frameskip'],
            reward_shaping=kwargs['reward_shaping'],
            reward_shaping_x=kwargs['reward_shaping_x'],
            feature_embellishment=kwargs['feature_embellishment'],
            relative_x_pos=kwargs['relative_x_pos'],
            relative_z_pos=kwargs['relative_z_pos'])
        eval_env.change_model(model=kwargs['model'].upper(), prosthetic=kwargs['prosthetic'], difficulty=kwargs['difficulty'], seed=seed)
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
    else:
        train_env = bench.Monitor(train_env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
        eval_env = None

    # training.train() doesn't like the extra keyword args added for controlling the prosthetics env, so remove them.
    del kwargs['model']
    del kwargs['prosthetic']
    del kwargs['difficulty']
    del kwargs['reward_shaping_x']
    del kwargs['frameskip']
    del kwargs['eval_frameskip']
    del kwargs['crowdai_submit']
    del kwargs['eval_only']

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = train_env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=train_env.action_space.shape, observation_shape=train_env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm, activation=kwargs['activation'])
    actor = Actor(nb_actions, layer_norm=layer_norm, activation=kwargs['activation'])

    del kwargs['activation']

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    train_env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    training.train(env=train_env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
    train_env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=500)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=None)
    boolean_flag(parser, 'evaluation', default=False)
    parser.add_argument('--difficulty', type=int, choices=[0,1,2], default=2)
    parser.add_argument('--model', type=str, choices=['2D', '3D'], default='3D')
    parser.add_argument('--activation', type=str, choices=['relu', 'selu', 'elu'], default='selu')
    boolean_flag(parser, 'prosthetic', default=True)
    parser.add_argument('--frameskip', type=int, default=1)
    parser.add_argument('--eval-frameskip', type=int, default=1)
    parser.add_argument('--saved-model-basename', type=str, default=None)  # all models are saved to saved-models/<saved_model_basename>-<epoch>
    parser.add_argument('--restore-model-name', type=str, default=None)  # all models are saved to saved-models/<restore_model_name>
    boolean_flag(parser, 'reward-shaping', default=False)
    parser.add_argument('--reward-shaping-x', type=float, default=1.)  # multiplier for reward shaping
    boolean_flag(parser, 'feature-embellishment', default=False)
    boolean_flag(parser, 'relative-x-pos', default=True)
    boolean_flag(parser, 'relative-z-pos', default=True)
    boolean_flag(parser, 'crowdai-submit', default=False)  # for submission to crowdai nips prosthetic challenge, must be used with --restore-model-name
    parser.add_argument('--crowdai-token', default='9c48765358e511504cf7731614afac30')
    boolean_flag(parser, 'eval-only', default=False)  # for running evaluation only, no training, must be used with --restore-model-name
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    dispatch(**args)

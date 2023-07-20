"""Random policy on RandomEnvs environments

    Example:
        (no-rendering)
        unset LD_PRELOAD; python test_random_policy.py --env RandomHopper-v0
        unset LD_PRELOAD; python test_random_policy.py --env RandomHopper-v0 --udr 0.5

        (rendering)
            export LD_PRELOAD=$CONDA_PREFIX/lib/libGLEW.so; python test_random_policy.py --env RandomHopper-v0 --render
            NOTE: the libGLEW.so could be located elsewhere (https://github.com/openai/mujoco-py/issues/268)
"""
import argparse
import pdb

import numpy as np
import gym
import random_envs

def main():
    env = gym.make(args.env)

    if args.udr is not None:
        env.set_dr_distribution(dr_type='uniform', distr=env.get_uniform_dr_by_percentage(percentage=args.udr))
        env.set_dr_training(True)

    state = env.reset()
    done = False

    print('============================')
    print('Env:', args.env)
    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Task dim:', env.task_dim)
    print('Current task:', env.get_task())

    while True:
        state, reward, done, info = env.step(env.action_space.sample())
        
        if args.render:
            env.render()

        if done:
            env.reset()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='RandomCartPole-v0', type=str, help='Random envs environments')
    parser.add_argument('--render', default=False, action='store_true', help='Rendering')
    parser.add_argument('--udr', default=None, type=float, help='Uniform domain randomization: sample new dynamics parameter at every reset. Uniform bounds deviated 25\% from the nominal values')

    return parser.parse_args()
args = parse_args()

if __name__ == '__main__':
    main()
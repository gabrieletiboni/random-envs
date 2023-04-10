"""Random policy on RandomEnvs environments

	Example:
		python test_random_policy.py --env RandomHopper-v0
"""
import argparse
import pdb

import gym
import random_envs

def main():
	render = True

	env = gym.make(args.env)

	state = env.reset()
	done = False

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Task dim:', env.task_dim)
	print('Current task:', env.get_task())

	while True:
		state, reward, done, info = env.step(env.action_space.sample())
		
		if render:
			env.render()

		if done:
			env.reset()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='RandomCartPole-v0', type=str, help='Random envs environments')

    return parser.parse_args()
args = parse_args()

if __name__ == '__main__':
	main()
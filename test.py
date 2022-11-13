"""Test script using random-envs

    Using random_envs to train a policy in a source, under-modeled
    domain with Domain Randomization and test the policy in the
    target domain.
"""
import random_envs
import gym

def main():
    source_env = gym.make('RandomHopperUnmodeled-v0')

    env.set_dr_distribution(dr_type='uniform', distr=[0.9, 1.1, 1.9, 2.1, 2.9, 3.1])  # Randomize link masses uniformly
    env.set_dr_training(True)
    # ... policy = ppo.train(source_env)
    env.set_dr_training(False)

    target_env = gym.make('RandomHopper-v0')
    # ... reward = evaluate_policy(policy, target_env)

if __name__ == '__main__':
    main()
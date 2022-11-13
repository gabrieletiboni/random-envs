# Random Gym Environments
Gym environments with domain randomization (DR) support for sim-to-real research in robot learning.
This repo uses the unmaintained version of [gym](https://github.com/openai/gym/), and the old mujoco bindings [mujoco_py](https://github.com/openai/mujoco-py).

Features:
- Gym environments: hopper, half-cheetah, walker2d, humanoid
- Noisy and unmodeled variants for each environment
- DR parametric distributions: uniform, normal, truncnormal
- Automatic sampling of new dynamics when env.reset() is called

## Environments
|                               | dim $\xi$ | $\xi$                             | state noise |
|-------------------------------|-----------|-----------------------------------|-------------|
| RandomHopper-v0               | 4         | Link masses                       | -           |
| RandomHopperNoisy-v0          | 4         | Link masses                       | $10^{-4}$   |
| RandomHopperUnmodeled-v0      | 3         | Link masses                       | -           |
| RandomHalfCheetah-v0          | 8         | Link masses, friction             | -           |
| RandomHalfCheetahNoisy-v0     | 8         | Link masses, friction             | $10^{-4}$   |
| RandomHalfCheetahUnmodeled-v0 | 5         | Link masses, friction             | -           |
| RandomWalker2d-v0             | 13        | Link masses and lengths, friction | -           |
| RandomWalker2dNoisy-v0        | 13        | Link masses and lengths, friction | $10^{-3}$   |
| RandomWalker2dUnmodeled-v0    | 9         | Link masses and lengths, friction | -           |
| RandomHumanoid-v0             | 30        | Link masses, joint damping        | -           |
| RandomHumanoidNoisy-v0        | 30        | Link masses, joint damping        | $10^{-3}$   |
| RandomHumanoidUnmodeled-v0    | 23        | Link masses, joint damping        | -           |

where $\xi \in \mathbb{R}^{dim \ \xi}$ is the dynamics parameter vector. The unmodeled variants represent under-modeled parameterizations of the environments where dynamics parameters not included are misidentified by 20% (Read more in Sec. 3.3 of our [work](https://arxiv.org/abs/2206.14661)).


## Installation
```
	# git clone <this repo>
	cd random-envs
	pip install .
```

## Getting Started
```
	import random_envs
	import gym

	env = gym.make('RandomHopper-v0')

	env.set_dr_distribution(dr_type='uniform', distr=[0.9, 1.1, 1.9, 2.1, 2.9, 3.1, 3.9, 4.1])  # Randomize link masses uniformly
	env.set_dr_training(True)

	# ... train a policy

	env.set_dr_trainin(False)

	# ... evaluate policy in non-randomized env

```
See `test.py` for a complete example in a sim-to-real transfer scenario. 

## Citing
If you use this repository, please consider citing
        
        @misc{tiboniadrbenchmark,
            title={Online vs. Offline Adaptive Domain Randomization Benchmark},
            author={Tiboni, Gabriele and Arndt, Karol and Averta, Giuseppe and Kyrki, Ville and Tommasi, Tatiana},
            year={2022},
            primaryClass={cs.RO},
            publisher={arXiv},
            doi={10.48550/ARXIV.2206.14661}
        }

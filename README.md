# Random Gym Environments
Gym environments with domain randomization (DR) support for sim-to-real research in robot learning.

This repo uses the unmaintained version of [gym](https://github.com/openai/gym/), and the old mujoco bindings [mujoco_py](https://github.com/openai/mujoco-py).

Features:
- Gym environments: hopper, half-cheetah, walker2d, humanoid
- Noisy and unmodeled variants for each environment
- DR parametric distributions: uniform, normal, truncnormal
- Automatic sampling of new dynamics when env.reset() is called

### Environments
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

where $\xi \in \mathbb{R}^{dim \xi}$ is the dynamics parameter vector.

### Getting Started

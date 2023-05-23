### Franka Panda environments
A pushing environment is implemented with Franka Panda robot.

### General notes
- Controller:
  - low-level robot is controlled at 1000Hz (sim dt = 0.001s)
  - by default
  - the policy is queried at 50hz (policy dt = 20ms)
  - Therefore
  - the 20 control timesteps in-between each policy query are handled by the action_interpolator (Repeater
  - LinearInterpolator
  - QuadraticInterpolator)
  - the env has a max_episode_steps=300, which translates to 6000 low-level control timesteps of the robot, for a total of 6 seconds.

- Initial state distribution:
  - box: randomly sampled from uniform(`init_box_low`, `init_box_high`)
  - robot jpos: randomly sampled from `init_joint_pos`+uniform(`-init_jpos_jitter`, `init_jpos_jitter`)
  - robot jvel: randomly sampled from default+uniform(`-init_jvel_jitter`, `init_jvel_jitter`)
  - goal: randomly sampled from uniform(`goal_low`, `goal_high`)
    - goal list (x,y):
      - A: [0.75, 0.0]
      - B: [0.7, 0.1]
      - C: [0.7, -0.1]
      - D: [0.65, -0.2]
      - E: [0.7, 0.05]

- Reward function:
  - target: distance penalty box-target
  - guide: distance penalty EE-box
  - control penalty: penalty for jpos, jvel and jacc which get close to the limits. Each of the three terms is bounded in [0,1].
    - Checkout commit f5e9938576778a517ff80631b174c6c6e0f6cc8a to visualize the plots
  - contact penalty: penalize contact-pairs proportional to their penetration distance
    - ("box", "table", 1e2)
    - ("panda0_finger1", "box", 1e2)
    - ("panda0_finger2", "box", 1e2)
    - ("panda0_finger1", "table", 3e7)
    - ("panda0_finger2", "table", 3e7)

- State space:
  - +7: robot joint pos
  - +7: robot joint vel
  - +2: box pos (x,y)
  - (optional) +1/+2: box orientation (z-euler angle rotation, or sin and cos of this z-euler angle)
  - +2: goal pos (x,y)

### Gym environments
Available envs are named: `PandaPush-PosCtrl-<Goal>-<DynType>[-InitJpos<val>][-InitBox<val>][-BoxHeight<val>][-Guide][-NormReward]-v0`

- Goal:
  - `GoalA`
- DynType:
  - `mf`  # mass, friction (x,y)
  - `mft`  # mass, friction (x,y), torsional friction
  - `mfcom`   # mass, friction (x,y), center of mass (x,y)
  - `mfcomy`  # mass, friction (x,y), center of mass (y)
  - `com`  # center of mass (x,y)
  - `comy`  # center of mass (y)
  - `mftcom`  # mass, friction (x,y), torsional friction, center of mass (x,y)
  - `mfcomd`  # mass, friction (x,y), center of mass (x,y), joint dampings 
  - `d`  # joint dampings
- InitJPos<val>
  - val: float, init jpos configuration uniform jitter
- InitBox<val>
  - val: float, init box position (x,y) uniform jitter
- BoxHeight<val>
  - val: float, box heigh uniform jitter
- Guide: if set, distance penalty EE-box added as reward term
- NormReward: if set, normalize reward terms in range ~[0, 1]
### Franka Panda environments
A pushing environment is implemented with Franka Panda robot.

### General notes
- Controller:
  - low-level robot is controlled at 1000Hz (sim dt = 0.001s)
  - by default, the policy is queried at 50hz (policy dt = 20ms)
  - Therefore, the 20 control timesteps in-between each policy query are handled by the action_interpolator (Repeater, LinearInterpolator, QuadraticInterpolator)
  - the env has a max_episode_steps=500, which translates to 10000 low-level control timesteps of the robot, for a total of 10 seconds.

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
- PandaPush-PosCtrl-GoalA-v0
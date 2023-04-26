### Franka Panda environments


### Common notes
- Controller:
  - low-level robot is controlled at 1000Hz (sim dt = 0.001s)
  - by default, the policy is queried at 50hz (policy dt = 20ms)
  - Therefore, the 20 control timesteps in-between each policy query are handled by the action_interpolator (Repeater, LinearInterpolator, QuadraticInterpolator)
#!/usr/bin/env python3

import time
import gym_minigrid
import gym
from gym_minigrid.wrappers import *

env_name='Minigrid-LavaGapDoorKeyEnv5x5-v0'

num_resets=200
num_frames=5000

env = gym.make(env_name)

# Benchmark env.reset
t0 = time.time()
for i in range(num_resets):
    env.reset()
t1 = time.time()
dt = t1 - t0
reset_time = (1000 * dt) / num_resets

# Benchmark rendering
t0 = time.time()
for i in range(num_frames):
    env.render('rgb_array')
t1 = time.time()
dt = t1 - t0
frames_per_sec = num_frames / dt

# Create an environment with an RGB agent observation
env = gym.make(env_name)
env = RGBImgPartialObsWrapper(env)
env = ImgObsWrapper(env)

# Benchmark rendering
t0 = time.time()
for i in range(num_frames):
    obs, reward, done, info = env.step(0)
t1 = time.time()
dt = t1 - t0
agent_view_fps = num_frames / dt

print('Env reset time: {:.1f} ms'.format(reset_time))
print('Rendering FPS : {:.0f}'.format(frames_per_sec))
print('Agent view FPS: {:.0f}'.format(agent_view_fps))
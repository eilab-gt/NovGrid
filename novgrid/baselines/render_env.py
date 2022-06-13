#!/usr/bin/env python3

import time
import novgrid
import gym_minigrid
import gym
from PIL import Image
from gym_minigrid.wrappers import *


env_name = 'MiniGrid-LavaShortcutMaze8x8-v0'
# env = RGBImgPartialObsWrapper(env)
# env = ImgObsWrapper(env)


env = gym.make(env_name)
env.reset()
outs = env.step(1)
outs2 = env.step(1)
outs3 = env.step(2)


# Simple rendering
img = Image.fromarray(env.render('rgb_array'),'RGB')
img.show()

# ## Video rendering with timing
# t0 = time.time()
# num_frames=5000
# images = []
# for i in range(num_frames):
#     img = Image.fromarray(env.render('rgb_array'),'RGB')
#     images.append(img)
#     # img.show()
#     obs, reward, done, info = env.step(0)
# images[0].save(env_name+'out.gif',
#                save_all=True,
#                append_images=images[1:],
#                optimize=False,
#                duration=40,
#                loop=0)
# t1 = time.time()
# dt = t1 - t0
# frames_per_sec = num_frames / dt
#
# print('Rendering FPS : {:.0f}'.format(frames_per_sec))

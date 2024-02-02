from setuptools import setup

import glob

setup(
    name='novgrid',
    version='0.0.2',
    keywords='novelty, grid, memory, environment, agent, rl, openaigym, openai-gym, gym, gymnasium',
    url='https://github.com/eilab-gt/NovGrid',
    description='A novelty experimentation wrapper for minigrid',
    packages=['novgrid'],
    install_requires=[
        'numpy>=1.15.0',
        'gymnasium',
        'minigrid'
    ],
    data_files=glob.glob('novgrid/env_configs/*.json')
)

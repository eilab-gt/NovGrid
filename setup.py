from setuptools import setup

setup(
    name='novgrid',
    version='0.0.1',
    keywords='novelty, grid, memory, environment, agent, rl, openaigym, openai-gym, gym',
    url='https://github.com/eilab-gt/NovGrid',
    description='A novelty experimentation wrapper for gym-minigrid',
    packages=['novgrid'],
    install_requires=[
        'gym>=0.9.6',
        'numpy>=1.15.0',
        'gym-minigrid'
    ]
)

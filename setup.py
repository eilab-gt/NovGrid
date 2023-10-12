from setuptools import setup

setup(
    name='novgrid',
    version='0.0.1',
    keywords='novelty, grid, memory, environment, agent, rl, openaigym, openai-gym, gym',
    url='https://github.com/eilab-gt/NovGrid',
    description='A novelty experimentation wrapper for minigrid',
    packages=['novgrid'],
    install_requires=[
        'gymnasium',
        'numpy',
        'minigrid'
    ]
)

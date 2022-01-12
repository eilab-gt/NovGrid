from setuptools import setup

setup(
    name='novgrid',
    version='0.0.1',
    keywords='novelty, grid, memory, environment, agent, rl, openaigym, openai-gym, gym',
    url='https://github.com/balloch/NovGrid',
    description='A novelty MiniGrid wrapper package for NovGrid',
    packages=['novgrid'],
    install_requires=[
        'gym>=0.9.6',
        'numpy>=1.15.0'
    ]
)

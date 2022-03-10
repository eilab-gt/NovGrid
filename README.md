# NovGrid

Novelty MiniGrid--NovGrid--is an extension of MiniGrid environment that allows for the world properties and dynamics to change according to a generalized novelty generator. The MiniGrid environment is a grid-world that facilitates reinforcement learning algorithm development with low environment integration overhead, which allows for rapid iteration and testing. In addition to necessary grid world objects of agents, floor, walls, and goals, MiniGrid implements actionable objects including doors, keys, balls, and boxes.  NovGrid extends the MiniGrid  environment by expanding the way the grid world and the agent interact to allow novelties to be injected into the environment. Specifically this is done by expanding the functionality of the actionable objects---doors, keys, lava, etc.---already in MiniGrid and creating a general environment wrapper that injects novelty at a certain point in the training process.

## Novelties

**GoalLocationChange**: This novelty changes the location of the goal object. In MiniGrid the Goal object is usually at fixed location.

**DoorLockToggle**: This novelty makes a door that is assumed to always be locked instead always unlocked and vice versa. In MiniGrid this is usually a static property. If a door that was unlocked before novelty injection is locked and requires a certain key after novelty injection, the policy learned before novelty injection will likely to fail. On the other hand, if novelty injection makes a previously locked door unlocked, an agent that does not ex- plore after novelty injection may always still seek out a key for a door that does not need it.

**DoorKeyChange**: This novelty changes which key that opens a locked door. In MiniGrid doors are always un- locked by keys of the same color as the door. This means that if key and door colors do not match after novelty, agents will have to find another key to open the door. This may cause a previously learned policy to fail until the agent learns to start using the other key.

**DoorNumKeys**: This novelty changes the number of keys needed to unlock a door. The default number of keys is one; this novelty tends to make policies fail because of the extra step of getting a second key.

**ImperviousToLava**: Lava becomes non-harmful, whereas in Minigrid lava always immediately ends the episode with no reward. This may result in new routes to the goal that potentially bypass doors.

**ActionRepetition**: This novelty changes the number of sequential timesteps an action will have to be repeated for it to occur. In MiniGrid it is usually assumed that for an action to occur it only needs to be issued once. So if an agent needed to command the pick-up action twice before novelty but only once afterwards, to reach its most effi- cient policy it would need to learn to not command pickup twice.

**ForwardMovementSpeed**: This novelty modifies the number of steps an agent takes each time the forward command is issued. In MiniGrid agents only move one gridsquare per time step. As a result, if the agent gets faster after novelty, the original policy may have a harder time controlling the agent, and will need to learn how to embrace this change that could make it reach the goal in fewer steps.

**ActionRadius**: This novelty is an example of a change to the relational preconditions of an action by changing the radius around the agent where an action works. In Mini- Grid this is usually assumed to be only a distance of one or zero, depending on the object. If an agent can pick up objects after novelty without being right next to them, it will have to realize this if it is to reach the optimum solu- tion.

**ColorRestriction**: This novelty restricts the objects one can interact with by color. In MiniGrid it is usually as- sumed that all objects can be interacted with. If an agent is trained with no blue interactions before novelty and then isn’t allowed to interact with yellow objects after novelty, the agent will have to learn to pay attention to the color of objects.

**Burdening**: This novelty changes the effect of actions based on whether the agent has any items in the inven- tory. In MiniGrid it is usually assumed that the inventory has no effect on actions. An agent experiencing this nov- elty, for example, might move twice as fast as usual when their inventory is empty, but half as fast as usual when in possession of the item, which it will have to compensate for strategically.

## Using the Package

The easiest way to use NovGrid is to install the package via `pip install novgrid`. Here is a usage example that trains a Stable Baselines3 implementation of PPO on a MiniGrid environment that experiences the DoorKeyChange novelty:

```python
import gym
import gym_minigrid
from stable_baselines3 import PPO
import novgrid

config = {
    'total_timesteps': 10000000,
    'novelty_episode': 10000
}
env = gym.make('MiniGrid-DoorKey-6x6-v0')
env = novgrid.novelty_generation.novelty_wrappers.DoorKeyChange(env, novelty_episode=config['novelty_episode'])
env = gym_minigrid.wrappers.FlatObsWrapper(env)

model = PPO('MlpPolicy', env)
model.learn(config['total_timesteps'])
```

## Manual Instructions

To run the baseline agent, clone the repository and follow the instructions below.

Get dependencies:
```shell
pip install requirements.txt
```

Move into baselines directory:
```shell
cd novgrid/baselines
```

Train from scratch on DoorKeyChange novelty:

```shell
python ppo_minigrid.py --novelty_wrapper=DoorKeyChange
```

Loading existing model and train on DoorKeyChange novelty:

```shell
python ppo_minigrid.py --load_model=models/ppo_minigrid_example_model.zip

```



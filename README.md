# NovGrid

Novelty MiniGrid (NovGrid) is an extension of [MiniGrid](https://github.com/Farama-Foundation/Minigrid) environment that allows for the world properties and dynamics to change according to a generalized novelty generator. The MiniGrid environment is a grid-world that facilitates reinforcement learning algorithm development with low environment integration overhead, which allows for rapid iteration and testing. In addition to necessary grid world objects of agents, floor, walls, and goals, MiniGrid implements actionable objects including doors, keys, balls, and boxes.  NovGrid extends the MiniGrid  environment by expanding the way the grid world and the agent interact to allow novelties to be injected into the environment. Specifically this is done by expanding the functionality of the actionable objects (doors, keys, lava, etc.) already in MiniGrid and creating a general environment wrapper that injects novelty at a certain point in the training process.


If you find this code useful, please reference in your paper:

```
@inproceedings{balloch2022novgrid,
  title={NovGrid: A Flexible Grid World for Evaluating Agent Response to Novelty},
  author={Balloch, Jonathan and Lin, Zhiyu and Hussain, Mustafa and Srinivas, Aarun and Peng, Xiangyu and Kim, Julia and Riedl, Mark},
  booktitle={In Proceedings of AAAI Symposium, Designing Artificial Intelligence for Open Worlds},
  year={2022},
}
```

## Installing the NovGrid Package
Requirements:
- Python 3.8+

From the NovGrid base directory run:
```shell
pip install -e .
```

## Examples using NovGrid

Here is an example that trains a [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) implementation of PPO on a NovGrid environment that experiences the DoorKeyChange novelty:

```python
import gymnasium as gym
from stable_baselines3 import PPO
import minigrid
import novgrid

config = {
    'env_configs': 'door_key_change'
    'total_timesteps': 10000000,
    'novelty_step': 10000
}

env = novgrid.NoveltyEnv(
    env_configs=config['env_configs'],
    novelty_step=config['novelty_step'],
    wrappers=[minigrid.wrappers.FlatObsWrapper]
)

model = PPO('MlpPolicy', env)
model.learn(config['total_timesteps'])
```

## Testing the Installation

To run the sample environment set follow the instructions below. 

```shell
python novgrid/example.py
```

The expected output should be:
```
pygame 2.5.2 (SDL 2.28.2, Python 3.8.18)
Hello from the pygame community. https://www.pygame.org/contribute.html
step_num: 0; env_idx: [0]; rewards: [0]; dones: [False]
step_num: 1; env_idx: [0]; rewards: [0]; dones: [False]
step_num: 2; env_idx: [0]; rewards: [0]; dones: [False]
step_num: 3; env_idx: [0]; rewards: [0]; dones: [False]
step_num: 4; env_idx: [0]; rewards: [0]; dones: [False]
step_num: 5; env_idx: [0]; rewards: [0]; dones: [False]
step_num: 6; env_idx: [0]; rewards: [0]; dones: [False]
step_num: 7; env_idx: [0]; rewards: [0]; dones: [False]
step_num: 8; env_idx: [0]; rewards: [0]; dones: [False]
step_num: 9; env_idx: [0]; rewards: [0]; dones: [False]
step_num: 10; env_idx: [1]; rewards: [0]; dones: [ True]
step_num: 11; env_idx: [1]; rewards: [0]; dones: [False]
step_num: 12; env_idx: [1]; rewards: [0]; dones: [False]
step_num: 13; env_idx: [1]; rewards: [0]; dones: [False]
step_num: 14; env_idx: [1]; rewards: [0]; dones: [False]
step_num: 15; env_idx: [1]; rewards: [0]; dones: [False]
step_num: 16; env_idx: [1]; rewards: [0]; dones: [False]
step_num: 17; env_idx: [1]; rewards: [0]; dones: [False]
step_num: 18; env_idx: [1]; rewards: [0]; dones: [False]
step_num: 19; env_idx: [1]; rewards: [0]; dones: [False]
step_num: 20; env_idx: [1]; rewards: [0]; dones: [False]
step_num: 21; env_idx: [2]; rewards: [0]; dones: [ True]
step_num: 22; env_idx: [2]; rewards: [0]; dones: [False]
step_num: 23; env_idx: [2]; rewards: [0]; dones: [False]
step_num: 24; env_idx: [2]; rewards: [0]; dones: [False]
step_num: 25; env_idx: [2]; rewards: [0]; dones: [False]
step_num: 26; env_idx: [2]; rewards: [0]; dones: [False]
step_num: 27; env_idx: [2]; rewards: [0]; dones: [False]
step_num: 28; env_idx: [2]; rewards: [0]; dones: [False]
step_num: 29; env_idx: [2]; rewards: [0]; dones: [False]
```

## Novelties
The following is a list and descriptions of the available novelty wrappers:

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

# NovGrid

Novelty MiniGrid--NovGrid--is an extension of MiniGrid environment that allows for the world properties and dynamics to change according to a generalized novelty generator. The MiniGrid environment is a grid-world that facilitates reinforcement learning algorithm development with low environment integration overhead, which allows for rapid iteration and testing. In addition to necessary grid world objects of agents, floor, walls, and goals, MiniGrid implements actionable objects including doors, keys, balls, and boxes.  NovGrid extends the MiniGrid  environment by expanding the way the grid world and the agent interact to allow novelties to be injected into the environment. Specifically this is done by expanding the functionality of the actionable objects---doors, keys, lava, etc.---already in MiniGrid and creating a general environment wrapper that injects novelty at a certain point in the training process.

import argparse

ENV_CONFIG_FILE = "sample"
TOTAL_TIME_STEPS = None
NOVELTY_STEP = 10
N_ENVS = 1
RENDER_DISPLAY = False
STEP_DELAY = 0.0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env-configs-file",
        "-ec",
        type=str,
        default=ENV_CONFIG_FILE,
        help="Use the path to a json file containing the env configs here.",
    )
    parser.add_argument(
        "--total-time-steps",
        "-t",
        type=int,
        default=TOTAL_TIME_STEPS,
        help="The total number of time steps to run.",
    )
    parser.add_argument(
        "--novelty-step",
        "-n",
        type=int,
        default=NOVELTY_STEP,
        help="The total number of time steps to run in an environment before injecting the next novelty.",
    )
    parser.add_argument(
        "--n-envs",
        "-e",
        type=int,
        default=N_ENVS,
        help="The number of envs to use when running the vectorized env.",
    )
    parser.add_argument(
        "--render-display",
        "-rd",
        type=lambda s: s.lower() in {"yes", "true", "t", "y"},
        default=RENDER_DISPLAY,
        help="Whether or not to render the display of the environment as the agent is stepping.",
    )
    parser.add_argument(
        "--step-delay",
        "-sd",
        type=float,
        default=STEP_DELAY,
        help="The amount of delay in seconds between each step call.",
    )

    return parser

import argparse
import time

from novgrid import NoveltyEnv
from novgrid.config import make_parser


def run_example(
    args: argparse.Namespace,
):
    env = NoveltyEnv(
        env_configs=args.env_configs_file,
        novelty_step=args.novelty_step,
        n_envs=args.n_envs,
        render_mode="human" if args.render_display else None,
    )

    env.reset()

    if args.total_time_steps is None:
        total_time_steps = (env.num_transfers + 1) * args.novelty_step
    else:
        total_time_steps = args.total_time_steps

    for step_num in range(0, total_time_steps, args.n_envs):
        observations, rewards, dones, infos = env.step(
            [env.action_space.sample() for _ in range(args.n_envs)]
        )
        if args.render_display:
            env.render("human")
        print(
            f"step_num: {step_num}; env_idx: {env.get_attr('env_idx')}; rewards: {rewards}; dones: {dones}"
        )

        if args.step_delay > 0:
            time.sleep(args.step_delay)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    run_example(args=args)

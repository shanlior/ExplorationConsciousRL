from baselines import deepq_base
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.common.atari_wrappers import make_atari
import os.path as osp
import datetime

def main():
    game = 'qbert'
    env = game.capitalize() + "NoFrameskip-v4"
#    env = game + "NoFrameskip-v4"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default=env)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=0)
    parser.add_argument('--dueling', type=int, default=0)
    parser.add_argument('--double', type=int, default=1)
    parser.add_argument('--lambda-double', type=int, default=0)
    parser.add_argument('--lam', type=float, default=0.6)
    parser.add_argument('--targets', type=int, default=1)
    parser.add_argument('--eps-val', type=float, default=0.1)
    parser.add_argument('--num-timesteps', type=int, default=int(50e6))
    parser.add_argument('--piecewise-schedule', type=int, default=1)
    parser.add_argument('--exploration-fraction', type=int, default=int(1e6))


    args = parser.parse_args()
    typename = ""
    if args.prioritized: typename += '_PRIORITY'
    if args.dueling: typename += '_DUEL'
    if args.lambda_double:
        typename += '_LAMBDA'
    if args.double: typename += '_DOUBLE'
    if args.piecewise_schedule:
        typename += '_PIECEWISE'
    if args.targets > 1: typename += '_{}_TARGETS_{}_LAM'.format(args.targets, args.lam)
    dir = osp.join('../experiments/EpsGreedy/' + game.lower(),
                   datetime.datetime.now().strftime("DQN_BASE_{}".format(args.eps_val) + typename + "-%d-%m-%H-%M-%f"))
    logger.configure(dir=dir)
    set_global_seeds(args.seed)
    env = make_atari(args.env)
    env = bench.Monitor(env, logger.get_dir())
    env = deepq_base.wrap_atari_dqn(env)
    model = deepq_base.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling),
    )
    act = deepq_base.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=10000,
        exploration_fraction=(float(args.exploration_fraction)/float(args.num_timesteps)),
        exploration_final_eps=args.eps_val,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        double=bool(args.double),
        lambda_double=bool(args.lambda_double),
        lam=args.lam,
        targets=args.targets,
        piecewise_schedule=bool(args.piecewise_schedule),
	test_agent=1e6
    )
    act.save()
    env.close()


if __name__ == '__main__':
    main()

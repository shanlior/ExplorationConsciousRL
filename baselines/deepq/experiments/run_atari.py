from baselines import deepq
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
    parser.add_argument('--expected-dueling', type=int, default=0)
    parser.add_argument('--double', type=int, default=1)
    parser.add_argument('--alpha', type=int, default=1) # Enables alpha-dqn
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--expected', type=int, default=1) # Turns on Expected-alpha-dqn
    parser.add_argument('--surrogate', type=int, default=0) # Turns on Surrogate-alpha-dqn
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--eps-val', type=float, default=0.01)
    parser.add_argument('--alpha-val', type=float, default=0.01)
    parser.add_argument('--piecewise-schedule', type=int, default=0)
    parser.add_argument('--alpha-epsilon', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(50e6))
    parser.add_argument('--exploration-fraction', type=int, default=int(1e6))
    parser.add_argument('--optimal-test', type=int, default=0)


    args = parser.parse_args()
    if args.alpha:
        assert(bool(args.expected) != bool(args.surrogate)) # both versions can't run together
    if args.surrogate:
        surrogate = True
    else:
        surrogate = False
    typename = ''
    if args.double: typename += 'D'

    typename += "DQN"
    typename += '_EPS_{}'.format(args.eps_val)
    if args.alpha:
        if args.eps_val != args.alpha_val:
            typename += '_ALPHA_{}'.format(args.alpha_val)
    if surrogate:
        typename += '_Surrogate'
    else:
        typename += '_Expected'
        if args.sample:
            typename += '_Sample'
    if args.piecewise_schedule:
        typename += '_Piecewise'
    if args.alpha_epsilon:
        typename += '_ALPHA=EPS'
    typename += '_{}_step'.format(args.steps)

    if args.prioritized: typename += '_PRIORITY'
    if args.dueling: typename += '_DUEL'
    game = args.env[:-14].lower()
    directory = 'AlphaGreedy/'
    dir = osp.join('../experiments/' + directory + game,
                   datetime.datetime.now().strftime(typename + "-%d-%m-%H-%M-%f"))
    logger.configure(dir=dir)
    set_global_seeds(args.seed)
    env = make_atari(args.env)
    env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
    env = deepq.wrap_atari_dqn(env)
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling),
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=10000,
        exploration_fraction=(float(args.exploration_fraction)/float(args.num_timesteps)),
        exploration_final_eps=args.eps_val,
        alpha_val=args.alpha_val,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        double=bool(args.double),
        epsilon=bool(args.alpha), # turns on alpha-dqn
        eps_val=args.eps_val,
        q1=surrogate, # q1 is surrogate, else, expected
        n_steps=args.steps,
        sample=bool(args.sample),
        piecewise_schedule=bool(args.piecewise_schedule),
	    test_agent=1e6
    )
    act.save()
    env.close()


if __name__ == '__main__':
    main()

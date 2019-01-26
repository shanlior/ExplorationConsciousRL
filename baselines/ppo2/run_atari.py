#!/usr/bin/env python3
import sys
from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
import multiprocessing
import tensorflow as tf
import os.path as osp
import datetime


def train(env_id, num_timesteps, seed, policy, alpha_optimal, alpha):

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()
    env = VecFrameStack(make_atari_env(env_id, 8, seed), 4)
    policy = {'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy}[policy]
    if alpha_optimal:
        ent_coef = 0.0
    else:
        ent_coef = 0.01
    ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=10,
        ent_coef=ent_coef,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1))

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--alpha', help='Alpha Value', type=float, default=0.01)
    parser.add_argument('--alpha_optimal', help='Alpha Optimal', type=bool, default=False)
    args = parser.parse_args()
    mode = ''
    if args.alpha_optimal:
        mode = 'Alpha_{}_'.format(args.alpha)
    dir = osp.join('Qbert', mode +
                   datetime.datetime.now().strftime("PPO-%d-%m-%H-%M-%f"))
    logger.configure(dir=dir)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, alpha_optimal=args.alpha_optimal, alpha=args.alpha)

if __name__ == '__main__':
    main()

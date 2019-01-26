import gym
from baselines.a2c.a2c import Model
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from policies import CnnPolicy, LstmPolicy, LnLstmPolicy


def load(path,args,env):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    nenvs = 1
    ob_space = env.observation_space
    ac_space = env.action_space

    make_model = lambda: Model(policy=args.policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=1)
    model = make_model()
    return model.load(path)

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    args = parser.parse_args()
    env = gym.make("BreakoutNoFrameskip-v4")
    env = wrap_deepmind(env)
    act = load("make_model.pkl",args,env)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()

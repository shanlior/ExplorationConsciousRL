import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import gym
import baselines.common.tf_util as U
from baselines import logger
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import BatchInput, FixedBatchInput, load_state, save_state
from collections import deque


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_state(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)


def load(path):
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
    return ActWrapper.load(path)


def learn(env,
          q_func,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.01,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          test_agent=1e6,
          param_noise=False,
          double=True,
          epsilon=True,
          eps_val=0.01,
          alpha_val=0.01,
          q1=False,
          n_steps=1,
          sample=False,
          piecewise_schedule=False,
          alpha_epsilon=False,
          callback=None):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    epsilon: if True, runs alpha-DQN
    Q1: if True, runs Surrogate version, else, runs Expected version.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.__enter__()

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph
    observation_space_shape = env.observation_space.shape
    def make_obs_ph(name):
        return BatchInput(observation_space_shape, name=name)
    def make_fixed_obs_ph(name):
        return FixedBatchInput(observation_space_shape, batch=batch_size*n_steps, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        make_fixed_obs_ph=make_fixed_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
        double_q=double,
        epsilon=epsilon,
        eps_val=alpha_val,
        q1=q1,
        n_steps=n_steps,
        batch_size=batch_size,
        sample=sample,
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer


    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha, n_steps=n_steps, gamma=gamma)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size, n_steps, gamma)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.

    if piecewise_schedule:
        exploration = PiecewiseSchedule(endpoints=[(0,1.0),(1e6,exploration_final_eps),(24e6,0.01)], outside_value=0.01)
    else:
        exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)
    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True
    epinfobuf = deque(maxlen=100)
    test_flag = False


    with tempfile.TemporaryDirectory() as td:
        model_saved = False
        model_file = os.path.join(td, "model")
        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True

            if epsilon:
                env_action, action_wanted, random_action_flag = act(np.array(obs)[None], update_eps=update_eps, **kwargs)

                env_action = env_action[0]
                if q1:
                    action = action_wanted[0]
                else:
                    action = env_action

            else:
                action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                env_action = action
            reset = False
            new_obs, rew, done, info = env.step(env_action)


            # Store transition in the replay buffer.

            replay_buffer.add(obs, action, env_action, rew, new_obs, float(done), float(random_action_flag), update_eps)
            obs = new_obs
            maybeepinfo = info.get('episode')
            if maybeepinfo:
                epinfobuf.extend([maybeepinfo])
            episode_rewards[-1] += rew
            if done:
                done_cnt = -1
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True


            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, env_actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, env_actions, rewards, obses_tp1, dones, random_action_flags, eps = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(actions), None
                if alpha_epsilon:
                    td_errors = train(obses_t, actions, env_actions, rewards, obses_tp1, dones, weights, random_action_flags, update_eps)
                else:
                    td_error = train(obses_t, actions, env_actions, rewards, obses_tp1, dones, weights, random_action_flags, alpha_val)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            if t > learning_starts and t % test_agent == 0:
                test_flag = True

            if done and test_flag:

                nEpisodes = 50
                rewards = deque(maxlen=nEpisodes)
                for i in range(nEpisodes):
                    obs, done = env.reset(), False
                    episode_rew = 0
                    reward = 0
                    maybeepinfo = None

                    while maybeepinfo is None:
                        curr_update_eps = 0.001
                        if env.unwrapped.ale.getEpisodeFrameNumber() > 108000: # Terminates episode by acting randomly
                            curr_update_eps=0.99999
                        obs, rew, done, info = env.step(act(obs[None], stochastic=True, update_eps=0.001, optimal_test=optimal_test)[0])
                        maybeepinfo = info.get('episode')
                        if maybeepinfo:
                            reward = maybeepinfo['r']
                            rewards.extend([reward])

                logger.record_tabular("test_reward_mean", np.mean([rew for rew in rewards]))
                logger.record_tabular("steps", t)
                logger.dump_tabular()
                obs = env.reset()
                test_flag = False


            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                mean_reward = safemean([epinfo['r'] for epinfo in epinfobuf])

                logger.record_tabular("episode_reward_mean", mean_reward)
                logger.record_tabular("eplenmean" , safemean([epinfo['l'] for epinfo in epinfobuf]))
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)

                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))

                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_reward > saved_mean_reward or ((mean_reward >= saved_mean_reward) and mean_reward > 0):
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_reward))
                    save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_reward
                    act.save()
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_state(model_file)

    return act

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

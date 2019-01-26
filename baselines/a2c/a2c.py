import os
import os.path as osp
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from collections import deque
from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common import tf_util

from utils import discount_with_dones, discount_moments_with_dones
from utils import Scheduler, make_path, find_trainable_variables
from utils import cat_entropy, mse

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps,
            ent_coef=0.01, vf_coef=0.5, mf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.make_session()
        nact = ac_space.n
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        ADV_MOMENT = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        R2 = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])
        ENT_COEF = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean((ADV) * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        mf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.mf), R2))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        ent_coef = Scheduler(v=ent_coef, nvalues=total_timesteps/10, schedule='step')
        mf_coef = 0.01
        loss = pg_loss - entropy*ENT_COEF + vf_loss * vf_coef + mf_loss * mf_coef
        # loss = pg_loss + vf_loss * vf_coef + mf_loss * mf_coef
        # loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef


        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, rewards_square, masks, actions, values, moments):
            values_random = np.random.normal(loc=values,scale=np.sqrt(np.maximum(moments - values ** 2,0)))
            # values_random = values - np.sqrt(np.maximum(moments - values ** 2,0))
            advs = rewards - values_random
            # advs = (1 - 2 * rewards) * rewards - values  + 2 * values * values
            advs_moment = rewards_square - moments
            # advs = (1 + 2 * rewards) * (rewards)
            # advs_moment = rewards_square
            for step in range(len(obs)):
                cur_lr = lr.value()
                cur_ent_coef = ent_coef.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, ADV_MOMENT: advs_moment, R:rewards, R2:rewards_square, LR:cur_lr, ENT_COEF:cur_ent_coef}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, moment_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, mf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, moment_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, nsteps=5, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc)
        self.obs = np.zeros((nenv, nh, nw, nc), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.counters = np.zeros(nenv)
        self.counters_fixed = []


    def run(self):
        mb_obs, mb_rewards, mb_rewards_square, mb_actions, mb_values, mb_moments, mb_dones = [],[],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for n in range(self.nsteps):
            actions, values, moments, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_moments.append(moments)
            mb_dones.append(self.dones)
            obs, rewards, dones, infos  = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
                    self.counters_fixed.append(self.counters[n])
                    self.counters[n] = 0
                else:
                    self.counters[n] += rewards[n]
            self.obs = obs
            rewards = np.sign(rewards)
            mb_rewards.append(rewards)
            mb_rewards_square.append(rewards) # MAYBE ERRRRORR
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_rewards_square = np.asarray(mb_rewards_square, dtype=np.float32).swapaxes(1, 0)

        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_moments = np.asarray(mb_moments, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        values_temp, moments_temp = self.model.value(self.obs, self.states, self.dones)
        last_values = values_temp.tolist()
        last_moments = moments_temp.tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value, moment) in enumerate(zip(mb_rewards, mb_dones, last_values, last_moments)):
            rewards_square = rewards.copy()
            rewards_square = rewards_square.tolist()
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                sigma = np.sqrt(np.maximum(moment - value ** 2, 0))
                prob_value = np.random.normal(loc=value,scale=sigma)
                # prob_value = value + sigma
                rewards = discount_with_dones(rewards+[prob_value], dones+[0], self.gamma)[:-1]
                rewards_square = discount_moments_with_dones(rewards_square, dones, self.gamma, flag=True, value=value, moment=moment)
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
                rewards_square = discount_moments_with_dones(rewards_square, dones, self.gamma)

            mb_rewards[n] = rewards
            mb_rewards_square[n] = rewards_square
        mb_rewards = mb_rewards.flatten()
        mb_rewards_square = mb_rewards_square.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_moments = mb_moments.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_rewards_square, mb_masks, mb_actions, mb_values, mb_moments, epinfos

def learn(policy, env, seed, nsteps=5, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100,save_interval=10000):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
    epinfobuf = deque(maxlen=100)


    nbatch = nenvs*nsteps
    tstart = time.time()
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, rewards_square, masks, actions, values, moments, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        policy_loss, value_loss, moment_loss, policy_entropy = model.train(obs, states, rewards, rewards_square, masks, actions, values, moments)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            em = explained_variance(moments, rewards_square)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("moment_loss", float(moment_loss))
            logger.record_tabular("explained_variance_rewards", float(ev))
            logger.record_tabular("explained_variance_moments", float(em))
            logger.record_tabular("episode_reward_mean", np.mean(np.array(runner.counters_fixed[-30:-1])))
            # logger.record_tabular("episode_length_mean", safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.dump_tabular()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
    if logger.get_dir():
        checkdir = osp.join(logger.get_dir(), 'checkpoints')
        os.makedirs(checkdir, exist_ok=True)
        savepath = osp.join(checkdir, '%.5i' % update)
        print('Saving to', savepath)
        model.save(savepath)
    env.close()


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
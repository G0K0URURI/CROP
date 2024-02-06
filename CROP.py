import time

import d4rl
import gym
import os
import numpy as np
from basicTrainer import BasicModelBasedOfflineRLTrainer, D4RLReplayBuffer
from model import ensembleModel, Normalizer, FCN
import torch.optim as optim
import torch
import torch.nn.functional as F
from urllib.error import HTTPError
import json
import random
from env_termination_fn import termination_fn
from myUtils import str2bool
from typing import List, Tuple, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CROPTrainer(BasicModelBasedOfflineRLTrainer):
    """CROP with ensemble environment model"""
    def __init__(self, state_dim, action_dim, state_mlp_hidden_size, reward_mlp_hidden_size,
                 critic_mlp_hidden_size, actor_mlp_hidden_size, critic_lr, actor_lr, alpha_lr=3e-5,
                 log_alpha=0., target_entropy=None, gamma=0.99, soft_tau=0.005,
                 offline_buffer=None, model_replay_buffer_capacity=1e5, env=None, env_lr=1e-4, beta=1.0, domain=None,
                 env_nums=7, best_env_nums=5, state_weight_decays=None, reward_weight_decays=None, max_reward=None,
                 min_reward=None, max_logstd=0.25,
                 min_logstd=-5):
        super(CROPTrainer, self).__init__(state_dim, action_dim, critic_mlp_hidden_size,
                                          actor_mlp_hidden_size, critic_lr,
                                          actor_lr, alpha_lr, log_alpha, target_entropy, gamma,
                                          soft_tau, offline_buffer,
                                          model_replay_buffer_capacity, env)
        self.env_nums = env_nums
        self.best_env_nums = best_env_nums
        self.best_env_state_model_index = []
        self.best_env_reward_model_index = []
        self.env_model = ensembleModel(state_dim=state_dim,
                                       action_dim=action_dim,
                                       state_mlp_hidden_size=state_mlp_hidden_size,
                                       reward_mlp_hidden_size=reward_mlp_hidden_size,
                                       num_ensemble=env_nums,
                                       num_elites=best_env_nums,
                                       state_weight_decays=state_weight_decays,
                                       reward_weight_decays=reward_weight_decays,
                                       max_logstd=max_logstd,
                                       min_logstd=min_logstd,
                                       max_reward=max_reward,
                                       min_reward=min_reward).to(device)
        self.termination_fn = termination_fn[domain]

        self.env_state_optimizer = optim.Adam([{'params': self.env_model.state_predictor.parameters()}],
                                              # {'params': self.env_models[i].max_logstd, 'weight_decay':0.0},
                                              # {'params': self.env_models[i].min_logstd, 'weight_decay':0.0}],
                                              lr=env_lr)
        self.env_reward_optimizer = optim.Adam([{'params': self.env_model.reward_predictor.parameters()}],
                                               lr=env_lr)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta = beta

        self.max_reward = max_reward
        self.min_reward = min_reward

    def env_state_loss(self, state, action, next_state):
        next_state_mean, next_state_std = self.env_model.pre_next_state_mean_and_std(state, action)
        next_state_var = next_state_std.square()
        next_state_mean_loss = torch.square(next_state_mean - next_state) / next_state_var
        # next_state_var_loss = next_state_var.log()
        next_state_var_loss = next_state_std.log()
        return next_state_mean_loss, next_state_var_loss

    def env_reward_loss(self, state, action, reward, random_action_num=1):
        reward_mean = self.env_model.pre_reward(state, action)

        reward_mse_loss = torch.square(reward_mean - reward)
        reward_advantage = torch.zeros_like(reward)
        for _ in range(random_action_num):
            random_action = torch.rand_like(action).to(device) * 2 - 1
            random_reward_mean = self.env_model.pre_reward(state, random_action)
            reward_advantage -= (random_reward_mean - self.min_reward) / (self.max_reward - self.min_reward)
        reward_advantage /= random_action_num
        return reward_mse_loss, reward_advantage

    def env_state_train_step(self, state, action, next_state):
        next_state_mean_loss, next_state_var_loss = self.env_state_loss(state, action, next_state)

        loss = next_state_mean_loss.mean(dim=(1, 2)).sum() + next_state_var_loss.mean(dim=(1, 2)).sum() # + \
               # 0.02 * torch.sum(self.env_model.state_predictor.max_logstd) - \
               # 0.02 * torch.sum(self.env_model.state_predictor.min_logstd)
        loss += self.env_model.state_predictor.get_decay_loss()
        self.env_state_optimizer.zero_grad()
        loss.backward()
        self.env_state_optimizer.step()

        with torch.no_grad():
            logger = {
                'next_state_mean_loss': next_state_mean_loss.mean().item(),
                'next_state_var_loss': next_state_var_loss.mean().item(),
                'state_loss': loss.item(),
            }
        return logger

    def env_reward_train_step(self, state, action, reward, random_action_num=10):
        reward_mse_loss, reward_advantage = self.env_reward_loss(state, action, reward, random_action_num)

        loss = reward_mse_loss.mean(dim=(1, 2)) - self.beta * reward_advantage.mean(dim=(1, 2))
        loss = loss.sum()
        loss += self.env_model.reward_predictor.get_decay_loss()
        self.env_reward_optimizer.zero_grad()
        loss.backward()
        self.env_reward_optimizer.step()

        with torch.no_grad():
            logger = {
                'reward_mse_loss': reward_mse_loss.mean().item(),
                'reward_advantage': reward_advantage.mean().item(),
                'reward_loss': loss.item(),
            }
        return logger

    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.env_model.state_predictor.num_elites)]
        return elites

    def env_state_train(self, batch_size, epoch, save_dir, valid_size=5000, max_epochs_since_update=5):

        log_key = ['next_state_mean_loss', 'next_state_var_loss', 'state_loss', 'state_mse_loss',
                   'state_mse_loss_valid', 'next_state_mean_loss_valid', 'next_state_var_loss_valid',
                   'state_loss_valid']

        self.env_model.state_predictor.normalizer.fit(
            np.concatenate([self.offline_buffer.observations, self.offline_buffer.actions],
                           axis=1))
        if valid_size < 1:
            valid_size = int(len(self.offline_buffer) * valid_size)
        valid_indexes = [random.sample(range(len(self.offline_buffer)), int(valid_size)) for _ in
                         range(self.env_nums)]
        train_indexes = [list(set(range(len(self.offline_buffer))) - set(valid_index)) for valid_index in
                         valid_indexes]
        valid_indexes = np.array(valid_indexes)
        train_indexes = np.array(train_indexes)
        train_size = train_indexes.shape[1]

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 训练状态预测器
        best_state_valid_loss = np.array([None]).repeat(self.env_nums)
        state_epochs_since_update = np.zeros(self.env_nums)
        state_trained_flag = np.zeros(self.env_nums)
        # state_epochs_since_update = 0
        info = 'epoch'
        for key in log_key:
            info += f',{key}'
        with open(os.path.join(save_dir, 'model_state_log.csv'), 'w') as f:
            f.write(info + '\n')
        logger = {}
        for i in range(int(epoch) + 1):
            train_indexes = np.array([np.random.permutation(train_index) for train_index in train_indexes])
            for batch_num in range(int(np.ceil(train_size / batch_size))):
                batch_index = train_indexes[:, batch_num * batch_size:min((batch_num + 1) * batch_size, train_size)]
                state = torch.FloatTensor(self.offline_buffer.observations[batch_index]).to(device)
                next_state = torch.FloatTensor(self.offline_buffer.next_observations[batch_index]).to(device)
                action = torch.FloatTensor(self.offline_buffer.actions[batch_index]).to(device)

                logger = self.env_state_train_step(state, action, next_state)
            with torch.no_grad():
                next_state_mean, _ = self.env_model.pre_next_state_mean_and_std(state, action)
                logger['state_mse_loss'] = torch.square(next_state_mean - next_state).mean().item()

                state = torch.FloatTensor(self.offline_buffer.observations[valid_indexes]).to(device)
                next_state = torch.FloatTensor(self.offline_buffer.next_observations[valid_indexes]).to(device)
                action = torch.FloatTensor(self.offline_buffer.actions[valid_indexes]).to(device)

                next_state_mean, _ = self.env_model.pre_next_state_mean_and_std(state, action)
                logger['state_mse_loss_valid'] = torch.square(next_state_mean - next_state).mean().item()
                next_state_mean_valid_loss, next_state_var_valid_loss = self.env_state_loss(state, action,
                                                                                            next_state)
                valid_loss = next_state_mean_valid_loss.mean(dim=(1, 2)) + next_state_var_valid_loss.mean(
                    dim=(1, 2))
                # logger.update(self.env_state_valid_step(state, action, next_state, env_index))
                valid_loss = valid_loss.cpu().numpy()
                logger['state_loss_valid'] = valid_loss.mean()
                valid_loss = list(valid_loss)

            self.save_env_state_part(os.path.join(save_dir, 'model_state_epoch%d.pt' % i))

            info = '%d' % i
            for key in log_key:
                if key in logger.keys():
                    info += ',%.4f' % logger[key]
                else:
                    info += ',None'
            with open(os.path.join(save_dir, 'model_state_log.csv'), 'a') as f:
                f.write(info + '\n')
            if i % 10 == 0:
                info = 'state epoch: %d' % i
                for key in log_key:
                    if key in logger.keys():
                        info += ' | %s: %.3f' % (key, logger[key])
                print(info)

            updated_indexes = []
            for j in range(self.env_nums):
                if state_trained_flag[j] == 0:
                    if best_state_valid_loss[j] is None or best_state_valid_loss[j] > valid_loss[j] + 1e-4:
                        best_state_valid_loss[j] = valid_loss[j]
                        state_epochs_since_update[j] = 0
                        updated_indexes.append(j)
                    else:
                        state_epochs_since_update[j] += 1
                        if state_epochs_since_update[j] > max_epochs_since_update:
                            state_trained_flag[j] = 1
            if len(updated_indexes) > 0:
                self.env_model.state_predictor.update_save(updated_indexes)
            elif (state_trained_flag == 1).all():
                break
        indexes = self.select_elites(best_state_valid_loss)
        self.env_model.state_predictor.set_elites(indexes)
        self.env_model.state_predictor.load_save()
        self.save_env_state_part(os.path.join(save_dir, 'model_state_best.pt'))
        with open(os.path.join(save_dir, 'model_state_log.csv'), 'a') as f:
            f.write(f'{best_state_valid_loss}\n')

    def env_reward_train(self, batch_size, epoch, save_dir, valid_size=5000, max_epochs_since_update=5,
                         random_action_num=10):
        log_key = ['reward_mse_loss', 'reward_advantage', 'reward_loss', 'reward_mse_loss_valid',
                   'reward_advantage_valid', 'reward_loss_valid']

        self.env_model.reward_predictor.normalizer.fit(
            np.concatenate([self.offline_buffer.observations, self.offline_buffer.actions],
                           axis=1))
        if valid_size < 1:
            valid_size = int(len(self.offline_buffer) * valid_size)
        valid_indexes = [random.sample(range(len(self.offline_buffer)), int(valid_size)) for _ in
                         range(self.env_nums)]
        train_indexes = [list(set(range(len(self.offline_buffer))) - set(valid_index)) for valid_index in
                         valid_indexes]
        valid_indexes = np.array(valid_indexes)
        train_indexes = np.array(train_indexes)
        train_size = train_indexes.shape[1]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 训练回报预测器
        best_reward_valid_loss = np.array([None]).repeat(self.env_nums)
        reward_epochs_since_update = np.zeros(self.env_nums)
        reward_trained_flag = np.zeros(self.env_nums)
        info = 'epoch'
        for key in log_key:
            info += f',{key}'
        with open(os.path.join(save_dir, 'model_reward_log.csv'), 'w') as f:
            f.write(info + '\n')
        logger = {}
        for i in range(int(epoch) + 1):
            train_indexes = np.array([np.random.permutation(train_index) for train_index in train_indexes])
            for batch_num in range(int(np.ceil(train_size / batch_size))):
                batch_index = train_indexes[:, batch_num * batch_size:min((batch_num + 1) * batch_size, train_size)]
                state = torch.FloatTensor(self.offline_buffer.observations[batch_index]).to(device)
                action = torch.FloatTensor(self.offline_buffer.actions[batch_index]).to(device)
                reward = torch.FloatTensor(self.offline_buffer.rewards[batch_index]).unsqueeze(-1).to(device)

                logger = self.env_reward_train_step(state, action, reward)
            with torch.no_grad():
                reward_mean = self.env_model.pre_reward(state, action)
                logger['reward_mse_loss'] = torch.square(reward_mean - reward).mean().item()

                state = torch.FloatTensor(self.offline_buffer.observations[valid_indexes]).to(device)
                action = torch.FloatTensor(self.offline_buffer.actions[valid_indexes]).to(device)
                reward = torch.FloatTensor(self.offline_buffer.rewards[valid_indexes]).unsqueeze(-1).to(device)

                reward_mean = self.env_model.pre_reward(state, action)
                logger['reward_mse_loss_valid'] = torch.square(reward_mean - reward).mean().item()
                reward_mse_loss, reward_advantage = self.env_reward_loss(state, action, reward,
                                                                         random_action_num)
                valid_loss = reward_mse_loss.mean(dim=(1, 2)) - self.beta * reward_advantage.mean(dim=(1, 2))
                valid_loss = valid_loss.cpu().numpy()
                logger['reward_loss_valid'] = valid_loss.mean()
                valid_loss = list(valid_loss)

            self.save_env_reward_part(os.path.join(save_dir, 'model_reward_epoch%d.pt' % i))

            info = '%d' % i
            for key in log_key:
                if key in logger.keys():
                    info += ',%.6f' % logger[key]
                else:
                    info += ',None'
            with open(os.path.join(save_dir, 'model_reward_log.csv'), 'a') as f:
                f.write(info + '\n')
            if i % 10 == 0:
                info = 'reward epoch: %d' % i
                for key in log_key:
                    if key in logger.keys():
                        info += ' | %s: %.3f' % (key, logger[key])
                print(info)

            updated_indexes = []
            for j in range(self.env_nums):
                if reward_trained_flag[j] == 0:
                    if best_reward_valid_loss[j] is None or best_reward_valid_loss[j] > valid_loss[j] + 1e-4:
                        best_reward_valid_loss[j] = valid_loss[j]
                        reward_epochs_since_update[j] = 0
                        updated_indexes.append(j)
                    else:
                        reward_epochs_since_update[j] += 1
                        if reward_epochs_since_update[j] > max_epochs_since_update:
                            reward_trained_flag[j] = 1
            if len(updated_indexes) > 0:
                self.env_model.reward_predictor.update_save(updated_indexes)
            elif (reward_trained_flag == 1).all():
                break
        indexes = self.select_elites(best_reward_valid_loss)
        self.env_model.reward_predictor.set_elites(indexes)
        self.env_model.reward_predictor.load_save()
        self.save_env_reward_part(os.path.join(save_dir, 'model_reward_best.pt'))
        with open(os.path.join(save_dir, 'model_reward_log.csv'), 'a') as f:
            f.write(f'{best_reward_valid_loss}\n')

    def env_train(self, batch_size, epoch, save_dir, valid_size=5000, max_epochs_since_update=5,
                  random_action_num=1, train_state=True, train_reward=True, state_path='state', reward_path='reward'):
        if train_state:
            self.env_state_train(batch_size, epoch, os.path.join(save_dir, state_path), valid_size,
                                 max_epochs_since_update)
        if train_reward:
            self.env_reward_train(batch_size, epoch, os.path.join(save_dir, reward_path), valid_size,
                                  max_epochs_since_update, random_action_num)
        return

    def roll_out(self, init_state_batch, length=1, random_roll_out=False):
        with torch.no_grad():
            state = init_state_batch
            for _ in range(length):
                if not random_roll_out:
                    action = self.actor.get_action(state, dtype='Tensor')
                else:
                    action = torch.rand(state.shape[0], self.action_dim).to(state.device) * 2 - 1
                next_state_mean_ensemble, next_state_std_ensemble, reward_ensemble = self.env_model(state, action)
                batch_size = next_state_mean_ensemble.shape[1]

                state_index = np.random.choice(self.env_model.state_predictor.elites.data.cpu().numpy(), batch_size)
                next_state_mean = next_state_mean_ensemble[state_index, np.arange(0, batch_size)]
                next_state_std = next_state_std_ensemble[state_index, np.arange(0, batch_size)]
                noise = torch.randn_like(next_state_std)
                next_state = next_state_mean + noise * next_state_std

                reward_index = self.env_model.reward_predictor.elites.data.cpu().numpy()
                reward = reward_ensemble[reward_index].mean(dim=0)

                state_np = state.detach().cpu().numpy()
                action_np = action.detach().cpu().numpy()
                reward_np = reward.detach().cpu().numpy()[:, 0]  # shape=[batch_size]
                next_state_np = next_state.detach().cpu().numpy()
                done_np = self.termination_fn(state_np, action_np, next_state_np)
                for i in range(state.shape[0]):
                    self.model_buffer.push(state_np[i], action_np[i], reward_np[i], next_state_np[i], done_np[i])
                state = next_state[~done_np]
                if state.shape[0] == 0:
                    break
        return

    def save_env_part(self, save_path):
        torch.save({'envModel': self.env_model.state_dict(),
                    'normalizer': self.env_model.state_predictor.normalizer}, save_path)

    def save_env_state_part(self, save_path):
        torch.save({'envStateModel': self.env_model.state_predictor.state_dict(),
                    'normalizer': self.env_model.state_predictor.normalizer}, save_path)

    def save_env_reward_part(self, save_path):
        torch.save({'envRewardModel': self.env_model.reward_predictor.state_dict(),
                    'normalizer': self.env_model.reward_predictor.normalizer}, save_path)

    def load_env_state_part(self, save_path):
        payload = torch.load(save_path)
        self.env_model.state_predictor.load_state_dict(payload['envStateModel'])
        self.env_model.state_predictor.normalizer = payload['normalizer']
        return

    def load_env_reward_part(self, save_path):
        payload = torch.load(save_path)
        self.env_model.reward_predictor.load_state_dict(payload['envRewardModel'])
        self.env_model.reward_predictor.normalizer = payload['normalizer']
        return

    def load_env_part(self, save_path):
        payload = torch.load(save_path)
        self.env_model.load_state_dict(payload['envModel'])
        self.env_model.state_predictor.normalizer = payload['normalizer']
        self.env_model.reward_predictor.normalizer = payload['normalizer']
        return

    def load_best_env_models(self, env_path, state_path, reward_path):
        self.load_env_state_part(os.path.join(env_path, state_path, 'model_state_best.pt'))
        self.load_env_reward_part(os.path.join(env_path, reward_path, 'model_reward_best.pt'))
        return

    def critic_train_step(self, state, action, reward, next_state, done):
        with torch.no_grad():
            next_action, next_log_prob, _, _, _ = self.actor.evaluate(next_state)
            next_q_value = torch.min(self.target_critic(next_state, next_action), dim=0)[
                               0] - self.alpha * next_log_prob
            target_q_value = reward + (1 - done) * self.gamma * torch.clamp(next_q_value,
                                                                            min=self.min_reward / (1 - self.gamma),
                                                                            max=self.max_reward / (1 - self.gamma))

        critic_loss = self.critic.qLoss(target_q_value, state, action, F.mse_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        logger = {
            'critic_loss': critic_loss.item()
        }
        return logger

    def recompute_reward(self, batch_size=64):
        data_size = len(self.offline_buffer)
        reward_index = self.env_model.reward_predictor.elites.data.cpu().numpy()
        for batch_num in range(int(np.ceil(len(self.offline_buffer) / batch_size))):
            state = torch.FloatTensor(self.offline_buffer.observations[
                                      batch_num * batch_size:min((batch_num + 1) * batch_size, data_size)]).to(device)
            action = torch.FloatTensor(
                self.offline_buffer.actions[batch_num * batch_size:min((batch_num + 1) * batch_size, data_size)]).to(
                device)
            reward = torch.FloatTensor(self.offline_buffer.rewards[
                                       batch_num * batch_size:min((batch_num + 1) * batch_size, data_size)]).unsqueeze(
                1).to(
                device)  # shape=(batch_size, 1)
            reward_ensemble = self.env_model.pre_reward(state, action)
            self.offline_buffer.rewards[
            batch_num * batch_size:min((batch_num + 1) * batch_size, data_size)] = reward_ensemble[reward_index].mean(
                dim=0).detach().cpu().numpy()[:, 0]

    def RL_train(self, offline_batch_size, model_batch_size, train_step, save_dir, roll_out_length=1,
                 critic_update_pre_step=1, BC_initial_step=0, roll_out_freq=1000, roll_out_batch_size=50000,
                 random_roll_out=False):
        assert critic_update_pre_step > 0
        log_key = ['critic_loss', 'actor_loss', 'alpha_loss', 'episode_reward_mean', 'episode_reward_std',
                   'normalized_score', 'reward_mean', 'log_prob', 'entropy', 'alpha', 'log_prob',
                   'expected_new_q_value']
        console_log_key = ['critic_loss', 'actor_loss', 'alpha_loss', 'episode_reward_mean', 'episode_reward_std',
                           'normalized_score', 'reward_mean', 'entropy', 'alpha']
        info = 'step'
        for key in log_key:
            info += f',{key}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'RL_log.csv'), 'w') as f:
            f.write(info + '\n')
        # self.actor_optimizer.state = collections.defaultdict(dict)  # 重置优化器的冲量等状态，避免之前的BC的冲量对RL的影响
        # self.critic_optimizer.state = collections.defaultdict(dict)  # 重置优化器的冲量等状态，避免之前的BC的冲量对RL的影响

        for i in range(int(BC_initial_step)):
            logger = {}
            offline_state, offline_action, offline_reward, offline_next_state, offline_done = self.offline_buffer.sample(
                offline_batch_size)
            offline_state = torch.FloatTensor(offline_state).to(device)
            offline_next_state = torch.FloatTensor(offline_next_state).to(device)
            offline_action = torch.FloatTensor(offline_action).to(device)
            offline_reward = torch.FloatTensor(offline_reward).unsqueeze(1).to(device)  # shape=(batch_size, 1)
            offline_done = torch.FloatTensor(np.float32(offline_done)).unsqueeze(1).to(device)
            logger.update(
                self.critic_train_step(offline_state, offline_action, offline_reward, offline_next_state, offline_done))
            self.soft_update_target_critic()
            logger.update(self.BC_train_step(offline_state, offline_action))
            # print(i, logger['log_prob'])
            if i % 10000 == 0:
                logger['episode_reward_mean'], logger['normalized_score'], logger['episode_reward_std'] = self.RL_test(
                    test_num=50)
                # logger['mean_episode_reward'] = 0
                info = '%dk' % ((i - BC_initial_step) / 1000)
                for key in log_key:
                    if key in logger.keys():
                        info += ',%.4f' % logger[key]
                    else:
                        info += ',None'
                with open(os.path.join(save_dir, 'RL_log.csv'), 'a') as f:
                    f.write(info + '\n')
                self.save_RL_part(os.path.join(save_dir, 'RL_part_%dk.pt' % ((i - BC_initial_step) / 1000)))
                info = 'step: %dk' % ((i - BC_initial_step) / 1000)
                # info = 'step: %d' % (i)
                for key in console_log_key:
                    if key in logger.keys():
                        info += ' | %s: %.3f' % (key, logger[key])
                    else:
                        info += ' | %s: None' % key
                print(info)

        self.model_buffer.clear()
        start_time = time.time()
        self.recompute_reward()
        print(time.time() - start_time)
        # self.actor_optimizer.state = collections.defaultdict(dict)  # 重置优化器的冲量等状态，避免之前的BC的冲量对RL的影响
        # self.critic_optimizer.state = collections.defaultdict(dict)  # 重置优化器的冲量等状态，避免之前的BC的冲量对RL的影响
        for i in range(int(train_step) + 1):
            if i % roll_out_freq == 0:
                init_states, _, _, _, _ = self.offline_buffer.sample(roll_out_batch_size)
                init_states = torch.FloatTensor(init_states).to(device)
                self.roll_out(init_states, length=roll_out_length, random_roll_out=random_roll_out)
            logger = {}
            for _ in range(critic_update_pre_step):
                with torch.no_grad():
                    offline_state, offline_action, offline_reward, offline_next_state, offline_done = self.offline_buffer.sample(
                        offline_batch_size)
                    offline_state = torch.FloatTensor(offline_state).to(device)
                    offline_next_state = torch.FloatTensor(offline_next_state).to(device)
                    offline_action = torch.FloatTensor(offline_action).to(device)
                    offline_reward = torch.FloatTensor(offline_reward).unsqueeze(1).to(device)  # shape=(batch_size, 1)
                    offline_done = torch.FloatTensor(np.float32(offline_done)).unsqueeze(1).to(device)
                    if model_batch_size > 0 and len(self.model_buffer) > 0:
                        model_state, model_action, model_reward, model_next_state, model_done = self.model_buffer.sample(
                            min(len(self.model_buffer), model_batch_size))
                        model_state = torch.FloatTensor(model_state).to(device)
                        model_next_state = torch.FloatTensor(model_next_state).to(device)
                        model_action = torch.FloatTensor(model_action).to(device)
                        model_reward = torch.FloatTensor(model_reward).unsqueeze(1).to(device)  # shape=(batch_size, 1)
                        model_done = torch.FloatTensor(np.float32(model_done)).unsqueeze(1).to(device)
                    else:
                        model_state = torch.zeros((0, offline_state.shape[1])).to(device)
                        model_action = torch.zeros((0, offline_action.shape[1])).to(device)
                        model_reward = torch.zeros(0).to(device)
                        model_next_state = torch.zeros((0, offline_next_state.shape[1])).to(device)
                        model_done = torch.zeros(0).to(device)

                    state = torch.cat((offline_state, model_state), dim=0)
                    next_state = torch.cat((offline_next_state, model_next_state), dim=0)
                    action = torch.cat((offline_action, model_action), dim=0)
                    reward = torch.cat((offline_reward, model_reward), dim=0)
                    done = torch.cat((offline_done, model_done), dim=0)
                    logger['reward_mean'] = reward.mean().item()

                logger.update(self.critic_train_step(state, action, reward, next_state, done))
                self.soft_update_target_critic()
            logger.update(self.actor_train_step(state))
            # assert logger['critic_loss'] < 1e5, logger['critic_loss']
            assert abs(logger['actor_loss']) < 1e5, logger
            assert abs(logger['alpha_loss']) < 1e5, logger
            if i % 10000 == 0:
                logger['episode_reward_mean'], logger['normalized_score'], logger['episode_reward_std'] = self.RL_test(
                    test_num=50)
                # logger['episode_reward_mean_model'], logger['normalized_score_model'], logger[
                #     'episode_reward_std_model'] \
                #     = self.RL_test_model(test_num=10)
                # logger['mean_episode_reward'] = 0
                info = '%dk' % (i / 1000)
                for key in log_key:
                    if key in logger.keys():
                        info += ',%.4f' % logger[key]
                    else:
                        info += ',None'
                with open(os.path.join(save_dir, 'RL_log.csv'), 'a') as f:
                    f.write(info + '\n')
                self.save_RL_part(os.path.join(save_dir, 'RL_part_%dk.pt' % (i / 1000)))
                info = 'step: %dk' % (i / 1000)
                # info = 'step: %d' % (i)
                for key in console_log_key:
                    if key in logger.keys():
                        info += ' | %s: %.3f' % (key, logger[key])
                    else:
                        info += ' | %s: None' % key

                print(info)
            # print(i)

    def save_RL_part(self, save_path):
        torch.save({'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'target_critic': self.target_critic.state_dict(),
                    'log_alpha': self.log_alpha,
                    },
                   save_path)
        return

    def load_RL_part(self, save_path):
        payload = torch.load(save_path)
        self.actor.load_state_dict(payload['actor'])
        self.critic.load_state_dict(payload['critic'])
        self.target_critic.load_state_dict(payload['target_critic'])
        self.log_alpha.data.copy_(payload['log_alpha'].data)
        return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--args-path', type=str)

    # # global hyperparameters
    # parser.add_argument('--env', type=str, default='hopper-medium-expert-v2')
    # parser.add_argument('--domain', type=str, default='hopper')
    # parser.add_argument("--save-path", type=str, help="the root dir of the experiment",
    #                     default='/data/lhdata/model-based-offline/CROP_ensembleEnv/hopper-m-e/')
    # parser.add_argument("--seed", type=int, default=1)
    #
    # # environment model learning hyperparameters
    # parser.add_argument("--train-state", type=str2bool, help='whether train to predict environment state',
    #                     default=False)
    # parser.add_argument("--train-reward", type=str2bool, help='whether train to predict environment reward',
    #                     default=False)
    # parser.add_argument("--env-path", type=str, help="the dir to save environment", default='env')
    # parser.add_argument("--state-path", type=str, help="the dir to save state part of environment", default='state-seed1')
    # parser.add_argument("--reward-path", type=str, help="the dir to save reward part of environment",
    #                     default='reward-beta.1-seed1')
    # parser.add_argument("--model-epoch", type=float, help="epochs to train model", default=1.5e4)
    # parser.add_argument('--state-mlp-hidden-size', type=Union[List[int], Tuple[int]],
    #                     help="hidden size of mlp in environment model", default=[200, 200, 200, 200])
    # parser.add_argument('--state-weight-decays', type=Union[List[int], Tuple[int]],
    #                     help="weight decays in environment model",
    #                     default=[5e-5, 5e-5, 5e-5, 5e-5, 5e-5])
    # parser.add_argument('--reward-mlp-hidden-size', type=Union[List[int], Tuple[int]],
    #                     help="hidden size of mlp in environment model", default=[200, 200, 200, 200])
    # parser.add_argument('--reward-weight-decays', type=Union[List[int], Tuple[int]],
    #                     help="weight decays in environment model",
    #                     default=[5e-5, 5e-5, 5e-5, 5e-5, 5e-5])
    # parser.add_argument("--env-lr", type=float, help="learning rate of environment model", default=1e-3)
    # parser.add_argument('--model-batch-size', type=int, help="batch size when training model", default=256)
    # parser.add_argument("--valid-size", type=float, help="the valid ratio or size for environment learning",
    #                     default=0.01)
    # parser.add_argument("--max-epochs-since-update", type=int, default=30)
    # parser.add_argument('--env-nums', type=int, help="number of environment models", default=7)
    # parser.add_argument('--best-env-nums', type=int, help="number of environment models chose to roll out", default=5)
    # parser.add_argument("--beta", type=float, help="negative punishment of generated samples", default=.1)
    # parser.add_argument('--random-action-num', type=int, default=10)
    # parser.add_argument("--max-logstd", type=float, default=.25)
    # parser.add_argument("--min-logstd", type=float, default=-5)
    # parser.add_argument("--reward-range", type=float, default=1)
    #
    # # reinforcement learning hyperparameters
    # parser.add_argument("--train-rl", type=str2bool, help='whether train reinforcement learning', default=True)
    # parser.add_argument("--rl-path", type=str, help="the dir to save RL", default='rl-test')
    # parser.add_argument("--rl-step", type=float, help="steps for RL", default=1e6)
    # parser.add_argument("--BC-initial-step", type=float, help="steps for behavior clone initialization", default=0)
    # parser.add_argument('--critic-mlp-hidden-size', type=int, help="hidden size of mlp in critic", default=256)
    # parser.add_argument('--actor-mlp-hidden-size', type=int, help="hidden size of mlp in actor", default=256)
    # parser.add_argument("--critic-lr", type=float, help="learning rate of critic", default=3e-4)
    # parser.add_argument("--actor-lr", type=float, help="learning rate of actor", default=1e-4)
    # parser.add_argument("--alpha-lr", type=float, help="learning rate of alpha", default=1e-4)
    # parser.add_argument("--log-alpha", type=float, help="initial alpha", default=0.0)
    # parser.add_argument("--target-entropy", type=float, help="target entropy in SAC", default=None)
    # parser.add_argument('--offline-batch-size', type=int, help="batch size of offline data in RL", default=256)
    # parser.add_argument('--fake-batch-size', type=int, help="batch size of model-generated data in RL", default=256)
    # parser.add_argument("--random-roll-out", type=str2bool, help='', default=False)
    # parser.add_argument("--roll-out-length", type=int, default=5)
    # parser.add_argument("--roll-out-freq", type=int, default=1000)
    # parser.add_argument("--roll-out-batch-size", type=int, default=50000)
    # parser.add_argument('--critic-update-pre-step', type=int, default=2)
    # parser.add_argument('--model-retain-epochs', type=int, default=5)
    # parser.add_argument("--soft-tau", type=float, default=0.005)

    args = parser.parse_args()
    with open(args.args_path, 'r') as f:
        args = f.read()
        args = json.loads(args)
        args = argparse.Namespace(**args)

    dataset = None
    d4rl_env = gym.make(args.env)  # d4rl env

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    d4rl_env.seed(args.seed)
    random.seed(args.seed)

    while dataset is None:
        try:
            dataset = d4rl.qlearning_dataset(d4rl_env)
        except (HTTPError, OSError):
            print('Unable to download dataset. Retry.')
            pass
    d4rl_buffer = D4RLReplayBuffer(dataset)
    max_reward = (0.5 + 0.5 * args.reward_range) * dataset['rewards'].max() + (0.5 - 0.5 * args.reward_range) * dataset['rewards'].min()
    min_reward = (0.5 + 0.5 * args.reward_range) * dataset['rewards'].min() + (0.5 - 0.5 * args.reward_range) * dataset['rewards'].max()
    print(f'Data size:{len(d4rl_buffer)}')
    trainer = CROPTrainer(state_dim=np.prod(d4rl_env.observation_space.shape),
                          action_dim=np.prod(d4rl_env.action_space.shape),
                          state_mlp_hidden_size=args.state_mlp_hidden_size,
                          reward_mlp_hidden_size=args.reward_mlp_hidden_size,
                          state_weight_decays=args.state_weight_decays,
                          reward_weight_decays=args.reward_weight_decays,
                          critic_mlp_hidden_size=args.critic_mlp_hidden_size,
                          actor_mlp_hidden_size=args.actor_mlp_hidden_size,
                          env_lr=args.env_lr,
                          critic_lr=args.critic_lr,
                          actor_lr=args.actor_lr,
                          alpha_lr=args.alpha_lr,
                          offline_buffer=d4rl_buffer,
                          env=d4rl_env,
                          beta=args.beta,
                          domain=args.domain,
                          env_nums=args.env_nums,
                          best_env_nums=args.best_env_nums,
                          model_replay_buffer_capacity=args.model_retain_epochs * args.roll_out_batch_size * args.roll_out_length,
                          max_reward=max_reward,
                          min_reward=min_reward,
                          soft_tau=args.soft_tau,
                          max_logstd=args.max_logstd,
                          min_logstd=args.min_logstd,
                          log_alpha=args.log_alpha,
                          target_entropy=args.target_entropy
                          )
    if args.train_state or args.train_reward:
        print(f'Start train environment model')
        if args.train_state:
            if not os.path.exists(os.path.join(args.save_path, args.env_path, args.state_path)):
                os.makedirs(os.path.join(args.save_path, args.env_path, args.state_path))
            with open(os.path.join(args.save_path, args.env_path, args.state_path, 'args.json'),
                      'w') as f:
                json.dump(vars(args), f, sort_keys=True, indent=4)

            trainer.env_state_train(args.model_batch_size, args.model_epoch,
                                    os.path.join(args.save_path, args.env_path, args.state_path),
                                    args.valid_size,
                                    args.max_epochs_since_update)
        if args.train_reward:
            if not os.path.exists(os.path.join(args.save_path, args.env_path, args.reward_path)):
                os.makedirs(os.path.join(args.save_path, args.env_path, args.reward_path))
            with open(os.path.join(args.save_path, args.env_path, args.reward_path, 'args.json'),
                      'w') as f:
                json.dump(vars(args), f, sort_keys=True, indent=4)

            trainer.env_reward_train(args.model_batch_size, args.model_epoch,
                                     os.path.join(args.save_path, args.env_path, args.reward_path),
                                     args.valid_size,
                                     args.max_epochs_since_update, args.random_action_num)

    if args.train_rl:
        if not os.path.exists(os.path.join(args.save_path, args.rl_path)):
            os.makedirs(os.path.join(args.save_path, args.rl_path))
        with open(os.path.join(args.save_path, args.rl_path, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)
        # trainer.load_env_part(os.path.join(args.save_path, args.env_path, 'model_best.pt'))
        trainer.load_best_env_models(os.path.join(args.save_path, args.env_path), args.state_path, args.reward_path)
        print('Start reinforcement learning')
        trainer.RL_train(offline_batch_size=args.offline_batch_size,
                         model_batch_size=args.fake_batch_size,
                         train_step=args.rl_step,
                         save_dir=os.path.join(args.save_path, args.rl_path),
                         roll_out_length=args.roll_out_length,
                         roll_out_freq=args.roll_out_freq,
                         roll_out_batch_size=args.roll_out_batch_size,
                         critic_update_pre_step=args.critic_update_pre_step,
                         BC_initial_step=args.BC_initial_step,
                         random_roll_out=args.random_roll_out)

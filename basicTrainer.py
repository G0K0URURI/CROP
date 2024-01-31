import collections
import os
from model import MultiQNetwork, GaussianPolicyNetwork
import torch.optim as optim
import torch
import torch.nn.functional as F
import random
import numpy as np
import csv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class D4RLReplayBuffer:
    def __init__(self, d4rl_dataset):
        self.observations = d4rl_dataset['observations']
        self.actions = d4rl_dataset['actions']
        self.next_observations = d4rl_dataset['next_observations']
        self.rewards = d4rl_dataset['rewards']
        self.terminals = d4rl_dataset['terminals']
        return

    def sample(self, batch_size):
        batch = random.sample(range(len(self)), batch_size)
        state = self.observations[batch]
        action = self.actions[batch]
        next_state = self.next_observations[batch]
        reward = self.rewards[batch]
        done = self.terminals[batch]
        return state, action, reward, next_state, done

    def devide(self, valid_size):
        if valid_size < 1:
            valid_size = int(len(self) * valid_size)
        valid_index = random.sample(range(len(self)), int(valid_size))
        train_index = list(set(range(len(self))) - set(valid_index))
        valid_buffer = D4RLReplayBuffer({
            'observations': self.observations[valid_index],
            'actions': self.actions[valid_index],
            'next_observations': self.next_observations[valid_index],
            'rewards': self.rewards[valid_index],
            'terminals': self.terminals[valid_index],
        })
        train_buffer = D4RLReplayBuffer({
            'observations': self.observations[train_index],
            'actions': self.actions[train_index],
            'next_observations': self.next_observations[train_index],
            'rewards': self.rewards[train_index],
            'terminals': self.terminals[train_index],
        })
        return train_buffer, valid_buffer

    def __len__(self):
        return self.observations.shape[0]


class ModelReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def clear(self):
        self.buffer = []
        self.position = 0
        return

    def __len__(self):
        return len(self.buffer)


class BasicModelBasedOfflineRLTrainer:
    def __init__(self, state_dim, action_dim, critic_mlp_hidden_size, actor_mlp_hidden_size, critic_lr, actor_lr,
                 alpha_lr=3e-5, log_alpha=0., target_entropy=None, gamma=0.99, soft_tau=0.005, offline_buffer=None,
                 model_replay_buffer_capacity=1e5, env=None):

        self.critic = MultiQNetwork(state_dim=state_dim,
                                    action_dim=action_dim,
                                    hidden_size=critic_mlp_hidden_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.target_critic = MultiQNetwork(state_dim=state_dim,
                                           action_dim=action_dim,
                                           hidden_size=critic_mlp_hidden_size).to(device)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor = GaussianPolicyNetwork(state_dim=state_dim,
                                           action_dim=action_dim,
                                           hidden_dim=actor_mlp_hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.log_alpha = torch.FloatTensor([log_alpha]).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim if target_entropy is None else target_entropy
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.gamma = gamma
        self.soft_tau = soft_tau

        self.offline_buffer = offline_buffer
        self.model_buffer = ModelReplayBuffer(model_replay_buffer_capacity)
        self.env = env

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def env_train(self, batch_size, train_step, save_dir, valid_size=5000):
        pass

    def roll_out(self, init_state_batch, length=5):
        pass

    def save_env_part(self, save_path):
        pass

    def load_env_part(self, save_path):
        pass

    def BC_train_step(self, state, action):
        log_prob = self.actor.get_log_prob(state, action)
        loss = -log_prob.mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        logger = {'log_prob': log_prob.mean().item()}
        return logger

    def BC_valid_step(self, state, action):
        with torch.no_grad():
            log_prob = self.actor.get_log_prob(state, action)
        logger = {'log_prob_valid': log_prob.mean().item()}
        return logger

    def BC_train(self, batch_size, train_step, save_dir, valid_size=5000):
        log_key = ['log_prob', 'log_prob_valid']
        info = 'step'
        train_buffer, valid_buffer = self.offline_buffer.devide(valid_size)
        for key in log_key:
            info += f',{key}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'BC_log.csv'), 'w') as f:
            f.write(info + '\n')
        for i in range(int(train_step) + 1):
            state, action, _, _, _ = train_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            logger = self.BC_train_step(state, action)
            if i % 10000 == 0:
                state, action, _, _, _ = valid_buffer.sample(len(valid_buffer))
                state = torch.FloatTensor(state).to(device)
                action = torch.FloatTensor(action).to(device)

                logger.update(self.BC_valid_step(state, action))

                info = '%dk' % (i / 1000)
                for key in log_key:
                    info += ',%.4f' % logger[key]
                with open(os.path.join(save_dir, 'BC_log.csv'), 'a') as f:
                    f.write(info + '\n')
                self.save_BC_part(os.path.join(save_dir, 'BC_%dk.pt' % (i / 1000)))
                info = 'step: %dk' % (i / 1000)
                for key in log_key:
                    info += ' | %s: %.4f' % (key, logger[key])
                print(info)
        return

    def save_BC_part(self, save_path):
        torch.save({'actor': self.actor.state_dict()},
                   save_path)
        return

    def load_BC_part(self, save_path):
        payload = torch.load(save_path)
        self.actor.load_state_dict(payload['actor'])
        return

    def load_best_BC_part(self, save_dir):
        with open(os.path.join(save_dir, 'BC_log.csv'), 'r') as f:
            log_file = csv.reader(f)
            headers = next(log_file)
            steps = []
            valid_loss = []
            for row in log_file:
                steps.append(row[0])
                valid_loss.append(float(row[2]))
        best_step = steps[np.argmax(valid_loss)]
        self.load_BC_part(os.path.join(save_dir, f'BC_{best_step}.pt'))
        print(f'Load model "{os.path.join(save_dir, f"BC_{best_step}.pt")}" with valid loss {np.max(valid_loss)}')
        return

    def critic_train_step(self, state, action, reward, next_state, done):
        with torch.no_grad():
            next_action, next_log_prob, _, _, _ = self.actor.evaluate(next_state)
            next_q_value = torch.min(self.target_critic(next_state, next_action), dim=0)[
                               0] - self.alpha * next_log_prob
            target_q_value = reward + (1 - done) * self.gamma * next_q_value

        critic_loss = self.critic.qLoss(target_q_value, state, action, F.mse_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        logger = {
            'critic_loss': critic_loss.item()
        }
        return logger

    def actor_train_step(self, state, mean_lambda=1e-3, std_lambda=1e-3,
                              z_lambda=0.0):
        new_action, log_prob, z, mean, log_std = self.actor.evaluate(state)
        expected_new_q_value = torch.min(self.critic(state, new_action), dim=0)[0]

        # log_prob_target = expected_new_q_value - expected_value
        # actor_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
        actor_loss = (self.alpha * log_prob - expected_new_q_value).mean()

        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss = std_lambda * log_std.pow(2).mean()
        z_loss = z_lambda * z.pow(2).sum(1).mean()

        actor_loss += mean_loss + std_loss + z_loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            new_action, log_prob, _, _, _ = self.actor.evaluate(state)
            alpha_loss = log_prob + self.target_entropy
        entropy = -alpha_loss.mean() + self.target_entropy
        alpha_loss = -self.alpha * alpha_loss.mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        logger = {
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'entropy': entropy.item(),
            'alpha': self.alpha.item(),
            'log_prob': log_prob.mean().item(),
            'expected_new_q_value': expected_new_q_value.mean().item()
        }
        return logger

    def soft_update_target_critic(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

    def RL_train(self, offline_batch_size, model_batch_size, train_step, save_dir, roll_out_length=1, critic_update_pre_step=1):
        log_key = ['critic_loss', 'actor_loss', 'alpha_loss', 'episode_reward_mean', 'episode_reward_std', 'normalized_score']
        info = 'step'
        for key in log_key:
            info += f',{key}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'RL_log.csv'), 'w') as f:
            f.write(info + '\n')
        self.actor_optimizer.state = collections.defaultdict(dict)
        self.critic_optimizer.state = collections.defaultdict(dict)
        for i in range(int(train_step) + 1):
            init_states, _, _, _, _ = self.offline_buffer.sample(1)
            init_states = torch.FloatTensor(init_states).to(device)
            self.roll_out(init_states, length=roll_out_length)
            for _ in range(critic_update_pre_step):
                offline_state, offline_action, offline_reward, offline_next_state, offline_done = self.offline_buffer.sample(
                    offline_batch_size)
                if model_batch_size > 0 and len(self.model_buffer) > 0:
                    model_state, model_action, model_reward, model_next_state, model_done = self.model_buffer.sample(
                        min(len(self.model_buffer), model_batch_size))
                else:
                    model_state = np.zeros((0, offline_state.shape[1]))
                    model_action = np.zeros((0, offline_action.shape[1]))
                    model_reward = np.zeros(0)
                    model_next_state = np.zeros((0, offline_next_state.shape[1]))
                    model_done = np.zeros(0)

                state = np.concatenate((offline_state, model_state), axis=0)
                next_state = np.concatenate((offline_next_state, model_next_state), axis=0)
                action = np.concatenate((offline_action, model_action), axis=0)
                reward = np.concatenate((offline_reward, model_reward), axis=0)
                done = np.concatenate((offline_done, model_done), axis=0)
                state = torch.FloatTensor(state).to(device)
                next_state = torch.FloatTensor(next_state).to(device)
                action = torch.FloatTensor(action).to(device)
                reward = torch.FloatTensor(reward).unsqueeze(1).to(device)  # shape=(batch_size, 1)
                done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

                logger = self.critic_train_step(state, action, reward, next_state, done)
                self.soft_update_target_critic()
            logger.update(self.actor_train_step(state))
            if i % 10000 == 0:
                logger['episode_reward_mean'], logger['normalized_score'], logger['episode_reward_std'] = self.RL_test()
                # logger['mean_episode_reward'] = 0
                info = '%dk' % (i / 1000)
                for key in log_key:
                    info += ',%.4f' % logger[key]
                with open(os.path.join(save_dir, 'RL_log.csv'), 'a') as f:
                    f.write(info + '\n')
                self.save_RL_part(os.path.join(save_dir, 'RL_part_%dk.pt' % (i / 1000)))
                info = 'step: %dk' % (i / 1000)
                # info = 'step: %d' % (i)
                for key in log_key:
                    info += ' | %s: %.3f' % (key, logger[key])
                print(info)

    def RL_test(self, test_num=10):
        episode_rewards = []
        for _ in range(test_num):
            done = False
            state = self.env.reset()
            episode_reward = 0
            while not done:
                state = torch.FloatTensor(state).to(device)
                action = self.actor.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state

            episode_rewards.append(episode_reward)
        episode_reward = np.array(episode_rewards)
        episode_reward_std = episode_reward.std()
        episode_reward = episode_reward.mean()
        return episode_reward, self.env.get_normalized_score(episode_reward)*100, episode_reward_std

    def save_RL_part(self, save_path):
        torch.save({'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'target_critic': self.target_critic.state_dict(),
                    'log_alpha': self.log_alpha},
                   save_path)
        return

    def load_RL_part(self, save_path):
        payload = torch.load(save_path)
        self.actor.load_state_dict(payload['actor'])
        self.critic.load_state_dict(payload['critic'])
        self.target_critic.load_state_dict(payload['target_critic'])
        self.log_alpha.data.copy_(payload['log_alpha'].data)
        return

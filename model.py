import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.distributions import Normal
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 解耦状态和回报预测的环境模型
class envModel(nn.Module):
    """
    注意MOPO源码中不直接计算next_state，而是计算next_state-state
    """
    def __init__(self, state_dim, action_dim, mlp_layer_number, mlp_hidden_size, max_logstd=0.25, min_logstd=-5,
                 max_reward=None, min_reward=None):
        super(envModel, self).__init__()
        assert mlp_layer_number > 1

        self.state_predictor = torch.nn.ModuleList([nn.Linear(state_dim + action_dim, mlp_hidden_size), nn.SiLU()])
        self.reward_predictor = torch.nn.ModuleList([nn.Linear(state_dim + action_dim, mlp_hidden_size), nn.SiLU()])

        for _ in range(mlp_layer_number - 1):
            self.state_predictor.append(nn.Linear(mlp_hidden_size, mlp_hidden_size))
            self.state_predictor.append(nn.SiLU())
        self.state_predictor.append(nn.Linear(mlp_hidden_size, 2 * state_dim))  # 输出为state分布（高斯分布）的均值和方差

        for _ in range(mlp_layer_number - 1):
            self.reward_predictor.append(nn.Linear(mlp_hidden_size, mlp_hidden_size))
            self.reward_predictor.append(nn.SiLU())
        self.reward_predictor.append(nn.Linear(mlp_hidden_size, 1))  # 输出为reward

        self.normalizer = Normalizer(state_dim + action_dim)

        self.state_dim = state_dim
        self.max_logstd = torch.nn.Parameter(torch.ones([1, state_dim]) * max_logstd, requires_grad=True)
        self.min_logstd = torch.nn.Parameter(torch.ones([1, state_dim]) * min_logstd, requires_grad=True)
        self.max_reward = max_reward
        self.min_reward = min_reward

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.normalizer.transform(x)

        state_pre = x
        for i in range(len(self.state_predictor)):
            state_pre = self.state_predictor[i](state_pre)
        state_pre_mean = state_pre[:, :self.state_dim] + state  # 模型计算的是state的变化量
        logstd = state_pre[:, self.state_dim:]
        logstd = self.reform_logstd(logstd)

        reward_pre = x
        for i in range(len(self.reward_predictor)):
            reward_pre = self.reward_predictor[i](reward_pre)

        reward_pre = self.reward_antinormalization(reward_pre)
        return state_pre_mean, logstd.exp(), reward_pre

    def reform_logstd(self, logstd):
        logstd = self.max_logstd - torch.nn.functional.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + torch.nn.functional.softplus(logstd - self.min_logstd)
        return logstd

    def reward_antinormalization(self, reward):
        reward = torch.sigmoid(reward) * (self.max_reward - self.min_reward) + self.min_reward
        return reward

    def pre_dist_and_reward(self, state, action):
        state_pre_mean, state_pre_std, reward_pre = self.forward(state, action)
        next_state_dist = Normal(state_pre_mean, state_pre_std)
        # reward_pre = self.reward_antinormalization(reward_pre)
        return next_state_dist, reward_pre

    def pre_next_state_mean_and_var_and_reward(self, state, action):
        state_pre_mean, state_pre_std, reward_pre = self.forward(state, action)
        # reward_pre = self.reward_antinormalization(reward_pre)
        return state_pre_mean, state_pre_std.square(), reward_pre

    def pre_next_state_mean_and_std(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.normalizer.transform(x)

        state_pre = x
        for i in range(len(self.state_predictor)):
            state_pre = self.state_predictor[i](state_pre)
        state_pre_mean = state_pre[:, :self.state_dim] + state  # 模型计算的是state的变化量
        logstd = state_pre[:, self.state_dim:]
        logstd = self.reform_logstd(logstd)

        state_pre_std = logstd.exp()
        return state_pre_mean, state_pre_std

    def pre_reward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.normalizer.transform(x)

        reward_pre = x
        for i in range(len(self.reward_predictor)):
            reward_pre = self.reward_predictor[i](reward_pre)
        reward_pre = self.reward_antinormalization(reward_pre)
        return reward_pre


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class MultiQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, ensembles=2, init_w=3e-3):
        super(MultiQNetwork, self).__init__()
        self.Qs = nn.ModuleList([SoftQNetwork(state_dim, action_dim, hidden_size, init_w) for _ in range(ensembles)])

    def forward(self, state, action):
        out = [q_net(state, action) for q_net in self.Qs]
        return torch.stack(out, dim=0)

    def qLoss(self, target, state, action, criterion):
        loss = 0
        for q_net in self.Qs:
            loss += criterion(q_net(state, action), target)
        return loss

    # def parameters(self, recurse: bool = True):
    #     p = []
    #     for q_net in self.Qs:
    #         p += q_net.parameters()
    #     return p


class GaussianPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3, log_std_min=-25, log_std_max=10):
        super(GaussianPolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def get_log_prob(self, state, action, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = 0.5 * torch.log((1 + action + epsilon) / (1 - action + epsilon))
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        return log_prob

    def get_action(self, state, dtype='ndarray'):
        # if not isinstance(state, torch.Tensor):
        #     state = torch.FloatTensor(state).to(device)
        # if len(state.shape) == 1:
        #     state = state.unsqueeze(0)
        mean, log_std = self.forward(state)

        std = log_std.exp()
        normal = Normal(mean, std)

        action = torch.tanh(normal.sample())
        if dtype == 'ndarray':
            action = action.detach().cpu().numpy()
        return action


class Normalizer:
    def __init__(self, dim):
        self.dim = dim
        self.fitted = False
        self.mu = torch.zeros((1, dim), requires_grad=False).to(device)
        self.sigma = torch.ones((1, dim), requires_grad=False).to(device)
        self.cached_mu, self.cached_sigma = np.zeros([1, dim]), np.ones([1, dim])

    def fit(self, data: np.ndarray):
        """
        Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        :param data: A numpy array containing the input
        :return:
        """
        mu = np.mean(data, axis=0, keepdims=True, dtype=np.float32)
        sigma = np.std(data, axis=0, keepdims=True, dtype=np.float32)
        sigma[sigma < 1e-12] = 1.0

        self.mu = torch.from_numpy(mu).to(device)
        self.sigma = torch.from_numpy(sigma).to(device)
        self.fitted = True
        self.cache()

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.sigma

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.
        Returns: (np.array) The transformed dataset.
        """
        return self.sigma * data + self.mu

    def cache(self):
        """Caches current values of this scaler.
        Returns: None.
        """
        self.cached_mu = self.mu.detach().cpu().numpy()
        self.cached_sigma = self.sigma.detach().cpu().numpy()

    def load_cache(self):
        """Loads values from the cache
        Returns: None.
        """
        self.mu = torch.from_numpy(self.cached_mu).to(device)
        self.sigma = torch.from_numpy(self.cached_sigma).to(device)


class FCN(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_layer_number, mlp_hidden_size):
        super(FCN, self).__init__()
        self.mlp = torch.nn.ModuleList([nn.Linear(input_dim, mlp_hidden_size), nn.SiLU()])
        for _ in range(mlp_layer_number - 1):
            self.mlp.append(nn.Linear(mlp_hidden_size, mlp_hidden_size))
            self.mlp.append(nn.SiLU())
        self.mlp.append(nn.Linear(mlp_hidden_size, output_dim))

    def forward(self, x):
        # x = torch.cat([state, action, reward, next_state], dim=-1)
        for i in range(len(self.mlp)):
            x = self.mlp[i](x)
        return x

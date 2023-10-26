import torch
import torch.nn as nn
from model import Normalizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

state_dim, action_dim, mlp_hidden_size, mlp_layer_number = 1, 1, 200, 4
reward_predictor = torch.nn.ModuleList([nn.Linear(state_dim + action_dim, mlp_hidden_size), nn.SiLU()])

for _ in range(mlp_layer_number - 1):
    reward_predictor.append(nn.Linear(mlp_hidden_size, mlp_hidden_size))
    reward_predictor.append(nn.SiLU())
reward_predictor.append(nn.Linear(mlp_hidden_size, 1))  # 输出为reward
normalizer = Normalizer(state_dim + action_dim)

data = pd.read_csv("simple-mdp/data/offline_data.csv",
                       usecols=['obss', 'actions', "next_obss", "rewards"])

dataset = {}
dataset["observations"] = np.array(data['obss']).reshape(-1, 1)
dataset["actions"] = np.array(data['actions']).reshape(-1, 1)
dataset["next_observations"] = np.array(data['next_obss']).reshape(-1, 1)
dataset["rewards"] = np.array(data['rewards']).reshape(-1, 1)
dataset['terminals'] = np.ones_like(dataset["rewards"])
normalizer.fit(np.concatenate([dataset["observations"], dataset["actions"]], axis=1))

payload = torch.load('simple-mdp/data/env-0/reward/model_reward_best.pt')
reward_predictor.load_state_dict(payload['envRewardModel'])

def reward_function(action):
    if -1 <= action < -0.6:
        T = -action + 0.2
    elif -0.6 <= action < 0.2:
        T = 5 * (action + 0.4) ** 2 + 0.6
    elif -0.2 <= action < 0.2:
        T = -5 * action ** 2 + 1
    elif 0.2 <= action < 0.6:
        T = 10 * (action - 0.4) ** 2 + 0.4
    elif 0.6 <= action <= 1:
        T = action + 0.2
    else:
        raise 0
    return T


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reward_predictor = reward_predictor.to(device)
actions = []
target_rewards = []
pre_rewards = []
for i in range(2000):
    action = i / 1000 - 1
    target_reward = reward_function(action)
    actions.append(action)
    target_rewards.append(target_reward)
    # state_tensor = torch.FloatTensor([[0]])
    # action_tensor = torch.FloatTensor([[action]])

    x = torch.tensor([0, action], dtype=torch.float32).to(device)
    x = normalizer.transform(x)
    x = x.to(device)
    reward_pre = x.to(device)
    for i in range(len(reward_predictor)):
        reward_pre = reward_predictor[i](reward_pre)
    reward_pre = torch.sigmoid(reward_pre) * 2 - 1
    pre_rewards.append(reward_pre.item())

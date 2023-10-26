import numpy as np
import pandas as pd
from scipy.stats import norm

def collect_data(length):
    dataset = {}
    obss = []
    actions = []
    next_obss = []
    rewards = []
    state = np.array([0])
    for _ in range(length):
        action = np.random.normal(0.1, 0.5, 1)
        while action < -1 or action > 1:
            action = np.random.normal(0.1, 0.5, 1)
        next_state = np.array([0])
        reward = dynamic_model(action)

        obss.append(state)
        actions.append(action)
        rewards.append(reward)
        next_obss.append(next_state)

        state = next_state

    # # save data
    obss = np.array(obss, dtype=np.float32).reshape(-1, 1)
    actions = np.array(actions, dtype=np.float32).reshape(-1, 1)
    next_obss = np.array(next_obss, dtype=np.float32).reshape(-1, 1)
    rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
    loggings = np.concatenate((obss, actions, next_obss, rewards), axis=1)
    df = pd.DataFrame(loggings, columns=["obss", "actions", "next_obss", "rewards"])
    df.to_csv(df.to_csv("simple-mdp/data/offline_data.csv"))

    # data = pd.read_csv("simple-mdp/data/offline_data.csv",
    #                    usecols=['obss', 'actions'])

    # form offline data
    dataset["observations"] = obss
    dataset["actions"] = actions
    dataset["next_observations"] = next_obss
    dataset["rewards"] = rewards.reshape(-1)
    dataset['terminals'] = np.ones_like(rewards)

    return dataset


mean1 = 0.5
std1 = 0.2
weight1 = 0.4
custom_normal_dist1 = norm(loc=mean1, scale=std1)
mean2 = -0.3
std2 = 0.5
weight2 = 1
custom_normal_dist2 = norm(loc=mean2, scale=std2)

def dynamic_model(action):
    # if -1 <= action < -0.6:
    #     T = np.random.normal(loc=-action + 0.2, scale=0.06, size=1)
    # elif -0.6 <= action < 0.2:
    #     T = np.random.normal(loc=5 * (action + 0.4) ** 2 + 0.6, scale=0.04, size=1)
    # elif -0.2 <= action < 0.2:
    #     T = np.random.normal(loc=-5 * action ** 2 + 1, scale=0.02, size=1)
    # elif 0.2 <= action < 0.6:
    #     T = np.random.normal(loc=10 * (action - 0.4) ** 2 + 0.4, scale=0.04, size=1)
    # elif 0.6 <= action <= 1:
    #     T = np.random.normal(loc=action + 0.2, scale=0.06, size=1)
    # else:
    #     raise 0
    # return T

    # 计算非标准正态分布下x的概率密度
    pdf_value = custom_normal_dist1.pdf(action)*weight1 + custom_normal_dist2.pdf(action)*weight2 + np.random.normal(loc=0, scale=1, size=1) * 0.1
    return pdf_value

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from basicTrainer import D4RLReplayBuffer
from model import envModel
import torch.optim as optim
import torch
import json
import random
from myUtils import str2bool
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CROPRewardTrainer:
    def __init__(self, state_dim, action_dim, env_mlp_layer_number, env_mlp_hidden_size, env_lr=1e-4, beta=1.0,
                 buffer=None,
                 env_nums=1, max_reward=None, min_reward=None, max_logstd=0.25, min_logstd=-5):
        self.env_nums = env_nums
        self.best_env_state_model_index = []
        self.best_env_reward_model_index = []
        self.env_models = [envModel(state_dim=state_dim,
                                    action_dim=action_dim,
                                    mlp_layer_number=env_mlp_layer_number,
                                    mlp_hidden_size=env_mlp_hidden_size,
                                    max_reward=max_reward,
                                    min_reward=min_reward,
                                    max_logstd=max_logstd,
                                    min_logstd=min_logstd).to(device)
                           for _ in range(env_nums)]
        self.env_reward_optimizers = [optim.Adam([{'params': self.env_models[i].reward_predictor.parameters()}],
                                                 lr=env_lr, weight_decay=0.0)
                                      for i in range(env_nums)]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta = beta
        self.offline_buffer = buffer
        self.max_reward = max_reward
        self.min_reward = min_reward

    def env_reward_loss(self, state, action, reward, env_index=0, random_action_num=1):
        reward_mean = self.env_models[env_index].pre_reward(state, action)

        reward_mse_loss = torch.square(reward_mean - reward)
        reward_advantage = torch.zeros_like(reward)
        for _ in range(random_action_num):
            random_action = torch.rand_like(action).to(device) * 2 - 1
            _, _, random_reward_mean = self.env_models[env_index].pre_next_state_mean_and_var_and_reward(state,
                                                                                                         random_action)
            reward_advantage -= (random_reward_mean - self.min_reward) / (self.max_reward - self.min_reward)
        reward_advantage /= random_action_num
        return reward_mse_loss, reward_advantage

    def env_reward_train_step(self, state, action, reward, env_index, random_action_num=1):
        reward_mse_loss, reward_advantage = self.env_reward_loss(state, action, reward, env_index, random_action_num)

        loss = reward_mse_loss.mean() - self.beta * reward_advantage.mean()
        self.env_reward_optimizers[env_index].zero_grad()
        loss.backward()
        self.env_reward_optimizers[env_index].step()

        with torch.no_grad():
            logger = {
                'reward_mse_loss': reward_mse_loss.mean().item(),
                'reward_advantage': reward_advantage.mean().item(),
                'reward_loss': loss.item(),
            }
        return logger

    def env_reward_valid_step(self, state, action, reward, env_index, random_action_num=1):
        #
        with torch.no_grad():
            reward_mse_loss, reward_advantage = self.env_reward_loss(state, action, reward, env_index,
                                                                     random_action_num)

            loss = reward_mse_loss.mean() - self.beta * reward_advantage.mean()
            logger = {
                'reward_mse_loss_valid': reward_mse_loss.mean().item(),
                'reward_advantage_valid': reward_advantage.mean().item(),
                'reward_loss_valid': loss.item()
            }
        return logger

    def env_reward_train(self, batch_size, epoch, save_dir, valid_size=5000, max_epochs_since_update=15, env_index=0,
                         random_action_num=1):
        log_key = ['reward_mse_loss', 'reward_advantage', 'reward_loss', 'reward_mse_loss_valid',
                   'reward_advantage_valid', 'reward_loss_valid']

        self.env_models[env_index].normalizer.fit(
            np.concatenate([self.offline_buffer.observations, self.offline_buffer.actions],
                           axis=1))
        train_buffer, valid_buffer = self.offline_buffer.devide(valid_size)
        train_size = len(train_buffer)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # train reward predictor
        best_valid_loss = None
        best_epoch = None
        epochs_since_update = 0
        info = 'epoch'
        for key in log_key:
            info += f',{key}'
        with open(os.path.join(save_dir, 'model_reward_log.csv'), 'w') as f:
            f.write(info + '\n')
        logger = {}
        for i in range(int(epoch) + 1):
            index = np.random.permutation(train_size)
            for batch_num in range(int(np.ceil(train_size / batch_size))):
                batch_index = index[batch_num * batch_size:min((batch_num + 1) * batch_size, train_size)]
                state = torch.FloatTensor(train_buffer.observations[batch_index]).to(device)
                next_state = torch.FloatTensor(train_buffer.next_observations[batch_index]).to(device)
                action = torch.FloatTensor(train_buffer.actions[batch_index]).to(device)
                reward = torch.FloatTensor(train_buffer.rewards[batch_index]).unsqueeze(1).to(
                    device)  # shape=(batch_size, 1)
                # done = torch.FloatTensor(np.float32(train_buffer.terminals[batch_index])).unsqueeze(1).to(device)

                logger = self.env_reward_train_step(state, action, reward, env_index, random_action_num)
            if i % 1 == 0:
                with torch.no_grad():
                    state, action, reward, next_state, _ = valid_buffer.sample(len(valid_buffer))
                    state = torch.FloatTensor(state).to(device)
                    next_state = torch.FloatTensor(next_state).to(device)
                    action = torch.FloatTensor(action).to(device)
                    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)  # shape=(batch_size, 1)
                    # done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

                    logger.update(self.env_reward_valid_step(state, action, reward, env_index, random_action_num))
                info = '%dk' % i
                for key in log_key:
                    if key in logger.keys():
                        info += ',%.4f' % logger[key]
                    else:
                        info += ',None'
                with open(os.path.join(save_dir, 'model_reward_log.csv'), 'a') as f:
                    f.write(info + '\n')
                self.save_env_reward_part(os.path.join(save_dir, 'model_reward_epoch%d.pt' % i), env_index)
                if i % 10 == 0:
                    info = 'reward epoch: %d' % i
                    for key in log_key:
                        if key in logger.keys():
                            info += ' | %s: %.3f' % (key, logger[key])
                    print(info)

                if best_valid_loss is None or best_valid_loss - logger['reward_loss_valid'] > 0.0001:
                    best_valid_loss = logger['reward_loss_valid']
                    epochs_since_update = 0
                    self.save_env_reward_part(os.path.join(save_dir, 'model_reward_best.pt'),
                                              env_index)  # 此时模型的状态和回报都被训练
                    best_epoch = i
                else:
                    epochs_since_update += 1
                    if epochs_since_update > max_epochs_since_update:
                        break
        with open(os.path.join(save_dir, 'model_reward_log.csv'), 'a') as f:
            f.write(f'best_epoch:{best_epoch},best_score:{best_valid_loss}\n')
        return

    def save_env_reward_part(self, save_path, env_index=0):
        torch.save({'envRewardModel': self.env_models[env_index].reward_predictor.state_dict()}, save_path)

    def load_env_reward_part(self, save_path, env_index=0):
        payload = torch.load(save_path)
        self.env_models[env_index].reward_predictor.load_state_dict(payload['envRewardModel'])
        return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--args-path', type=str)

    # global hyperparameters
    parser.add_argument("--save-path", type=str, help="the root dir of the experiment",
                        default='simple-mdp/data/')
    parser.add_argument("--seed", type=int, default=1)

    # environment model learning hyperparameters
    parser.add_argument("--train-reward", type=str2bool, help='whether train to predict environment reward',
                        default=True)
    parser.add_argument("--env-path", type=str, help="the dir to save environment", default='env')
    parser.add_argument("--reward-path", type=str, help="the dir to save reward part of environment", default='reward')
    parser.add_argument("--model-epoch", type=float, help="epochs to train model", default=1.5e4)
    parser.add_argument('--env-mlp-layer-number', type=int, help="number of mlp in environment model", default=4)
    parser.add_argument('--env-mlp-hidden-size', type=int, help="hidden size of mlp in environment model", default=200)
    parser.add_argument("--env-lr", type=float, help="learning rate of environment model", default=1e-4)
    parser.add_argument('--model-batch-size', type=int, help="batch size when training model", default=256)
    parser.add_argument("--valid-size", type=float, help="the valid ratio or size for environment learning",
                        default=0.01)
    parser.add_argument("--max-epochs-since-update", type=int, default=30)
    parser.add_argument('--env-nums', type=int, help="number of environment models", default=1)
    # parser.add_argument("--beta", type=float, help="negative punishment of generated samples", default=.1)
    parser.add_argument('--random-action-num', type=int, default=10)
    parser.add_argument("--max-logstd", type=float, default=.25)
    parser.add_argument("--min-logstd", type=float, default=-5)

    args = parser.parse_args()

    # dataset = None
    from simple_mdp import collect_data

    # dataset = collect_data(10000)
    data = pd.read_csv("simple-mdp/data/offline_data.csv",
                       usecols=['obss', 'actions', "next_obss", "rewards"])
    dataset = {}
    dataset["observations"] = np.array(data['obss']).reshape(-1, 1)
    dataset["actions"] = np.array(data['actions']).reshape(-1, 1)
    dataset["next_observations"] = np.array(data['next_obss']).reshape(-1, 1)
    dataset["rewards"] = np.array(data['rewards']).reshape(-1)
    dataset['terminals'] = np.ones_like(dataset["rewards"])
    args.obs_shape = 1
    args.action_dim = 1

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    buffer = D4RLReplayBuffer(dataset)
    for beta in [10]:
        trainer = CROPRewardTrainer(state_dim=args.obs_shape,
                                    action_dim=args.action_dim,
                                    env_mlp_layer_number=args.env_mlp_layer_number,
                                    env_mlp_hidden_size=args.env_mlp_hidden_size,
                                    env_lr=args.env_lr,
                                    buffer=buffer,
                                    beta=beta,
                                    env_nums=args.env_nums,
                                    max_reward=1.5 * dataset["rewards"].max() - 0.5 * dataset["rewards"].min(),
                                    min_reward=1.5 * dataset["rewards"].min() - 0.5 * dataset["rewards"].max(),
                                    max_logstd=args.max_logstd,
                                    min_logstd=args.min_logstd,
                                    )
        for i in range(args.env_nums):
            if not os.path.exists(
                    os.path.join(args.save_path, args.env_path + f'-{i}', args.reward_path + f'-beta{beta}')):
                os.makedirs(os.path.join(args.save_path, args.env_path + f'-{i}', args.reward_path + f'-beta{beta}'))
            with open(os.path.join(args.save_path, args.env_path + f'-{i}', args.reward_path + f'-beta{beta}',
                                   'args.json'),
                      'w') as f:
                json.dump(vars(args), f, sort_keys=True, indent=4)

            trainer.env_reward_train(args.model_batch_size, args.model_epoch,
                                     os.path.join(args.save_path, args.env_path + f'-{i}',
                                                  args.reward_path + f'-beta{beta}'),
                                     args.valid_size,
                                     args.max_epochs_since_update, i, args.random_action_num)

import warnings

import numpy as np
import pandas as pd
from pathlib import Path

import json
import torch


def support_to_scalar(logits):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=-1)
    support_size = logits.shape[-1] // 2
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
            .expand(probabilities.shape)
            .float()
            .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=-1)
    return x


def select_top_p(input_tensor, top_p):
    # Sort the last dimension of the input tensor
    sorted_tensor, _ = torch.sort(input_tensor, dim=-1, descending=True)

    # calculate cumulative sum
    cumulative_sums = torch.cumsum(sorted_tensor, dim=-1)

    # Judge whether the cumulative sum is greater than top_ p, return the bool tensor
    bool_tensor = cumulative_sums > top_p

    # Calculate whether the index of each position is less than the maximum n.
    index_tensor = torch.arange(input_tensor.shape[-1]).expand_as(input_tensor)
    max_n_tensor = bool_tensor.sum(dim=-1, keepdim=True)
    bool_tensor = index_tensor < max_n_tensor

    return bool_tensor


class InsulinArmAgent(object):
    def __init__(self, config_path, model_path, device=None):
        super(InsulinArmAgent, self).__init__()
        if isinstance(model_path, str) or isinstance(model_path, Path):
            model_path = Path(model_path)
            if model_path.name == 'scripted_model.zip':
                self.model = torch.jit.load(str(model_path))
            else:
                self.model = torch.load(str(model_path))
        else:
            self.model = model_path

        if config_path is None:
            warnings.warn('config_path is None. using default max_len=128')
            self.config = {
                'max_len': 128,
                'gamma': 0.90,
            }
        else:
            self.config = json.loads(Path(config_path).read_text())

        self.max_len = 128
        if 'max_len' in self.config:
            self.max_len = self.config['max_len']
        elif 'max_seq_len' in self.config:
            self.max_len = self.config['max_seq_len']
        else:
            warnings.warn(f'max_len not found in config: {config_path}. using 128')
        self.gamma = self.config['gamma']

        self.device = device

        if 'n_input' in self.config:
            self.n_input = self.config['n_input']
        else:
            self.n_input = self.model.n_input
        self.model.eval()
        if self.device is not None:
            self.model.to(device)

    @staticmethod
    def build_from_dir(model_dir, device=None):
        model_dir = Path(model_dir)
        if (model_dir / 'scripted_model.zip').exists():
            model_path = str(model_dir / 'scripted_model.zip')
            config_path = model_dir / 'scripted_model.config.json'
        else:
            model_path = str(model_dir / 'model.pt')
            config_path = model_dir / 'model_data.json'
        agent = InsulinArmAgent(config_path, model_path, device)
        return agent

    def self_rollout(self, obs, action_all, option_all, t0, tt, beam_size=5, top_p=0.9):
        '''
        obs: np.array
            torch.Size([1, 128, 381])
        action_all: np.array
            torch.Size([1, 128])
        option_all: np.array
            torch.Size([1, 128])
        t0: int
            [0,t0): observed time point
        tt: int
            [t0,tt): time point to be predicted

        NOTE:
            [0,t0): observed time point
            [t0,tt): time point to be predicted
            [tt,max_len): padding

        '''
        model = self.model

        shape_valid = self.max_len, self.n_input
        assert len(obs.shape) == 3

        if obs.shape[1:] != shape_valid:
            print(obs.shape, shape_valid)
        assert obs.shape[1:] == shape_valid

        # obs = torch.tensor(obs, dtype=torch.float32)
        # option_all = torch.tensor(option_all, dtype=torch.int)

        padding = torch.zeros(obs.shape[:2], dtype=torch.bool, device=self.device)
        padding[:, t0:] = True

        # initialization
        with torch.no_grad():
            try:
                value0, reward0, policy0, state0 = model.initial_inference(obs, action_all, option_all, padding)
            except Exception as e:
                # raise e
                value0, reward0, policy0, state0 = model.initial_inference(obs, padding)
            # def initial_inference(self, obs, padding_mask: Optional[torch.Tensor] = None, dynamics: bool = False):
            # aux0 = model.auxiliary(state0)
        action0 = policy0.argmax(dim=-1)

        # iteration
        with torch.no_grad():
            # shape: (batch, feature)

            actions = None
            rewards = None
            values = support_to_scalar(value0)[:, t0 - 1]
            values_prev = values
            action_prev = action_all[:, :t0 + 1]

            state0 = torch.cat([state0[:, :t0 + 1]])
            state_temp = torch.cat([state0[:, :t0 + 1]])
            policy_temp = torch.cat([policy0[:, t0]])
            score_temp = torch.cat([support_to_scalar(reward0[:, t0])])
            for i, t in enumerate(range(t0, tt)):
                option_temp = option_all[:, t-t0:t + 1]

                # policy_available = select_top_p(policy, top_p=top_p)
                # score = reward + (policy_available * (-1000))

                # expand
                if option_temp[:, -1].item() == 0:
                    n_expand = 1
                    _, action_temp = torch.topk(policy_temp, n_expand, dim=-1, largest=True)
                    action_temp = action_temp.view(-1)
                    action_temp = torch.zeros_like(action_temp).long()
                    option_temp = option_temp.repeat_interleave(action_temp.shape[0], dim=0)

                else:
                    n_expand = beam_size
                    _, action_temp = policy_temp.topk(n_expand, dim=-1, largest=True)
                    action_temp = action_temp.view(-1)

                    state0 = state0.repeat_interleave(n_expand, dim=0)
                    state_temp = state_temp.repeat_interleave(n_expand, dim=0)
                    option_temp = option_temp.repeat_interleave(action_temp.shape[0], dim=0)

                    action_prev = action_prev.repeat_interleave(n_expand, dim=0)
                    values_prev = values_prev.repeat_interleave(n_expand, dim=0)

                action_prev = torch.cat([action_prev[:,1:], action_temp[:, None]], dim=-1)

                # calculate 
                # value, reward, policy, next_state = model.recurrent_inference(state_temp, action_prev, option_temp)
                value, reward, policy, next_state = model.recurrent_inference(state_temp, action_prev, option_temp, state0)
                value = value[..., -1, :]
                reward = reward[..., -1, :]
                policy = policy[..., -1, :]
                value = support_to_scalar(value)
                reward = support_to_scalar(reward)

                # select sort method
                score_temp = (reward + value)*(self.gamma**i) + score_temp.repeat_interleave(n_expand, dim=0) - values_prev
                # score_temp = reward + score_temp.repeat_interleave(n_expand, dim=0)
                k = min(beam_size, len(score_temp))
                _, selected = score_temp.topk(k, dim=0, largest=True)

                # prune
                state0 = state0[selected]
                state_temp = next_state[selected]
                policy_temp = policy[selected]
                score_temp = score_temp[selected]
                action_selected = action_temp[selected].unsqueeze(1)
                reward_selected = reward[selected].unsqueeze(1)
                value_selected = value[selected].unsqueeze(1)
                values_prev = (value_selected*(self.gamma**i)).squeeze(1)
                action_prev = action_prev[selected]

                if actions is None:
                    actions = action_selected
                    rewards = reward_selected
                    values = value_selected
                else:
                    actions = torch.cat([actions.repeat_interleave(n_expand, dim=0)[selected], action_selected], dim=1)
                    rewards = torch.cat([rewards.repeat_interleave(n_expand, dim=0)[selected], reward_selected], dim=1)
                    values = torch.cat([values.repeat_interleave(n_expand, dim=0)[selected], value_selected], dim=1)

        options = option_all[:, t0:tt].view(-1)
        actions = actions[0]
        values = values[0][:-1]
        rewards = rewards[0]
        return actions, options, values, rewards

    def get_sample_data(self):
        obs = torch.rand(1, self.config['max_len'], self.config['n_input'], dtype=torch.float32)
        action = torch.randint(0, 40, (1, self.config['max_len']))
        option = torch.randint(0, 4, (1, self.config['max_len']))
        return obs, action, option


if __name__ == '__main__':
    from pathlib import Path
    import json
    import torch

    model_dir = Path('assets/models/glu_risk')
    model_dir = Path('tests/output/')

    device = 'cpu'
    model = torch.jit.load(str(model_dir / 'scripted_model.zip'))
    model.to(device)
    config_path = model_dir / 'scripted_model.config.json'
    agent = InsulinArmAgent(config_path, model, device=device)
    obs, action_all, option_all = agent.get_sample_data()
    obs = obs
    action_all = action_all
    option_all = option_all
    t0 = 20
    tt = 40

    obs = obs.to(device)
    action_all = action_all.to(device)
    option_all = option_all.to(device)
    actions, options, values, rewards = agent.self_rollout(obs, action_all, option_all, t0, tt)
    print(action_all[:, t0:tt])
    print(actions, actions.shape)
    print(options, options.shape)
    print(values, values.shape)
    print(rewards, rewards.shape)

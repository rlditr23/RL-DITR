import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable

from ts.models.baseline import TransformerCausalEncoder, TransformerModel, MLP, TransformerCausalDecoder


class TransformerPlanningModel(nn.Module):
    '''
    def representation(self, observation):
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        return next_encoded_state_normalized, reward

    def prediction(self, encoded_state):
        return policy, value

    def initial_inference(self, observation):
        return value, reward, policy_logits, encoded_state,
    def recurrent_inference(self, encoded_state, action):
        return value, reward, policy_logits, next_encoded_state
    '''

    def __init__(self, n_action, n_option, n_reward_max, n_value_max, n_aux,
                 n_input, n_hidden=256, nhead=8, nhid=2048, nlayers=3, dropout: float = 0.5,
                 max_len=512, n_step=7,
                 ):
        super(TransformerPlanningModel, self).__init__()

        # print(n_outputs, n_input, n_hidden, nhead, nhid, nlayers, max_len, dropout)
        self.model_type = 'Transformer'
        self.n_action = n_action
        self.n_option = n_option
        self.n_reward_max = n_reward_max
        self.n_value_max = n_value_max
        self.n_aux = n_aux

        self.max_len = max_len
        self.n_step = n_step

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.nhead = nhead
        self.nlayers = nlayers
        self.nhid = nhid
        self.dropout = dropout
        self.n_reward_size = n_reward_max * 2 + 1
        self.n_value_size = n_value_max * 2 + 1

        # self.encoder = TransformerCausalEncoder(n_input+n_hidden*2, n_hidden, nhead, nhid, nlayers, max_len=max_len, dropout=dropout)
        self.encoder = TransformerCausalEncoder(n_input+n_hidden*2, n_hidden, nhead, nhid, nlayers, max_len=max_len, dropout=dropout)
        # self.encoder = TransformerCausalEncoder(n_input+n_action+n_option, n_hidden, nhead, nhid, nlayers, max_len=max_len, dropout=dropout)
        # self.encoder = TransformerCausalEncoder(n_input, n_hidden, nhead, nhid, 1, max_len=max_len, dropout=dropout)
        self.embedding_action = nn.Embedding(n_action + 1, n_hidden)
        self.embedding_option = nn.Embedding(n_option + 1, n_hidden)

        # self.output_policy_func = MLP(n_hidden,[n_hidden],n_action)
        # self.output_value_func = MLP(n_hidden,[n_hidden],self.n_value_size)
        # self.output_reward_func = MLP(n_hidden*3,[n_hidden],self.n_reward_size)
        # self.output_aux_func = MLP(n_hidden,[n_hidden],n_aux)
        # self.dynamics_func = MLP(n_hidden*3,[n_hidden],n_hidden)

        # self.output_policy_func = MLP(n_hidden*2,[n_hidden],n_action)
        # self.output_value_func = MLP(n_hidden*2,[n_hidden],self.n_value_size)
        # self.output_reward_func = MLP(n_hidden*4,[n_hidden],self.n_reward_size)
        # self.output_aux_func = MLP(n_hidden*2,[n_hidden],n_aux)
        # self.dynamics_func = MLP(n_hidden*4,[n_hidden],n_hidden)

        self.output_policy_func = MLP(n_hidden, [n_hidden], n_action)
        self.output_value_func = MLP(n_hidden, [n_hidden], self.n_value_size)
        self.output_reward_func = MLP(n_hidden * 3, [n_hidden], self.n_reward_size)
        self.output_aux_func = MLP(n_hidden, [n_hidden], n_aux)
        self.output_glu_func = MLP(n_hidden, [n_hidden], 1)
        # self.output_aux_func = nn.Linear(n_hidden,n_aux)
        # self.output_glu_func = nn.Linear(n_hidden,1)

        self.dynamics_func = TransformerCausalDecoder(n_hidden*3, n_hidden, nhead, nhid, nlayers, max_len=max_len, dropout=dropout) # View historical information, Attention
        # self.dynamics_func = TransformerCausalEncoder(n_hidden+n_action+n_option, n_hidden, nhead, nhid, nlayers, max_len=max_len, dropout=dropout) # View historical information, Attention

        # self.output_reward_func = MLP(n_hidden+n_action+n_option,[n_hidden],self.n_reward_size)

    def ext_feat(self, state, action, option):
        action_embed = self.embedding_action(action)
        option_embed = self.embedding_option(option)
        # action_embed = F.one_hot(action,num_classes=self.n_action)
        # option_embed = F.one_hot(option,num_classes=self.n_option)
        feature = torch.cat([state, action_embed, option_embed], dim=-1)
        return feature

    @torch.jit.export
    def representation(self, obs, action_prev, option, padding_mask: Optional[torch.Tensor] = None, dynamics: bool = False):
        feature = self.ext_feat(obs, action_prev, option)
        state = self.encoder(feature, padding_mask, dynamics)
        return state

    @torch.jit.export
    def prediction(self, state: torch.Tensor):
        policy = self.output_policy_func(state)
        value = self.output_value_func(state)
        return policy, value

    @torch.jit.export
    def auxiliary(self, state):
        aux = self.output_aux_func(state)
        return aux

    @torch.jit.export
    def dynamics(self, state, action, option, state_0, padding_mask: Optional[torch.Tensor] = None, offset=0):
        '''
        state: (..., t, dim)
        action: (..., )
        option: (..., )
        Historical information is required
        '''
        feature = self.ext_feat(state, action, option)
        next_state = self.dynamics_func(feature,state_0,padding_mask=padding_mask,offset=offset)
        reward = self.output_reward_func(feature)
        return next_state, reward

    @torch.jit.export
    def initial_inference(self, obs, action, option, padding_mask: Optional[torch.Tensor] = None,
    # def initial_inference(self, obs, padding_mask: Optional[torch.Tensor] = None,
                          dynamics: bool = False):
        state = self.representation(obs, action, option, padding_mask, dynamics)
        # state = self.representation(obs, padding_mask, dynamics)
        policy, value = self.prediction(state)

        reward = torch.log(
            (
                torch.zeros(1, self.n_reward_size)
                    .scatter(1, torch.tensor([[self.n_reward_size // 2]]).long(), 1.0)
                    .unsqueeze(1)
                    .repeat(obs.shape[0], obs.shape[1], 1)
                    .to(obs.device)
            )
        )

        return value, reward, policy, state

    @torch.jit.export
    def recurrent_inference(self, state, action, option, state_0, padding_mask: Optional[torch.Tensor] = None):
        '''
        state: (..., dim)
        action: (..., )
        option: (..., )
        '''
        # state_t_with_b6 = torch.cat([state,torch.roll(state,shifts=6,dims=1)],dim=-1)
        next_state, reward = self.dynamics(state, action, option, state_0=state_0, padding_mask=padding_mask)
        policy, value = self.prediction(next_state)
        return value, reward, policy, next_state


    def forward(self, obs, action_prev, option, padding_mask: Optional[torch.Tensor] = None, n_step=None, sample=False, gamma=0.9):
        '''
        obs: (B,L,E) float
        action: (B,L) long
        option: (B,L) long
        padding_mask: (B,L) (bool)
            - True/Non zero means ignore
            - bool type tensor
            - https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py

        Outputs:
        ---------------------
        state: (B,H,L,D)
        policy: (B,H,L,D)
        value: (B,H,L,D)
        reward: (B,H,L,D)
        aux: (B,H,L,D)
        glu: (B,H,L,D)

        H: lookahead
        '''
        if n_step is None:
            n_step = self.n_step

        state_0 = self.representation(obs, action_prev, option, padding_mask, dynamics=False)
        state_t = state_0
        action_t = torch.roll(action_prev, shifts=-1, dims=1)
        option_t = option
        policy_train_list = [] # Supervised Training
        state_list = []
        aux_list = []
        value_list = []
        reward_list = []
        glu_list = []
        for i in range(n_step):
            # The current setting is: the input is (s, a, s, a), ending with a, and predicting the next (s, a)
            # instead of starting with s, for example (s, a, s)

            # In fact
            policy_t, value_t = self.prediction(state_t) # Should use the previous state to generate actions
            state_t_next, reward_t = self.dynamics(state_t, action_t, option_t, padding_mask=padding_mask, state_0=state_0, offset=i)
            aux_t = self.output_aux_func(state_t_next)
            glu_t = self.output_glu_func(state_t_next)

            state_list += [state_t]
            aux_list += [aux_t]
            value_list += [value_t]
            reward_list += [reward_t]
            glu_list += [glu_t]
            # policy_train_list += [policy_t] # change

            state_t = state_t_next
            action_t = torch.roll(action_t, shifts=-1, dims=1)
            option_t = torch.roll(option_t, shifts=-1, dims=1)

        # Greedy path, imitating doctors
        state_t = state_0
        option_t = option
        policy_list = [] # For testing
        policy_action_logprob_list = [] # Reinforcement learning training
        policy_reward_list = []
        for i in range(n_step):
            policy_t, value_t = self.prediction(state_t)
            dist = torch.distributions.Categorical(logits=policy_t)
            sample = False # change
            if sample:
                action_t_pred = dist.sample()
                action_t_logprob = dist.log_prob(action_t_pred)
            else:
                action_t_pred = policy_t.argmax(dim=-1)
                action_t_logprob = dist.log_prob(action_t_pred)
            state_t_next_pred, reward_t_pred = self.dynamics(state_t, action_t_pred, option_t, padding_mask=padding_mask, state_0=state_0, offset=i)

            policy_train_list += [policy_t] # change
            policy_list += [policy_t]
            policy_action_logprob_list += [action_t_logprob]
            policy_reward_list += [reward_t_pred.detach()] # This is the conference, which is used for patient model learning

            state_t = state_t_next_pred
            option_t = torch.roll(option_t, shifts=-1, dims=1)

        # return calculation
        policy_return_list = []
        running_add = torch.zeros_like(policy_reward_list[0])
        for i in reversed(range(n_step)):
            running_add = running_add * gamma + policy_reward_list[i]
            policy_return_list += [running_add]
        policy_return_list = policy_return_list[::-1]

        state = torch.stack(state_list, dim=1)
        policy = torch.stack(policy_list, dim=1)
        value = torch.stack(value_list, dim=1)
        reward = torch.stack(reward_list, dim=1)
        aux = torch.stack(aux_list, dim=1)
        glu = torch.stack(glu_list, dim=1)

        policy_train = torch.stack(policy_train_list, dim=1)
        policy_action_logprob = torch.stack(policy_action_logprob_list, dim=1)
        policy_reward = torch.stack(policy_reward_list, dim=1)
        policy_return = torch.stack(policy_return_list, dim=1)

        outputs = {
            'state': state,
            'policy': policy,
            'value': value,
            'reward': reward,
            'aux': aux,
            'glu': glu,

            'policy_train': policy_train, # For Reinforcement learning
            'policy_action_logprob': policy_action_logprob, # For Reinforcement learning
            'policy_reward': policy_reward,
            'policy_return': policy_return,
        }

        return outputs


if __name__ == '__main__':
    n_input = 147
    n_hidden = 256
    nhead = 8
    nhid = 512
    nlayers = 3
    n_outputs = 4
    max_len = 128
    model = TransformerModel(n_outputs, n_input, n_hidden, nhead, nhid, nlayers,
                             max_len=max_len, dropout=0.5)
    # model = LSTM(n_outputs, n_input, n_hidden, nhead, nhid, nlayers, max_len=max_len, dropout=0.5)

    batch_size = 3

    x = torch.rand(batch_size, max_len, n_input)
    y = model(x)
    print(y.shape)

    n_action = 40
    n_option = 5
    n_reward_max = 21
    n_value_max = 22
    n_aux = 23

    model = TransformerPlanningModel(n_action, n_option, n_reward_max, n_value_max, n_aux,
                                     n_input, n_hidden, nhead, nhid, nlayers,
                                     max_len=max_len, dropout=0.0)
    batch_size = 3
    n_reward_size = n_reward_max * 2 + 1
    n_value_size = n_value_max * 2 + 1

    # (obs, action, option, padding)
    obs = torch.rand(batch_size, max_len, n_input)
    action = torch.torch.randint(3, n_action, [batch_size, max_len])
    option = torch.torch.randint(3, n_option, [batch_size, max_len])
    outputs = model(x, action, option)
    for k, v in outputs.items():
        print(k, v.shape)

    model_scripted = torch.jit.script(model)

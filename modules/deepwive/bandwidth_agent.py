import math
import ipdb
import random
from collections import deque
from compressai.layers import ResidualBlock, ResidualBlockWithStride, AttentionBlock

import torch
import torch.nn as nn
import torch.autograd
# import torch.utils.checkpoint as checkpoint

from modules.attention_feature import AFModule
from utils import split_list_by_val, perms_without_reps


class ActorCell(nn.Module):
    def __init__(self, c_in, c_feat, output_nc):
        super().__init__()

        self.conv_layers = nn.ModuleDict({
            'rbws1': ResidualBlockWithStride(
                in_ch=c_in,
                out_ch=c_feat,
                stride=2),

            'af1': AFModule(c_in=c_feat),

            'rb1': ResidualBlock(
                in_ch=c_feat,
                out_ch=c_feat),

            'rbws2': ResidualBlockWithStride(
                in_ch=c_feat,
                out_ch=c_feat,
                stride=2),

            'af2': AFModule(c_in=c_feat),

            'a1': AttentionBlock(c_feat),

            'rb2': ResidualBlock(
                in_ch=c_feat,
                out_ch=c_feat),

            'rbws3': ResidualBlockWithStride(
                in_ch=c_feat,
                out_ch=c_feat,
                stride=2),

            'af3': AFModule(c_in=c_feat),

            'rb3': ResidualBlock(
                in_ch=c_feat,
                out_ch=c_feat),

            'rbws4': ResidualBlockWithStride(
                in_ch=c_feat,
                out_ch=c_feat,
                stride=2),

            'af4': AFModule(c_in=c_feat),

            'a2': AttentionBlock(c_feat),

            'pool': nn.AdaptiveAvgPool2d(1),
        })

        self.fc = nn.Sequential(
            nn.Linear(c_feat+1, c_feat),
            nn.LeakyReLU(),
            nn.Linear(c_feat, c_feat),
            nn.LeakyReLU(),
            nn.Linear(c_feat, output_nc)
        )

        # self.fc_modules = [module for k, module in self.fc._modules.items()]

    def run_fn(self, module_key):
        def custom_forward(*inputs):
            if module_key[:2] == 'af':
                x, snr = inputs
                x = self.conv_layers[module_key](x, snr)
            else:
                x = inputs[0]
                x = self.conv_layers[module_key](x)
            return x
        return custom_forward

    def forward(self, x, snr):
        B, _, _, _ = x.size()
        # for key in self.conv_layers:
        #     if key[:2] == 'af':
        #         x = checkpoint.checkpoint(
        #             self.run_fn(key), x, snr)
        #     else:
        #         x = checkpoint.checkpoint(
        #             self.run_fn(key), x)

        for key in self.conv_layers:
            if key[:2] == 'af':
                x = self.conv_layers[key](x, snr)
            else:
                x = self.conv_layers[key](x)

        snr_context = snr.repeat_interleave(B//snr.size(0), dim=0)
        fc_input = torch.cat((x.view(x.size(0), -1), snr_context), dim=1)
        out = self.fc(fc_input)
        # out = checkpoint.checkpoint_sequential(self.fc_modules, len(self.fc_modules),
        #                                        fc_input.requires_grad_())
        return out


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state):
        batch_size = action.size(0)
        (state, snr) = state
        (next_state, next_snr) = next_state

        state = torch.chunk(state, chunks=batch_size, dim=0)
        action = torch.chunk(action, chunks=batch_size, dim=0)
        reward = torch.chunk(reward, chunks=batch_size, dim=0)
        next_state = torch.chunk(next_state, chunks=batch_size, dim=0)

        for batch_idx in range(batch_size):
            experience = ((state[batch_idx].detach().cpu(), snr.detach().cpu()),
                          action[batch_idx].detach().cpu(),
                          reward[batch_idx].detach().cpu(),
                          (next_state[batch_idx].detach().cpu(), next_snr.detach().cpu())
                          )
            self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        snr_batch = []
        next_state_batch = []
        next_snr_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state = experience
            (state, snr) = state
            (next_state, next_snr) = next_state

            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            snr_batch.append(snr)
            next_state_batch.append(next_state)
            next_snr_batch.append(next_snr)

        state_batch = (torch.cat(state_batch, dim=0), torch.cat(snr_batch, dim=0))
        action_batch = torch.cat(action_batch, dim=0)
        reward_batch = torch.cat(reward_batch, dim=0)
        next_state_batch = (torch.cat(next_state_batch, dim=0), torch.cat(next_snr_batch, dim=0))
        return (state_batch, action_batch, reward_batch, next_state_batch)

    def __len__(self):
        return len(self.buffer)


class BWAllocator(nn.Module):
    def __init__(self, gop_size, n_chunks, actor_feat,
                 batch_size, gamma=0.99, tau=0.05,
                 max_memory_size=300, device='cpu'):
        super().__init__()
        self.num_actions = (math.factorial(n_chunks + gop_size - 1)
                            // (math.factorial(gop_size - 1)
                                * math.factorial(n_chunks)))
        self.n_chunks = n_chunks
        self.gop_size = gop_size
        self.device = device

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.eps_start = 0.9
        self.eps_end = 0.10
        self.eps_decay = 400
        self.steps_done = 0

        self.action_set = self._get_action_set(n_chunks, gop_size).to(device)

        self.actor = ActorCell(c_in=21*(gop_size - 1)+6,
                               c_feat=actor_feat,
                               output_nc=self.num_actions)

        self.memory = Memory(max_memory_size)

    def _get_action_set(self, n_chunks, gop_size):
        action_set = [1] * n_chunks + [0] * (gop_size - 1)
        action_set = perms_without_reps(action_set)
        action_set = [split_list_by_val(action, 0) for action in action_set]
        action_set = [[sum(bw) for bw in action] for action in action_set]
        assert len(action_set) == self.num_actions
        print('Action set size: {}'.format(len(action_set)))
        return torch.tensor(action_set)

    def push_experience(self, experience):
        self.memory.push(
            experience['state'],
            experience['action'],
            experience['reward'],
            experience['next_state']
        )

    def get_action(self, state, mode):
        (state, snr) = state

        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1. * self.steps_done / self.eps_decay)

        if mode == 'train':
            self.steps_done += 1

        if random.random() < eps_threshold and mode == 'train':
            action_idx = torch.randint(0, self.num_actions, (state.size(0),), device=self.device)
            action_sample = torch.index_select(self.action_set, 0, action_idx)
        else:
            with torch.no_grad():
                policy = self.actor(state, snr)
                action_idx = torch.argmax(policy, dim=1).view(-1)
                action_sample = torch.index_select(self.action_set, 0, action_idx)

        return action_sample, {'policy': action_idx,
                               'eps_threshold': eps_threshold}

    def get_loss(self):
        if len(self.memory) < self.batch_size * 10:
            return False, torch.ones(1)
        else:
            (states, actions, rewards, next_states) = self.memory.sample(self.batch_size)
            states, snr = states
            next_states, next_snr = next_states

            states = states.to(self.device)
            snr = snr.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            next_snr = next_snr.to(self.device)

            actor_Q = self.actor(states, snr).gather(1, actions.unsqueeze(1))
            next_action_Q = self.actor(next_states, next_snr).max(1)[0]
            Qprime = (rewards + self.gamma * next_action_Q).unsqueeze(1)
            loss = (actor_Q - Qprime).pow(2).mean()
            return True, loss

    def __str__(self):
        return f'BWAllocator({self.gop_size},{self.n_chunks})'

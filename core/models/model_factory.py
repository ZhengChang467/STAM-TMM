import os
import torch
import torch.nn as nn
from torch.optim import Adam
from core.models import STAM

import torch.optim.lr_scheduler as lr_scheduler

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        # self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = configs.num_layers
        networks_map = {
            'stam': STAM.RNN,
        }
        num_hidden = []
        for i in range(configs.num_layers):
            num_hidden.append(configs.num_hidden)
        self.num_hidden = num_hidden
        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        print("Network state:")
        for param_tensor in self.network.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
            print(param_tensor, '\t', self.network.state_dict()[param_tensor].size())

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=configs.lr_decay)
        self.MSE_criterion = nn.MSELoss()
        self.L1_loss = nn.L1Loss()

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt' + '-' + str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])

    def train(self, frames, mask, itr):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()
        next_frames, _ = self.network(frames_tensor, mask_tensor)
        loss_l1 = self.L1_loss(next_frames, frames_tensor[:, 1:])
        loss_l2 = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        # loss = loss_l1 + loss_l2
        loss_l2.backward()
        self.optimizer.step()
        if itr >= self.configs.sampling_stop_iter and itr % self.configs.delay_interval == 0:
            self.scheduler.step()
            print('Lr decay to:%.8f', self.optimizer.param_groups[0]['lr'])
        return next_frames, loss_l1.detach().cpu().numpy(), loss_l2.detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames, gates_visual = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy(), gates_visual

import torch
import torch.nn as nn
from core.layers.STAMCell import STAMCell
import math


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()
        # print(configs.srcnn_tf)
        self.configs = configs
        self.frame_channel = configs.patch_size * \
            configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.time = 2
        cell_list = []

        width = configs.img_width // configs.patch_size // configs.sr_size
        height = configs.img_height // configs.patch_size // configs.sr_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                STAMCell(in_channel, num_hidden[i], height, width, self.time, configs.filter_size,
                         configs.stride, self.frame_channel)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Encoder-Decoder Module, if it is necessary
        self.conv_encoder = nn.Sequential()
        n = int(math.log2(configs.sr_size))
        for i in range(n):
            self.conv_encoder.add_module(name='encoder{0}'.format(i),
                                         module=nn.Conv3d(in_channels=self.frame_channel,
                                                          out_channels=self.frame_channel,
                                                          stride=(1, 2, 2),
                                                          padding=(1, 1, 1),
                                                          kernel_size=(3, 3, 3)
                                                          ))
            self.conv_encoder.add_module(name='encoder_relu{0}'.format(i),
                                         module=nn.ReLU())
        self.conv_decoder = nn.Sequential()
        for i in range(n - 1):
            self.conv_decoder.add_module(name='decoder{0}'.format(i),
                                         module=nn.ConvTranspose2d(in_channels=self.frame_channel,
                                                                   out_channels=self.frame_channel,
                                                                   stride=(
                                                                       2, 2),
                                                                   padding=(
                                                                       1, 1),
                                                                   kernel_size=(
                                                                       3, 3),
                                                                   output_padding=(
                                                                       1, 1)
                                                                   ))

            self.conv_decoder.add_module(name='decoder_relu{0}'.format(i),
                                         module=nn.ReLU())
        if n > 0:
            self.conv_decoder.add_module(name='decoder' + str(n),
                                         module=nn.ConvTranspose2d(in_channels=self.frame_channel,
                                                                   out_channels=self.frame_channel,
                                                                   stride=(
                                                                       2, 2),
                                                                   padding=(
                                                                       1, 1),
                                                                   kernel_size=(
                                                                       3, 3),
                                                                   output_padding=(
                                                                       1, 1)
                                                                   ))
        self.conv_last = nn.Conv3d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=(self.time, 1, 1), stride=[self.time, 1, 1], padding=0)

    def forward(self, frames, mask_true):
        # [batch, length, time, height, width, channel] -> [batch, length, channel, time, height, width]
        # frames = frames.permute(0, 1, 5, 2, 3, 4).contiguous()
        gates_visual = []
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch_size = frames.shape[0]
        # time = frames.shape[3]
        height = frames.shape[3] // self.configs.sr_size
        width = frames.shape[4] // self.configs.sr_size
        frame_channels = frames.shape[2]
        # print(frames.shape, mask_true.shape)
        next_frames = []
        h_t = []
        c_t = []
        c_net = []
        time = 2
        time_stride = 1
        for i in range(self.num_layers):
            gates_visual.append([])
            zeros = torch.zeros([batch_size, self.num_hidden[i], time, height, width]).to(
                self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
        c_net.append(c_t)
        memory = torch.zeros([batch_size, self.num_hidden[0], time, height, width]).to(
            self.configs.device)
        x_gen = self.conv_last(h_t[self.num_layers - 1])
        m_net = []
        m_net.append(memory)
        input_list = []
        for time_step in range(time - 1):
            input_list.append(
                torch.zeros([batch_size, frame_channels, height*self.configs.sr_size, width*self.configs.sr_size]).to(self.configs.device))

        for t in range(self.configs.total_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                time_diff = t - self.configs.input_length
                net = mask_true[:, time_diff] * frames[:, t] + \
                    (1 - mask_true[:, time_diff]) * x_gen
            input_list.append(net)

            if t % (time - time_stride) == 0:
                input_frm = torch.stack(input_list[t:])
                input_frm = input_frm.permute(1, 2, 0, 3, 4).contiguous()
                frames_feature = self.conv_encoder(input_frm)

                # c_att
                c_att = []
                m_att = []
                if len(c_net) <= self.configs.tau:
                    for idx in range(len(c_net)):
                        c_att.append(c_net[idx][0])
                else:
                    for idx in range(len(c_net) - self.configs.tau, len(c_net)):
                        c_att.append(c_net[idx][0])
                # m_att.append(m_t)
                if len(m_net) <= self.configs.tau:
                    for idx in range(len(m_net)):
                        m_att.append(m_net[idx])
                else:
                    for idx in range(len(m_net) - self.configs.tau, len(m_net)):
                        m_att.append(m_net[idx])

                h_t[0], c_t[0], memory, gates = self.cell_list[0](
                    frames_feature, h_t[0], c_t[0], memory, c_att, m_att)
                gates_visual[0].append(gates)
                m_net.append(memory)
                for i in range(1, self.num_layers):
                    # c_att
                    c_att = []
                    m_att = []
                    if len(c_net) <= self.configs.tau:
                        for idx in range(len(c_net)):
                            c_att.append(c_net[idx][i])
                    else:
                        for idx in range(len(c_net) - self.configs.tau, len(c_net)):
                            c_att.append(c_net[idx][i])

                    if len(m_net) <= self.configs.tau:
                        for idx in range(len(m_net)):
                            m_att.append(m_net[idx])
                    else:
                        for idx in range(len(m_net) - self.configs.tau, len(m_net)):
                            m_att.append(m_net[idx])

                    h_t[i], c_t[i], memory, gates = self.cell_list[i](
                        h_t[i - 1], h_t[i], c_t[i], memory, c_att, m_att)
                    gates_visual[i].append(gates)
                    m_net.append(memory)
                c_net.append(c_t)
                frame_out = self.conv_last(
                    h_t[self.num_layers - 1]).squeeze(dim=2)
                x_gen = self.conv_decoder(frame_out)
                # print(x_gen.shape)
                # next_frames.append(x_gen)
                next_frames.append(x_gen)

        # [length, batch, channel, time, height, width] -> [batch, length, height, time, width, channel]
        # next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 5, 2).contiguous()
        # next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4, 5).contiguous()
        next_frames = torch.stack(next_frames, dim=0).permute(
            1, 0, 2, 3, 4).contiguous()

        # print('out:',next_frames.shape)
        return next_frames, gates_visual

# import argparse
#
# parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')
#
# # ['cuda', 'cpu:0']
# parser.add_argument('--device', type=str, default='cuda:0')
#
# # data
# parser.add_argument('--dataset_name', type=str, default='mnist')
# parser.add_argument('--train_data_paths', type=str, default='./data/moving-mnist-example/moving-mnist-train.npz')
# parser.add_argument('--valid_data_paths', type=str, default='./data/moving-mnist-example/moving-mnist-valid.npz')
# parser.add_argument('--input_length', type=int, default=5)
# parser.add_argument('--total_length', type=int, default=14)
# parser.add_argument('--img_width', type=int, default=64)
# parser.add_argument('--img_channel', type=int, default=3)
# parser.add_argument('--img_time', type=int, default=3)
#
# parser.add_argument('--reverse_input', type=bool, default=True)
# parser.add_argument('--patch_size', type=int, default=4)
# parser.add_argument('--sr', type=bool, default=True)
# parser.add_argument('--tau', type=int, default=5)
#
# parser.add_argument('--sr_size', type=int, default=4)
#
#
# # model
# parser.add_argument('--model_name', type=str, default='predrnn')
# parser.add_argument('--num_hidden', type=str, default='32,32,32,32')
# parser.add_argument('--filter_size', type=int, default=3)
# parser.add_argument('--stride', type=int, default=1)
# parser.add_argument('--layer_norm', type=bool, default=True)
#
# # training
# parser.add_argument('--is_training', type=bool, default=True)
# parser.add_argument('--lr', type=float, default=0.001)
# parser.add_argument('--batch_size', type=int, default=4)
# parser.add_argument('--max_iterations', type=int, default=80000)
# parser.add_argument('--display_interval', type=int, default=1)
# parser.add_argument('--test_interval', type=int, default=10)
# parser.add_argument('--snapshot_interval', type=int, default=10)
# parser.add_argument('--num_save_samples', type=int, default=10)
# parser.add_argument('--n_gpu', type=int, default=1)
# parser.add_argument('--pretrained_model', type=str, default='')
# parser.add_argument('--save_dir', type=str, default='checkpoints/mnist_predrnn')
# parser.add_argument('--gen_frm_dir', type=str, default='results/mnist_predrnn')
#
# # scheduled sampling
# parser.add_argument('--scheduled_sampling', type=bool, default=True)
# parser.add_argument('--sampling_stop_iter', type=int, default=50000)
# parser.add_argument('--sampling_start_value', type=float, default=1.0)
# parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)
# parser.add_argument('--srcnn_f1', type=int, default=32)
# parser.add_argument('--srcnn_f2', type=int, default=32)
#
# args = parser.parse_args()
# args.tied = True
# #
# ST_LSTM = RNN(num_layers=4,
#               num_hidden=[32, 32, 32, 32],
#               configs=args).cuda()
# frames = torch.zeros((4, 14, 48, 3, 16, 16)).cuda()
# mask_true = torch.zeros((args.batch_size,
#                          args.total_length - args.input_length - 1,
#                          args.img_time,
#                          args.img_width // args.patch_size,
#                          args.img_width // args.patch_size,
#                          args.patch_size ** 2 * args.img_channel)).cuda()
#
# output = ST_LSTM(frames=frames, mask_true=mask_true)
# print(output.shape)

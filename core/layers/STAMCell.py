import torch
import torch.nn as nn


# from core.utils.BiCubic_interpolation import BiCubic_interpolation


class STAMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, time, filter_size, stride, frame_channel):
        super(STAMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = (filter_size[0] // 2,
                        filter_size[1] // 2, filter_size[2] // 2)
        self._forget_bias = 1.0
        self.frame_channel = frame_channel
        # self.c_srcnn = c_srcnn
        # self.m_srcnn = m_srcnn
        # self.o_srcnn = o_srcnn
        self.conv_x = nn.Sequential(
            nn.Conv3d(in_channel, num_hidden * 7, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 7, time, height, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv3d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, time, height, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv3d(num_hidden, num_hidden * 3, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 3, time, height, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv3d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, time, height, width])
        )
        self.conv_last = nn.Conv3d(
            num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)
        # self.conv_residual = nn.Conv3d(in_channel, num_hidden, kernel_size=filter_size,
        #                                stride=stride, padding=self.padding)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_t, h_t, c_t, m_t, c_att, m_att):
        gates = []
        # print(x_t.shape)
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(
            x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)
        # Temporal module
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        gates.append(i_t)
        gates.append(f_t)
        gates.append(g_t)

        # EAIFG_temporal
        c_att_merge = torch.stack(c_att, dim=1)
        f_t_merge = f_t.unsqueeze(dim=1)
        cf = c_att_merge * f_t_merge

        # cf = cf.sum(dim=[2, 3, 4, 5])
        # cf = self.softmax(cf).view(*cf.shape, 1, 1, 1, 1)
        # eafig = cf * c_att_merge
        # eafig = eafig.sum(dim=1)
        # c_new = i_t * g_t + self.conv_cf(torch.cat([c_t, eafig], dim=1))

        cf = cf.view(*cf.shape[0:2], -1)
        cf = self.softmax(cf)
        # print(cf[0,0,:].sum())
        c_att_merge = c_att_merge.view(*c_att_merge.shape[0:2], -1)
        eafig_c = cf * c_att_merge
        eafig_c = eafig_c.sum(dim=1)
        eafig_c = eafig_c.view(f_t.shape)
        c_new = i_t * g_t + f_t * eafig_c
        # c_new = i_t * g_t + f_t * self.conv_cf(torch.cat([c_t, eafig_c], dim=1))
        # c_new = f_t * c_t + i_t * g_t
        # Spatial Module
        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        gates.append(i_t_prime)
        gates.append(f_t_prime)
        gates.append(g_t_prime)

        # EIFAG_spatial
        m_att_merge = torch.stack(m_att, dim=1)
        f_t_prime_merge = f_t_prime.unsqueeze(dim=1)
        mf = m_att_merge * f_t_prime_merge

        mf = mf.view(*mf.shape[0:2], -1)
        mf = self.softmax(mf)
        # print(cf[0,0,:].sum())
        m_att_merge = m_att_merge.view(*m_att_merge.shape[0:2], -1)
        eafig_m = mf * m_att_merge
        eafig_m = eafig_m.sum(dim=1)
        eafig_m = eafig_m.view(f_t_prime.shape)
        m_new = i_t_prime * g_t_prime + f_t_prime * eafig_m
        # m_new = i_t_prime * g_t_prime + f_t_prime * self.conv_mf(torch.cat([m_t, eafig_m], dim=1))
        # m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        gates.append(o_t)
        h_new = o_t * torch.tanh(self.conv_last(mem))
        # x_residual = self.conv_residual(x_t)
        # print(h_new.shape, x_t.shape)
        return h_new, c_new, m_new, gates

# model = EAST3DLSTMCell(in_channel=16,
#                        num_hidden=64,
#                        width=16,
#                        time=3,
#                        filter_size=5,
#                        stride=1,
#                        layer_norm=True).cuda()
# c_att = []
# c_att.append(torch.rand((64, 64, 3, 16, 16)).cuda())
# c_att.append(torch.rand((64, 64, 3, 16, 16)).cuda())
# c_att.append(torch.rand((64, 64, 3, 16, 16)).cuda())
# h_t = torch.rand((64, 64, 3, 16, 16)).cuda()
# c_t = torch.rand((64, 64, 3, 16, 16)).cuda()
# m_t = torch.rand((64, 64, 3, 16, 16)).cuda()
#
# input = torch.rand((64, 16, 3, 16, 16)).cuda()
# h, c, m = model(input, h_t, c_t, m_t, c_att)
# print(h_t.shape)

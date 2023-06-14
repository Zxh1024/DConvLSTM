import torch
import torch.nn as nn
from core.layers.DeformConv2D import DeformConv2d

class DeformableConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(DeformableConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            DeformConv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            # LayerNorm是对每个样本的所有特征做归一化
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_h = nn.Sequential(
            DeformConv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.Wci = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()
        self.Wcf = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()
        self.Wcg = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()
        self.Wco = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()
        """
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, width, width])
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)
        """

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t).cuda()
        h_concat = self.conv_h(h_t).cuda()

        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)

        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h + self.Wci * c_t)

        # 为了减少训练过程中遗忘的规模
        # 因为sigmoid(x)的特点是x越大越接近1，这样始终让里面的值有一个固定偏置
        # 这样会使得sigmoid(f_x + f_h + self.Wcf * c_t + self._forget_bias)不会太小，能记住之前更多的信息

        f_t = torch.sigmoid(f_x + f_h + self.Wcf * c_t + self._forget_bias)

        # g_t = torch.tanh(g_x + g_h + self.Wcg * c_t)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        o_t = torch.sigmoid(o_x + o_h + self.Wco * c_t)

        h_new = o_t * torch.tanh(c_new)

        return h_new, c_new
import torch
import torch.nn as nn

class ConvLSTM_SAC_Cell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(ConvLSTM_SAC_Cell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv = nn.Sequential(
            nn.Conv2d(2, 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([3, width, width])
        )
        self.Wci = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()
        self.Wcf = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()
        self.Wcg = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()
        self.Wco = nn.Parameter(torch.zeros(1, num_hidden, width, width)).cuda()

    def forward(self, x_t, h_t, c_t, m, m_t):
        x_concat = self.conv_x(x_t).cuda()
        h_concat = self.conv_h(h_t).cuda()

        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)

        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h + self.Wci * c_t)

        f_t = torch.sigmoid(f_x + f_h + self.Wcf * c_t + self._forget_bias)

        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        o_t = torch.sigmoid(o_x + o_h + self.Wco * c_t)

        h_new = o_t * torch.tanh(c_new)

        combined = self.conv(torch.cat([m, m_t], dim=1))
        mo, mg, mi = torch.split(combined, 1, dim=1)

        mi = torch.sigmoid(mi)
        m_new = (1 - mi) * m + mi * torch.tanh(mg)

        h_new = (1 - torch.sigmoid(m_new)) * x_t + torch.sigmoid(m_new) * h_new

        return h_new, c_new, m_new

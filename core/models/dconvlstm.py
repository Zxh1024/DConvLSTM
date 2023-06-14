import numpy as np
import torch
import torch.nn as nn
from core.layers.ConvLSTMCell import ConvLSTMCell
from core.layers.DeformableConvLSTMCell import DeformableConvLSTMCell

class DConvLSTM(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(DConvLSTM, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.index = 0
        cell_list = []
        cell_list1 = []

        width = configs.img_width // configs.patch_size

        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = 1 if i == 0 else num_hidden[i - 1]
            if i % 2 == 0:
                cell_list.append(
                    DeformableConvLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                           configs.stride, configs.layer_norm)
                )
                cell_list1.append(
                    ConvLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                 configs.stride, configs.layer_norm)
                )
            else:
                cell_list.append(
                    ConvLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                 configs.stride, configs.layer_norm)
                )
                cell_list1.append(
                    DeformableConvLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                           configs.stride, configs.layer_norm)
                )

        self.cell_list = nn.ModuleList(cell_list)
        self.cell_list1 = nn.ModuleList(cell_list1)

        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true):

        # [batch, seq, height, width, channel] -> [batch, seq, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen
            if self.index == 0:

                h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])
                for i in range(1, self.num_layers):
                    h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

                x_gen = self.conv_last(h_t[self.num_layers - 1])
                next_frames.append(x_gen)
                self.index = 1

            else:

                h_t[0], c_t[0] = self.cell_list1[0](net, h_t[0], c_t[0])
                for i in range(1, self.num_layers):
                    h_t[i], c_t[i] = self.cell_list1[i](h_t[i - 1], h_t[i], c_t[i])

                x_gen = self.conv_last(h_t[self.num_layers - 1])
                next_frames.append(x_gen)
                self.index = 0

        # [seq, batch, channel, height, width] -> [batch, seq, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        return next_frames


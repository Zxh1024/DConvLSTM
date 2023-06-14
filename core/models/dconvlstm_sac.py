import numpy as np
import torch
import torch.nn as nn
from core.layers.ConvLSTM_SAC_Cell import ConvLSTM_SAC_Cell
from core.layers.DeformableConvLSTM_SAC_Cell import DeformableConvLSTM_SAC_Cell


class DConvLSTM_SAC(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(DConvLSTM_SAC, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.width = configs.img_width // configs.patch_size
        self.index = 0
        cell_list = []
        cell_list1 = []

        width = configs.img_width // configs.patch_size

        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = 1 if i == 0 else num_hidden[i - 1]
            if i % 2 == 0:
                cell_list.append(
                    DeformableConvLSTM_SAC_Cell(in_channel, num_hidden[i], width, configs.filter_size,
                                           configs.stride, configs.layer_norm)
                )
                cell_list1.append(
                    ConvLSTM_SAC_Cell(in_channel, num_hidden[i], width, configs.filter_size,
                                 configs.stride, configs.layer_norm)
                )
            else:
                cell_list.append(
                    ConvLSTM_SAC_Cell(in_channel, num_hidden[i], width, configs.filter_size,
                                 configs.stride, configs.layer_norm)
                )
                cell_list1.append(
                    DeformableConvLSTM_SAC_Cell(in_channel, num_hidden[i], width, configs.filter_size,
                                           configs.stride, configs.layer_norm)
                )

        self.cell_list = nn.ModuleList(cell_list)
        self.cell_list1 = nn.ModuleList(cell_list1)

        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true):

        moran = frames_tensor[:, :, :, :, 1].detach().cpu().numpy()
        moran = np.array(moran).reshape(self.configs.batch_size, self.configs.total_length, self.configs.img_width, self.configs.img_width, self.frame_channel)
        moran = torch.FloatTensor(moran).to(self.configs.device).permute(0, 1, 4, 2, 3).contiguous()

        tp = frames_tensor[:, :, :, :, 0].detach().cpu().numpy()
        tp = np.array(tp).reshape(self.configs.batch_size, self.configs.total_length, self.configs.img_width, self.configs.img_width, self.frame_channel)
        frames = torch.FloatTensor(tp).to(self.configs.device).permute(0, 1, 4, 2, 3).contiguous()

        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        m_t = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            zeros1 = torch.zeros([batch, 1, height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            m_t.append(zeros1)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                net = frames[:, t]
                m = moran[:, t]
            else:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                       (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            if self.index == 0:

                h_t[0], c_t[0], m_t[0] = self.cell_list1[0](net, h_t[0], c_t[0], m, m_t[0])
                for i in range(1, self.num_layers):
                    h_t[i], c_t[i], m_t[i] = self.cell_list1[i](h_t[i - 1], h_t[i], c_t[i], m_t[i - 1], m_t[i])

                x_gen = self.conv_last(h_t[self.num_layers - 1])
                next_frames.append(x_gen)
                self.index = 1

            else:

                h_t[0], c_t[0], m_t[0] = self.cell_list[0](net, h_t[0], c_t[0], m, m_t[0])
                for i in range(1, self.num_layers):
                    h_t[i], c_t[i], m_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], m_t[i - 1], m_t[i])

                x_gen = self.conv_last(h_t[self.num_layers - 1])
                next_frames.append(x_gen)
                self.index = 0

        # [seq, batch, channel, height, width] -> [batch, seq, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        return next_frames

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, padding=d * (kernel_size - 1) // 2, dilation=d),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, padding=1, dilation=1)
            ) for d in dilation
        ])

    def forward(self, x):
        for conv in self.convs:
            x = x + conv(x)
        return x

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.upsample_rates = config["upsample_rates"]
        self.upsample_kernel_sizes = config["upsample_kernel_sizes"]
        self.upsample_initial_channel = config["upsample_initial_channel"]
        self.resblock_kernel_sizes = config["resblock_kernel_sizes"]
        self.resblock_dilation_sizes = config["resblock_dilation_sizes"]

        self.conv_pre = nn.Conv1d(config["num_mels"], self.upsample_initial_channel, 7, padding=3)

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        channels = self.upsample_initial_channel

        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(channels, channels // 2, k, stride=u, padding=(k - u) // 2)
            )
            channels //= 2
            for d in self.resblock_dilation_sizes:
                self.resblocks.append(ResBlock(channels, self.resblock_kernel_sizes[i], d))

        self.conv_post = nn.Conv1d(channels, 1, 7, padding=3)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(len(self.ups)):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = 0
            for j in range(len(self.resblock_dilation_sizes)):
                xs = xs + self.resblocks[i * len(self.resblock_dilation_sizes) + j](x)
            x = xs / len(self.resblock_dilation_sizes)
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)
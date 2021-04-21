import torch
from torch import nn

# DCGAN ------------------------------------------------------------------------------
class ConvBatchRelu(nn.Module):
    def __init__(self, in_chennels=3, out_channels=128, kernel_size=3, stride=1, padding=1):
        super(ConvBatchRelu, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_chennels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.25)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class UpsampleConvBatchRelu(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1):
        super(UpsampleConvBatchRelu, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.block = nn.Sequential(
            ConvBatchRelu(3, out_channels=64, kernel_size=3, stride=2, padding=1),
            ConvBatchRelu(64, out_channels=64, kernel_size=3, stride=2, padding=1),
            ConvBatchRelu(64, out_channels=128, kernel_size=3, stride=2, padding=1),
            ConvBatchRelu(128, out_channels=128, kernel_size=3, stride=2, padding=1),
        )

        self.last = nn.Linear(8192, 1)
        # self.last = nn.Linear(32768, 1)


    def forward(self, x):
        x = self.block(x)
        x = x.view(x.size()[0], -1)
        x = self.last(x)

        return x


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.first = nn.Sequential(
            nn.Linear(z_dim, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True)
        )

        self.block = nn.Sequential(
            UpsampleConvBatchRelu(128, 128),
            UpsampleConvBatchRelu(128, 64),
            UpsampleConvBatchRelu(64, 64)
        )

        self.last = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.first(x)
        x = torch.reshape(x, (-1, 128, 8, 8))
        x = self.block(x)
        x = self.last(x)

        return x


if __name__ == '__main__':
    z = torch.randn(4, 200)

    G = Generator(z_dim=200)
    D = Discriminator()
    out = G(z)
    print(out.size())

    out = D(out)
    print(out.size())
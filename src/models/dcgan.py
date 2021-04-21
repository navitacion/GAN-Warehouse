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
    def __init__(self, img_size=128):
        super(Discriminator, self).__init__()

        self.block = nn.Sequential(
            ConvBatchRelu(3, img_size // 2, kernel_size=3, stride=2, padding=1),
            ConvBatchRelu(img_size // 2, img_size // 2, kernel_size=3, stride=2, padding=1),
            ConvBatchRelu(img_size // 2, img_size, kernel_size=3, stride=2, padding=1),
            ConvBatchRelu(img_size, img_size, kernel_size=3, stride=2, padding=1),
        )

        after_shape = img_size // (2 ** 4)

        self.last = nn.Linear(img_size * after_shape * after_shape, 1)


    def forward(self, x):
        x = self.block(x)
        x = x.view(x.size()[0], -1)
        x = self.last(x)

        return x


class Generator(nn.Module):
    def __init__(self, z_dim=100, img_size=128):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.after_shape = img_size // (2 ** 4)

        self.first = nn.Sequential(
            nn.Linear(z_dim, int(self.img_size * self.after_shape * self.after_shape)),
            nn.BatchNorm1d(int(self.img_size * self.after_shape * self.after_shape)),
            nn.ReLU(inplace=True)
        )

        self.block = nn.Sequential(
            UpsampleConvBatchRelu(self.img_size, self.img_size),
            UpsampleConvBatchRelu(self.img_size, self.img_size // 2),
            UpsampleConvBatchRelu(self.img_size // 2, self.img_size // 2)
        )

        self.last = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.img_size // 2, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.first(x)
        x = torch.reshape(x, (-1, self.img_size, self.after_shape, self.after_shape))
        x = self.block(x)
        x = self.last(x)

        return x


if __name__ == '__main__':
    z = torch.randn(4, 200)

    G = Generator(z_dim=200, img_size=256)
    D = Discriminator(img_size=256)
    out = G(z)
    print(out.size())

    out = D(out)
    print(out.size())
import torch
from torch import nn

# WGAN-GP  ------------------------------------------------------------------------------
class GP_ConvBatchRelu(nn.Module):
    def __init__(self, in_chennels=3, out_channels=128, kernel_size=3, stride=1, padding=1):
        super(GP_ConvBatchRelu, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_chennels, out_channels, kernel_size, stride, padding),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.25)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class GP_UpsampleConvBatchRelu(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1):
        super(GP_UpsampleConvBatchRelu, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

# DiscriminatorとGeneratorのネットワークは左右対称にならないとうまく行かない
# レイヤ数とin/out_channelsも合わせる必要あるかも
class WGAN_GP_Discriminator(nn.Module):
    def __init__(self, img_size=128):
        super(WGAN_GP_Discriminator, self).__init__()

        self.block = nn.Sequential(
            GP_ConvBatchRelu(3, img_size // 2, kernel_size=3, stride=2, padding=1),
            GP_ConvBatchRelu(img_size // 2, img_size // 2, kernel_size=3, stride=2, padding=1),
            GP_ConvBatchRelu(img_size // 2, img_size, kernel_size=3, stride=2, padding=1),
            GP_ConvBatchRelu(img_size, img_size, kernel_size=3, stride=2, padding=1)
        )

        after_shape = img_size // (2 ** 4)

        self.last = nn.Linear(img_size * after_shape * after_shape, 1)


    def forward(self, x):
        x = self.block(x)
        x = x.view(x.size()[0], -1)
        x = self.last(x)

        return x


class WGAN_GP_Generator(nn.Module):
    def __init__(self, z_dim=100, img_size=128):
        super(WGAN_GP_Generator, self).__init__()
        self.img_size = img_size
        self.after_shape = img_size // (2 ** 4)

        self.first = nn.Sequential(
            nn.Linear(z_dim, int(self.img_size * self.after_shape * self.after_shape)),
            nn.BatchNorm1d(int(self.img_size * self.after_shape * self.after_shape)),
            nn.ReLU(inplace=True)
        )

        self.block = nn.Sequential(
            GP_UpsampleConvBatchRelu(self.img_size, self.img_size),
            GP_UpsampleConvBatchRelu(self.img_size, self.img_size // 2),
            GP_UpsampleConvBatchRelu(self.img_size // 2, self.img_size // 2)
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
    z = torch.randn(4, 50)

    G = WGAN_GP_Generator(z_dim=50, img_size=512)
    D = WGAN_GP_Discriminator(img_size=512)
    out = G(z)
    print(out.size())

    out = D(out)
    print(out.size())
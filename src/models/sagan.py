import gc
import torch
from torch import nn

# Self-Attention GAN  ------------------------------------------------------------------------------
class Self_Attention(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        proj_query = self.query_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3])
        proj_key = self.key_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3])
        proj_value = self.value_conv(x).view(x.shape[0], -1, x.shape[2] * x.shape[3])

        S = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        del proj_query, proj_key

        o = torch.bmm(proj_value, self.softmax(S)).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        x = x + self.gamma * o
        del proj_value, S

        return x


class SA_ConvBatchRelu(nn.Module):
    def __init__(self, in_chennels=3, out_channels=128, kernel_size=4, stride=1, padding=1):
        super(SA_ConvBatchRelu, self).__init__()
        self.block = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_chennels, out_channels, kernel_size, stride, padding)),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class SA_UpsampleConvBatchRelu(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, kernel_size=4, stride=1, padding=0):
        super(SA_UpsampleConvBatchRelu, self).__init__()
        self.block = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class SAGAN_Discriminator(nn.Module):
    def __init__(self, filter=128):
        super(SAGAN_Discriminator, self).__init__()

        # self-attentionのin_dimが小さいほど処理が重くなる
        self.block = nn.Sequential(
            SA_ConvBatchRelu(3, filter * 1, kernel_size=4, stride=2, padding=1),
            SA_ConvBatchRelu(filter * 1, filter * 1, kernel_size=4, stride=2, padding=1),
            SA_ConvBatchRelu(filter * 1, filter * 2, kernel_size=4, stride=2, padding=1),
            SA_ConvBatchRelu(filter * 2, filter * 4, kernel_size=4, stride=2, padding=1),
            Self_Attention(in_dim=filter * 4),
            SA_ConvBatchRelu(filter * 4, filter * 4, kernel_size=4, stride=2, padding=1),
            SA_ConvBatchRelu(filter * 4, filter * 8, kernel_size=4, stride=2, padding=1),
            Self_Attention(in_dim=filter * 8),
        )

        self.last = nn.utils.spectral_norm(nn.Conv2d(filter * 8, 1, kernel_size=4, stride=1))


    def forward(self, x):
        x = self.block(x)
        x = self.last(x)

        return x


class SAGAN_Generator(nn.Module):
    def __init__(self, z_dim=100, filter=128):
        super(SAGAN_Generator, self).__init__()
        self.block = nn.Sequential(
            SA_UpsampleConvBatchRelu(z_dim, filter * 8),
            Self_Attention(in_dim=filter * 8),
            SA_UpsampleConvBatchRelu(filter * 8, filter * 4, stride=2, padding=1),
            SA_UpsampleConvBatchRelu(filter * 4, filter * 4, stride=2, padding=1),
            Self_Attention(in_dim=filter * 4),
            SA_UpsampleConvBatchRelu(filter * 4, filter * 2, stride=2, padding=1),
            SA_UpsampleConvBatchRelu(filter * 2, filter * 1, stride=2, padding=1),
            SA_UpsampleConvBatchRelu(filter * 1, filter * 1, stride=2, padding=1),
        )

        self.last = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(filter, 3, kernel_size=4, stride=2, padding=1)),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.block(x)
        x = self.last(x)

        return x


if __name__ == '__main__':
    z = torch.randn(4, 20, 1, 1)

    G = SAGAN_Generator(z_dim=20, filter=32)
    D = SAGAN_Discriminator(filter=32)
    out = G(z)
    print(out.size())
    out = D(out)
    print(out.size())
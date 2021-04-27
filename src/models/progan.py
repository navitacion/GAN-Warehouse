import gc

import numpy as np
import torch
from torch import nn

# Progressive GAN  ------------------------------------------------------------------------------

class P_Conv(nn.Module):
    def __init__(self, in_chennels=3, out_channels=128, kernel_size=4, stride=2, padding=1):
        super(P_Conv, self).__init__()
        self.block = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_chennels, out_channels, kernel_size, stride, padding)),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class P_ConvTranspose(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1):
        super(P_ConvTranspose, self).__init__()
        self.block = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Pro_Discriminator(nn.Module):
    def __init__(self, filter, max_img_size=1024):
        super(Pro_Discriminator, self).__init__()

        # 最初の層はz_dimが入力 出力はfilterで固定
        self.block = [P_Conv(3, filter)]

        # 最終的に出力したい画像サイズに応じてレイヤ数を調整する
        cnt = int(np.log2(max_img_size))
        for _ in range(cnt - 1):
            self.block.append(P_Conv(filter, filter))

        self.block = nn.ModuleList(self.block)

        # 生成画像はRGBチャンネルに合わせた3次元に
        self.last = nn.utils.spectral_norm(nn.Conv2d(in_channels=filter, out_channels=1, kernel_size=1))


    def forward(self, x, residual):
        for i, net in enumerate(self.block):
            x = net(x)
            # residualに指定した分だけ入力
            # 学習の過程で増やしていくように設計
            if i == residual - 1:
                break

        x = self.last(x)

        return x


class Pro_Generator(nn.Module):
    def __init__(self, z_dim, filter, max_img_size=1024):
        super(Pro_Generator, self).__init__()

        # 最初の層はz_dimが入力 出力はfilterで固定
        self.block = [P_ConvTranspose(z_dim, filter)]

        # 最終的に出力したい画像サイズに応じてレイヤ数を調整する
        cnt = int(np.log2(max_img_size))
        for _ in range(cnt - 1):
            self.block.append(P_ConvTranspose(filter, filter))

        self.block = nn.ModuleList(self.block)

        # 生成画像はRGBチャンネルに合わせた3次元に
        self.last = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=filter, out_channels=3, kernel_size=1)),
            nn.Tanh()
        )

    def forward(self, x, residual):
        for i, net in enumerate(self.block):
            x = net(x)
            # residualに指定した分だけ入力
            # 学習の過程で増やしていくように設計
            if i == residual - 1:
                break

        x = self.last(x)

        return x



if __name__ == '__main__':

    z = torch.randn(4, 20, 1, 1)

    net = Pro_Generator(z_dim=20, filter=32, max_img_size=1024)
    out = net(z, residual=2)
    print(out.size())


    # z = torch.randn(4, 3, 16, 16)

    net = Pro_Discriminator(filter=32, max_img_size=1024)
    out = net(out, residual=2)
    print(out.size())

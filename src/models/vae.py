import torch
from torch import nn

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


class ConvTBatchRelu(nn.Module):
    def __init__(self, in_chennels=3, out_channels=128, kernel_size=2, stride=2):
        super(ConvTBatchRelu, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_chennels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.25)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, z_dim=200):
        super(Encoder, self).__init__()

        self.block = nn.Sequential(
            ConvBatchRelu(3, 32, kernel_size=3, stride=2, padding=1),
            ConvBatchRelu(32, 64, kernel_size=3, stride=2, padding=1),
            ConvBatchRelu(64, 64, kernel_size=3, stride=2, padding=1),
            ConvBatchRelu(64, 64, kernel_size=3, stride=2, padding=1),
        )

        self.mu_layer = nn.Linear(4096, z_dim)
        self.log_var_layer = nn.Linear(4096, z_dim)

    def _reparameterization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x):
        b, c, h, w = x.size()
        x = self.block(x)
        x = x.view(b, -1)

        mu = self.mu_layer(x)
        log_var = self.log_var_layer(x)
        z = self._reparameterization_trick(mu, log_var)

        return z, mu, log_var


class Decoder(nn.Module):
    def __init__(self, z_dim=200):
        super(Decoder, self).__init__()
        self.z_dim = z_dim

        self.first = nn.Linear(z_dim, 4096)

        self.block = nn.Sequential(
            ConvTBatchRelu(64, 64, kernel_size=2, stride=2),
            ConvTBatchRelu(64, 64, kernel_size=2, stride=2),
            ConvTBatchRelu(64, 32, kernel_size=2, stride=2)
        )

        self.last = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.first(x)
        x = torch.reshape(x, (-1, 64, 8, 8))
        x = self.block(x)
        x = self.last(x)

        return x


class VAE(nn.Module):
    def __init__(self, z_dim=200):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.z_dim = z_dim

    def forward(self, img):
        z, mu, log_var = self.encoder(img)
        out = self.decoder(z)

        return out, mu, log_var


if __name__ == '__main__':

    img = torch.randn(4, 3, 128, 128)

    net = Encoder()
    out, mu, log = net(img)
    print(out.size())
    print(mu.size())
    print(log.size())

    net = Decoder()

    out = net(out)

    print(out.size())
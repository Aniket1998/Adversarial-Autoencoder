import torch
import torch.nn as nn
from math import ceil, log2

__all__ = ['Encoder', 'Decoder', 'Discriminator']


class Encoder(nn.Module):
    def __init__(self, input_size=32, input_channels=3, step_channels=64, z_dim=10, nonlinearity=None):
        super(Encoder, self).__init__()
        if input_size < 16 or ceil(log2(input_size)) != log2(input_size):
            raise Exception('Input Size must be at least 16*16 and the dimension must be an exact power of 2')
        step_down = input_size.bit_length() - 4  # Fast computation of log2(x) - 3
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        d = step_channels
        model = [nn.Sequential(
            nn.Conv2d(input_channels, d, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d), nl)]
        for i in range(step_down):
            model.append(nn.Sequential(
                nn.Conv2d(d, d * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 2), nl))
            d *= 2
        model.append(nn.Sequential(
            nn.Conv2d(d, d, 4, 1, 0, bias=False),
            nn.BatchNorm2d(d), nl))
        self.model = nn.Sequential(*model)
        self.testmodel = model
        self.fc_mean = nn.Sequential(
                nn.Linear(d, z_dim, bias=False),
                nn.BatchNorm1d(z_dim), nl)
        self.fc_logvar = nn.Sequential(
                nn.Linear(d, z_dim, bias=False),
                nn.BatchNorm1d(z_dim), nl)

    def reparametrize(self, mu, logvar):
        if self.training:
            eps = torch.randn_like(mu)
            std = torch.exp(0.5 * logvar)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, x.size(1))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return self.reparametrize(mean, logvar), mean, logvar


class Decoder(nn.Module):
    def __init__(self, target_size=32, target_channels=3, step_channels=64, z_dim=10, nonlinearity=None, last_nonlinearity=None):
        super(Decoder, self).__init__()
        if target_size < 16 or ceil(log2(target_size)) != log2(target_size):
            raise Exception('Target Image Size must be at least 16*16 and the dimension must be an exact power of 2')
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        last_nl = nn.Tanh() if last_nonlinearity is None else last_nonlinearity
        step_up = target_size.bit_length() - 4
        d = step_channels * (2 ** step_up)
        self.fc = nn.Sequential(
                nn.Linear(z_dim, d),
                nn.BatchNorm1d(d), nl)
        model = [nn.Sequential(
            nn.ConvTranspose2d(d, d, 4, 1, 0, bias=False),
            nn.BatchNorm2d(d), nl)]
        for i in range(step_up):
            model.append(nn.Sequential(
                nn.ConvTranspose2d(d, d // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d // 2), nl))
            d = d // 2
        model.append(nn.Sequential(
            nn.ConvTranspose2d(d, target_channels, 4, 2, 1, bias=True), last_nl))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, x.size(1), 1, 1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, z_dim=10, nonlinearity=None, last_nonlinearity=None):
        super(Discriminator, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        last_nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        self.model = nn.Sequential(
                nn.Linear(z_dim, 512, bias=True), nl,
                nn.Linear(512, 256, bias=False),
                nn.BatchNorm1d(256), nl,
                nn.Linear(256, 128, bias=False),
                nn.BatchNorm1d(128), nl,
                nn.Linear(128, 1, bias=True), last_nl)

    def forward(self, x):
        return self.model(x)

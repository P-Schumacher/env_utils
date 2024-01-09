import torch
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_dims, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dims, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class Decoder(nn.Module):
    def __init__(self, output_dims, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, output_dims)
        self.output_dims = output_dims

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, self.output_dims))


class CVAE(nn.Module):
    def __init__(self, observation, latent_dims, condition_dim=1):
        input_shape = observation.shape[-1] + condition_dim
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_shape, latent_dims)
        self.decoder = Decoder(input_shape, latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train(autoencoder, data, epochs=20):
    device = 'cpu'
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        print(f'{epoch=} of {epochs}')
        for x, y in zip(data[0], data[1]):
            x = x.to(device)  # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat) ** 2).sum()
            loss.backward()
            opt.step()
    return autoencoder

if __name__ == '__main__':
    observation = torch.randn((1000, 1, 200), device='cpu')
    data = [observation, observation]
    generator = CVAE(observation[0], latent_dims=100, condition_dim=0)
    train(generator, data, epochs=2000000)
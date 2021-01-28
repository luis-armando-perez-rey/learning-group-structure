import torch.nn as nn
import torch
import torch.nn.functional as F

def init_weights(m, gain):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d) | (type(m) == nn.ConvTranspose2d):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)


class Encoder(nn.Module):

    def __init__(self, n_out=4, n_hid=128, weight_scale=5, n_channels = 3):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(n_channels, 16, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(16, 32, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU())

        self.output = nn.Sequential(nn.Linear(32 * 7 * 7, n_hid),
                                    nn.ReLU(),
                                    nn.Linear(n_hid, n_out))

        self.conv.apply(lambda x: init_weights(x, weight_scale))
        self.output.apply(lambda x: init_weights(x, weight_scale))

    def forward(self, obs):
        if len(obs.shape) != 4:
            obs = obs.unsqueeze(0)
        obs = obs.permute(0, 3, 1, 2)
        obs = obs / 255
        obs = self.conv(obs)
        obs = obs.contiguous().view(obs.size(0), -1)
        return F.normalize(self.output(obs)).squeeze()


class Decoder(nn.Module):

    def __init__(self, n_in=4, n_hid=128, weight_scale=5, n_channels = 3):
        super().__init__()

        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, 32 * 7 * 7)

        self.conv = nn.Sequential(nn.ConvTranspose2d(32, 32, 3, stride=1),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(32, 16, 4, stride=2),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(16, n_channels, 8, stride=4),
                                  )

        self.conv.apply(lambda x: init_weights(x, weight_scale))
        init_weights(self.fc1, weight_scale)
        init_weights(self.fc2, weight_scale)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 32, 7, 7)
        x = self.conv(x).permute(0, 2, 3, 1)
        return torch.sigmoid(x).squeeze()

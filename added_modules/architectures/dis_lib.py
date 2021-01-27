import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self,
                 z_dim=4,
                 n_channels=3,
                 image_size=(64, 64),
                 conv_hid=32,
                 conv_kernel=(4, 4),
                 conv_stride=(2, 2),):
        super().__init__()

        self.conv1 = nn.Conv2d(n_channels, conv_hid, conv_kernel, stride=conv_stride, padding=1)
        self.conv2 = nn.Conv2d(conv_hid, conv_hid, conv_kernel, stride=conv_stride, padding=1)
        self.conv3 = nn.Conv2d(conv_hid, conv_hid*2, conv_kernel, stride=conv_stride, padding=1)
        self.conv4 = nn.Conv2d(conv_hid*2, conv_hid*2, conv_kernel, stride=conv_stride, padding=1)

        self.fc1 = nn.Linear(64*4*4, 256)
        self.fc2 = nn.Linear(256, z_dim)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x).squeeze()


class Decoder(nn.Module):

    def __init__(self,
                 n_in=4,
                 conv_kernel=(4, 4),
                 conv_stride=(2, 2),
                 n_channels=3
                 ):
        super().__init__()

        self.fc1 = nn.Linear(n_in, 256)
        self.fc2 = nn.Linear(256, 1024)

        self.deconv1 = nn.ConvTranspose2d(64, 64, conv_kernel, stride=conv_stride, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, conv_kernel, stride=conv_stride, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, conv_kernel, stride=conv_stride, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, n_channels, conv_kernel, stride=conv_stride, padding=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.reshape(x, (-1, 64, 4, 4))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))

        return x.squeeze(dim = 0)

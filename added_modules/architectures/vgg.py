import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


def calculate_pad_same(image_size, kernel_size, stride):
    """
    Calculates the padding to get the "same" size as in Tensorflow
    Only works for images were filter covers the complete image in the convolution
    """
    print((image_size[0] - (kernel_size[0] - 1) - 1) % stride[0] == 0)
    print("Image size", image_size)
    print("Kernel size", kernel_size)
    print("Stride size", stride)
    assert (image_size[0] - (kernel_size[0] - 1) - 1) % stride[
        0] == 0, "Image can't be convoluted on the height exactly"
    assert (image_size[1] - (kernel_size[1] - 1) - 1) % stride[1] == 0, "Image can't be convoluted on the width exactly"

    pad = tuple(
        [(image_size[num] * (stride[num] - 1) - stride[num] + kernel_size[num]) // 2 for num in range(len(image_size))])
    return pad


class Encoder(nn.Module):

    def __init__(self,
                 z_dim=4,
                 n_channels=3,
                 image_size=(64, 64),
                 conv_hid=64,
                 conv_kernel=(3, 3),
                 conv_stride=(1, 1),
                 maxpool_kernel=(2, 2)):
        super().__init__()

        conv_pad = calculate_pad_same(image_size, conv_kernel, conv_stride)
        maxpool_pad = calculate_pad_same(image_size, maxpool_kernel, maxpool_kernel)
        self.maxpool_pad = [maxpool_pad[1], maxpool_pad[1], maxpool_pad[0], maxpool_pad[0]]
        self.conv1 = nn.Conv2d(n_channels, conv_hid, conv_kernel, stride=conv_stride, padding=conv_pad)
        self.maxpool1 = nn.MaxPool2d(maxpool_kernel, None)
        self.conv2 = nn.Conv2d(conv_hid, conv_hid, conv_kernel, stride=conv_stride, padding=conv_pad)
        self.maxpool2 = nn.MaxPool2d(maxpool_kernel, None)
        self.conv3 = nn.Conv2d(conv_hid, conv_hid, conv_kernel, stride=conv_stride, padding=conv_pad)
        self.maxpool3 = nn.MaxPool2d(maxpool_kernel, None)

        final_size = np.product((conv_hid, image_size[0], image_size[1]))
        self.fc1 = nn.Linear(final_size, conv_hid)
        self.fc2 = nn.Linear(conv_hid, z_dim)

    def forward(self, x):
        # x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x).squeeze()


class Decoder(nn.Module):

    def __init__(self, image_size,
                 n_in=4,
                 conv_hid=64,
                 conv_kernel=(3, 3),
                 conv_stride=(1, 1),
                 n_channels=3
                 ):
        super().__init__()

        convdim = (conv_hid, image_size[0], image_size[1])
        self.fc1 = nn.Linear(n_in, conv_hid)
        self.fc2 = nn.Linear(conv_hid, np.product(convdim))

        conv_pad = calculate_pad_same(image_size, conv_kernel, conv_stride)
        self.up1 = nn.Upsample(scale_factor=1)

        self.conv1 = nn.Conv2d(conv_hid, conv_hid, conv_kernel, stride=conv_stride, padding=conv_pad)
        self.conv2 = nn.Conv2d(conv_hid, conv_hid, conv_kernel, stride=conv_stride, padding=conv_pad)
        self.conv3 = nn.Conv2d(conv_hid, n_channels, conv_kernel, stride=conv_stride, padding=conv_pad)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.reshape(x, (-1, 64, 64, 64))
        x = self.up1(x)
        x = F.relu(self.conv1(x))
        x = self.up1(x)
        x = F.relu(self.conv2(x))
        x = self.up1(x)
        x = self.conv3(x)

        return torch.sigmoid(x).squeeze(dim = 0)

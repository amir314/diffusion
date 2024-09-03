import torch
from torch import nn


class Downsampler(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               padding="same")
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding="same")
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        self.copy = x.clone() # save value for skip connection
        x = self.max_pool(x)
        return x


class Upsampler(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               padding="same")
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding="same")
        self.up_conv = nn.ConvTranspose2d(out_channels,
                                          out_channels,
                                          kernel_size=2,
                                          stride=2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.up_conv(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self,
                 in_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               2*in_channels,
                               kernel_size=3,
                               padding="same")
        self.conv2 = nn.Conv2d(2*in_channels,
                               2*in_channels,
                               kernel_size=3,
                               padding="same")
        self.up_conv = nn.ConvTranspose2d(2*in_channels,
                                          in_channels,
                                          kernel_size=2,
                                          stride=2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.up_conv(x)
        return x


class Output(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               hidden_channels,
                               kernel_size=3,
                               padding="same")
        self.conv2 = nn.Conv2d(hidden_channels,
                               hidden_channels,
                               kernel_size=3,
                               padding="same")
        self.conv3 = nn.Conv2d(hidden_channels,
                               out_channels,
                               kernel_size=1,
                               padding="same")
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class UNET(nn.Module):
    def __init__(self,
                 in_channels: int) -> None:
        """
        Initializes a U-Net architecture. See arXiv:1505.04597.
        """

        super().__init__()

        # Downsamplers
        self.down1 = Downsampler(in_channels, 64)
        self.down2 = Downsampler(64, 128)
        self.down3 = Downsampler(128, 256)

        # Upsamplers
        self.up1 = Upsampler(512, 128)
        self.up2 = Upsampler(256, 64)

        # Bottleneck
        self.bottle_neck = Bottleneck(256)

        # Output layer
        self.final = Output(128, 64, in_channels)

    def forward(self, x) -> torch.Tensor:
        # Downsample block
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        # Bottleneck
        x = self.bottle_neck(x) # shape same as self.down3(x)

        # Upsample block
        x = torch.concat((x, self.down3.copy), dim=1)
        x = self.up1(x)
        x = torch.concat((x, self.down2.copy), dim=1)
        x = self.up2(x)
        x = torch.concat((x, self.down1.copy), dim=1)

        # Final output layer
        x = self.final(x)

        return x

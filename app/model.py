import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.PReLU()

        self.conv2 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x + residual


class SuperResolutionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: 1-channel grayscale image
        self.head = nn.Sequential(
            nn.Conv2d(
                1,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.PReLU()
        )

        # 8 residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock() for _ in range(8)]
        )

        self.mid_conv = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            padding=1
        )

        # 2x upscaling
        self.upsample = nn.Sequential(
            nn.Conv2d(
                64,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        # Output: 1-channel grayscale image
        self.tail = nn.Sequential(
            nn.Conv2d(
                64,
                1,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):
        x = self.head(x)

        for block in self.res_blocks:
            x = block(x)

        x = self.mid_conv(x) + x

        x = self.upsample(x)

        x = self.tail(x)

        return x


def load_model(model_path: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SuperResolutionModel()

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, device

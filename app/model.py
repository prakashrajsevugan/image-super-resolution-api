import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.PReLU(
            num_parameters=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
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
            nn.PReLU(
                num_parameters=1
            )
        )

        # 8 residual blocks
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock()
                for _ in range(8)
            ]
        )

        # Feature refinement
        self.mid_conv = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            padding=1
        )

        # 2x upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(
                64,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.PixelShuffle(2),
            nn.PReLU(
                num_parameters=1
            )
        )

        # Output residual
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

        # Global residual connection
        x = self.mid_conv(x) + x

        x = self.upsample(x)

        x = self.tail(x)

        return x


def load_model(model_path):
    device = torch.device("cpu")

    print(f"Loading model: {model_path}")

    model = SuperResolutionModel()

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("Model loaded successfully.")
    print(f"Device: {device}")

    return model, device
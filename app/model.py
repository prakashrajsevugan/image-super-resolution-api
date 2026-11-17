import torch
import torch.nn as nn


# ============================================================
# Residual Block
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # The checkpoint contains:
        # res_blocks.X.relu.weight
        # Therefore this must be PReLU, not ReLU.
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

        # Residual connection
        x = x + residual

        return x


# ============================================================
# Super Resolution Model
# ============================================================

class SuperResolutionModel(nn.Module):

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Input
        #
        # 1 channel grayscale
        # ----------------------------------------------------

        self.head = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.PReLU(
                num_parameters=1
            )
        )


        # ----------------------------------------------------
        # 8 Residual Blocks
        # ----------------------------------------------------

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock()
                for _ in range(8)
            ]
        )


        # ----------------------------------------------------
        # Middle convolution
        # ----------------------------------------------------

        self.mid_conv = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )


        # ----------------------------------------------------
        # 2x Upsampling
        #
        # 64 -> 256
        # PixelShuffle(2)
        # 256 -> 64
        # ----------------------------------------------------

        self.upsample = nn.Sequential(

            nn.Conv2d(
                in_channels=64,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(
                upscale_factor=2
            ),

            nn.PReLU(
                num_parameters=1
            )
        )


        # ----------------------------------------------------
        # Output
        #
        # 64 -> 1 channel
        # ----------------------------------------------------

        self.tail = nn.Sequential(

            nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=3,
                padding=1
            )
        )


    def forward(self, x):

        # Head
        x = self.head(x)


        # Save feature before residual blocks
        # The checkpoint architecture uses residual blocks
        # followed by a middle convolution and skip connection.
        for block in self.res_blocks:
            x = block(x)


        # Middle convolution + global residual connection
        x = self.mid_conv(x) + x


        # 2x upsampling
        x = self.upsample(x)


        # Output
        x = self.tail(x)


        return x


# ============================================================
# Load Model
# ============================================================

def load_model(model_path: str):

    # Render will normally use CPU.
    # If CUDA is available, this also supports GPU.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


    print(
        f"Loading model from: {model_path}"
    )

    print(
        f"Using device: {device}"
    )


    # Create architecture
    model = SuperResolutionModel()


    # Load checkpoint
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )


    # Load weights
    model.load_state_dict(
        checkpoint
    )


    # Move model to device
    model.to(device)


    # Evaluation mode
    model.eval()


    print(
        "Model loaded successfully."
    )


    return model, device
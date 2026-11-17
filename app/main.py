import io
import os

import torch
import numpy as np

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from app.model import load_model


app = FastAPI(
    title="Image Super Resolution API",
    description="2x image super-resolution using a PyTorch model",
    version="1.0.0"
)


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/best_model.pth"
)


# Load model once when the server starts
model, device = load_model(MODEL_PATH)


@app.get("/")
def root():
    return {
        "message": "Image Super Resolution API",
        "model": "PyTorch Super Resolution",
        "scale": "2x",
        "device": str(device)
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(device)
    }


@app.post("/upscale")
async def upscale_image(
    file: UploadFile = File(...)
):

    # Check file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    try:

        # Read uploaded image
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        )

        # Convert to grayscale because
        # the model expects 1 channel
        image = image.convert("L")

        # Convert PIL image -> numpy
        image_np = np.array(
            image,
            dtype=np.float32
        )

        # Normalize 0-255 -> 0-1
        image_np /= 255.0

        # H,W -> 1,1,H,W
        tensor = torch.from_numpy(
            image_np
        ).unsqueeze(0).unsqueeze(0)

        tensor = tensor.to(device)

        # Run model
        with torch.inference_mode():

            output = model(tensor)

        # Move CPU
        output = output.squeeze(0).squeeze(0)

        output = output.cpu().numpy()

        # Clamp output
        output = np.clip(
            output,
            0.0,
            1.0
        )

        # Convert 0-1 -> 0-255
        output = (
            output * 255.0
        ).round().astype(np.uint8)

        # numpy -> PIL
        output_image = Image.fromarray(
            output,
            mode="L"
        )

        # Save to memory
        buffer = io.BytesIO()

        output_image.save(
            buffer,
            format="PNG"
        )

        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={
                "Content-Disposition":
                    'attachment; filename="upscaled.png"'
            }
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


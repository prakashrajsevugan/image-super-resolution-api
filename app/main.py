import io
import os

import numpy as np
import torch

from PIL import Image

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import StreamingResponse

from app.model import load_model


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Image Super Resolution API",
    description="2x Image Super Resolution using PyTorch",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

# Allow requests from your HTML frontend.
#
# For testing, "*" is fine.
# Later, replace "*" with your actual frontend URL.

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# Model Configuration
# ============================================================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/best_model.pth"
)


# ============================================================
# Load Model Once
# ============================================================

try:

    model, device = load_model(
        MODEL_PATH
    )

except Exception as e:

    print(
        f"ERROR: Could not load model: {e}"
    )

    raise


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Image Super Resolution API",

        "model": "PyTorch Super Resolution",

        "scale": "2x",

        "device": str(device),

        "status": "running"
    }


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",

        "model_loaded": True,

        "device": str(device)
    }


# ============================================================
# Image Upscaling Endpoint
# ============================================================

@app.post("/upscale")
async def upscale_image(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="Could not determine file type."
        )


    if not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )


    try:

        # ----------------------------------------------------
        # Read uploaded image
        # ----------------------------------------------------

        contents = await file.read()


        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )


        print(
            f"Received file: {file.filename}"
        )

        print(
            f"Content type: {file.content_type}"
        )

        print(
            f"File size: {len(contents)} bytes"
        )


        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        image = Image.open(
            io.BytesIO(contents)
        )


        print(
            f"Original image size: {image.size}"
        )


        # ----------------------------------------------------
        # Convert to grayscale
        #
        # Your model expects:
        #
        # [batch, 1, height, width]
        #
        # ----------------------------------------------------

        image = image.convert("L")


        # ----------------------------------------------------
        # PIL -> NumPy
        # ----------------------------------------------------

        image_np = np.asarray(
            image,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Normalize
        #
        # 0-255 -> 0-1
        # ----------------------------------------------------

        image_np = image_np / 255.0


        # ----------------------------------------------------
        # NumPy -> PyTorch
        #
        # Current:
        #
        # H x W
        #
        # Required:
        #
        # 1 x 1 x H x W
        # ----------------------------------------------------

        tensor = torch.from_numpy(
            image_np
        )


        tensor = tensor.unsqueeze(0)


        tensor = tensor.unsqueeze(0)


        # Move to CPU/GPU
        tensor = tensor.to(device)


        print(
            f"Model input shape: {tuple(tensor.shape)}"
        )


        # ----------------------------------------------------
        # Run model
        # ----------------------------------------------------

        with torch.inference_mode():

            output = model(
                tensor
            )


        print(
            f"Model output shape: {tuple(output.shape)}"
        )


        # ----------------------------------------------------
        # Remove batch and channel dimensions
        #
        # 1 x 1 x H x W
        #
        # ->
        #
        # H x W
        # ----------------------------------------------------

        output = output.squeeze(
            0
        ).squeeze(
            0
        )


        # ----------------------------------------------------
        # Move to CPU
        # ----------------------------------------------------

        output = output.cpu().numpy()


        # ----------------------------------------------------
        # Clamp values
        # ----------------------------------------------------

        output = np.clip(
            output,
            0.0,
            1.0
        )


        # ----------------------------------------------------
        # Convert
        #
        # 0-1 -> 0-255
        # ----------------------------------------------------

        output = (
            output * 255.0
        )


        output = np.round(
            output
        ).astype(
            np.uint8
        )


        # ----------------------------------------------------
        # NumPy -> PIL
        # ----------------------------------------------------

        output_image = Image.fromarray(
            output,
            mode="L"
        )


        print(
            f"Output image size: {output_image.size}"
        )


        # ----------------------------------------------------
        # Save image into memory
        # ----------------------------------------------------

        buffer = io.BytesIO()


        output_image.save(
            buffer,
            format="PNG"
        )


        # Move pointer to beginning
        buffer.seek(0)


        # ----------------------------------------------------
        # Return image
        # ----------------------------------------------------

        return StreamingResponse(

            buffer,

            media_type="image/png",

            headers={
                "Content-Disposition":
                    'inline; filename="upscaled.png"'
            }
        )


    except HTTPException:

        raise


    except Exception as e:

        print(
            "===================================="
        )

        print(
            "IMAGE PROCESSING ERROR"
        )

        print(
            str(e)
        )

        print(
            "===================================="
        )


        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {str(e)}"
        )